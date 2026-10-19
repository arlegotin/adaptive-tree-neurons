"""Commit fresh seed identities before generation; never authorize T access.

The supplied registry authenticates known seed exclusions only. Complete teacher
and historical-registry qualification remains a separate production freeze gate.
"""
from dataclasses import dataclass,field
import hashlib
import os
from pathlib import Path
import secrets
import time

from treeneuron.config import canonical_hash,canonical_json
from treeneuron.protocol import _execution,_read,_reference
from .ledger import write_new_json
from .protocol_v2 import load_provisional_method,_digest
from .seeds import seed64


_RELATIVE='src/treeneuron/research/confirmation_seeds.py'
_PUBLIC_FIELDS={'study_id','claim','matrix','teacher_count','created_utc_ns',
    'method_file_sha256','method_lock_hash','execution_sha256','code_binding',
    'excluded_registry','private_file_sha256','root_sha256','teacher_seed_set_sha256',
    'protected_access_authorized','teacher_id_exclusions_verified','registry_completeness_qualified'}
_PRIVATE_FIELDS={'schema','root','matrix','teacher_count','method_file_sha256',
    'execution_sha256','excluded_registry_sha256'}


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file(path,expected):
    path=Path(path)
    if not _digest(expected) or path.is_symlink() or not path.is_file() or _sha(path)!=expected:
        raise ValueError('Seed commitment file differs from its trusted byte reference')
    return path


def _code(method,root):
    digest=_sha(__file__)
    if (method.payload['source']['files'].get(_RELATIVE)!=digest
            or _sha(root/_RELATIVE)!=digest):
        raise ValueError('Seed commitment code must match the locked committed source')
    return dict(path=_RELATIVE,sha256=digest)


def _registry(root,reference):
    value=_read(_reference(root,reference))
    if (type(value) is not dict or set(value)!={'schema','teacher_seeds'}
            or value['schema']!='research/v2-teacher-seed-registry'
            or type(value['teacher_seeds']) is not list
            or any(type(v) is not int or not 0<=v<2**63 for v in value['teacher_seeds'])
            or value['teacher_seeds']!=sorted(set(value['teacher_seeds']))):
        raise ValueError('A byte-authenticated sorted unique excluded teacher seed registry is required')
    return set(value['teacher_seeds'])


def _teachers(root,matrix,count,excluded):
    if type(root) is not int or not 0<=root<2**63:
        raise ValueError('Invalid private confirmation seed root')
    values=tuple(seed64(root,matrix,'primary',f'teacher-{i}','teacher') for i in range(count))
    if len(set(values))!=count or set(values)&excluded:
        raise ValueError('Confirmation teacher seed collision with another or an excluded teacher')
    return values


def _root_hash(root):
    return canonical_hash(dict(namespace='research/v2-confirmation-root',root=root))


def _new_output(path,root):
    path=Path(path)
    if path.exists() or path.is_symlink():raise ValueError('Seed output already exists; never regenerate in place')
    if path.resolve().is_relative_to(root) and _execution(path.resolve().relative_to(root).as_posix()):
        raise ValueError('Seed outputs must not change execution inputs')
    return path


@dataclass(frozen=True)
class SeedCommitment:
    path:Path
    file_sha256:str
    root_sha256:str


@dataclass(frozen=True)
class ConfirmationSeedPlan:
    claim:str
    matrix:str
    teacher_seeds:tuple[int,...]=field(repr=False)
    root_sha256:str
    protected_access_authorized:bool=False


def commit_confirmation_seeds(*,method_path,expected_method_sha256,repository_root,
        excluded_registry,teacher_count,public_output,private_output):
    """Draw one fresh root, commit new seed IDs, and leave observations untouched.

    Private publication precedes the public commitment. A failed public write
    retains that private file for diagnosis; it is never silently regenerated.
    """
    root=Path(repository_root).resolve()
    method=load_provisional_method(method_path,expected_file_sha256=expected_method_sha256,repository_root=root)
    if type(teacher_count) is not int or teacher_count not in range(32,129,8):
        raise ValueError('Confirmation teacher count must be on the fixed32..128 grid')
    public=_new_output(public_output,root);private=_new_output(private_output,root)
    if public.resolve()==private.resolve():raise ValueError('Public commitment and private root require distinct paths')
    excluded=_registry(root,excluded_registry);binding=_code(method,root)
    p=method.payload;matrix='C2' if p['method']['claim']=='residual_capacity' else 'C1'
    fresh=secrets.randbits(63);teachers=_teachers(fresh,matrix,teacher_count,excluded)
    # Entropy acquisition is outside our control. Recheck the exact original
    # commitments before publishing a root under possibly changed inputs.
    load_provisional_method(method_path,expected_file_sha256=expected_method_sha256,repository_root=root)
    _registry(root,excluded_registry)
    private_value=dict(schema='research/v2-private-confirmation-root',root=fresh,matrix=matrix,
        teacher_count=teacher_count,method_file_sha256=expected_method_sha256,
        execution_sha256=p['source']['execution_sha256'],excluded_registry_sha256=excluded_registry['sha256'])
    private.parent.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(private,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as stream:
        stream.write(canonical_json(private_value)+'\n');stream.flush();os.fsync(stream.fileno())
    payload=dict(study_id=p['study_id'],claim=p['method']['claim'],matrix=matrix,
        teacher_count=teacher_count,created_utc_ns=time.time_ns(),method_file_sha256=expected_method_sha256,
        method_lock_hash=method.lock_hash,execution_sha256=p['source']['execution_sha256'],code_binding=binding,
        excluded_registry=dict(excluded_registry),private_file_sha256=_sha(private),root_sha256=_root_hash(fresh),
        teacher_seed_set_sha256=canonical_hash(dict(teacher_seeds=list(teachers))),
        protected_access_authorized=False,teacher_id_exclusions_verified=False,registry_completeness_qualified=False)
    value=dict(schema='research/v2-confirmation-seed-commitment',payload=payload,commitment_hash=canonical_hash(payload))
    write_new_json(public,value)
    return SeedCommitment(public,_sha(public),payload['root_sha256'])


def load_confirmation_seeds(path,*,expected_file_sha256,private_path,method_path,
        expected_method_sha256,repository_root):
    """Authenticate committed seed lineage without granting any data capability."""
    path=_file(path,expected_file_sha256);value=_read(path)
    if (type(value) is not dict or set(value)!={'schema','payload','commitment_hash'}
            or value['schema']!='research/v2-confirmation-seed-commitment'
            or type(value['payload']) is not dict or set(value['payload'])!=_PUBLIC_FIELDS
            or canonical_hash(value['payload'])!=value['commitment_hash']):
        raise ValueError('Invalid seed commitment envelope')
    p=value['payload'];root=Path(repository_root).resolve()
    method=load_provisional_method(method_path,expected_file_sha256=expected_method_sha256,repository_root=root)
    m=method.payload;matrix='C2' if m['method']['claim']=='residual_capacity' else 'C1'
    if (p['study_id']!=m['study_id'] or p['claim']!=m['method']['claim'] or p['matrix']!=matrix
            or type(p['teacher_count']) is not int or p['teacher_count'] not in range(32,129,8)
            or type(p['created_utc_ns']) is not int or p['created_utc_ns']<=0
            or p['method_file_sha256']!=expected_method_sha256 or p['method_lock_hash']!=method.lock_hash
            or p['execution_sha256']!=m['source']['execution_sha256'] or p['code_binding']!=_code(method,root)
            or any(p[k] is not False for k in ('protected_access_authorized',
                'teacher_id_exclusions_verified','registry_completeness_qualified'))):
        raise ValueError('Seed commitment method/source/phase changed')
    excluded=_registry(root,p['excluded_registry'])
    private=_file(private_path,p['private_file_sha256'])
    if private.stat().st_mode&0o077:raise ValueError('Private confirmation root must retain owner-only permissions')
    v=_read(private)
    if (type(v) is not dict or set(v)!=_PRIVATE_FIELDS or v['schema']!='research/v2-private-confirmation-root'
            or v['matrix']!=matrix or v['teacher_count']!=p['teacher_count']
            or v['method_file_sha256']!=expected_method_sha256 or v['execution_sha256']!=p['execution_sha256']
            or v['excluded_registry_sha256']!=p['excluded_registry']['sha256']):
        raise ValueError('Private root lineage differs from the public commitment')
    teachers=_teachers(v['root'],matrix,p['teacher_count'],excluded)
    if (_root_hash(v['root'])!=p['root_sha256']
            or canonical_hash(dict(teacher_seeds=list(teachers)))!=p['teacher_seed_set_sha256']):
        raise ValueError('Committed confirmation root or teacher seed identities differ')
    return ConfirmationSeedPlan(p['claim'],matrix,teachers,p['root_sha256'])
