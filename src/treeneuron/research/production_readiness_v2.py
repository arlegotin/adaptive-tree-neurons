"""Actual final-source readiness; software/review records alone grant no gate."""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from .deployment_worker_v2 import _ROOT,_source,_digest,_json
from .protected_journal_v2 import _reference,_write_new
from .native_verification_v2 import verification_identity
from treeneuron.config import canonical_hash,canonical_json


def _commit(value):return type(value) is str and re.fullmatch('[0-9a-f]{40}',value) is not None


def _metadata_reference(value):
    return (type(value) is dict and set(value)=={'path','sha256'} and type(value['path']) is str
        and Path(value['path']).is_absolute() and Path(value['path']).resolve()==Path(value['path']) and _digest(value['sha256']))


def _same_execution_snapshot(saved,current):
    fields={'pinned_commit','checkout_commit','files','execution_sha256'}
    return (type(saved) is dict and type(current) is dict and set(saved)==set(current)==fields
        and _commit(saved['checkout_commit']) and _commit(current['checkout_commit'])
        and all(saved[k]==current[k] for k in ('pinned_commit','files','execution_sha256')))


def _inspect_review_metadata(value,snapshot,*,source_sha256,changed_execution_files):
    """Supplied metadata validation only, never a native/source capability."""
    fields={'schema','reviewed_commit','final_commit','final_execution_sha256','source_sha256',
        'review_package_reference','review_response_reference','reviewer','findings'}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-production-branch-review'
            or not _commit(value['reviewed_commit']) or value['final_commit']!=snapshot['pinned_commit']
            or value['final_execution_sha256']!=snapshot['execution_sha256'] or value['source_sha256']!=source_sha256
            or any(not _metadata_reference(value[k]) for k in ('review_package_reference','review_response_reference'))
            or type(value['reviewer']) is not dict or set(value['reviewer'])!={'agent_id','model','reasoning_effort'}
            or any(type(s) is not str or not s.strip() for s in value['reviewer'].values()) or type(value['findings']) is not list
            or type(changed_execution_files) is not list or any(type(p) is not str for p in changed_execution_files)
            or len(set(changed_execution_files))!=len(changed_execution_files)):
        raise ValueError('Exact fresh branch-review/final scientific binding required')
    fixed=set();seen=set();deferred=[]
    for finding in value['findings']:
        fields={'finding_id','severity','resolution','changed_files','red_reference','green_reference'}
        if (type(finding) is not dict or set(finding)!=fields or type(finding['finding_id']) is not str
                or not finding['finding_id'] or finding['finding_id'] in seen or finding['severity'] not in ('Critical','Important','Minor')
                or type(finding['changed_files']) is not list or any(type(p) is not str for p in finding['changed_files'])
                or len(set(finding['changed_files']))!=len(finding['changed_files'])):
            raise ValueError('Exact unique branch-review finding dispositions required')
        seen.add(finding['finding_id'])
        if finding['severity']=='Minor':
            if (finding['resolution']!='deferred' or finding['changed_files'] or finding['red_reference'] is not None
                    or finding['green_reference'] is not None):raise ValueError('Deferred minors cannot hide scientific changes')
            deferred.append(finding['finding_id'])
        else:
            if finding['resolution']!='fixed' or any(not _metadata_reference(finding[k]) for k in ('red_reference','green_reference')):
                raise ValueError('No unresolved material finding may enter production readiness')
            fixed.update(finding['changed_files'])
    if fixed!=set(changed_execution_files):raise ValueError('Every reviewed-to-final scientific change needs an explicit verified fix disposition')
    return dict(reviewed_commit=value['reviewed_commit'],final_commit=value['final_commit'],
        changed_execution_files=sorted(fixed),unresolved_material_findings=0,deferred_minor_ids=deferred,
        production_source_ready=False,protected_access_authorized=False,
        scope='Review metadata arithmetic only; actual source/native/software/reference checks remain separate')


def _review(reference,snapshot):
    _,raw=_reference(reference);value=_json(raw)
    from treeneuron.protocol import _git,_execution
    try:
        reviewed=value.get('reviewed_commit') if type(value) is dict else None
        if not _commit(reviewed):raise ValueError('Full reviewed commit required')
        _git(_ROOT,'merge-base','--is-ancestor',reviewed,snapshot['pinned_commit'])
        changed=sorted(p for p in _git(_ROOT,'diff','--name-only',reviewed,snapshot['pinned_commit']).decode().splitlines() if _execution(p))
    except subprocess.CalledProcessError as error:raise ValueError('Review must belong to the final scientific branch ancestry') from error
    checked=_inspect_review_metadata(value,snapshot,source_sha256=_source(),changed_execution_files=changed)
    for key in ('review_package_reference','review_response_reference'):_reference(value[key])
    for finding in value['findings']:
        if finding['resolution']=='fixed':
            for key in ('red_reference','green_reference'):_reference(finding[key])
    if _reference(reference)[1]!=raw:raise ValueError('Branch review changed during inspection')
    return checked,value


def _snapshot(pinned,saved=None):
    from treeneuron.protocol import execution_snapshot,_git
    if not _commit(pinned):raise ValueError('Full committed scientific revision required')
    current=execution_snapshot(_ROOT,pinned)
    if saved is None:return current
    if not _same_execution_snapshot(saved,current):raise ValueError('Readiness scientific package differs from current pinned execution files')
    try:checkout=_git(_ROOT,'rev-parse',saved['checkout_commit']+'^{commit}').decode().strip()
    except subprocess.CalledProcessError as error:raise ValueError('Saved readiness checkout commit is unavailable') from error
    if checkout!=saved['checkout_commit']:raise ValueError('Saved readiness checkout identity differs')
    return copy.deepcopy(saved)


def _assemble(registry,*,qualification_reference,verification_references,review_reference,pinned_commit,created_utc_ns,saved_snapshot=None):
    from .production_verification_v2 import production_verification_profiles,load_production_verification
    if (type(verification_references) is not dict or set(verification_references)!=set(production_verification_profiles())
            or type(created_utc_ns) is not int or created_utc_ns<=0):raise ValueError('All fixed production verification profiles required')
    snapshot=_snapshot(pinned_commit,saved_snapshot);identity=verification_identity()
    if registry['identity']!=identity:raise ValueError('Actual native qualification must bind this exact source/test/lock revision')
    proofs={name:load_production_verification(ref,expected_profile=name) for name,ref in verification_references.items()}
    if any(p['identity']!=identity or p['software_verified'] is not True for p in proofs.values()):
        raise ValueError('Final production fixture proof source differs')
    checked,review=_review(review_reference,snapshot)
    payload=dict(schema='research/v2-production-readiness-payload',created_utc_ns=created_utc_ns,
        source_sha256=identity['scientific_source_sha256'],qualification_reference=copy.deepcopy(qualification_reference),
        native_registry_identity_sha256=canonical_hash(identity),execution_snapshot=snapshot,
        verification_references=copy.deepcopy(verification_references),verification_case_counts={k:p['test_cases'] for k,p in proofs.items()},
        review_reference=copy.deepcopy(review_reference),review_binding=checked,
        production_source_ready=True,protected_access_authorized=False,T_accessed=False,
        scope='Actual native qualification, committed execution package, fixed production fixtures and fresh branch review bound; this record alone never opens protected data')
    if verification_identity()!=identity or not _same_execution_snapshot(snapshot,_snapshot(pinned_commit)):
        raise ValueError('Scientific source/test/lock/package changed during readiness assembly')
    return payload,review


def _publication_location(output,registry,qualification_reference,verification_references,review_reference,review):
    from .native_qualification_registry_v2 import native_registry_storage_roots
    from .primary_program_v2 import _check_program_location
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=qualification_reference)
    files=[review_reference['path'],review['review_package_reference']['path'],review['review_response_reference']['path']]
    files.extend(f[k]['path'] for f in review['findings'] if f['resolution']=='fixed' for k in ('red_reference','green_reference'))
    path=_check_program_location(output,working_roots=roots['working_roots'],
        proof_roots=[*roots['proof_roots'],*(ref['root'] for ref in verification_references.values())],input_files=files)
    if path.exists() or path.suffix!='.json':raise ValueError('Fresh immutable readiness JSON required')
    return path


@dataclass(frozen=True)
class ProductionReadiness:
    reference:dict
    _payload_json:str

    @property
    def payload(self):return json.loads(self._payload_json)


def seal_production_readiness(*,qualification_reference,verification_references,review_reference,pinned_commit,output):
    from .native_qualification_registry_v2 import load_native_qualification_registry
    registry=load_native_qualification_registry(qualification_reference)  # First actual gate.
    payload,review=_assemble(registry,qualification_reference=qualification_reference,verification_references=verification_references,
        review_reference=review_reference,pinned_commit=pinned_commit,created_utc_ns=time.time_ns())
    path=_publication_location(output,registry,qualification_reference,verification_references,review_reference,review)
    if load_native_qualification_registry(qualification_reference)['identity']!=registry['identity']:
        raise ValueError('Actual native gate changed before readiness publication')
    if not _same_execution_snapshot(payload['execution_snapshot'],_snapshot(pinned_commit)):
        raise ValueError('Scientific package changed before readiness publication')
    _write_new(path,dict(schema='research/v2-production-readiness',payload=payload,payload_hash=canonical_hash(payload)))
    reference=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return ProductionReadiness(reference,canonical_json(payload))


def load_production_readiness(reference,*,qualification_reference):
    from .native_qualification_registry_v2 import load_native_qualification_registry
    registry=load_native_qualification_registry(qualification_reference)  # Before any saved payload.
    _,raw=_reference(reference);value=_json(raw)
    if (type(value) is not dict or set(value)!={'schema','payload','payload_hash'} or value['schema']!='research/v2-production-readiness'
            or type(value['payload']) is not dict or canonical_hash(value['payload'])!=value['payload_hash']):
        raise ValueError('Exact immutable production readiness envelope required')
    saved=value['payload']
    try:
        payload,_=_assemble(registry,qualification_reference=qualification_reference,verification_references=saved['verification_references'],
            review_reference=saved['review_reference'],pinned_commit=saved['execution_snapshot']['pinned_commit'],
            created_utc_ns=saved['created_utc_ns'],saved_snapshot=saved['execution_snapshot'])
    except (KeyError,TypeError) as error:raise ValueError('Malformed production readiness payload') from error
    if canonical_json(saved)!=canonical_json(payload) or _reference(reference)[1]!=raw:raise ValueError('Readiness source, native, software or review evidence changed')
    return ProductionReadiness(copy.deepcopy(reference),canonical_json(payload))
