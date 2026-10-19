"""Expand a committed private root into identities only; generate no observations."""
from dataclasses import dataclass,field
import hashlib
import json
from pathlib import Path

from treeneuron.config import canonical_hash,canonical_json
from treeneuron.protocol import _read
from .confirmation_seeds import load_confirmation_seeds,_file
from .protocol_v2 import load_provisional_method
from .seeds import seed64,seed32


_RELATIVE='src/treeneuron/research/confirmation_namespaces.py'


@dataclass(frozen=True)
class ConfirmationNamespacePlan:
    claim:str
    matrix:str
    teacher_count:int
    root_sha256:str
    namespace_sha256:str
    _payload_json:str=field(repr=False)
    protected_access_authorized:bool=False
    observation_generation_authorized:bool=False
    registry_completeness_qualified:bool=False
    teacher_id_exclusions_verified:bool=False

    @property
    def teachers(self):return json.loads(self._payload_json)['teachers']


def expand_confirmation_namespaces(path,*,expected_file_sha256,private_path,method_path,
        expected_method_sha256,repository_root):
    """Derive paired FGVT/R2 seed records; never draw entropy or generate data.

    This object is metadata for the trusted coordinator. Fitting workers must
    receive only their declared subset; it is not a protected-data capability.
    """
    arguments=dict(path=path,expected_file_sha256=expected_file_sha256,private_path=private_path,
        method_path=method_path,expected_method_sha256=expected_method_sha256,repository_root=repository_root)
    plan=load_confirmation_seeds(**arguments)
    root=Path(repository_root).resolve()
    method=load_provisional_method(method_path,expected_file_sha256=expected_method_sha256,repository_root=root)
    code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (method.payload['source']['files'].get(_RELATIVE)!=code
            or hashlib.sha256((root/_RELATIVE).read_bytes()).hexdigest()!=code):
        raise ValueError('Namespace expansion code differs from the committed chosen method')
    public=_read(_file(path,expected_file_sha256))['payload']
    private=_read(_file(private_path,public['private_file_sha256']))
    seed_root=private['root'];rows=[];independent={};used32=set()
    def reserve(value,label):
        if type(value) is not int or not 0<=value<2**63:raise ValueError('Invalid derived63-bit identity')
        if value in independent:raise ValueError(f'Independent seed namespace collision: {independent[value]} and {label}')
        independent[value]=label
        return value
    # Reserve every teacher first so a later observation cannot collide with a
    # teacher that has not yet been visited by this expansion loop.
    for i,value in enumerate(plan.teacher_seeds):reserve(value,f'teacher-{i}/teacher')
    for index,teacher_seed in enumerate(plan.teacher_seeds):
        base=(plan.matrix,'primary',f'teacher-{index}')
        if seed64(seed_root,*base,'teacher')!=teacher_seed:raise ValueError('Teacher identity changed during namespace expansion')
        def derive(*labels):return reserve(seed64(seed_root,*base,*labels),'/'.join((*base,*labels)))
        reference=derive('reference');observations={role:derive(role) for role in 'FGVT'};repetitions=[]
        for repeat in range(2):
            labels=('repeat',str(repeat));values={role:derive(*labels,role)
                for role in ('optimizer','continuation_stream','allocation','contrast')}
            library,counter=seed32(seed_root,*base,*labels,'optimizer',used=used32)
            repetitions.append(dict(repetition=repeat,**values,optimizer_library32=library,optimizer_library32_counter=counter))
        rows.append(dict(teacher_index=index,teacher_seed=teacher_seed,reference_seed=reference,
            observation_seeds=observations,repetitions=repetitions))
    payload=dict(schema='research/v2-confirmation-namespace-plan',claim=plan.claim,matrix=plan.matrix,
        root_sha256=plan.root_sha256,teacher_count=len(rows),teachers=rows,
        source_execution_sha256=method.payload['source']['execution_sha256'],
        public_commitment_file_sha256=expected_file_sha256,method_file_sha256=expected_method_sha256,
        code_binding=dict(path=_RELATIVE,sha256=code),protected_access_authorized=False,
        observation_generation_authorized=False,registry_completeness_qualified=False,teacher_id_exclusions_verified=False)
    # Identity derivation cannot authorize continued work under drifted source,
    # private/public bytes or the exclusion registry even though it wrote no data.
    if load_confirmation_seeds(**arguments)!=plan:raise ValueError('Committed seed lineage changed during expansion')
    return ConfirmationNamespacePlan(plan.claim,plan.matrix,len(rows),plan.root_sha256,
        canonical_hash(payload),canonical_json(payload))
