"""Secondary native evidence after the real global barrier, without promotion.

Private role-context cores also support unqualified toy fixtures. The public
routes always reopen the real barrier before role metadata or model evidence.
"""
import copy
from pathlib import Path

from .native_generation_v2 import _audit_native_generation
from .native_workload_v2 import _inspect_native_workload
from .native_candidate_v2 import _inspect_native_candidate
from .native_inspection_reader_v2 import _load_native_inspection
from .validation_scoring_v2 import _score_validation_prediction


def _generation(store,**args):return _audit_native_generation(store=store,role_scope='secondary',**args)


def _score(store,**args):return _score_validation_prediction(store=store,role_scope='secondary',**args)


def _workload(store,args):
    return _inspect_native_workload(args,audit_generation=lambda **kw:_generation(store,**kw),role_scope='secondary')


def _candidate(store,**args):
    return _inspect_native_candidate(**args,audit_generation=lambda **kw:_generation(store,**kw),
        score_validation=lambda **kw:_score(store,**kw),role_scope='secondary')


def _inspection(store,reference,**args):
    return _load_native_inspection(reference,**args,audit_generation=lambda **kw:_generation(store,**kw),
        score_validation=lambda **kw:_score(store,**kw),role_scope='secondary')


def _released(dataset,barrier_reference,qualification_reference,args):
    from .released_transfer_v2 import ReleasedTransferStore
    from .workload import _checked_manifest
    store=ReleasedTransferStore(dataset,barrier_reference=barrier_reference,qualification_reference=qualification_reference)
    # No caller-specified source root/fold may replace the genuinely frozen one.
    from .method_worker import _sha
    if (Path(args['prepared_root'])!=store.root or Path(args['prepared_root']).resolve()!=store.root
            or _sha(store.root/'manifest.json')!=args['expected_manifest_sha256']):
        raise ValueError('Secondary native evidence must use the exact frozen prepared root/manifest')
    _checked_manifest(dataset,store.manifest);store._roles._split(args['outer_fold'])
    return store


def inspect_secondary_native_workload(*,dataset,barrier_reference,qualification_reference,generation_arguments):
    args=copy.deepcopy(generation_arguments)
    store=_released(dataset,barrier_reference,qualification_reference,args)
    result=_workload(store._roles,args)
    _released(dataset,barrier_reference,qualification_reference,args)
    return result


def inspect_secondary_native_candidate(*,dataset,barrier_reference,qualification_reference,generation_arguments,**kwargs):
    args=copy.deepcopy(generation_arguments)
    store=_released(dataset,barrier_reference,qualification_reference,args)
    result=_candidate(store._roles,generation_arguments=args,**kwargs)
    _released(dataset,barrier_reference,qualification_reference,args)
    return result


def load_secondary_native_inspection(reference,*,dataset,barrier_reference,qualification_reference,generation_arguments,immutable_generation_root):
    args=copy.deepcopy(generation_arguments)
    store=_released(dataset,barrier_reference,qualification_reference,args)
    result=_inspection(store._roles,reference,generation_arguments=args,immutable_generation_root=immutable_generation_root)
    _released(dataset,barrier_reference,qualification_reference,args)
    return result
