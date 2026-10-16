"""Audited synthetic T access. Predictors receive X only; labels never leave here.

Task10 registers immutable G/V-selected model files before access. Cooperating
adapters cannot construct grants. File/process owners can bypass this protocol;
this module deliberately makes no cryptographic access-control claim.
"""
from __future__ import annotations

from collections.abc import Mapping, Callable
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..artifacts import _atomic_json, _utc
from ..config import canonical_hash, canonical_json
from ..data.types import SplitManifest
from ..protocol import (FrozenProtocol, validate_frozen, _read, _reference, _keys,
                        _hash, _manifest, fit_variance)

_KEY=object()


def _test_ids(namespace, seed, count):
    if not namespace.startswith('confirmatory/') or not namespace.endswith('/T'):
        raise ValueError('Protected T requires separate confirmation namespace')
    if type(seed) is not int or seed<0 or type(count) is not int or count<1:
        raise ValueError('Invalid T seed/count')
    return tuple(int.from_bytes(hashlib.sha256(f'{namespace}/{seed}/row/{i}'.encode()).digest()[:8],
        'big') & ((1<<63)-1) for i in range(count))


def test_manifest(development: SplitManifest, *, namespace: str, seed: int, n_test: int) -> SplitManifest:
    """Commit T row identities without generating predictors or reading labels."""
    if 'T' in development.assignments or set(development.assignments)!=set('FGV'):
        raise ValueError('Require F/G/V development manifest')
    rows=_test_ids(namespace,seed,n_test)
    if set(rows)&(set(development.row_id)|set(development.group_id)):
        raise ValueError('T rows/groups overlap development')
    return SplitManifest(development.dataset_id,development.row_id+rows,development.group_id+rows,
        development.assignments+('T',)*n_test,teacher_id=development.teacher_id,phase='confirmatory',
        hashes=development.hashes,feature_names=development.feature_names)


@contextmanager
def _locked(directory):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(lock,fcntl.LOCK_UN)


def _directory(protocol):
    return protocol.path.parent/'access'/protocol.protocol_hash


def _events(directory):
    path=directory/'access.jsonl'
    if not path.exists(): return []
    records=[]; previous=None
    for line in path.read_text().splitlines():
        envelope=json.loads(line); record=envelope['record']
        if envelope['sha256']!=canonical_hash(record) or record['previous']!=previous:
            raise ValueError('Access ledger hash chain drift')
        records.append(record); previous=envelope['sha256']
    return records


def _event(directory,event,**fields):
    records=_events(directory)
    record=dict(event=event,utc=_utc(),previous=canonical_hash(records[-1]) if records else None,**fields)
    with (directory/'access.jsonl').open('a') as stream:
        stream.write(canonical_json(dict(record=record,sha256=canonical_hash(record)))+'\n')
        stream.flush(); os.fsync(stream.fileno())


def access_events(protocol: FrozenProtocol) -> list[dict]:
    return _events(_directory(protocol))


def _validate(protocol,manifest,run_id):
    if not isinstance(protocol,FrozenProtocol): raise ValueError('Validated frozen protocol required')
    value=_read(protocol.path); root=Path(value['payload']['sources']['root'])
    checked=validate_frozen(protocol.path,{'root':str(root)})
    if checked.protocol_hash!=protocol.protocol_hash: raise ValueError('Protocol capability identity differs')
    units=value['payload']['sources']['cohort']['units']
    found=[u for u in units if run_id in u['run_ids']]
    if len(found)!=1 or found[0]['manifest']!=json.loads(canonical_json(manifest.to_dict())):
        raise ValueError('Run/study/unit/manifest linkage differs')
    return root,value['payload'],found[0],checked


def register_run(protocol: FrozenProtocol, manifest: SplitManifest, *, run_id: str,
                 models: Mapping[str,dict], f_target_variance: float,
                 operational: Mapping | None=None) -> Path:
    """Bind selected model artifacts before T. Resume may only reopen the same run."""
    root,payload,unit,checked=_validate(protocol,manifest,run_id)
    if set(models)!=set(payload['sources']['scientific']['evaluation_methods']):
        raise ValueError('Prediction methods differ from complete frozen control set')
    for reference in models.values(): _reference(root,reference)
    if type(f_target_variance) not in (int,float) or not np.isfinite(f_target_variance) or f_target_variance<=0:
        raise ValueError('Positive F-only target variance required')
    if f_target_variance!=fit_variance(root,unit): raise ValueError('F variance differs from retained F labels')
    operational=dict(operational or {})
    if set(operational)-{'resume_run_id','verbosity'} or operational.get('verbosity','normal') not in ('quiet','normal'):
        raise ValueError('Unsupported operational override')
    record=dict(protocol_hash=protocol.protocol_hash,study_id=payload['study']['study_id'],
        unit_id=unit['unit_id'],run_id=run_id,manifest_sha256=canonical_hash(manifest.to_dict()),
        models=dict(models),f_target_variance=float(f_target_variance))
    directory=_directory(protocol)
    with _locked(directory):
        path=directory/'runs'/run_id/'link.json'
        if path.exists():
            if operational.get('resume_run_id')!=run_id or _read(path)!=record:
                raise ValueError('Resume requires identical existing run linkage')
            _event(directory,'operational_resume',run_id=run_id,operational=operational,
                   checkout_commit=checked.checkout_commit)
        else:
            if 'resume_run_id' in operational: raise ValueError('Resume requires existing run linkage')
            path.parent.mkdir(parents=True,exist_ok=False); _atomic_json(path,record)
            _event(directory,'run_registered',run_id=run_id,link_sha256=_hash(path),
                   checkout_commit=checked.checkout_commit,operational=operational)
        return path


def _link(protocol,root,run_id):
    path=_directory(protocol)/'runs'/run_id/'link.json'; value=_read(path)
    registrations=[e for e in _events(_directory(protocol)) if e['event']=='run_registered' and e['run_id']==run_id]
    if len(registrations)!=1 or registrations[0]['link_sha256']!=_hash(path):
        raise ValueError('Unlogged or changed run linkage')
    for reference in value['models'].values(): _reference(root,reference)
    return value


def open_test(protocol: FrozenProtocol, manifest: SplitManifest, *, run_id: str) -> ProtectedEvaluation:
    """Validate identity before issuing an evaluator; no T labels are read here."""
    root,payload,unit,checked=_validate(protocol,manifest,run_id)
    directory=_directory(protocol)
    with _locked(directory):
        _link(protocol,root,run_id)
        events=_events(directory)
        if any(e['event']=='labels_access' and e['run_id']==run_id for e in events) and not any(
                e['event']=='evaluation_completed' and e['run_id']==run_id for e in events):
            _event(directory,'repeat_rejected',run_id=run_id,reason='incomplete previous T access')
            raise ValueError('Prior T access incomplete; automatic reread forbidden')
        _event(directory,'access_granted',run_id=run_id,protocol_hash=protocol.protocol_hash,
            study_id=payload['study']['study_id'],unit_id=unit['unit_id'],checkout_commit=checked.checkout_commit)
    return ProtectedEvaluation(protocol,manifest,run_id=run_id,_key=_KEY)


def _rng(namespace,seed,stream):
    digest=hashlib.sha256(f'{namespace}/{seed}/{stream}'.encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8],'big')))


def _test_labels(teacher,X,namespace,seed,sigma):
    from ..data.synthetic import evaluate_teacher
    target=evaluate_teacher(teacher,X)
    return target+sigma*_rng(namespace,seed,'noise').standard_normal(len(X)),target


class ProtectedEvaluation:
    """Capability created solely by open_test; disk is revalidated on every use."""
    def __init__(self,protocol,manifest,*,run_id,_key=None):
        if _key is not _KEY: raise TypeError('Use open_test to acquire protected evaluation')
        self._protocol=protocol; self._manifest=manifest; self._run_id=run_id

    def evaluate(self,predictors: Mapping[str,Callable[[np.ndarray],np.ndarray]]) -> dict:
        from ..data.synthetic import TeacherSpec,_rotate_inputs
        protocol,manifest,run_id=self._protocol,self._manifest,self._run_id
        root,payload,unit,checked=_validate(protocol,manifest,run_id)
        directory=_directory(protocol)
        with _locked(directory):
            link=_link(protocol,root,run_id)
            if set(predictors)!=set(link['models']): raise ValueError('Prediction methods differ from registered models')
            completed=[e for e in _events(directory) if e['event']=='evaluation_completed' and e['run_id']==run_id]
            output=directory/'runs'/run_id/'predictions.json'
            if completed:
                if len(completed)!=1 or _hash(output)!=completed[0]['predictions_sha256']:
                    raise ValueError('Completed prediction artifact drift')
                return {k:v for k,v in _read(output).items() if k!='observations'}
            if any(e['event']=='labels_access' and e['run_id']==run_id for e in _events(directory)):
                _event(directory,'repeat_rejected',run_id=run_id,reason='previous T access')
                raise ValueError('Previous T access forbids an unlogged selection')
            namespace=f'confirmatory/{payload["study"]["study_id"]}/{payload["study"]["cohort_id"]}/T'
            seed=unit['seeds']['test']; count=payload['design']['n_T']
            rows=_test_ids(namespace,seed,count)
            if rows!=tuple(r for r,p in zip(manifest.row_id,manifest.assignments) if p=='T'):
                raise ValueError('T identity differs from frozen seed commitment')
            teacher=TeacherSpec.from_dict(_read(_reference(root,unit['teacher'])))
            X=_rng(namespace,seed,'inputs').standard_normal((count,teacher.d))
            X=_rotate_inputs(X,teacher.rotation,teacher.d-teacher.nuisance_dimensions)
            X=np.frombuffer(X.tobytes(),dtype=X.dtype).reshape(X.shape)
            predictions={}
            for name,predict in predictors.items():
                result=np.asarray(predict(X),dtype=np.float64)
                if result.shape!=(count,) or not np.isfinite(result).all(): raise ValueError('Invalid test predictions')
                predictions[name]=result.copy()
            # Revalidate after callbacks before granting any label access.
            _validate(protocol,manifest,run_id); _link(protocol,root,run_id)
            _event(directory,'labels_access',run_id=run_id,protocol_hash=protocol.protocol_hash,
                manifest_sha256=canonical_hash(manifest.to_dict()),checkout_commit=checked.checkout_commit)
            try:
                config=_read(_reference(root,payload['sources']['raw']))['config']
                y,target=_test_labels(teacher,X,namespace,seed,config['data']['noise_sigma'])
                metrics={name:dict(mse=float(np.mean((values-y)**2)),
                    normalized_risk=float(np.mean((values-y)**2)/link['f_target_variance']))
                    for name,values in predictions.items()}
                result=dict(protocol_hash=protocol.protocol_hash,run_id=run_id,row_id=rows,metrics=metrics,
                    predictions={name:values.tolist() for name,values in predictions.items()},
                    observations=dict(y=y.tolist(),noise_free_target=target.tolist(),group_id=rows))
                result=json.loads(canonical_json(result)); _atomic_json(output,result)
                _event(directory,'evaluation_completed',run_id=run_id,predictions_sha256=_hash(output))
                return {k:v for k,v in result.items() if k!='observations'}
            except BaseException as error:
                _event(directory,'evaluation_failed',run_id=run_id,error=type(error).__name__)
                raise


def load_evaluation(protocol: FrozenProtocol, *, run_id: str) -> dict:
    """Read sealed authorized rows for analysis; never predict or generate T again.

    Returns y, noise-free targets, row/group IDs and predictions only after the
    original access event, immutable model/run linkage and output seal validate.
    This evidence read has its own audit event, distinct from labels_access.
    """
    value=_read(protocol.path)
    units=value['payload']['sources']['cohort']['units']
    matches=[unit for unit in units if run_id in unit['run_ids']]
    if len(matches)!=1: raise ValueError('Unknown frozen run')
    manifest=_manifest(matches[0]['manifest'])
    root,payload,unit,checked=_validate(protocol,manifest,run_id)
    directory=_directory(protocol)
    with _locked(directory):
        _link(protocol,root,run_id)
        events=_events(directory)
        accessed=[e for e in events if e['event']=='labels_access' and e['run_id']==run_id]
        completed=[e for e in events if e['event']=='evaluation_completed' and e['run_id']==run_id]
        output=directory/'runs'/run_id/'predictions.json'
        if len(accessed)!=1 or len(completed)!=1 or not output.is_file() or _hash(output)!=completed[0]['predictions_sha256']:
            raise ValueError('Authorized prediction evidence missing or drift')
        result=_read(output)
        if result['protocol_hash']!=protocol.protocol_hash or result['run_id']!=run_id:
            raise ValueError('Prediction evidence identity drift')
        _event(directory,'evidence_read',run_id=run_id,predictions_sha256=_hash(output),
               checkout_commit=checked.checkout_commit)
        return result
