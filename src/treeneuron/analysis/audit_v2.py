"""Byte-authenticated development arrays to a canonical report record.

E/V arithmetic only. No protected access, model-generation authentication,
population inference, operational-ledger or cost authentication is granted.
"""
import hashlib
import io
from pathlib import Path
import re

import numpy as np

from treeneuron.protocol import _read,execution_snapshot
from treeneuron.research.ledger import TERMINAL
from . import prediction_arithmetic_v2,report_v2


def _keys(value,names):
    if type(value) is not dict or set(value)!=set(names):raise ValueError('Development audit fields differ')


def _name(value):
    if type(value) is not str or not value:raise ValueError('Nonempty declared identity required')


def _digest(value,length=64):
    if type(value) is not str or re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is None:
        raise ValueError('Trusted byte/source commitment required')


def _path(root,name):
    _name(name);relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Relative evidence path without traversal required')
    path=root
    for part in relative.parts:
        path=path/part
        if path.is_symlink():raise ValueError('Symlink evidence components are forbidden')
    if not path.resolve().is_relative_to(root):raise ValueError('Evidence path escapes repository')
    return path


def _source(root,value):
    _keys(value,('commit','package_sha256','execution_snapshot'))
    _digest(value['commit'],40);_digest(value['package_sha256']);saved=value['execution_snapshot']
    _keys(saved,('pinned_commit','checkout_commit','files','execution_sha256'))
    if saved['checkout_commit']!=value['commit']:raise ValueError('Recorded source commit differs from execution snapshot')
    current=execution_snapshot(root,saved['pinned_commit'])
    original=execution_snapshot(root,saved['checkout_commit'])
    if any(current[k]!=saved[k] or original[k]!=saved[k] for k in ('files','execution_sha256')):
        raise ValueError('Declared committed execution bytes differ')
    modules=(('audit_v2.py',Path(__file__)),('prediction_arithmetic_v2.py',Path(prediction_arithmetic_v2.__file__)),
        ('report_v2.py',Path(report_v2.__file__)))
    for name,running in modules:
        expected=hashlib.sha256(running.read_bytes()).hexdigest()
        if current['files'].get('src/treeneuron/analysis/'+name)!=expected:
            raise ValueError('Running audit/arithmetic/report code differs from committed bytes')
    digest=hashlib.sha256();count=0
    for name in sorted(current['files']):
        if name.startswith('src/treeneuron/') and name.endswith('.py'):
            digest.update(name[len('src/treeneuron/'):].encode());digest.update((root/name).read_bytes());count+=1
    if not count or digest.hexdigest()!=value['package_sha256']:raise ValueError('Scientific package identity differs')


def audit_development_study(manifest_path,*,expected_manifest_sha256,repository_root):
    """Recompute every eligible raw prediction; retain every missing method.

    The manifest is externally committed. Its independent-unit/attempt declarations
    remain declarations, not proof of a complete frozen study or model lineage.
    """
    root=Path(repository_root).resolve();path=Path(manifest_path)
    if not path.is_absolute():path=root/path
    try:relative=path.relative_to(root).as_posix()
    except ValueError as error:raise ValueError('Manifest must be repository-relative evidence') from error
    _digest(expected_manifest_sha256);path=_path(root,relative)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_manifest_sha256:
        raise ValueError('External study manifest commitment differs')
    value=_read(path)
    _keys(value,('schema','study_id','status','source','independent_units','cases','attempts','limitations'))
    if (value['schema']!='research/v2-development-audit-input'
            or value['status'] not in ('completed','estimation','retired','blocked')
            or type(value['cases']) is not list or type(value['attempts']) is not list
            or type(value['limitations']) is not list):raise ValueError('Development study/status/inventories required')
    _name(value['study_id']);_name(value['independent_units'])
    for item in value['limitations']:_name(item)
    case_ids=set();prediction_ids=set();attempt_ids=set()
    for attempt in value['attempts']:
        _keys(attempt,('attempt_id','status'));_name(attempt['attempt_id'])
        if attempt['attempt_id'] in attempt_ids or attempt['status'] not in TERMINAL|{'absent','running'}:
            raise ValueError('Declared attempt inventory differs')
        attempt_ids.add(attempt['attempt_id'])
    # Validate every evaluation role and inventory before decoding any arrays.
    for case in value['cases']:
        _keys(case,('case_id','task_id','repetition','role','grouping','arrays','methods'))
        _name(case['case_id']);_name(case['task_id'])
        if (case['case_id'] in case_ids or type(case['repetition']) is not int or case['repetition'] not in (0,1)
                or case['role'] not in ('E','V') or case['grouping'] not in ('row','group','day')
                or type(case['methods']) is not list or not case['methods']):
            raise ValueError('Only unique declared E/V evaluation cases are admitted')
        case_ids.add(case['case_id']);methods=set()
        for method in case['methods']:
            _keys(method,('method_id','prediction_id','status','prediction'))
            _name(method['method_id']);_name(method['prediction_id'])
            if (method['method_id'] in methods or method['prediction_id'] in prediction_ids
                    or method['status'] not in TERMINAL|{'absent','running'}
                    or (method['prediction'] is None)!=(method['status']!='completed_eligible')):
                raise ValueError('Every planned method requires its exact status and eligible-only prediction reference')
            methods.add(method['method_id']);prediction_ids.add(method['prediction_id'])
    if not value['cases'] and value['status'] not in ('retired','blocked'):
        raise ValueError('A completed/estimation development study requires evaluation cases')
    _source(root,value['source'])
    sources={relative:expected_manifest_sha256};cache={}
    def arrays(reference,keys):
        _keys(reference,('path','sha256'));_digest(reference['sha256'])
        name=reference['path'];target=_path(root,name)
        if name in sources and sources[name]!=reference['sha256']:raise ValueError('Conflicting file commitments')
        if name not in cache:
            try:content=target.read_bytes()
            except OSError as error:raise ValueError('Missing declared raw arrays') from error
            if hashlib.sha256(content).hexdigest()!=reference['sha256']:raise ValueError('Raw array bytes changed')
            cache[name]=content
        sources[name]=reference['sha256']
        try:
            with np.load(io.BytesIO(cache[name]),allow_pickle=False) as archive:
                if set(archive.files)!=set(keys) or len(archive.files)!=len(keys):raise ValueError('Exact raw array fields required')
                return {key:archive[key].copy() for key in keys}
        except (OSError,TypeError,EOFError) as error:raise ValueError('Invalid closed raw-array archive') from error
    predictions=[];planned=[]
    for case in value['cases']:
        keys=['F_y','y','row_id','group_id']+(['timestamps'] if case['grouping']=='day' else [])
        raw=arrays(case['arrays'],keys);raw.setdefault('timestamps',None)
        methods=case['methods'];statuses={m['method_id']:m['status'] for m in methods}
        values={m['method_id']:arrays(m['prediction'],['row_id','values']) for m in methods if m['prediction'] is not None}
        result=prediction_arithmetic_v2.audit_prediction_arrays(**raw,grouping=case['grouping'],
            planned_methods=[m['method_id'] for m in methods],statuses=statuses,predictions=values)
        for method in methods:
            name=method['method_id'];metrics=result['methods'].get(name);planned.append(method['prediction_id'])
            predictions.append(dict(prediction_id=method['prediction_id'],method_id=name,task_id=case['task_id'],
                repetition=case['repetition'],status=method['status'],
                normalized_risk=metrics['normalized_risk'] if metrics else None,
                normalization_status=('defined' if result['F_target_variance']>0 else 'undefined_F_variance') if metrics else 'unavailable_prediction',
                **{k:metrics[k] if metrics else None for k in ('primary_MSE','row_RMSE','row_MAE')}))
    for name,digest in sources.items():
        target=_path(root,name)
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError('Study manifest/raw evidence changed during audit')
    canonical=dict(schema='research/v2-canonical-audit',study_id=value['study_id'],status=value['status'],
        evidence_scope='development',source={k:value['source'][k] for k in ('commit','package_sha256')},
        provenance=dict(source_authenticated=True,raw_predictions_authenticated=bool(predictions) and any(p['status']=='completed_eligible' for p in predictions),
            protected_protocol_authenticated=False),protocol_sha256=None,independent_units=value['independent_units'],
        planned_prediction_ids=planned,predictions=predictions,attempts=value['attempts'],comparisons=[],
        costs=dict(physical_elapsed_ns=None,standalone_attributed_ns=None,scope='Cost authentication is outside this development prediction audit.'),
        limitations=value['limitations']+['Development array bytes/arithmetic only; model-generation provenance, complete operational ledger, independent-unit declaration, protected protocol and population inference are not authenticated.'],
        sources=[dict(path=name,sha256=digest) for name,digest in sorted(sources.items())])
    report_v2.render_report(canonical)
    return canonical
