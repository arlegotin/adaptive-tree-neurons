"""Once-only scoring of fixed saved predictions, after genuine public gates.

Private known-label fixtures carry no native or protected-data authority. This
component publishes metric/comparison inputs; inference, independent audit and
containing whole-pipeline cost remain separate production prerequisites.
"""
import copy
import hashlib
import io
import math
from pathlib import Path
import time
import zipfile

from .deployment_worker_v2 import _json,_source
from .protected_journal_v2 import (_path,_reference,_canonical,_hash,_write_new,
    _read_metadata_study,_append_metadata_event)
from .protected_prediction_queue_v2 import _context_payload
from .protected_prediction_gate_v2 import _inspect_complete_stage

_FALSE=dict(native_qualification_authenticated=False,protected_access_authorized=False,T_accessed=False,
    whole_training_procedure_cost_authenticated=False,claim_authorized=False)


def _ref(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _npz(raw,fields):
    import numpy as np
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members=archive.infolist()
            if (len(members)!=len(fields) or {m.filename for m in members}!={f'{k}.npy' for k in fields}
                    or sum(m.file_size for m in members)>64*1024**2):
                raise ValueError('Exact bounded numeric role archive required; extra fields forbidden')
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:return {k:archive[k].copy() for k in fields}
    except (zipfile.BadZipFile,KeyError) as error:raise ValueError('Invalid numeric role archive') from error


def _score_fold_arrays(*,F,labels,predictions,temporal):
    """Pure arithmetic, deliberately no provenance or label authorization."""
    import numpy as np
    from treeneuron.analysis.prediction_arithmetic_v2 import audit_prediction_arrays,_mean
    if (type(temporal) is not bool or type(F) is not dict or not {'y','row_id'}<=set(F)
            or type(labels) is not dict or set(labels)!={'y','row_id'} or type(predictions) is not dict or not predictions):
        raise ValueError('Explicit F targets, exact evaluation target/row role and fixed predictions required')
    Frows=np.asarray(F['row_id']);rows=np.asarray(labels['row_id'])
    if (Frows.dtype!=np.int64 or Frows.ndim!=1 or len(Frows)!=len(F['y']) or len(np.unique(Frows))!=len(Frows)
            or rows.dtype!=np.int64 or rows.ndim!=1 or len(np.intersect1d(Frows,rows))):
        raise ValueError('Disjoint unique F and evaluation row identities required')
    first=next(iter(predictions.values()));groups=first['group_id'];times=first.get('timestamps')
    fields={'prediction','row_id','group_id'}|({'timestamps'} if temporal else set())
    converted={}
    for qid,packet in predictions.items():
        if (type(qid) is not str or not qid or type(packet) is not dict or set(packet)!=fields
                or not np.array_equal(packet['row_id'],rows) or not np.array_equal(packet['group_id'],groups)
                or packet['group_id'].dtype!=np.int64
                or (temporal and not np.array_equal(packet['timestamps'],times))
                or packet['prediction'].dtype not in (np.dtype('float32'),np.dtype('float64'))):
            raise ValueError('Every fixed prediction must share exact target rows/group/day geometry')
        converted[qid]=dict(row_id=packet['row_id'],values=packet['prediction'].astype(np.float64))
    audit=audit_prediction_arrays(F_y=F['y'],y=labels['y'],row_id=rows,group_id=groups,
        grouping='day' if temporal else 'group',timestamps=times,planned_methods=list(predictions),
        statuses={k:'completed_eligible' for k in predictions},predictions=converted)
    ids=sorted(set(int(v) for v in groups));variance=audit['F_target_variance'];losses={}
    for qid,packet in predictions.items():
        by_group={g:[] for g in ids}
        for y,p,g in zip(labels['y'],packet['prediction'],groups):by_group[int(g)].append((float(p)-float(y))**2)
        losses[qid]=[_mean(by_group[g])/variance for g in ids] if variance>0 else None
        if losses[qid] is not None and not all(math.isfinite(v) for v in losses[qid]):raise ValueError('Unrepresentable normalized group losses')
    out=dict(arithmetic=audit,group_ids=ids,normalized_group_losses=losses)
    if temporal:out['day_ordinals']=ids.copy()
    return out


def _label_cases(payload,inspection):
    """Role/path metadata only. Never open any role file here."""
    queries=payload['prediction_manifest']['queries'];commits={c['query_id']:c for c in inspection['commits']}
    if len(commits)!=len(queries) or set(commits)!={q['query_id'] for q in queries}:
        raise ValueError('Every planned finite prediction must exist before label access')
    views=payload['prediction_manifest'].get('view_bindings')
    if type(views) is not list or not views:raise ValueError('Frozen complete view bindings required')
    indexed={q['query_id']:q for q in queries};used=set();keys=set()
    for v in views:
        fields={'job_id','dataset','outer_fold','family_id','search_repeat','view','status','deadline_ns','query_id'}
        if type(v) is not dict or set(v)!=fields or type(v['search_repeat']) is not int or v['search_repeat'] not in (0,1):
            raise ValueError('Exact frozen view identity required')
        key=(v['job_id'],v['view'])
        if key in keys:raise ValueError('Duplicate frozen model view')
        keys.add(key);qid=v['query_id']
        if qid is None:
            if v['status'] not in ('unavailable_at_prefix','unavailable_no_eligible_model'):raise ValueError('Missing view is unavailable, never an imputed loss')
            continue
        q=indexed.get(qid)
        if (q is None or v['status']!='retained_model_bytes'
                or any(v[k]!=q[k] for k in ('job_id','dataset','outer_fold','family_id'))):
            raise ValueError('Frozen view substituted another prediction or test-selected model')
        used.add(qid)
    if used!=set(indexed):raise ValueError('Orphan fixed query is not a declared view')
    cases=[]
    for name,item in sorted(payload['data'].items()):
        data_root=_path(item['root'])
        for fold in item['folds']:
            qs=[q for q in queries if q['dataset']==name and q['outer_fold']==fold['outer_fold']]
            if not qs:continue
            index=fold['outer_fold'];roles=fold['roles'];temporal=name=='bike'
            for role,relative in [('F',f'outer-{index}/F.npz'),('T_labels',f'protected/outer-{index}/T-labels.npz')]:
                r=roles[role]
                if (type(r) is not dict or set(r)!={'reference','bytes'} or type(r['bytes']) is not int or r['bytes']<=0
                        or _path(r['reference']['path'])!=data_root/relative):
                    raise ValueError('Label evaluator requires original explicit F and protected T-label roles')
            if any(q['feature_reference']!=roles['T_features']['reference'] or q['split_hash']!=fold['split_hash'] for q in qs):
                raise ValueError('Label role cannot substitute a split or feature identity')
            cases.append(dict(dataset=name,outer_fold=index,primary=fold['primary'],split_hash=fold['split_hash'],
                temporal=temporal,F=copy.deepcopy(roles['F']),T_labels=copy.deepcopy(roles['T_labels']),queries=[q['query_id'] for q in qs]))
    if set(q for c in cases for q in c['queries'])!=set(indexed):raise ValueError('Every query needs its frozen data role')
    return cases


def _read_role(case,role,*,root,source_sha256,protocol_sha256,genesis_sha256,index,actor='evaluator'):
    """Witness access intent before opening bytes; retain failed outcomes."""
    if _source()!=source_sha256:raise ValueError('Frozen source changed before label role access')
    if role not in ('F','T_labels'):raise ValueError('Only explicitly bound F and protected T-label roles may be read')
    if actor not in ('evaluator','auditor'):raise ValueError('Explicit frozen evaluator or auditor actor required')
    reference=case[role]['reference'];stem=root/'accesses'/f'{index:04}'
    start=dict(schema='research/v2-frozen-label-role-access-start',actor=actor,source_sha256=source_sha256,
        protocol_sha256=protocol_sha256,genesis_sha256=genesis_sha256,dataset=case['dataset'],outer_fold=case['outer_fold'],
        role=role,reference=reference,origin_ns=time.perf_counter_ns(),standalone_authority=False)
    _write_new(Path(str(stem)+'-start.json'),start);error=None;status='failed';byte_count=None
    try:
        _,raw=_reference(reference);byte_count=len(raw)
        if byte_count!=case[role]['bytes']:raise ValueError('Frozen role byte count changed')
        fields=['y','row_id'] if role=='T_labels' else ['X','y','row_id','group_id']+(['timestamps'] if case['temporal'] else [])
        arrays=_npz(raw,fields);status='read_and_decoded'
        return arrays
    except BaseException as exc:
        error=dict(kind=type(exc).__name__,reason=str(exc));raise
    finally:
        end=time.perf_counter_ns()
        _write_new(Path(str(stem)+'-end.json'),dict(schema='research/v2-frozen-label-role-access-end',
            start_reference=_ref(Path(str(stem)+'-start.json')),status=status,error=error,bytes_read=byte_count,
            end_ns=end,elapsed_ns=end-start['origin_ns'],standalone_authority=False))


def _closed_metrics(root,*,payload,reference,inspection,cases,expected_reference=None):
    """Validate retained scoring evidence without reopening any label archive."""
    completion_ref=_ref(root/'completion.json') if expected_reference is None else expected_reference
    if completion_ref['path']!=str(root/'completion.json'):raise ValueError('Metrics completion must belong to this label stage')
    _,raw=_reference(completion_ref);complete=_json(raw)
    if set(complete)!={'schema','metrics_reference','clock_reference','standalone_authority'} or complete['schema']!='research/v2-closed-label-metrics' or complete['standalone_authority'] is not False:
        raise ValueError('Exact closed metric evidence required')
    _,raw=_reference(complete['metrics_reference']);metrics=_json(raw)
    if (metrics.get('schema')!='research/v2-frozen-prediction-metrics' or metrics.get('source_sha256')!=payload['source_sha256']
            or metrics.get('protocol_reference')!=reference or metrics.get('genesis_sha256')!=inspection['genesis_sha256']
            or metrics.get('prediction_manifest_sha256')!=payload['prediction_manifest']['manifest_hash']
            or metrics.get('view_bindings')!=payload['prediction_manifest']['view_bindings']
            or metrics.get('cases')!=cases or any(metrics.get(k) is not False for k in _FALSE)):
        raise ValueError('Closed metrics changed frozen source/protocol/owner/view/data identities')
    _,raw=_reference(complete['clock_reference']);clock=_json(raw)
    _,raw=_reference(clock['start_reference']);start=_json(raw)
    if (clock.get('schema')!='research/v2-label-evaluation-clock' or start.get('schema')!='research/v2-label-evaluation-start'
            or start.get('source_sha256')!=payload['source_sha256'] or start.get('protocol_reference')!=reference
            or start.get('genesis_sha256')!=inspection['genesis_sha256'] or clock.get('metrics_reference')!=complete['metrics_reference']
            or any(type(clock.get(k)) is not int or clock[k]<=0 for k in ('origin_ns','end_ns','elapsed_ns'))
            or clock['origin_ns']!=start.get('origin_ns') or clock['end_ns']-clock['origin_ns']!=clock['elapsed_ns']):
        raise ValueError('Closed metric invocation requires matching observed clock/source/owner')
    from .execution_clock_v2 import validate_clock_domain
    validate_clock_domain(start['clock_domain'])
    if clock['clock_domain']!=start['clock_domain']:raise ValueError('Metric clock domain changed')
    expected_accesses=[(c,r) for c in cases for r in ('F','T_labels')]
    accesses=metrics.get('access_references')
    if type(accesses) is not list or len(accesses)!=len(expected_accesses):raise ValueError('Every fixed role read needs a closed access record')
    expected_names=set()
    for index,(ref,(case,role)) in enumerate(zip(accesses,expected_accesses)):
        start_path=root/'accesses'/f'{index:04}-start.json';end_path=root/'accesses'/f'{index:04}-end.json'
        if ref['path']!=str(end_path):raise ValueError('Metric access history cannot substitute another read')
        _,raw=_reference(ref);end=_json(raw);_,raw=_reference(end['start_reference']);access=_json(raw)
        if (end['start_reference']['path']!=str(start_path) or access.get('actor')!='evaluator'
                or access.get('source_sha256')!=payload['source_sha256'] or access.get('protocol_sha256')!=reference['sha256']
                or access.get('genesis_sha256')!=inspection['genesis_sha256'] or access.get('dataset')!=case['dataset']
                or access.get('outer_fold')!=case['outer_fold'] or access.get('role')!=role or access.get('reference')!=case[role]['reference']
                or end.get('status')!='read_and_decoded' or end.get('error') is not None or end.get('bytes_read')!=case[role]['bytes']
                or type(access.get('origin_ns')) is not int or type(end.get('end_ns')) is not int
                or end.get('elapsed_ns')!=end['end_ns']-access['origin_ns']
                or not clock['origin_ns']<=access['origin_ns']<end['end_ns']<=clock['end_ns']):
            raise ValueError('Closed access differs from its fixed label role or observed metric clock')
        expected_names.update((start_path.name,end_path.name))
    if {p.name for p in (root/'accesses').iterdir()}!=expected_names:raise ValueError('Unknown/partial/forgotten role accesses prevent completion')
    if len(metrics['folds'])!=len(cases) or any(row['queries']!=case['queries'] for row,case in zip(metrics['folds'],cases)):
        raise ValueError('Metrics must preserve every planned fixed prediction')
    _context_payload(payload,reference)
    return completion_ref,clock['elapsed_ns']


def _evaluate_label_stage(*,protocol_payload,protocol_reference,directory,owner_root,guard):
    """Private fixture integration only; actual public gates are not injectable."""
    origin=time.perf_counter_ns();payload=_context_payload(protocol_payload,protocol_reference);study=_path(directory)
    inspection=_inspect_complete_stage(protocol_payload=payload,protocol_reference=protocol_reference,directory=study,owner_root=owner_root)
    cases=_label_cases(payload,inspection);root=study/'labels'
    from .ledger import StudyLease
    with StudyLease(study):
        state=_read_metadata_study(study);phase=state['phase'];guard()
        if phase in ('labels_started','metrics_committed'):
            if not (root/'completion.json').is_file():raise ValueError('Ambiguous/incomplete label stage lacks closed metric evidence; cannot reset or reread labels')
            event=next((_json(line) for line in (study/'events.jsonl').read_bytes().splitlines() if _json(line)['kind']=='metrics_committed'),None)
            completion_ref,elapsed=_closed_metrics(root,payload=payload,reference=protocol_reference,inspection=inspection,cases=cases,
                expected_reference=None if event is None else event['details']['metrics_reference'])
            if event is None:
                guard();_append_metadata_event(study,kind='metrics_committed',details=dict(metrics_reference=completion_ref,elapsed_ns=elapsed))
            elif event['details']['elapsed_ns']!=elapsed:raise ValueError('Committed metric clock differs from witnessed cost')
            return dict(status='metrics_committed',metrics_reference=completion_ref,label_roles_read_this_call=0,**_FALSE)
        if phase!='all_predictions_committed':raise ValueError('Exact completed prediction barrier required before labels')
        if root.exists():raise ValueError('Unexpected label artifacts cannot reset once-only evaluation')
        from .execution_clock_v2 import observe_clock_domain
        start=dict(schema='research/v2-label-evaluation-start',source_sha256=payload['source_sha256'],protocol_reference=protocol_reference,
            genesis_sha256=inspection['genesis_sha256'],origin_ns=origin,clock_domain=observe_clock_domain(),standalone_authority=False)
        _write_new(root/'clock-start.json',start)
        _append_metadata_event(study,kind='labels_started',details={})
        folds=[];accesses=[];metrics_ref=None;error=None
        commits={c['query_id']:c for c in inspection['commits']}
        try:
            for case in cases:
                role_args=dict(root=root,source_sha256=payload['source_sha256'],protocol_sha256=protocol_reference['sha256'],genesis_sha256=inspection['genesis_sha256'])
                F=_read_role(case,'F',index=len(accesses),**role_args);accesses.append(_ref(root/'accesses'/f'{len(accesses):04}-end.json'))
                target=_read_role(case,'T_labels',index=len(accesses),**role_args);accesses.append(_ref(root/'accesses'/f'{len(accesses):04}-end.json'))
                predictions={}
                for qid in case['queries']:
                    _,raw=_reference(commits[qid]['owned_readback']['prediction_reference'])
                    predictions[qid]=_npz(raw,['prediction','row_id','group_id']+(['timestamps'] if case['temporal'] else []))
                result=_score_fold_arrays(F=F,labels=target,predictions=predictions,temporal=case['temporal'])
                folds.append(dict(dataset=case['dataset'],outer_fold=case['outer_fold'],primary=case['primary'],split_hash=case['split_hash'],queries=case['queries'],**result))
            metrics=dict(schema='research/v2-frozen-prediction-metrics',source_sha256=payload['source_sha256'],protocol_reference=protocol_reference,
                genesis_sha256=inspection['genesis_sha256'],prediction_manifest_sha256=payload['prediction_manifest']['manifest_hash'],
                cases=cases,folds=folds,view_bindings=copy.deepcopy(payload['prediction_manifest']['view_bindings']),access_references=accesses,
                **_FALSE,scope='Fixed original-unit metric and comparison inputs only; native/access provenance, independent audit, inferential claims and full pipeline cost require their actual gates')
            _write_new(root/'metrics.json',metrics);metrics_ref=_ref(root/'metrics.json');guard()
        except BaseException as exc:
            error=dict(kind=type(exc).__name__,reason=str(exc));raise
        finally:
            end=time.perf_counter_ns();clock=dict(schema='research/v2-label-evaluation-clock',start_reference=_ref(root/'clock-start.json'),
                origin_ns=origin,end_ns=end,elapsed_ns=end-origin,clock_domain=start['clock_domain'],metrics_reference=metrics_ref,error=error,
                standalone_authority=False,scope='Evaluator entry through fixed scoring/full final guard; own clock/completion/journal publication and return excluded')
            _write_new(root/'clock-end.json',clock)
            if metrics_ref is not None:_write_new(root/'completion.json',dict(schema='research/v2-closed-label-metrics',metrics_reference=metrics_ref,
                clock_reference=_ref(root/'clock-end.json'),standalone_authority=False))
        completion_ref,elapsed=_closed_metrics(root,payload=payload,reference=protocol_reference,inspection=inspection,cases=cases)
        _append_metadata_event(study,kind='metrics_committed',details=dict(metrics_reference=completion_ref,elapsed_ns=elapsed))
        return dict(status='metrics_committed',metrics_reference=completion_ref,label_roles_read_this_call=len(cases),**_FALSE)


def evaluate_protected_labels(reference,*,prediction_stage_reference,directory,models_reference,global_barrier_reference,
        qualification_reference,preT_costs_reference,readiness_reference):
    """Only actual complete gates can enter the public once-only label stage."""
    from .protected_prediction_gate_v2 import _actual_protocol,load_protected_prediction_stage
    from .protected_journal_v2 import _public_locations
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol=_actual_protocol(reference,kwargs)
    stage=load_protected_prediction_stage(prediction_stage_reference,protocol_reference=reference,directory=directory,**kwargs)
    root,owners=_public_locations(directory,protocol);payload=protocol.payload
    def guard():
        actual=_actual_protocol(reference,kwargs)
        current=load_protected_prediction_stage(prediction_stage_reference,protocol_reference=reference,directory=root,**kwargs)
        if _canonical(actual.payload)!=_canonical(payload) or current.payload!=stage.payload:raise ValueError('Genuine frozen protocol/prediction evidence changed before label commit')
    result=_evaluate_label_stage(protocol_payload=payload,protocol_reference=protocol.reference,directory=root,owner_root=owners,guard=guard)
    return dict(result,native_qualification_authenticated=True,T_accessed=True,genuine_frozen_label_stage_authenticated=True,
        T_labels_accessed_this_call=result['label_roles_read_this_call']>0,
        scope='Actual frozen once-only metric stage; independent raw audit, inferential report and whole pipeline/deployment costs remain separate')
