"""Bind fixed metric inputs to the predeclared practical comparisons.

This is neither an independent raw-target audit nor a whole-cost/quality claim.
Private numeric fixtures do not authenticate native models or protected access.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import time

from .deployment_worker_v2 import _source,_json
from .protected_journal_v2 import _canonical,_hash,_reference,_write_new,_read_metadata_study

_NAMES=('airfoil','concrete','ccpp','parkinsons','bike','energy')
_FAMILIES=tuple(f'P{i:02}' for i in range(14))
_VIEWS=('native_quality','600000000000','1800000000000','7200000000000')
_FALSE=dict(protected_access_authorized=False,native_models_authenticated=False,raw_model_provenance_authenticated=False,
    whole_training_procedure_cost_authenticated=False,claim_authorized=False,population_superiority_confirmed=False)


def _compile_comparison_inputs(payload,reference,metrics):
    """Compile supplied metadata only; genuine protocol/metric loaders wrap it."""
    if (payload.get('source_sha256')!=_source() or metrics.get('source_sha256')!=payload['source_sha256']
            or metrics.get('schema')!='research/v2-frozen-prediction-metrics' or metrics.get('protocol_reference')!=reference
            or metrics.get('prediction_manifest_sha256')!=payload['prediction_manifest']['manifest_hash']
            or metrics.get('view_bindings')!=payload['prediction_manifest']['view_bindings'] or set(payload['data'])!=set(_NAMES)):
        raise ValueError('Exact current-source frozen protocol, complete view and metric bindings required')
    query_rows=payload['prediction_manifest']['queries'];queries={q['query_id']:q for q in query_rows}
    if len(queries)!=len(query_rows):raise ValueError('Duplicate frozen prediction query')
    folds={}
    for name in _NAMES:
        rows=payload['data'][name]['folds']
        if len(rows)!=5 or {r['outer_fold'] for r in rows}!=set(range(5)):raise ValueError('All five frozen outer folds required')
        for r in rows:
            key=(name,r['outer_fold'])
            if r['primary'] is not (r['outer_fold']==(4 if name=='bike' else 0)):raise ValueError('Primary fold barrier changed')
            folds[key]=r
    metric_folds={(r['dataset'],r['outer_fold']):r for r in metrics['folds']}
    if len(metric_folds)!=len(metrics['folds']) or set(metric_folds)!=set(folds):raise ValueError('Exactly thirty fixed metric folds required')
    secondary=[]
    for key,row in metric_folds.items():
        frozen=folds[key];qs={qid for qid,q in queries.items() if (q['dataset'],q['outer_fold'])==key}
        ids=row['group_ids'];losses=row['normalized_group_losses'];methods=row['arithmetic']['methods'];variance=row['arithmetic']['F_target_variance']
        if (row['split_hash']!=frozen['split_hash'] or row['primary'] is not frozen['primary']
                or len(row['queries'])!=len(qs) or set(row['queries'])!=qs or set(losses)!=qs or set(methods)!=qs
                or type(ids) is not list or len(ids)<2 or any(type(g) is not int for g in ids) or ids!=sorted(set(ids))
                or type(variance) not in (int,float) or not math.isfinite(variance) or variance<0):
            raise ValueError('Fixed metric queries/split/groups/normalization changed')
        if key[0]=='bike' and (row.get('day_ordinals')!=ids or any(b-a!=1 for a,b in zip(ids,ids[1:]))):
            raise ValueError('Bike requires actual consecutive held-out day identities')
        for qid in qs:
            values=losses[qid];risk=methods[qid]['normalized_risk']
            if variance==0:
                if values is not None or risk is not None:raise ValueError('Undefined F variance cannot be replaced with invented normalized losses')
                continue
            if (type(values) is not list or len(values)!=len(ids)
                    or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values)
                    or type(risk) not in (int,float) or not math.isfinite(risk) or risk<0
                    or abs(math.fsum(values)/len(values)-risk)>1e-12*max(1.,abs(risk))):
                raise ValueError('Normalized group/day losses disagree with fixed query risk')
        if not row['primary']:
            secondary.append(dict(dataset=key[0],outer_fold=key[1],query_risks={q:methods[q]['normalized_risk'] for q in sorted(qs)},
                scope='Dependent secondary fold, descriptive only; no feedback to primary procedure'))
    bindings={};used=set()
    for v in metrics['view_bindings']:
        key=(v['dataset'],v['outer_fold'],v['search_repeat'],v['family_id'],v['view'])
        if (key in bindings or key[:2] not in folds or type(v['search_repeat']) is not int or v['search_repeat'] not in (0,1)
                or v['family_id'] not in _FAMILIES or v['view'] not in _VIEWS
                or v['job_id']!=f"{v['dataset']}/outer-{v['outer_fold']}/repeat-{v['search_repeat']}/{v['family_id']}"
                or v['deadline_ns']!=(None if v['view']=='native_quality' else int(v['view']))):
            raise ValueError('Duplicate or substituted frozen model view')
        qid=v['query_id'];q=queries.get(qid)
        if qid is None:
            if v['status'] not in ('unavailable_at_prefix','unavailable_no_eligible_model'):raise ValueError('Unavailable views have explicit non-loss status')
        elif (q is None or v['status']!='retained_model_bytes'
                or any(q[k]!=v[k] for k in ('dataset','outer_fold','family_id','job_id'))
                or q['split_hash']!=folds[key[:2]]['split_hash']):
            raise ValueError('Model view substituted another query, job, family or split')
        else:used.add(qid)
        bindings[key]=v
    expected={(n,f,r,family,view) for n in _NAMES for f in range(5) for r in (0,1) for family in _FAMILIES for view in _VIEWS}
    if set(bindings)!=expected or used!=set(queries):raise ValueError('Complete3360views and exact used query inventory required')
    views={}
    for view in _VIEWS:
        packets={};champions={};missing=[]
        for name in _NAMES:
            fold=4 if name=='bike' else 0;row=metric_folds[name,fold];families={}
            for family in _FAMILIES:
                pair=[]
                for r in (0,1):
                    v=bindings[name,fold,r,family,view];qid=v['query_id']
                    if qid is None or row['normalized_group_losses'][qid] is None:
                        missing.append(dict(dataset=name,outer_fold=fold,search_repeat=r,family_id=family,view=view,
                            status=v['status'] if qid is None else 'undefined_F_variance'))
                        pair.append(None)
                    else:pair.append(row['normalized_group_losses'][qid])
                if all(p is not None for p in pair):families[family]=[list(p) for p in zip(*pair)]
            packet=dict(group_ids=row['group_ids'],family_losses=families)
            if name=='bike':packet['day_ordinals']=row['day_ordinals']
            packets[name]=packet;chosen=[]
            for r in (0,1):
                choice=payload['frozen_primary_champions']['choices'][name][str(r)][view]
                actual_missing=[f for f in _FAMILIES[2:] if bindings[name,fold,r,f,view]['query_id'] is None]
                if choice['status']=='unavailable_portfolio':
                    if not actual_missing or choice['missing_families']!=actual_missing or choice['family_id'] is not None:
                        raise ValueError('Unavailable frozen portfolio differs from its missing family views')
                    chosen.append(None);continue
                if actual_missing:raise ValueError('A missing portfolio family cannot retain a complete V champion')
                family=choice.get('family_id')
                if family not in _FAMILIES[2:] or choice.get('status')!='V_selected_champion':raise ValueError('Frozen V champion must come from P02-P13')
                v=bindings[name,fold,r,family,view];q=queries.get(v['query_id'])
                if (q is None or choice.get('job_id')!=v['job_id'] or choice.get('model_binding')!=q['model_binding']
                        or choice.get('missing_families')!=[]):raise ValueError('Frozen V champion changed its exact job/model/query binding')
                chosen.append(family)
            champions[name]=chosen
        views[view]=dict(status='incomplete_primary_predictions' if missing else 'complete',losses=packets,champions=champions,missing=missing)
    return dict(schema='research/v2-frozen-comparison-inputs',planned_views=len(bindings),fold_count=len(folds),views=views,
        secondary_fold_summaries=sorted(secondary,key=lambda r:(r['dataset'],r['outer_fold'])),**_FALSE)


def _compact(value):
    """Draws regenerate from retained inputs/seed/source; retain their pair hashes."""
    if isinstance(value,dict):return {k:_compact(v) for k,v in value.items() if k not in ('aggregate_bootstrap_log_benefits','dataset_bootstrap_log_benefits')}
    if isinstance(value,(list,tuple)):return [_compact(v) for v in value]
    if hasattr(value,'tolist'):return value.tolist()
    return value


def _point_summary(view):
    from treeneuron.analysis.practical_v2 import practical_benefit
    risks={}
    for name,packet in view['losses'].items():
        fs=packet['family_losses'];choice=view['champions'][name];n=len(packet['group_ids'])
        risks[name]=dict(selected=[math.fsum(a[r] for a in fs['P00'])/n for r in (0,1)],
            baseline=[math.fsum(a[r] for a in fs[choice[r]])/n for r in (0,1)])
    if any(v<=0 for name,packet in risks.items() if name!='energy' for values in packet.values() for v in values):return None
    observational={k:v for k,v in risks.items() if k!='energy'}
    if all(v>0 for values in risks['energy'].values() for v in values):observational['energy']=risks['energy']
    return practical_benefit(observational)


def _analyze_fixed_metrics(payload,reference,metrics):
    from treeneuron.analysis.practical_comparisons_v2 import compare_practical_families
    inputs=_compile_comparison_inputs(payload,reference,metrics);policy=payload['policy']['analysis']
    if (policy['bootstrap_iterations']!=20000 or type(policy['bootstrap_seed']) is not int or not 0<=policy['bootstrap_seed']<2**63
            or policy['bike_primary_block_days']!=7 or policy['bike_sensitivity_block_days']!=[3,14]):
        raise ValueError('Exact frozen20k/seed/seven-day/3and14sensitivity policy required')
    native=inputs['views']['native_quality'];primary=None;sensitivity={}
    if native['status']=='complete':
        for days in (7,3,14):
            result=_compact(compare_practical_families(native['losses'],champions=native['champions'],seed=policy['bootstrap_seed'],
                iterations=20000,bike_block_days=days))
            if days==7:primary=result
            else:sensitivity[str(days)]=result
    prefixes={}
    for view in _VIEWS[1:]:
        row=inputs['views'][view];prefixes[view]=dict(status=row['status'],missing=row['missing'],
            point_summary=_point_summary(row) if row['status']=='complete' else None,inferential_claim_authorized=False,
            scope='Descriptive fixed search prefix; unavailable predictions are not model losses')
    return dict(schema='research/v2-fixed-practical-metric-analysis',source_sha256=payload['source_sha256'],protocol_reference=reference,
        metric_inputs_sha256=_hash(metrics),inputs=inputs,native_quality=dict(status=native['status'],missing=native['missing'],
            primary_seven_day=primary,bike_sensitivities=sensitivity),descriptive_prefixes=prefixes,**_FALSE,
        bootstrap_storage='Draw arrays omitted; exact inputs, frozen source/seed/law, iteration count and paired index hashes retained for recomputation',
        scope='Fixed metric binding and conditional arithmetic only; independent raw target/model/access audit, dependence qualifications and whole costs remain required. No dataset-population inference.')


def _actual_metrics(reference,stage_reference,directory,kwargs):
    from .protected_prediction_gate_v2 import _actual_protocol,load_protected_prediction_stage,_inspect_complete_stage
    from .protected_journal_v2 import _public_locations
    from .protected_label_evaluator_v2 import _label_cases,_closed_metrics
    from .ledger import StudyLease
    protocol=_actual_protocol(reference,kwargs)
    stage=load_protected_prediction_stage(stage_reference,protocol_reference=reference,directory=directory,**kwargs)
    root,owners=_public_locations(directory,protocol)
    with StudyLease(root):
        state=_read_metadata_study(root)
        if state['phase']!='metrics_committed':raise ValueError('Frozen witnessed metric completion required; analysis cannot start label scoring')
        inspection=_inspect_complete_stage(protocol_payload=protocol.payload,protocol_reference=reference,directory=root,owner_root=owners)
        cases=_label_cases(protocol.payload,inspection)
        events=[_json(line) for line in (root/'events.jsonl').read_bytes().splitlines()]
        event=next(e for e in events if e['kind']=='metrics_committed');completion_ref,elapsed=_closed_metrics(root/'labels',
            payload=protocol.payload,reference=reference,inspection=inspection,cases=cases,expected_reference=event['details']['metrics_reference'])
        if event['details']['elapsed_ns']!=elapsed:raise ValueError('Witnessed metric cost differs')
        _,raw=_reference(completion_ref);completion=_json(raw);_,raw=_reference(completion['metrics_reference']);metrics=_json(raw)
        if (_canonical(_actual_protocol(reference,kwargs).payload)!=_canonical(protocol.payload)
                or load_protected_prediction_stage(stage_reference,protocol_reference=reference,directory=root,**kwargs).payload!=stage.payload):
            raise ValueError('Actual frozen analysis context changed')
    return protocol,stage,metrics,completion_ref


@dataclass(frozen=True)
class ProtectedPracticalAnalysis:
    reference:dict
    _payload_json:str

    @property
    def payload(self):return json.loads(self._payload_json)


def seal_protected_practical_analysis(reference,*,prediction_stage_reference,directory,output,models_reference,
        global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol,stage,metrics,completion=_actual_metrics(reference,prediction_stage_reference,directory,kwargs)
    from .protected_prediction_gate_v2 import _publication_location
    path=_publication_location(output,protocol=protocol,inspection=stage.payload['inspection'])
    analysis=_analyze_fixed_metrics(protocol.payload,reference,metrics)
    again,_,actual,current=_actual_metrics(reference,prediction_stage_reference,directory,kwargs)
    if _canonical(again.payload)!=_canonical(protocol.payload) or _hash(actual)!=_hash(metrics) or current!=completion:
        raise ValueError('Fixed metrics changed during practical analysis')
    payload=dict(schema='research/v2-bound-protected-practical-analysis',created_utc_ns=time.time_ns(),protocol_reference=reference,
        prediction_stage_reference=prediction_stage_reference,metrics_completion_reference=completion,directory=str(directory),analysis=analysis,
        genuine_frozen_metric_binding_authenticated=True,**_FALSE)
    _write_new(path,dict(schema='research/v2-sealed-protected-practical-analysis',payload=payload,payload_sha256=_hash(payload)))
    return ProtectedPracticalAnalysis(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),_canonical(payload))


def load_protected_practical_analysis(reference,*,protocol_reference,prediction_stage_reference,directory,models_reference,
        global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol,_,metrics,completion=_actual_metrics(protocol_reference,prediction_stage_reference,directory,kwargs)
    _,raw=_reference(reference);saved=_json(raw)
    if (type(saved) is not dict or set(saved)!={'schema','payload','payload_sha256'}
            or saved['schema']!='research/v2-sealed-protected-practical-analysis'
            or type(saved['payload']) is not dict or _hash(saved['payload'])!=saved['payload_sha256']):
        raise ValueError('Exact immutable protected practical analysis required')
    payload=saved['payload'];expected_fields={'schema','created_utc_ns','protocol_reference','prediction_stage_reference',
        'metrics_completion_reference','directory','analysis','genuine_frozen_metric_binding_authenticated',*_FALSE}
    if (set(payload)!=expected_fields or payload['schema']!='research/v2-bound-protected-practical-analysis'
            or type(payload['created_utc_ns']) is not int or payload['created_utc_ns']<=0
            or payload['protocol_reference']!=protocol_reference or payload['prediction_stage_reference']!=prediction_stage_reference
            or payload['metrics_completion_reference']!=completion or payload['directory']!=str(directory)
            or payload['genuine_frozen_metric_binding_authenticated'] is not True or any(payload[k] is not False for k in _FALSE)
            or payload['analysis']!=_analyze_fixed_metrics(protocol.payload,protocol_reference,metrics)):
        raise ValueError('Protected analysis differs from actual frozen metrics, source, policy or model binding')
    again,_,current,current_completion=_actual_metrics(protocol_reference,prediction_stage_reference,directory,kwargs)
    if _canonical(again.payload)!=_canonical(protocol.payload) or _hash(current)!=_hash(metrics) or current_completion!=completion:
        raise ValueError('Actual frozen metric context changed during analysis readback')
    _reference(reference)
    return ProtectedPracticalAnalysis(copy.deepcopy(reference),_canonical(payload))
