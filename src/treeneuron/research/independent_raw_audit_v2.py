"""Independent raw prediction arithmetic; no model or configuration fitting.

The public route authenticates the actual frozen procedure first. Private
known-target fixtures never authenticate native models or protected access.
"""
import hashlib
import math
from pathlib import Path
import time

from .deployment_worker_v2 import _source,_json
from .protected_journal_v2 import _canonical,_hash,_reference,_write_new,_read_metadata_study

_FALSE=dict(native_models_authenticated=False,protected_access_authorized=False,T_accessed=False,
    raw_model_provenance_authenticated=False,whole_training_procedure_cost_authenticated=False,claim_authorized=False)


def _check_audit_policy(payload):
    from .practical_procedure_v2 import PROTECTED_AUDIT_POLICY
    if (payload.get('source_sha256')!=_source()
            or payload.get('policy',{}).get('information',{}).get('protected_audit')!=PROTECTED_AUDIT_POLICY):
        raise ValueError('Exact current-source fixed verification-only auditor policy required')


def _independent_fold_arrays(*,F,labels,predictions,temporal):
    """Separate NumPy reductions; no producer loss or arithmetic helper calls."""
    import numpy as np
    from treeneuron.data.types import array_hash
    def vector(value,dtype,name):
        a=np.asarray(value)
        if a.dtype!=np.dtype(dtype) or a.ndim!=1 or not len(a) or (a.dtype.kind=='f' and not np.isfinite(a).all()):
            raise ValueError(f'Invalid independent {name} array')
        return a
    if type(labels) is not dict or set(labels)!={'y','row_id'} or not predictions:raise ValueError('Exact fixed target and prediction roles required')
    y=vector(labels['y'],'float64','targets');rows=vector(labels['row_id'],'int64','rows')
    fy=vector(F['y'],'float64','F targets');fr=vector(F['row_id'],'int64','F rows')
    if (len(fy)<2 or len(fr)!=len(fy) or len(rows)!=len(y) or len(np.unique(rows))!=len(rows)
            or len(np.unique(fr))!=len(fr) or np.intersect1d(rows,fr).size):raise ValueError('Independent disjoint/aligned row identities required')
    first=next(iter(predictions.values()));groups=vector(first['group_id'],'int64','groups');times=first.get('timestamps')
    if groups.shape!=y.shape:raise ValueError('Independent group geometry differs')
    if temporal:
        if (times is None or times.dtype.kind!='M' or times.shape!=y.shape or np.isnat(times).any()
                or np.datetime_data(times.dtype)[0] not in ('h','m','s','ms','us','ns') or np.any(times[1:]<=times[:-1])
                or not np.array_equal(groups,times.astype('datetime64[D]').astype(np.int64))):raise ValueError('Independent actual day/time geometry required')
    elif times is not None:raise ValueError('Unexpected temporal metadata')
    unique=np.unique(groups);variance=float(np.var(fy,ddof=0));methods={};group_losses={};prediction_hashes={}
    if not math.isfinite(variance):raise ValueError('Unrepresentable independent target variance')
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for qid,packet in predictions.items():
                fields={'prediction','row_id','group_id'}|({'timestamps'} if temporal else set())
                values=np.asarray(packet['prediction'])
                if (set(packet)!=fields or values.dtype not in (np.dtype('float32'),np.dtype('float64')) or values.shape!=y.shape
                        or not np.isfinite(values).all() or packet['row_id'].dtype!=np.int64 or not np.array_equal(packet['row_id'],rows)
                        or packet['group_id'].dtype!=np.int64 or not np.array_equal(packet['group_id'],groups)
                        or (temporal and not np.array_equal(packet['timestamps'],times))):raise ValueError('Independent prediction identities/dtype/geometry differ')
                p=values.astype(np.float64);errors=p-y;squares=np.square(errors);absolute=np.abs(errors)
                gm=np.array([np.mean(squares[groups==g]) for g in unique]);ga=np.array([np.mean(absolute[groups==g]) for g in unique])
                primary=float(np.mean(gm));row=float(np.mean(squares))
                methods[qid]=dict(primary_MSE=primary,primary_RMSE=math.sqrt(primary),primary_MAE=float(np.mean(ga)),
                    normalized_risk=primary/variance if variance>0 else None,row_MSE=row,row_RMSE=math.sqrt(row),row_MAE=float(np.mean(absolute)))
                group_losses[qid]=(gm/variance).tolist() if variance>0 else None
                prediction_hashes[qid]=dict(row_id=array_hash(rows),values=array_hash(p))
    except (FloatingPointError,OverflowError) as error:raise ValueError('Unrepresentable independent raw arithmetic') from error
    return dict(F_target_variance=variance,methods=methods,group_ids=unique.tolist(),normalized_group_losses=group_losses,
        F_targets_sha256=array_hash(fy),evaluation_targets_sha256=array_hash(y),row_ids_sha256=array_hash(rows),group_ids_sha256=array_hash(groups),
        timestamps_sha256=array_hash(times) if temporal else None,prediction_hashes=prediction_hashes)


def _compare_reconstruction(expected,actual):
    audit=expected['arithmetic'];differences=[]
    def compare(a,b):
        if a is None or b is None:
            if a is not None or b is not None:raise ValueError('Independent reconstruction changed undefined normalization')
            return
        if type(a) not in (int,float) or type(b) not in (int,float) or not math.isfinite(a) or not math.isfinite(b):raise ValueError('Independent metric reconstruction must remain finite')
        delta=abs(a-b);differences.append(delta)
        if delta>1e-12*max(1.,abs(a),abs(b)):raise ValueError('Independent raw metric reconstruction disagrees with producer')
    if (set(audit['methods'])!=set(actual['methods']) or expected['group_ids']!=actual['group_ids']
            or set(expected['normalized_group_losses'])!=set(actual['normalized_group_losses'])):
        raise ValueError('Independent reconstruction lost a frozen method/group')
    compare(audit['F_target_variance'],actual['F_target_variance'])
    for key in ('F_targets_sha256','evaluation_targets_sha256','row_ids_sha256','group_ids_sha256','timestamps_sha256','prediction_hashes'):
        if audit[key]!=actual[key]:raise ValueError('Independent raw reconstruction differs in exact target/prediction identities')
    for qid,metrics in actual['methods'].items():
        if set(audit['methods'][qid])!=set(metrics):raise ValueError('Independent metric inventory differs')
        for name,value in metrics.items():compare(audit['methods'][qid][name],value)
        a=expected['normalized_group_losses'][qid];b=actual['normalized_group_losses'][qid]
        if a is None or b is None:compare(a,b)
        elif len(a)!=len(b):raise ValueError('Independent group loss vector differs')
        else:
            for left,right in zip(a,b):compare(left,right)
    return dict(numeric_comparisons=len(differences),max_absolute_difference=max(differences,default=0.))


def _ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _audit_raw_files(*,protocol_payload,protocol_reference,directory,owner_root,output,guard):
    """Known-fixture composition only; the public actual loaders are separate."""
    from .protected_prediction_queue_v2 import _context_payload
    from .protected_prediction_gate_v2 import _inspect_complete_stage,_check_stage_output
    from .protected_label_evaluator_v2 import _label_cases,_closed_metrics,_read_role,_npz
    from .protected_journal_v2 import _path
    from .ledger import StudyLease
    from .execution_clock_v2 import observe_clock_domain
    started=time.perf_counter_ns();payload=_context_payload(protocol_payload,protocol_reference);_check_audit_policy(payload)
    study=_path(directory);root=_path(output)
    with StudyLease(study):
        inspection=_inspect_complete_stage(protocol_payload=payload,protocol_reference=protocol_reference,directory=study,owner_root=owner_root)
        state=_read_metadata_study(study)
        if state['phase']!='metrics_committed':raise ValueError('Independent audit requires the sole witnessed metric stage completed')
        cases=_label_cases(payload,inspection);event=next(_json(line) for line in (study/'events.jsonl').read_bytes().splitlines() if _json(line)['kind']=='metrics_committed')
        completion_ref,_=_closed_metrics(study/'labels',payload=payload,reference=protocol_reference,inspection=inspection,cases=cases,
            expected_reference=event['details']['metrics_reference'])
        _,raw=_reference(completion_ref);completion=_json(raw);_,raw=_reference(completion['metrics_reference']);metrics=_json(raw)
        _check_stage_output(root/'audit.json',inspection=inspection,protocol_payload=payload,protected_roots=[],input_files=[protocol_reference['path']])
        if root.exists():raise ValueError('Fresh immutable raw-audit output required; retain previous attempts/costs')
        guard();start=dict(schema='research/v2-independent-raw-audit-start',actor='auditor',source_sha256=payload['source_sha256'],
            protocol_reference=protocol_reference,metrics_completion_reference=completion_ref,genesis_sha256=inspection['genesis_sha256'],
            origin_ns=started,clock_domain=observe_clock_domain(),standalone_authority=False)
        _write_new(root/'start.json',start);accesses=[];checks=[];reconstructions=[];status='audit_failure';error=None
        commits={c['query_id']:c for c in inspection['commits']};folds={(r['dataset'],r['outer_fold']):r for r in metrics['folds']}
        try:
            for case in cases:
                kwargs=dict(root=root,source_sha256=payload['source_sha256'],protocol_sha256=protocol_reference['sha256'],
                    genesis_sha256=inspection['genesis_sha256'],actor='auditor')
                F=_read_role(case,'F',index=len(accesses),**kwargs);accesses.append(_ref(root/'accesses'/f'{len(accesses):04}-end.json'))
                target=_read_role(case,'T_labels',index=len(accesses),**kwargs);accesses.append(_ref(root/'accesses'/f'{len(accesses):04}-end.json'))
                predictions={}
                for qid in case['queries']:
                    _,raw=_reference(commits[qid]['owned_readback']['prediction_reference'])
                    predictions[qid]=_npz(raw,['prediction','row_id','group_id']+(['timestamps'] if case['temporal'] else []))
                actual=_independent_fold_arrays(F=F,labels=target,predictions=predictions,temporal=case['temporal'])
                checks.append(_compare_reconstruction(folds[case['dataset'],case['outer_fold']],actual))
                reconstructions.append(dict(dataset=case['dataset'],outer_fold=case['outer_fold'],split_hash=case['split_hash'],**actual))
            guard();_context_payload(payload,protocol_reference);status='matched_raw_metrics'
            result=dict(schema='research/v2-independent-raw-arithmetic',status=status,source_sha256=payload['source_sha256'],
                protocol_reference=protocol_reference,metrics_completion_reference=completion_ref,metric_inputs_sha256=_hash(metrics),
                genesis_sha256=inspection['genesis_sha256'],query_count=len(commits),fold_count=len(cases),access_references=accesses,
                numeric_comparisons=sum(c['numeric_comparisons'] for c in checks),max_absolute_difference=max(c['max_absolute_difference'] for c in checks),
                reconstructions=reconstructions,raw_arithmetic_checked=True,**_FALSE,
                scope='Independent NumPy raw arithmetic and exact byte/row joins; private fixture proof alone grants no actual native/protected/whole-cost authority')
            _write_new(root/'audit.json',result)
            return result
        except BaseException as exc:
            error=dict(kind=type(exc).__name__,reason=str(exc));raise
        finally:
            end=time.perf_counter_ns()
            _write_new(root/'terminal.json',dict(schema='research/v2-independent-raw-audit-terminal',status=status,actor='auditor',
                start_reference=_ref(root/'start.json'),origin_ns=started,end_ns=end,elapsed_ns=end-started,
                clock_domain=start['clock_domain'],error=error,audit_reference=_ref(root/'audit.json') if status=='matched_raw_metrics' else None,
                **_FALSE,scope='Audit entry through arithmetic/full final guard/output; own terminal publication/return and containing observer excluded'))


def audit_protected_raw_predictions(reference,*,analysis_reference,prediction_stage_reference,directory,output,
        models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    from .frozen_practical_analysis_v2 import load_protected_practical_analysis,_actual_metrics
    from .protected_prediction_gate_v2 import _publication_location
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    analysis=load_protected_practical_analysis(analysis_reference,protocol_reference=reference,prediction_stage_reference=prediction_stage_reference,
        directory=directory,**kwargs)
    protocol,stage,metrics,completion=_actual_metrics(reference,prediction_stage_reference,directory,kwargs)
    if analysis.payload['analysis']['metric_inputs_sha256']!=_hash(metrics):raise ValueError('Actual frozen analysis substituted its raw metric inputs')
    _check_audit_policy(protocol.payload);root=Path(output)
    _publication_location(root/'audit.json',protocol=protocol,inspection=stage.payload['inspection'])
    def guard():
        current=load_protected_practical_analysis(analysis_reference,protocol_reference=reference,prediction_stage_reference=prediction_stage_reference,
            directory=directory,**kwargs)
        actual,again,new_metrics,new_completion=_actual_metrics(reference,prediction_stage_reference,directory,kwargs)
        if (current.payload!=analysis.payload or actual.payload!=protocol.payload or again.payload!=stage.payload
                or _hash(new_metrics)!=_hash(metrics) or new_completion!=completion):raise ValueError('Actual raw-audit context changed')
    result=_audit_raw_files(protocol_payload=protocol.payload,protocol_reference=reference,directory=directory,
        owner_root=stage.payload['inspection']['owner_root'],output=output,guard=guard)
    bound=dict(schema='research/v2-bound-independent-raw-audit',analysis_reference=analysis_reference,
        prediction_stage_reference=prediction_stage_reference,protocol_reference=reference,raw_audit_reference=_ref(root/'audit.json'),
        terminal_reference=_ref(root/'terminal.json'),genuine_frozen_raw_audit_authenticated=True,
        native_models_authenticated=True,raw_model_provenance_authenticated=True,T_accessed=True,
        protected_access_authorized=False,whole_training_procedure_cost_authenticated=False,claim_authorized=False,
        scope='Actual frozen raw target/prediction/metric/model/access joins; final dependence qualification, containing costs/deployment and report gates remain required')
    _write_new(root/'bound-audit.json',bound)
    return dict(result,bound_audit_reference=_ref(root/'bound-audit.json'),genuine_frozen_raw_audit_authenticated=True,
        native_models_authenticated=True,raw_model_provenance_authenticated=True,T_accessed=True)


def _inspect_raw_audit_evidence(*,protocol_payload,protocol_reference,inspection,metrics,completion_reference,raw_reference,terminal_reference):
    """Read retained audit records only; supplied metadata grants no authority."""
    from .protected_journal_v2 import _path
    from .execution_clock_v2 import validate_clock_domain
    payload=protocol_payload;_check_audit_policy(payload)
    raw_path,raw_bytes=_reference(raw_reference);root=raw_path.parent
    terminal_path,terminal_bytes=_reference(terminal_reference);audit=_json(raw_bytes);terminal=_json(terminal_bytes)
    if raw_path!=root/'audit.json' or terminal_path!=root/'terminal.json':raise ValueError('One exact original audit directory required')
    required={'start.json','audit.json','terminal.json','accesses'};entries={p.name for p in root.iterdir()}
    if entries not in (required,required|{'bound-audit.json'}):raise ValueError('Unknown/partial audit artifacts cannot be omitted from completed evidence')
    for path in root.iterdir():
        _path(path)
        if not (path.is_dir() if path.name=='accesses' else path.is_file()):raise ValueError('Ordinary audit files/access directory required')
    start_reference=terminal.get('start_reference');start_path,start_bytes=_reference(start_reference);start=_json(start_bytes)
    source=payload['source_sha256'];genesis=inspection['genesis_sha256'];cases=metrics['cases']
    start_fields={'schema','actor','source_sha256','protocol_reference','metrics_completion_reference','genesis_sha256','origin_ns','clock_domain','standalone_authority'}
    terminal_fields={'schema','status','actor','start_reference','origin_ns','end_ns','elapsed_ns','clock_domain','error','audit_reference','scope',*_FALSE}
    if (type(start) is not dict or set(start)!=start_fields or start_path!=root/'start.json'
            or start['schema']!='research/v2-independent-raw-audit-start' or start['actor']!='auditor'
            or start['source_sha256']!=source or start['protocol_reference']!=protocol_reference
            or start['metrics_completion_reference']!=completion_reference or start['genesis_sha256']!=genesis or start['standalone_authority'] is not False
            or type(terminal) is not dict or set(terminal)!=terminal_fields or terminal['schema']!='research/v2-independent-raw-audit-terminal'
            or terminal['status']!='matched_raw_metrics' or terminal['actor']!='auditor' or terminal['error'] is not None
            or terminal['audit_reference']!=raw_reference or any(terminal[k] is not False for k in _FALSE)
            or any(type(terminal[k]) is not int or terminal[k]<=0 for k in ('origin_ns','end_ns','elapsed_ns'))
            or terminal['origin_ns']!=start['origin_ns'] or terminal['end_ns']-terminal['origin_ns']!=terminal['elapsed_ns']
            or terminal['clock_domain']!=start['clock_domain']
            or terminal['scope']!='Audit entry through arithmetic/full final guard/output; own terminal publication/return and containing observer excluded'):
        raise ValueError('Strict completed source/protocol/owner/metric/clock audit evidence required')
    validate_clock_domain(start['clock_domain'])
    fields={'schema','status','source_sha256','protocol_reference','metrics_completion_reference','metric_inputs_sha256','genesis_sha256',
        'query_count','fold_count','access_references','numeric_comparisons','max_absolute_difference','reconstructions','raw_arithmetic_checked','scope',*_FALSE}
    if (type(audit) is not dict or set(audit)!=fields or audit['schema']!='research/v2-independent-raw-arithmetic'
            or audit['status']!='matched_raw_metrics' or audit['source_sha256']!=source or audit['protocol_reference']!=protocol_reference
            or audit['metrics_completion_reference']!=completion_reference or audit['metric_inputs_sha256']!=_hash(metrics)
            or audit['genesis_sha256']!=genesis or type(audit['query_count']) is not int or audit['query_count']!=len(inspection['commits'])
            or type(audit['fold_count']) is not int or audit['fold_count']!=len(cases) or audit['raw_arithmetic_checked'] is not True
            or any(audit[k] is not False for k in _FALSE)):
        raise ValueError('Exact independent raw arithmetic and frozen metric inputs required')
    planned=[(case,role) for case in cases for role in ('F','T_labels')];accesses=audit['access_references']
    if type(accesses) is not list or len(accesses)!=len(planned):raise ValueError('All auditor role accesses required')
    refs=[raw_reference,terminal_reference,start_reference];expected_files=set();previous=start['origin_ns']
    for index,(reference,(case,role)) in enumerate(zip(accesses,planned)):
        end_path,end_bytes=_reference(reference);end=_json(end_bytes);access_ref=end.get('start_reference');path,content=_reference(access_ref);access=_json(content)
        access_fields={'schema','actor','source_sha256','protocol_sha256','genesis_sha256','dataset','outer_fold','role','reference','origin_ns','standalone_authority'}
        end_fields={'schema','start_reference','status','error','bytes_read','end_ns','elapsed_ns','standalone_authority'}
        if (end_path!=root/'accesses'/f'{index:04}-end.json' or path!=root/'accesses'/f'{index:04}-start.json'
                or set(access)!=access_fields or access['schema']!='research/v2-frozen-label-role-access-start' or access['actor']!='auditor'
                or access['source_sha256']!=source or access['protocol_sha256']!=protocol_reference['sha256'] or access['genesis_sha256']!=genesis
                or any(access[k]!=case[k] for k in ('dataset','outer_fold')) or access['role']!=role or access['reference']!=case[role]['reference']
                or set(end)!=end_fields or end['schema']!='research/v2-frozen-label-role-access-end' or end['status']!='read_and_decoded'
                or end['error'] is not None or type(end['bytes_read']) is not int or end['bytes_read']!=case[role]['bytes']
                or access['standalone_authority'] is not False or end['standalone_authority'] is not False
                or any(type(v) is not int or v<=0 for v in (access['origin_ns'],end['end_ns'],end['elapsed_ns']))
                or end['elapsed_ns']!=end['end_ns']-access['origin_ns'] or not previous<=access['origin_ns']<end['end_ns']<=terminal['end_ns']):
            raise ValueError('Auditor actor/role/source/owner/access clock differs from the fixed raw read')
        previous=end['end_ns'];expected_files.update((path.name,end_path.name));refs.extend((reference,access_ref))
    if {p.name for p in (root/'accesses').iterdir()}!=expected_files:raise ValueError('Orphaned or missing auditor access cannot be ignored')
    metric_folds={(f['dataset'],f['outer_fold']):f for f in metrics['folds']};reconstructions=audit['reconstructions']
    if type(reconstructions) is not list or len(reconstructions)!=len(cases):raise ValueError('All fixed independent fold reconstructions required')
    comparisons=[]
    for case,row in zip(cases,reconstructions):
        key=(case['dataset'],case['outer_fold'])
        fields={'dataset','outer_fold','split_hash','F_target_variance','methods','group_ids','normalized_group_losses','F_targets_sha256',
            'evaluation_targets_sha256','row_ids_sha256','group_ids_sha256','timestamps_sha256','prediction_hashes'}
        if type(row) is not dict or set(row)!=fields or any(row[k]!=case[k] for k in ('dataset','outer_fold','split_hash')):
            raise ValueError('Independent reconstruction substituted or omitted a fixed fold')
        comparisons.append(_compare_reconstruction(metric_folds[key],row))
    count=sum(v['numeric_comparisons'] for v in comparisons);difference=max(v['max_absolute_difference'] for v in comparisons)
    if (type(audit['numeric_comparisons']) is not int or audit['numeric_comparisons']!=count
            or type(audit['max_absolute_difference']) not in (int,float) or audit['max_absolute_difference']!=difference):
        raise ValueError('Independent comparison summary differs from its retained reconstructions')
    if _source()!=source:raise ValueError('Raw audit source changed during readback')
    for reference in refs:_reference(reference)
    return dict(schema='research/v2-independent-raw-audit-readback',raw_reference=raw_reference,terminal_reference=terminal_reference,
        source_sha256=source,protocol_reference=protocol_reference,metrics_completion_reference=completion_reference,metric_inputs_sha256=_hash(metrics),
        raw_arithmetic_evidence_checked=True,numeric_comparisons=count,max_absolute_difference=difference,auditor_role_reads=len(planned),
        observed_audit_elapsed_ns=terminal['elapsed_ns'],clock_domain=terminal['clock_domain'],evidence_references=refs,**_FALSE,
        scope='Retained raw-auditor arithmetic/access/clock join only; no third target decoding, model operation or native/protected/whole-cost capability')


def load_protected_raw_audit(reference,*,protocol_reference,analysis_reference,prediction_stage_reference,directory,
        models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    from .frozen_practical_analysis_v2 import load_protected_practical_analysis,_actual_metrics
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    analysis=load_protected_practical_analysis(analysis_reference,protocol_reference=protocol_reference,prediction_stage_reference=prediction_stage_reference,directory=directory,**kwargs)
    protocol,stage,metrics,completion=_actual_metrics(protocol_reference,prediction_stage_reference,directory,kwargs)
    bound_path,raw=_reference(reference);bound=_json(raw)
    fields={'schema','analysis_reference','prediction_stage_reference','protocol_reference','raw_audit_reference','terminal_reference',
        'genuine_frozen_raw_audit_authenticated','native_models_authenticated','raw_model_provenance_authenticated','T_accessed',
        'protected_access_authorized','whole_training_procedure_cost_authenticated','claim_authorized','scope'}
    if (type(bound) is not dict or set(bound)!=fields or bound['schema']!='research/v2-bound-independent-raw-audit'
            or bound_path.name!='bound-audit.json' or bound['analysis_reference']!=analysis_reference
            or bound['prediction_stage_reference']!=prediction_stage_reference or bound['protocol_reference']!=protocol_reference
            or any(bound[k] is not True for k in ('genuine_frozen_raw_audit_authenticated','native_models_authenticated','raw_model_provenance_authenticated','T_accessed'))
            or any(bound[k] is not False for k in ('protected_access_authorized','whole_training_procedure_cost_authenticated','claim_authorized'))
            or bound['raw_audit_reference']['path']!=str(bound_path.parent/'audit.json') or bound['terminal_reference']['path']!=str(bound_path.parent/'terminal.json')
            or analysis.payload['analysis']['metric_inputs_sha256']!=_hash(metrics)):
        raise ValueError('Exact actual bound raw-audit/model/protocol/analysis evidence required')
    result=_inspect_raw_audit_evidence(protocol_payload=protocol.payload,protocol_reference=protocol_reference,inspection=stage.payload['inspection'],
        metrics=metrics,completion_reference=completion,raw_reference=bound['raw_audit_reference'],terminal_reference=bound['terminal_reference'])
    again=load_protected_practical_analysis(analysis_reference,protocol_reference=protocol_reference,prediction_stage_reference=prediction_stage_reference,directory=directory,**kwargs)
    actual,new_stage,new_metrics,new_completion=_actual_metrics(protocol_reference,prediction_stage_reference,directory,kwargs)
    if (again.payload!=analysis.payload or actual.payload!=protocol.payload or new_stage.payload!=stage.payload or _hash(new_metrics)!=_hash(metrics)
            or new_completion!=completion):raise ValueError('Actual raw-audit context changed during readback')
    _reference(reference)
    return dict(result,bound_audit_reference=reference,genuine_frozen_raw_audit_authenticated=True,native_models_authenticated=True,
        raw_model_provenance_authenticated=True,T_accessed=True,scope='Actual frozen native/model/protocol/timing-plan/metrics and independent retained audit evidence; costs/dependence/report claims remain separate')
