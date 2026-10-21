"""Fixed practical confirmation orchestration; no method or analysis search.

Private metadata fixtures never authorize native inference or test access.
Public routes require actual full frozen prerequisites before output.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import secrets
import time

from .deployment_worker_v2 import _source,_json,_digest
from .protected_journal_v2 import _path,_reference,_hash,_canonical,_write_new

_STEPS=('native_timing_plan','predictions','prediction_stage','metrics','analysis','raw_audit','native_measurements')
_FALSE=dict(native_qualification_authenticated=False,protected_access_authorized=False,whole_training_procedure_cost_authenticated=False,T_accessed=False)
_SCOPE='Fixed step executor through return/exception; own end publication and enclosing invocation excluded'


def _ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _make_fixed_programme(payload,reference,inputs,*,directory,output,rss_limit_bytes):
    if payload.get('source_sha256')!=_source() or type(rss_limit_bytes) is not int or not 64*1024**2<=rss_limit_bytes<=16*1024**3:
        raise ValueError('Exact current source and bounded root RSS required')
    _reference(reference);directory=_path(directory);root=_path(output)
    required={'models_reference','global_barrier_reference','qualification_reference','preT_costs_reference','readiness_reference'}
    if type(inputs) is not dict or set(inputs)!=required:raise ValueError('Exact five frozen prerequisite references required')
    for key,value in inputs.items():
        if key=='qualification_reference':
            if type(value) is not dict or set(value)!={'root','inventory_sha256'} or not _digest(value['inventory_sha256']):raise ValueError('Exact qualification inventory required')
            _path(value['root'])
        else:_reference(value)
    if root.is_relative_to(directory) or directory.is_relative_to(root):raise ValueError('Programme and protected study must have separate roots')
    outputs=dict(native_timing_plan=str(root/'native-timing-plan/deployment-plan.json'),prediction_stage=str(root/'prediction-stage.json'),
        analysis=str(root/'analysis.json'),raw_audit=str(root/'raw-audit/bound-audit.json'),native_measurements=str(root/'native-measurements'))
    value=dict(schema='research/v2-fixed-practical-confirmation-programme',source_sha256=_source(),protocol_reference=reference,
        protocol_payload_sha256=_hash(payload),inputs=copy.deepcopy(inputs),protected_study_directory=str(directory),directory=str(root),
        outputs=outputs,observer_directory=str(root.parent/(root.name+'-observations')),steps=list(_STEPS),rss_limit_bytes=rss_limit_bytes,pause_after_ns=None,worker_concurrency=1,
        required_memory_proxy_bytes=(5*rss_limit_bytes+3)//4+512*1024**2,required_disk_proxy_bytes=512*1024**2,**_FALSE)
    value['programme_hash']=_hash(value)
    return value


def _checked_programme(reference,plan):
    path,raw=_reference(reference);stored=_json(raw)
    if path!=_path(plan['directory'])/'programme.json' or stored!=plan:raise ValueError('Exact original programme bytes required')
    _,content=_reference(plan['protocol_reference']);payload=_json(content)
    if payload.get('schema')=='research/v2-final-practical-prediction-protocol':payload=payload['payload']
    expected=_make_fixed_programme(payload,plan['protocol_reference'],plan['inputs'],directory=_path(plan['protected_study_directory']),
        output=_path(plan['directory']),rss_limit_bytes=plan['rss_limit_bytes'])
    if expected!=plan:raise ValueError('Fixed programme/source/protocol/prerequisites/paths changed')
    return path


def _result(value):
    if type(value) is not dict or set(value)!={'status','evidence_references'} or value['status'] not in ('completed','paused','blocked'):
        raise ValueError('Explicit fixed-step status and immutable evidence required')
    refs=value['evidence_references']
    if type(refs) is not list or value['status']=='completed' and not refs:raise ValueError('Complete fixed step needs immutable evidence')
    for reference in refs:_reference(reference)
    return value


def _call_history(reference,plan):
    """Read every closed step, including failures; orphaned starts block reuse."""
    from .execution_clock_v2 import validate_clock_domain
    root=_path(plan['directory'])/'calls';history=[]
    if not root.exists():return history
    for step in root.iterdir():
        _path(step)
        if not step.is_dir() or step.name not in _STEPS:raise ValueError('Unknown programme call cannot disappear')
        episodes=sorted(step.iterdir())
        if [p.name for p in episodes]!=[f'call-{i:04}' for i in range(len(episodes))]:raise ValueError('Complete sequential programme call history required')
        for index,episode in enumerate(episodes):
            _path(episode)
            if not episode.is_dir() or {p.name for p in episode.iterdir()}!={'start.json','end.json'}:raise ValueError('Every programme step start needs its closed end; no orphan reset')
            start_ref=_ref(episode/'start.json');_,raw=_reference(start_ref);start=_json(raw);end_ref=_ref(episode/'end.json');_,raw=_reference(end_ref);end=_json(raw)
            fields={'schema','source_sha256','programme_reference','programme_hash','step','call_index','origin_ns','clock_domain','standalone_authority'}
            end_fields={'schema','start_reference','origin_ns','end_ns','elapsed_ns','clock_domain','result','error','clock_scope','standalone_authority'}
            if (set(start)!=fields or start['schema']!='research/v2-fixed-protected-step-start' or start['source_sha256']!=plan['source_sha256']
                    or start['programme_reference']!=reference or start['programme_hash']!=plan['programme_hash'] or start['step']!=step.name
                    or type(start['call_index']) is not int or start['call_index']!=index or start['standalone_authority'] is not False
                    or set(end)!=end_fields or end['schema']!='research/v2-fixed-protected-step-end' or end['start_reference']!=start_ref
                    or end['origin_ns']!=start['origin_ns'] or end['clock_domain']!=start['clock_domain'] or end['clock_scope']!=_SCOPE
                    or any(type(end[k]) is not int or end[k]<=0 for k in ('origin_ns','end_ns','elapsed_ns'))
                    or end['end_ns']-end['origin_ns']!=end['elapsed_ns'] or end['standalone_authority'] is not False
                    or end['error'] is not None and (type(end['error']) is not dict or set(end['error'])!={'type','reason'}
                        or any(type(end['error'][k]) is not str for k in ('type','reason')))):
                raise ValueError('Strict source/programme/step/clock result history required')
            validate_clock_domain(start['clock_domain'])
            if end['error'] is None:_result(end['result'])
            elif end['result'] is not None:raise ValueError('Failed call cannot retain an eligible result')
            history.append(dict(step=step.name,start_reference=start_ref,end_reference=end_ref,start=start,end=end))
    return history


def _invocation_history(reference,plan,calls):
    """Reconcile every local call with exactly one closed invocation."""
    from .execution_clock_v2 import validate_clock_domain
    root=_path(plan['directory'])/'invocations'
    if not root.exists():
        if calls:raise ValueError('Step calls lack their containing invocation')
        return []
    starts={};ends={}
    for path in root.iterdir():
        _path(path);match=re.fullmatch(r'([0-9a-f]{32})-(start|end)\.json',path.name)
        if not path.is_file() or match is None:raise ValueError('Unknown programme invocation cannot be ignored')
        (starts if match[2]=='start' else ends)[match[1]]=path
    if set(starts)!=set(ends):raise ValueError('Unclosed programme invocation blocks complete recovery/costs')
    known={c['end_reference']['path']:c for c in calls};covered=set();records=[]
    for identity in sorted(starts):
        start_ref=_ref(starts[identity]);_,raw=_reference(start_ref);start=_json(raw);end_ref=_ref(ends[identity]);_,raw=_reference(end_ref);end=_json(raw)
        fields={'schema','source_sha256','programme_reference','origin_ns','clock_domain','standalone_authority'}
        end_fields={'schema','start_reference','origin_ns','end_ns','elapsed_ns','clock_domain','status','error','completed_steps','call_references','clock_scope',*_FALSE}
        if (set(start)!=fields or start['schema']!='research/v2-fixed-protected-invocation-start' or start['source_sha256']!=plan['source_sha256']
                or start['programme_reference']!=reference or start['standalone_authority'] is not False
                or set(end)!=end_fields or end['schema']!='research/v2-fixed-protected-invocation-end' or end['start_reference']!=start_ref
                or end['origin_ns']!=start['origin_ns'] or end['clock_domain']!=start['clock_domain']
                or any(type(end[k]) is not int or end[k]<=0 for k in ('origin_ns','end_ns','elapsed_ns'))
                or end['end_ns']-end['origin_ns']!=end['elapsed_ns'] or any(end[k] is not False for k in _FALSE if k!='T_accessed')
                or end['T_accessed'] is not None
                or end['clock_scope']!='Invocation through fixed steps/full context guard; own end publication/return and outer observation excluded'
                or type(end['completed_steps']) is not list or end['completed_steps']!=list(_STEPS[:len(end['completed_steps'])])
                or type(end['call_references']) is not list or end['status'] not in ('completed','operator_pause','blocked','interrupted_or_failed')
                or end['status']=='completed' and end['completed_steps']!=list(_STEPS)
                or (end['error'] is None)!=(end['status']!='interrupted_or_failed')
                or end['error'] is not None and (type(end['error']) is not dict or set(end['error'])!={'type','reason'}
                    or any(type(end['error'][k]) is not str for k in ('type','reason')))):
            raise ValueError('Strict closed source/programme/clock invocation required')
        validate_clock_domain(start['clock_domain']);previous=end['origin_ns']
        for ref in end['call_references']:
            _reference(ref);call=known.get(ref['path'])
            if call is None or call['end_reference']!=ref or ref['path'] in covered:raise ValueError('Each full step call belongs to one containing invocation')
            a,b=call['start'],call['end']
            if a['clock_domain']!=start['clock_domain'] or not previous<=a['origin_ns']<b['end_ns']<=end['end_ns']:
                raise ValueError('Serial fixed-step calls must stay inside their invocation')
            previous=b['end_ns'];covered.add(ref['path'])
        records.append(dict(reference=end_ref,start_reference=start_ref,clock=end))
    if covered!=set(known):raise ValueError('Missing containing invocation cannot erase paid step work')
    for record in records:_reference(record['reference']);_reference(record['start_reference'])
    return records


def _run_fixed_steps(reference,plan,*,execute_step,resolve_completed,guard,max_step_starts=None):
    _checked_programme(reference,plan)
    if max_step_starts is not None and (type(max_step_starts) is not int or max_step_starts<=0):raise ValueError('Positive operational step prefix required')
    from .ledger import StudyLease
    from .execution_clock_v2 import observe_clock_domain
    root=_path(plan['directory']);completed=[];call_refs=[];status='blocked';error=None;escaped=None;started=0
    with StudyLease(root):
        _invocation_history(reference,plan,_call_history(reference,plan))
        invocations=root/'invocations';identity=secrets.token_hex(16);domain=observe_clock_domain();origin=time.perf_counter_ns()
        start_file=invocations/f'{identity}-start.json'
        _write_new(start_file,dict(schema='research/v2-fixed-protected-invocation-start',source_sha256=plan['source_sha256'],programme_reference=reference,
            origin_ns=origin,clock_domain=domain,standalone_authority=False))
        try:
            guard()
            for step in _STEPS:
                _checked_programme(reference,plan)
                found=resolve_completed(step)
                if found is not None:
                    if _result(found)['status']!='completed':raise ValueError('Only actually complete proofs may skip execution')
                    completed.append(step);continue
                if max_step_starts is not None and started>=max_step_starts:status='operator_pause';break
                base=root/'calls'/step;index=len(list(base.glob('call-*')));episode=base/f'call-{index:04}'
                begin=time.perf_counter_ns();start_ref=episode/'start.json'
                _write_new(start_ref,dict(schema='research/v2-fixed-protected-step-start',source_sha256=plan['source_sha256'],programme_reference=reference,
                    programme_hash=plan['programme_hash'],step=step,call_index=index,origin_ns=begin,clock_domain=domain,standalone_authority=False))
                returned=None;call_error=None;started+=1
                try:returned=_result(execute_step(step))
                except BaseException as caught:
                    call_error=dict(type=type(caught).__name__,reason=str(caught));raise
                finally:
                    end=time.perf_counter_ns();target=episode/'end.json'
                    _write_new(target,dict(schema='research/v2-fixed-protected-step-end',start_reference=_ref(start_ref),origin_ns=begin,end_ns=end,
                        elapsed_ns=end-begin,clock_domain=domain,result=returned,error=call_error,clock_scope=_SCOPE,standalone_authority=False));call_refs.append(_ref(target))
                if returned['status']!='completed':status='operator_pause' if returned['status']=='paused' else 'blocked';break
                completed.append(step)
            else:status='completed'
            guard();_checked_programme(reference,plan);_call_history(reference,plan)
            return dict(status=status,completed_steps=completed,started_steps=started,programme_reference=reference,**dict(_FALSE,T_accessed=None),
                scope='Private fixed-step execution only; actual operation/protocol/native/owner/cost qualification remains separate')
        except BaseException as caught:
            escaped=caught;error=dict(type=type(caught).__name__,reason=str(caught));status='interrupted_or_failed';raise
        finally:
            end=time.perf_counter_ns()
            final=invocations/f'{identity}-end.json'
            _write_new(final,dict(schema='research/v2-fixed-protected-invocation-end',start_reference=_ref(start_file),
                origin_ns=origin,end_ns=end,elapsed_ns=end-origin,clock_domain=domain,status=status,error=error,completed_steps=completed,
                call_references=call_refs,clock_scope='Invocation through fixed steps/full context guard; own end publication/return and outer observation excluded',**dict(_FALSE,T_accessed=None)))
            if escaped is not None:escaped.confirmation_invocation_reference=_ref(final)


def _actual_protocol(reference,kwargs):
    from .practical_T_protocol_v2 import load_practical_T_protocol
    return load_practical_T_protocol(reference,**kwargs)


def seal_practical_confirmation_programme(reference,*,directory,output,models_reference,global_barrier_reference,qualification_reference,
        preT_costs_reference,readiness_reference,rss_limit_bytes=4*1024**3):
    inputs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol=_actual_protocol(reference,inputs)
    from .protected_journal_v2 import _public_locations
    from .protected_prediction_gate_v2 import _publication_location
    study,owners=_public_locations(directory,protocol)
    root=_publication_location(output,protocol=protocol,inspection=dict(study_directory=str(study),owner_root=str(owners),invocation_references=[]),kind='directory',fresh=False)
    plan=_make_fixed_programme(protocol.payload,reference,inputs,directory=study,output=root,rss_limit_bytes=rss_limit_bytes);path=root/'programme.json'
    if path.exists():
        _checked_programme(_ref(path),plan);return dict(reference=_ref(path),**_FALSE)
    if root.exists():raise ValueError('Fresh programme namespace required; preserve partial publications')
    if _actual_protocol(reference,inputs).payload!=protocol.payload:raise ValueError('Actual frozen programme context changed')
    _write_new(path,plan)
    return dict(reference=_ref(path),**_FALSE)


def _step_operations(protocol,plan):
    """Actual public operations only; no injected qualification or label paths."""
    kwargs=plan['inputs'];reference=protocol.reference;study=_path(plan['protected_study_directory']);paths={k:_path(p) for k,p in plan['outputs'].items()}
    from .native_deployment_plan_v2 import seal_native_deployment_plan,load_native_deployment_plan
    from .protected_prediction_gate_v2 import seal_protected_prediction_stage,load_protected_prediction_stage,_actual_stage
    from .protected_prediction_queue_v2 import run_protected_predictions
    from .protected_label_evaluator_v2 import evaluate_protected_labels
    from .frozen_practical_analysis_v2 import seal_protected_practical_analysis,load_protected_practical_analysis
    from .independent_raw_audit_v2 import audit_protected_raw_predictions,load_protected_raw_audit
    from .native_deployment_execution_v2 import run_native_deployment_measurements,collect_native_deployment_measurements
    def result(*refs,status='completed'):return _result(dict(status=status,evidence_references=list(refs)))
    def resolve(step):
        path=paths.get(step)
        if step=='predictions':
            if not study.exists():return None
            from .protected_journal_v2 import _read_metadata_study
            state=_read_metadata_study(study)
            if state['phase'] not in ('all_predictions_committed','labels_started','metrics_committed'):return None
            _actual_stage(reference,study,kwargs)
            ends=sorted((study/'invocations').glob('*-end.json'))
            refs=[_ref(p) for p in ends if _json(p.read_bytes())['status']=='all_predictions_committed']
            if not refs:raise ValueError('Complete predictions lack immutable invocation evidence')
            return result(*refs)
        if step=='metrics':
            if not study.exists():return None
            from .protected_journal_v2 import _read_metadata_study
            if _read_metadata_study(study)['phase']!='metrics_committed':return None
            evaluate_protected_labels(reference,prediction_stage_reference=_ref(paths['prediction_stage']),directory=study,**kwargs)
            return result(_ref(study/'labels/completion.json'))
        if path is None or not path.exists():return None
        if step=='native_timing_plan':load_native_deployment_plan(_ref(path),protocol_reference=reference,directory=study,**kwargs)
        elif step=='prediction_stage':load_protected_prediction_stage(_ref(path),protocol_reference=reference,directory=study,**kwargs)
        elif step=='analysis':load_protected_practical_analysis(_ref(path),protocol_reference=reference,prediction_stage_reference=_ref(paths['prediction_stage']),directory=study,**kwargs)
        elif step=='raw_audit':load_protected_raw_audit(_ref(path),protocol_reference=reference,analysis_reference=_ref(paths['analysis']),prediction_stage_reference=_ref(paths['prediction_stage']),directory=study,**kwargs)
        elif step=='native_measurements':
            observed=collect_native_deployment_measurements(_ref(paths['native_timing_plan']),protocol_reference=reference,directory=study,output=path,rss_limit_bytes=plan['rss_limit_bytes'],**kwargs)
            if observed['status']!='complete_descriptive_measurements':return None
            return result(*observed['context_references'])
        return result(_ref(path))
    def execute(step):
        if step=='native_timing_plan':
            value=seal_native_deployment_plan(reference,directory=study,output=paths[step].parent,**kwargs);return result(value.reference)
        if step=='predictions':
            value=run_protected_predictions(reference,directory=study,rss_limit_bytes=plan['rss_limit_bytes'],**kwargs)
            return result(value['invocation_reference'],status='completed' if value['status']=='all_predictions_committed' else 'blocked' if value['status'].startswith('blocked') else 'paused')
        if step=='prediction_stage':return result(seal_protected_prediction_stage(reference,directory=study,output=paths[step],**kwargs).reference)
        if step=='metrics':
            evaluate_protected_labels(reference,prediction_stage_reference=_ref(paths['prediction_stage']),directory=study,**kwargs)
            return result(_ref(study/'labels/completion.json'))
        if step=='analysis':return result(seal_protected_practical_analysis(reference,prediction_stage_reference=_ref(paths['prediction_stage']),directory=study,output=paths[step],**kwargs).reference)
        if step=='raw_audit':
            value=audit_protected_raw_predictions(reference,analysis_reference=_ref(paths['analysis']),prediction_stage_reference=_ref(paths['prediction_stage']),
                directory=study,output=paths[step].parent,**kwargs);return result(value['bound_audit_reference'])
        if step=='native_measurements':
            value=run_native_deployment_measurements(_ref(paths['native_timing_plan']),protocol_reference=reference,directory=study,output=paths[step],
                resume=paths[step].exists(),rss_limit_bytes=plan['rss_limit_bytes'],**kwargs)
            found=resolve(step)
            if found is not None:return found
            contexts=sorted(paths[step].glob('context-*.json'),key=lambda p:int(p.stem.split('-')[-1]))
            return result(*[_ref(p) for p in contexts],status='blocked' if value['status']=='completed_with_failures' else 'paused')
        raise ValueError('Unknown fixed programme step')
    return execute,resolve


def run_practical_confirmation(reference,*,protocol_reference,models_reference,global_barrier_reference,qualification_reference,
        preT_costs_reference,readiness_reference,max_step_starts=None):
    inputs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol=_actual_protocol(protocol_reference,inputs);_,raw=_reference(reference);plan=_json(raw)
    if plan.get('inputs')!=inputs or plan.get('protocol_reference')!=protocol_reference:raise ValueError('Actual programme prerequisites changed')
    _checked_programme(reference,plan)
    from .protected_journal_v2 import _public_locations
    from .protected_prediction_gate_v2 import _publication_location
    study,owners=_public_locations(plan['protected_study_directory'],protocol)
    _publication_location(plan['directory'],protocol=protocol,inspection=dict(study_directory=str(study),owner_root=str(owners),invocation_references=[]),kind='directory',fresh=False)
    from .native_deployment_plan_v2 import _study
    from .protected_journal_v2 import _read_metadata_study,_append_metadata_event
    from .ledger import StudyLease
    _study(protocol,study,prepare=True)
    with StudyLease(study):
        state=_read_metadata_study(study)
        if state['programme_reference'] is None:_append_metadata_event(study,kind='programme_frozen',details=dict(programme_reference=reference))
        elif state['programme_reference']!=reference:raise ValueError('Another programme already owns this protected execution history')
    def guard():
        if (_actual_protocol(protocol_reference,inputs).payload!=protocol.payload
                or _read_metadata_study(study)['programme_reference']!=reference):raise ValueError('Actual frozen confirmation/programme context changed')
    execute,resolve=_step_operations(protocol,plan)
    value=_run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=guard,max_step_starts=max_step_starts)
    return dict(value,actual_frozen_practical_context_authenticated=True,
        scope='Actual fixed protected programme gates; complete outer observations/standalone costs, dependence qualifications and report claims remain separate')
