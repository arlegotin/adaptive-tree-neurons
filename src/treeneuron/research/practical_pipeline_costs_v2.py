"""Observed practical pipeline accounting; arithmetic is never provenance.

The public join reopens actual frozen native/model/raw/metric evidence. Private
interval or owned-negative fixtures cannot certify a complete procedure.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

from .deployment_worker_v2 import _source,_json
from .protected_journal_v2 import _path,_reference,_hash,_canonical,_write_new
from .execution_clock_v2 import validate_clock_domain,same_clock_domain
from .confirm_v2 import _STEPS,_checked_programme,_call_history,_invocation_history

_FALSE=dict(native_qualification_authenticated=False,whole_training_procedure_cost_authenticated=False,
    protected_access_authorized=False,cost_claim_authorized=False,T_accessed=None)


def _ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _interval(row,fields):
    if type(row) is not dict or set(row)!=fields:raise ValueError('Exact pipeline interval fields required')
    validate_clock_domain(row['clock_domain'])
    if (row['clock_domain']['status']!='observed' or any(type(row[k]) is not int or not 0<row[k]<2**63 for k in ('origin_ns','end_ns','elapsed_ns'))
            or row['end_ns']-row['origin_ns']!=row['elapsed_ns']):raise ValueError('Positive closed intervals and observed domains required for nesting')


def _overlap(a,b):return max(a['origin_ns'],b['origin_ns'])<min(a['end_ns'],b['end_ns'])


def _summarize_postT_intervals(frames,invocations,calls):
    """Pure supplied metadata; no file, native or physical provenance inferred."""
    if type(frames) is not list or not frames or type(invocations) is not list or not invocations or type(calls) is not list or not calls:
        raise ValueError('Complete explicit physical/invocation/call observations required')
    fs={};ivs={};seen=set();total={step:0 for step in _STEPS};routed={}
    for f in frames:
        _interval(f,{'frame_id','clock_domain','origin_ns','end_ns','elapsed_ns','invocation_ids'})
        if type(f['frame_id']) is not str or not f['frame_id'] or f['frame_id'] in fs or type(f['invocation_ids']) is not list or any(type(x) is not str for x in f['invocation_ids']) or len(set(f['invocation_ids']))!=len(f['invocation_ids']):raise ValueError('Unique frame and linked invocation identities required')
        fs[f['frame_id']]=f
    for index,f in enumerate(frames):
        if any(same_clock_domain(f['clock_domain'],g['clock_domain']) and _overlap(f,g) for g in frames[:index]):raise ValueError('Serial programme physical frames cannot overlap')
    for inv in invocations:
        _interval(inv,{'invocation_id','clock_domain','origin_ns','end_ns','elapsed_ns'})
        key=inv['invocation_id']
        if type(key) is not str or not key or key in ivs:raise ValueError('Unique programme invocation required')
        if any(same_clock_domain(inv['clock_domain'],other['clock_domain']) and _overlap(inv,other) for other in ivs.values()):
            raise ValueError('Serial programme invocations cannot overlap')
        matching=[f for f in frames if key in f['invocation_ids'] and same_clock_domain(f['clock_domain'],inv['clock_domain']) and f['origin_ns']<=inv['origin_ns']<inv['end_ns']<=f['end_ns']]
        if len(matching)!=1:raise ValueError('Every invocation needs exactly one committed containing owned frame')
        ivs[key]=inv;routed[key]=[]
    for call in calls:
        _interval(call,{'call_id','step','invocation_id','clock_domain','origin_ns','end_ns','elapsed_ns'})
        key=call['call_id'];step=call['step'];inv=ivs.get(call['invocation_id'])
        if type(key) is not str or not key or key in seen or step not in total or inv is None:raise ValueError('Unique fixed-step call and containing invocation required')
        if (not same_clock_domain(call['clock_domain'],inv['clock_domain']) or not inv['origin_ns']<=call['origin_ns']<call['end_ns']<=inv['end_ns']
                or any(_overlap(call,c) for c in routed[call['invocation_id']])):raise ValueError('Serial full calls must remain inside their observed invocation')
        seen.add(key);routed[call['invocation_id']].append(call);total[step]+=call['elapsed_ns']
    if any(n<=0 for n in total.values()):raise ValueError('All seven fixed operations require observed full-call costs')
    physical=sum(f['elapsed_ns'] for f in frames);nested=sum(v['elapsed_ns'] for v in invocations);full=sum(total.values())
    if not full<=nested<=physical:raise ValueError('Nested clocks cannot exceed containing physical work')
    domains=[]
    for f in frames:
        if not any(same_clock_domain(f['clock_domain'],d) for d in domains):domains.append(f['clock_domain'])
    span=max(f['end_ns'] for f in frames)-min(f['origin_ns'] for f in frames) if len(domains)==1 else None
    return dict(physical_worker_ns=physical,nested_invocation_ns=nested,full_step_call_ns=total,common_programme_overhead_ns=physical-full,
        observed_clock_domain_count=len(domains),observed_calendar_span_ns=span,observed_idle_ns=None if span is None else span-physical,**_FALSE,
        scope='Untrusted supplied intervals only; closed durations summed, nested calls not added, no native or actual physical provenance')


def _discover_confirmation_envelopes(reference,plan):
    """Discover ALL immutable bound slots, including denied/failed workers."""
    import fcntl
    from .program_process_v2 import inspect_program_process
    _checked_programme(reference,plan);root=_path(plan['observer_directory'])
    if not root.is_dir() or (root/'worker-lease.json').is_symlink():raise ValueError('Complete inactive owned observer archive required')
    with (root/'worker-lease.json').open('r') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as error:raise PermissionError('Cannot cost an active confirmation observer') from error
        try:
            paths=sorted(p for p in root.iterdir() if p.name!='worker-lease.json')
            if not paths or [p.name for p in paths]!=[f'episode-{i:04}' for i in range(len(paths))] or any(p.is_symlink() or not p.is_dir() for p in paths):raise ValueError('Complete contiguous observer history required')
            frames=[]
            for path in paths:
                if {p.name for p in path.iterdir()}!={'envelope.json','request.json','process'}:
                    raise ValueError('Unknown/partial observer episode evidence cannot be omitted')
                target=path/'envelope.json'
                if not target.is_file():raise ValueError('Missing outer worker terminal cannot be omitted or zero-imputed')
                ref=_ref(target);proof=inspect_program_process(ref);_,raw=_reference(ref);envelope=_json(raw);_,raw=_reference(envelope['request_reference']);request=_json(raw)
                args=request['arguments']
                if (request['tranche']!='confirmation' or args.get('programme_reference')!=reference or args.get('directory')!=plan['directory']
                        or args.get('protocol_reference')!=plan['protocol_reference'] or any(args.get(k)!=v for k,v in plan['inputs'].items())
                        or proof['child_started'] and not proof['worker_protocol_closed']):raise ValueError('Outer source/programme/input/worker closure binding differs')
                linked=[]
                if envelope['body_reference'] is not None:
                    _,raw=_reference(envelope['body_reference']);linked=_json(raw)['program_invocation_references']
                frames.append(dict(reference=ref,proof=proof,linked_invocation_references=linked))
            for frame in frames:_reference(frame['reference'])
            _checked_programme(reference,plan)
            return dict(frames=frames,observer_directory=str(root),**_FALSE,scope='Readonly owned archive/clock metadata only; actual frozen pipeline gates remain separate')
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def _actual_inputs(reference,kwargs,*,analysis_reference,raw_audit_reference,prediction_stage_reference,directory):
    from .confirm_v2 import _actual_protocol
    from .independent_raw_audit_v2 import load_protected_raw_audit
    from .frozen_practical_analysis_v2 import _actual_metrics,load_protected_practical_analysis
    from .practical_preT_costs_v2 import load_practical_preT_costs
    protocol=_actual_protocol(reference,kwargs)
    raw=load_protected_raw_audit(raw_audit_reference,protocol_reference=reference,analysis_reference=analysis_reference,
        prediction_stage_reference=prediction_stage_reference,directory=directory,**kwargs)
    actual,stage,metrics,completion=_actual_metrics(reference,prediction_stage_reference,directory,kwargs)
    analysis=load_protected_practical_analysis(analysis_reference,protocol_reference=reference,prediction_stage_reference=prediction_stage_reference,directory=directory,**kwargs)
    pre=load_practical_preT_costs(kwargs['preT_costs_reference'],**{k:kwargs[k] for k in ('models_reference','global_barrier_reference','qualification_reference')})
    if actual.payload!=protocol.payload:raise ValueError('Actual pipeline inputs changed')
    return protocol,stage,metrics,completion,analysis,raw,pre


def _cost_output_location(output,programme,reference):
    from .primary_program_v2 import _check_program_location
    path=_check_program_location(output,working_roots=[programme['directory'],programme['observer_directory'],programme['protected_study_directory']],
        proof_roots=[],input_files=[reference['path']])
    if path.exists():raise ValueError('Fresh immutable final-cost output required')
    return path


@dataclass(frozen=True)
class PracticalPipelineCosts:
    reference:dict
    _payload_json:str
    @property
    def payload(self):return json.loads(self._payload_json)


def seal_practical_pipeline_costs(reference,*,programme_reference,analysis_reference,raw_audit_reference,prediction_stage_reference,
        directory,output,models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    inputs=_actual_inputs(reference,kwargs,analysis_reference=analysis_reference,raw_audit_reference=raw_audit_reference,
        prediction_stage_reference=prediction_stage_reference,directory=directory)
    payload=_assemble_pipeline(inputs,programme_reference=programme_reference,directory=directory)
    from .protected_prediction_gate_v2 import _publication_location
    path=_publication_location(output,protocol=inputs[0],inspection=inputs[1].payload['inspection'])
    _,content=_reference(programme_reference);_cost_output_location(path,_json(content),programme_reference)
    again=_actual_inputs(reference,kwargs,analysis_reference=analysis_reference,raw_audit_reference=raw_audit_reference,
        prediction_stage_reference=prediction_stage_reference,directory=directory)
    if _assemble_pipeline(again,programme_reference=programme_reference,directory=directory)!=payload:raise ValueError('Actual pipeline costs changed before sealing')
    _write_new(path,dict(schema='research/v2-sealed-practical-pipeline-costs',payload=payload,payload_sha256=_hash(payload)))
    return PracticalPipelineCosts(_ref(path),_canonical(payload))


def _attribute_query_calls(queries,calls):
    """All attempted calls, not only winners; aliased views reuse their query."""
    if type(queries) is not list or not queries or type(calls) is not list or not calls:raise ValueError('Explicit complete query/call inventory required')
    mapping={};per_job={};per_query={};seen=set()
    for query in queries:
        qid=query['query_id'];job=query['job_id']
        if type(qid) is not str or not qid or qid in mapping or type(job) is not str or not job:raise ValueError('Unique frozen query/job mapping required')
        mapping[qid]=job;per_job.setdefault(job,0);per_query[qid]=0
    for call in calls:
        if type(call) is not dict or set(call)!={'call_id','query_id','elapsed_ns'} or type(call['call_id']) is not str or not call['call_id'] or call['call_id'] in seen or call['query_id'] not in mapping or type(call['elapsed_ns']) is not int or call['elapsed_ns']<=0:
            raise ValueError('Each failed/successful query call must be paid once')
        seen.add(call['call_id']);qid=call['query_id'];per_query[qid]+=call['elapsed_ns'];per_job[mapping[qid]]+=call['elapsed_ns']
    if any(value<=0 for value in per_query.values()):raise ValueError('Missing planned query calls cannot be costed as zero')
    return dict(per_job_ns=per_job,per_query_ns=per_query,call_count=len(calls),**_FALSE,scope='Supplied query attribution only; no observed execution/native provenance')


def _collect_postT_programme(reference,plan):
    _checked_programme(reference,plan);calls=_call_history(reference,plan);invocations=_invocation_history(reference,plan,calls)
    if not any(i['clock']['status']=='completed' and i['clock']['completed_steps']==list(_STEPS) for i in invocations):
        raise ValueError('A complete seven-operation programme is required for complete pipeline costs')
    discovered=_discover_confirmation_envelopes(reference,plan);frames=[];inv_rows=[];call_rows=[];parents={}
    for frame in discovered['frames']:
        proof=frame['proof'];frames.append(dict(frame_id=frame['reference']['path'],**{k:proof[k] for k in ('clock_domain','origin_ns','end_ns','elapsed_ns')},
            invocation_ids=[r['path'] for r in frame['linked_invocation_references']]))
    for inv in invocations:
        clock=inv['clock'];identity=inv['reference']['path'];inv_rows.append(dict(invocation_id=identity,**{k:clock[k] for k in ('clock_domain','origin_ns','end_ns','elapsed_ns')}))
        for ref in clock['call_references']:
            if ref['path'] in parents:raise ValueError('Full step call cannot be double-counted')
            parents[ref['path']]=identity
    for call in calls:
        end=call['end'];call_rows.append(dict(call_id=call['end_reference']['path'],step=call['step'],invocation_id=parents[call['end_reference']['path']],
            **{k:end[k] for k in ('clock_domain','origin_ns','end_ns','elapsed_ns')}))
    summary=_summarize_postT_intervals(frames,inv_rows,call_rows)
    return dict(programme_reference=reference,programme_hash=plan['programme_hash'],summary=summary,frames=discovered['frames'],
        invocations=invocations,calls=calls,**_FALSE,scope='Complete owned programme/call observations; actual frozen raw/native/model gates remain external')


def _inside_operation(clock,operations,step):
    validate_clock_domain(clock['clock_domain'])
    if clock['clock_domain']['status']!='observed' or any(type(clock[k]) is not int or clock[k]<=0 for k in ('origin_ns','end_ns','elapsed_ns')) or clock['end_ns']-clock['origin_ns']!=clock['elapsed_ns']:
        raise ValueError('Observed closed nested component clock required')
    matches=[call for call in operations if call['step']==step and same_clock_domain(clock['clock_domain'],call['end']['clock_domain'])
        and call['end']['origin_ns']<=clock['origin_ns']<clock['end_ns']<=call['end']['end_ns']]
    if len(matches)!=1:raise ValueError('Every nested component must belong to exactly one full fixed-operation call')
    return matches[0]['end_reference']


def _assemble_pipeline(inputs,*,programme_reference,directory):
    protocol,stage,metrics,completion,analysis,raw_audit,pre=inputs;payload=protocol.payload
    _,content=_reference(programme_reference);programme=_json(content);_checked_programme(programme_reference,programme)
    from .protected_journal_v2 import _read_metadata_study
    from .confirm_v2 import _step_operations
    from .native_deployment_execution_v2 import collect_native_deployment_measurements
    study=_path(directory);state=_read_metadata_study(study)
    expected_inputs={k:payload[k] for k in ('models_reference','global_barrier_reference','qualification_reference','preT_costs_reference','readiness_reference')}
    if (programme['protocol_reference']!=protocol.reference or programme['protected_study_directory']!=str(study)
            or programme['inputs']!=expected_inputs or state['programme_reference']!=programme_reference
            or pre.payload['source_sha256']!=payload['source_sha256'] or raw_audit['genuine_frozen_raw_audit_authenticated'] is not True
            or raw_audit['metrics_completion_reference']!=completion or analysis.payload['analysis']['metric_inputs_sha256']!=_hash(metrics)):
        raise ValueError('Actual pipeline programme/owner/native/model/raw/metric inputs differ')
    _,resolve=_step_operations(protocol,programme)
    for step in _STEPS:
        value=resolve(step)
        if value is None or value['status']!='completed':raise ValueError('Every frozen operation must have its actual complete proof')
    observed=_collect_postT_programme(programme_reference,programme);operations=observed['calls'];nested=[];query_calls=[];queries=payload['prediction_manifest']['queries']
    inspection=stage.payload['inspection']
    for reference in inspection['invocation_references']:
        _,content=_reference(reference);clock=_json(content);nested.append(dict(kind='prediction_invocation',reference=reference,containing_call=_inside_operation(clock,operations,'predictions')))
        for call_reference in clock['call_references']:
            _,content=_reference(call_reference);call=_json(content)
            query_calls.append(dict(call_id=call_reference['path'],query_id=call['query_id'],elapsed_ns=call['elapsed_ns']))
    attributed=_attribute_query_calls(queries,query_calls)
    if sum(attributed['per_query_ns'].values())!=inspection['nested_executor_call_ns']:raise ValueError('Complete successful/failed query costs differ from prediction evidence')
    _,content=_reference(completion);metrics_completion=_json(content);_,content=_reference(metrics_completion['clock_reference']);clock=_json(content)
    nested.append(dict(kind='sole_label_stage',reference=metrics_completion['clock_reference'],containing_call=_inside_operation(clock,operations,'metrics')))
    _,content=_reference(raw_audit['terminal_reference']);clock=_json(content)
    nested.append(dict(kind='independent_raw_audit',reference=raw_audit['terminal_reference'],containing_call=_inside_operation(clock,operations,'raw_audit')))
    timing_ref=_ref(_path(programme['outputs']['native_timing_plan']))
    timing=collect_native_deployment_measurements(timing_ref,protocol_reference=protocol.reference,directory=study,
        output=_path(programme['outputs']['native_measurements']),rss_limit_bytes=programme['rss_limit_bytes'],**expected_inputs)
    if timing['status']!='complete_descriptive_measurements' or not timing['all_attempt_costs_observed'] or timing['completed_measurements']!=9360 or timing['complete_30_block_comparisons']!=156:
        raise ValueError('Complete9360 original-model timings and all failed/paused costs required')
    for reference in timing['context_references']:
        _,content=_reference(reference);clock=_json(content);nested.append(dict(kind='native_measurement_context',reference=reference,containing_call=_inside_operation(clock,operations,'native_measurements')))
    pre_summary=pre.payload['interval_summary'];standalone=pre_summary['standalone_full_call_ns'];query_job=attributed['per_job_ns']
    if not set(query_job)<=set(standalone) or len(standalone)!=840:raise ValueError('Frozen840 procedure jobs must own all query costs')
    totals={job:amount+query_job.get(job,0) for job,amount in standalone.items()}
    no_queries=sorted(set(standalone)-set(query_job));physical=pre_summary['physical_worker_ns']+observed['summary']['physical_worker_ns']
    # Physical programme work includes query/evaluation/timing frames. The
    # standalone row is a different view and is never added to that total.
    result=dict(schema='research/v2-practical-pipeline-cost-payload',source_sha256=payload['source_sha256'],protocol_reference=protocol.reference,
        programme_reference=programme_reference,preT_costs_reference=payload['preT_costs_reference'],prediction_stage_reference=stage.reference,
        analysis_reference=analysis.reference,raw_audit_reference=raw_audit['bound_audit_reference'],metrics_completion_reference=completion,
        postT_observations=observed,nested_component_routes=nested,timing_summary=timing,
        physical_owned_execution_ns=physical,preT_physical_worker_ns=pre_summary['physical_worker_ns'],postT_physical_worker_ns=observed['summary']['physical_worker_ns'],
        standalone_training_full_call_ns=standalone,all_frozen_query_full_call_ns=attributed['per_job_ns'],standalone_training_plus_all_frozen_queries_ns=totals,
        jobs_with_no_planned_frozen_prediction_queries=no_queries,
        standalone_scope='Each of840 complete training/search procedures plus all its frozen native/prefix prediction calls and failures. Zero query cost only where no query was planned, never for a missing planned call. Common evaluation and timing-study overhead is separate.',
        common_postT_full_step_call_ns=observed['summary']['full_step_call_ns'],common_postT_programme_overhead_ns=observed['summary']['common_programme_overhead_ns'],
        costed_job_count=840,costed_query_count=len(queries),native_qualification_authenticated=True,owned_pipeline_execution_cost_authenticated=True,
        whole_training_procedure_cost_authenticated=True,protected_access_authorized=False,cost_claim_authorized=False,T_accessed=True,
        dedicated_timing_qualified=False,quality_precision_equivalence_authenticated=False,
        excluded_or_separately_reported_costs=['Engineering/data acquisition/qualification work outside the frozen procedure',
            'Observer final receipt publication/return, final cost/audit/report computation and common cold parent startup outside declared frames; not zero-imputed'],
        scope='Actual complete pre-test training and owned protected evaluation/timing execution. Physical frames, standalone procedures and nested diagnostic clocks are distinct views; never sum them. No formal efficiency claim or test-population inference.')
    if _source()!=payload['source_sha256']:raise ValueError('Actual pipeline source changed during cost reconstruction')
    for row in nested:_reference(row['reference'])
    _reference(programme_reference)
    return result


def load_practical_pipeline_costs(reference,*,protocol_reference,programme_reference,analysis_reference,raw_audit_reference,prediction_stage_reference,
        directory,models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    inputs=_actual_inputs(protocol_reference,kwargs,analysis_reference=analysis_reference,raw_audit_reference=raw_audit_reference,
        prediction_stage_reference=prediction_stage_reference,directory=directory)
    _,content=_reference(reference);saved=_json(content)
    if type(saved) is not dict or set(saved)!={'schema','payload','payload_sha256'} or saved['schema']!='research/v2-sealed-practical-pipeline-costs' or type(saved['payload']) is not dict or _hash(saved['payload'])!=saved['payload_sha256']:
        raise ValueError('Exact immutable practical pipeline cost envelope required')
    actual=_assemble_pipeline(inputs,programme_reference=programme_reference,directory=directory)
    if saved['payload']!=actual:raise ValueError('Actual complete pipeline/observer/call costs changed')
    again=_actual_inputs(protocol_reference,kwargs,analysis_reference=analysis_reference,raw_audit_reference=raw_audit_reference,
        prediction_stage_reference=prediction_stage_reference,directory=directory)
    if _assemble_pipeline(again,programme_reference=programme_reference,directory=directory)!=actual:raise ValueError('Actual cost context changed during readback')
    _reference(reference)
    return PracticalPipelineCosts(copy.deepcopy(reference),_canonical(actual))
