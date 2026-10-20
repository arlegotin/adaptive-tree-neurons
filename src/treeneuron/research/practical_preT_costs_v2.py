"""Practical training/search cost joins; helpers grant no scientific gate.

Elapsed worker time, full method calls, nested caller clocks and common study
overhead are distinct. Absolute timestamps are compared only on observed equal
boot domains. Public actual-model/native admission is a separate prerequisite.
"""
import hashlib
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import time

from .deployment_worker_v2 import _source,_json
from .execution_clock_v2 import validate_clock_domain,same_clock_domain,_read_reference,inspect_program_invocation_clock


_FALSE=dict(native_qualification_authenticated=False,whole_training_procedure_cost_authenticated=False,
    protected_access_authorized=False,T_accessed=False)


def _interval(row,fields):
    if type(row) is not dict or set(row)!=fields:raise ValueError('Exact execution interval fields required')
    validate_clock_domain(row['clock_domain'])
    if (any(type(row[k]) is not int or not 0<row[k]<2**63 for k in ('origin_ns','end_ns','elapsed_ns'))
            or row['end_ns']-row['origin_ns']!=row['elapsed_ns']):raise ValueError('Strict positive execution intervals required')


def _overlap(a,b):return max(a['origin_ns'],b['origin_ns'])<min(a['end_ns'],b['end_ns'])


def _summarize_intervals(outer,calls,*,expected_job_ids,controller_paid_ns):
    """Arithmetic on supplied metadata only, never source/process authority."""
    if (type(outer) is not list or not outer or type(calls) is not list or not calls
            or type(expected_job_ids) is not list or not expected_job_ids
            or any(type(j) is not str or not j for j in expected_job_ids)
            or len(set(expected_job_ids))!=len(expected_job_ids)
            or type(controller_paid_ns) is not dict or set(controller_paid_ns)!=set(expected_job_ids)):
        raise ValueError('Explicit complete job/call/worker accounting required')
    frames={}
    for row in outer:
        _interval(row,{'envelope_id','clock_domain','origin_ns','end_ns','elapsed_ns'})
        name=row['envelope_id']
        if type(name) is not str or not name or name in frames:raise ValueError('Unique outer execution identities required')
        if row['clock_domain']['status']!='observed':raise ValueError('Observed domains required for complete interval routing')
        frames[name]=row
    for i,first in enumerate(outer):
        for second in outer[i+1:]:
            if same_clock_domain(first['clock_domain'],second['clock_domain']) and _overlap(first,second):
                raise ValueError('Serial program root intervals cannot overlap')
    totals={name:0 for name in expected_job_ids};seen=set();inner=0;per_frame={name:[] for name in frames}
    for row in calls:
        _interval(row,{'call_id','job_id','envelope_id','clock_domain','origin_ns','end_ns','elapsed_ns','caller_elapsed_ns'})
        identity=row['call_id'];job=row['job_id'];parent=row['envelope_id']
        if (type(identity) is not str or not identity or identity in seen or type(job) is not str or job not in totals
                or type(parent) is not str or parent not in frames or type(row['caller_elapsed_ns']) is not int
                or not 0<row['caller_elapsed_ns']<=row['elapsed_ns']):raise ValueError('Unique bound calls and known inner intervals required')
        frame=frames[parent]
        if (not same_clock_domain(row['clock_domain'],frame['clock_domain'])
                or not frame['origin_ns']<=row['origin_ns']<row['end_ns']<=frame['end_ns']):
            raise ValueError('Full call cannot escape or substitute its observed outer domain')
        if any(_overlap(row,other) for other in per_frame[parent]):raise ValueError('Sequential method calls cannot overlap')
        per_frame[parent].append(row);seen.add(identity);totals[job]+=row['elapsed_ns'];inner+=row['caller_elapsed_ns']
    if any(total<=0 for total in totals.values()):raise ValueError('Every planned method needs full observed calls')
    for job,paid in controller_paid_ns.items():
        if type(paid) is not int or not 0<=paid<=totals[job]:raise ValueError('Paid HPO cannot exceed its fully observed method calls')
    physical=sum(r['elapsed_ns'] for r in outer);full=sum(totals.values());domains=[]
    for row in outer:
        if not any(same_clock_domain(row['clock_domain'],d) for d in domains):domains.append(row['clock_domain'])
    calendar=max(r['end_ns'] for r in outer)-min(r['origin_ns'] for r in outer) if len(domains)==1 else None
    return dict(physical_worker_ns=physical,standalone_full_call_ns=totals,known_inner_caller_ns=inner,
        common_program_overhead_ns=physical-full,observed_calendar_span_ns=calendar,
        observed_idle_ns=None if calendar is None else calendar-physical,observed_clock_domain_count=len(domains),
        scope='Untrusted supplied intervals only; nested clocks not added; no measured physical/native provenance inferred',**_FALSE)


def _collect_program(program_reference,*,expected_journal_sha256):
    """Reopen actual queue/call/invocation bytes, without native/model authority."""
    from treeneuron.config import canonical_hash
    from .primary_program_v2 import _progress
    from .program_call_clock_v2 import inspect_program_call_clock,_program
    saved,program_raw,program=_read_reference(program_reference);root=_program(program)
    if saved!=root/'plan.json':raise ValueError('Exact immutable program plan required')
    journal=root/'events.jsonl';raw=journal.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_journal_sha256:raise ValueError('Externally pinned final program journal required')
    progress=_progress(program,None)
    if progress['pending'] is not None:raise PermissionError('Unclosed queue episode prevents complete cost collection')
    calls=[];expected_files=set();paid={}
    for line in raw.splitlines():
        event=_json(line)
        if event['kind']!='close_episode':continue
        job=event['job_id'];index=event['episode_index'];episode=Path(event['episode_root'])
        path=root/'call-clocks'/job/f'episode-{index:04}.json';before=path.read_bytes()
        reference=dict(path=str(path),sha256=hashlib.sha256(before).hexdigest())
        proof=inspect_program_call_clock(reference,program=program)
        if not proof['caller_boundary_checked']:raise ValueError('Closed queue cost requires its actual caller boundary')
        call=_json(before)
        if call['caller_reference']!=event['completion_reference']:raise ValueError('Full call replaced the queue-selected caller')
        calls.append(dict(reference=reference,**proof));expected_files.add(path)
        outcome=_json((episode/'episode.json').read_bytes())
        if 'closed_body_reference' in outcome:
            body=_json((episode/'process/body-outcome.json').read_bytes())
            result_path,_,result=_read_reference(body['search_result_reference'])
            paid[job]=result['total_charged_ns']
    if set((root/'call-clocks').rglob('*.json'))!=expected_files:
        raise ValueError('Orphan or missing full-call observations prevent complete cost collection')
    invocations=[];committed={}
    for path in sorted((root/'invocations').glob('*.json')):
        before=path.read_bytes();reference=dict(path=str(path),sha256=hashlib.sha256(before).hexdigest())
        proof=inspect_program_invocation_clock(reference,program_reference=program_reference)
        value=_json(before)
        for ref in value['call_clock_references']:
            if ref['path'] in committed and committed[ref['path']]!=ref['sha256']:
                raise ValueError('An earlier invocation committed different full-call bytes')
            committed[ref['path']]=ref['sha256']
        invocations.append(dict(reference=reference,proof=proof,call_clock_references=value['call_clock_references']))
    if not invocations or committed!={r['reference']['path']:r['reference']['sha256'] for r in calls}:
        raise ValueError('Every complete call needs its containing invocation byte commitment')
    if (saved.read_bytes()!=program_raw or journal.read_bytes()!=raw or _source()!=program['source_sha256']):
        raise ValueError('Program cost inputs changed while collecting')
    return dict(program=program,program_reference=program_reference,journal_sha256=expected_journal_sha256,
        completed_jobs=progress['completed_jobs'],calls=calls,invocations=invocations,controller_paid_ns=paid,
        collection_sha256=canonical_hash(dict(program_reference=program_reference,journal_sha256=expected_journal_sha256,
            calls=calls,invocations=invocations,controller_paid_ns=paid)),**_FALSE)


def _actual_models(models_reference,global_barrier_reference,qualification_reference):
    from .qualified_practical_models_v2 import load_qualified_practical_models
    return load_qualified_practical_models(models_reference,global_barrier_reference=global_barrier_reference,
        qualification_reference=qualification_reference)


def _assemble(*,models,models_reference,global_barrier_reference,qualification_reference,program_bundles,created_utc_ns):
    """Internal composition; public callers must reopen actual models first."""
    from treeneuron.config import canonical_hash,canonical_json
    from .primary_selection_v2 import _ref
    from .workload import PracticalPlan
    from .primary_program_v2 import build_primary_program_plan
    from .secondary_program_v2 import build_secondary_program_plan
    from .program_process_v2 import inspect_program_process
    from .native_resource_forecast_v2 import load_native_practical_resource_forecast
    payload=models.payload
    if (type(program_bundles) is not dict or set(program_bundles)!={'primary','secondary'}
            or type(created_utc_ns) is not int or created_utc_ns<=0):raise ValueError('Exact two-tranche execution bundles required')
    practical=PracticalPlan(canonical_json(_ref(payload['practical_plan_reference'],{},decode=True)))
    plan=practical.to_dict();model_rows={r['job_id']:r for r in payload['models']}
    outer=[];calls=[];paid={};collections={};proof_paths=[]
    for tranche in ('primary','secondary'):
        bundle=program_bundles[tranche]
        if (type(bundle) is not dict or set(bundle)!={'program_reference','journal_sha256','envelope_references'}
                or type(bundle['envelope_references']) is not list or not bundle['envelope_references']):
            raise ValueError('Program plan, final journal and all owned outer invocations required')
        collected=_collect_program(bundle['program_reference'],expected_journal_sha256=bundle['journal_sha256'])
        program=collected['program'];jobs=[j for j in plan['jobs'] if j['tranche']==tranche]
        expected_kwargs=dict(practical_plan=practical,practical_plan_reference=payload['practical_plan_reference'],
            expected_practical_plan_hash=payload['expected_practical_plan_hash'],support_reference=payload['support_reference'],
            procedure_reference=payload['procedure_reference'],qualification_reference=qualification_reference,
            resource_forecast_reference=program['resource_forecast_reference'],directory=program['directory'],rss_limit_bytes=program['rss_limit_bytes'])
        expected=(build_primary_program_plan(**expected_kwargs) if tranche=='primary' else
            build_secondary_program_plan(**expected_kwargs,global_barrier_reference=global_barrier_reference))
        if canonical_json(expected)!=canonical_json(program) or program['jobs']!=jobs or collected['completed_jobs']!=len(jobs):
            raise ValueError('Every fixed primary168/secondary672 job must terminate in its exact queue')
        if any(model_rows[j['job_id']]['search_directory']!=str(Path(program['directory'])/'searches'/j['job_id']) for j in jobs):
            raise ValueError('Full models must belong to the costed actual program searches')
        load_native_practical_resource_forecast(program['resource_forecast_reference'],practical_plan=practical,
            expected_practical_plan_hash=payload['expected_practical_plan_hash'],support_reference=payload['support_reference'],qualification_reference=qualification_reference)
        envelopes=[];seen=set()
        for reference in bundle['envelope_references']:
            path,_,value=_read_reference(reference);proof=inspect_program_process(reference)
            if str(path) in seen:raise ValueError('An outer invocation cannot be paid twice')
            seen.add(str(path));proof_paths.append(str(path.parent))
            _,_,request=_read_reference(value['request_reference']);args=request['arguments']
            if (request['tranche']!=tranche or request['source_sha256']!=payload['source_sha256']
                    or args['qualification_reference']!=qualification_reference
                    or args['resource_forecast_reference']!=program['resource_forecast_reference']
                    or args['directory']!=program['directory'] or args['rss_limit_bytes']!=program['rss_limit_bytes']):
                raise ValueError('Outer worker substituted its frozen program identity')
            if tranche=='primary':
                if any(args[k]!=program[k] for k in ('practical_plan_reference','expected_practical_plan_hash','support_reference','procedure_reference')):
                    raise ValueError('Primary outer worker changed frozen recipe references')
            elif args['barrier_reference']!=global_barrier_reference:raise ValueError('Secondary outer worker changed global barrier')
            linked=[]
            if value['body_reference'] is not None:
                _,_,body=_read_reference(value['body_reference']);linked=body['program_invocation_references']
            if proof['child_started'] and not proof['worker_protocol_closed']:
                raise ValueError('Unknown/unclosed outer worker prevents complete execution accounting')
            entry=dict(envelope_id=reference['sha256'],**{k:proof[k] for k in ('clock_domain','origin_ns','end_ns','elapsed_ns')})
            outer.append(entry);envelopes.append(dict(interval=entry,linked=linked))
        assigned={}
        for invocation in collected['invocations']:
            proof=invocation['proof'];reference=invocation['reference']
            matches=[e for e in envelopes if reference in e['linked'] and same_clock_domain(e['interval']['clock_domain'],proof['clock_domain'])
                and e['interval']['origin_ns']<=proof['origin_ns']<=proof['end_ns']<=e['interval']['end_ns']]
            if len(matches)!=1:raise ValueError('Each actual program invocation needs one committed containing outer worker')
            for call_ref in invocation['call_clock_references']:
                assigned.setdefault(call_ref['path'],[]).append((proof,matches[0]['interval']))
        for call in collected['calls']:
            matches=[frame for inv,frame in assigned.get(call['reference']['path'],[]) if same_clock_domain(inv['clock_domain'],call['clock_domain'])
                and inv['origin_ns']<=call['origin_ns']<=call['end_ns']<=inv['end_ns']]
            if len(matches)!=1 or not call['caller_interval_containment_observed']:
                raise ValueError('Every full method call needs its actual observed invocation/caller containment')
            calls.append(dict(call_id=call['reference']['sha256'],job_id=call['job_id'],envelope_id=matches[0]['envelope_id'],
                **{k:call[k] for k in ('clock_domain','origin_ns','end_ns','elapsed_ns','caller_elapsed_ns')}))
        if set(collected['controller_paid_ns'])!={j['job_id'] for j in jobs}:
            raise ValueError('Every planned search needs its actual paid HPO total')
        paid.update(collected['controller_paid_ns']);collections[tranche]=collected['collection_sha256'];proof_paths.append(program['directory'])
    summary=_summarize_intervals(outer,calls,expected_job_ids=[j['job_id'] for j in plan['jobs']],controller_paid_ns=paid)
    if _source()!=payload['source_sha256']:raise ValueError('Practical cost source changed during collection')
    return dict(schema='research/v2-practical-preT-cost-payload',created_utc_ns=created_utc_ns,source_sha256=payload['source_sha256'],
        models_reference=copy.deepcopy(models_reference),global_barrier_reference=copy.deepcopy(global_barrier_reference),
        qualification_reference=copy.deepcopy(qualification_reference),program_bundles=copy.deepcopy(program_bundles),
        collection_hashes=collections,interval_summary=summary,controller_paid_ns=paid,
        costed_job_count=len(model_rows),preT_training_execution_cost_authenticated=True,protected_access_authorized=False,T_accessed=False,
        whole_training_procedure_cost_authenticated=False,cost_claim_authorized=False,
        proof_roots=sorted(set(proof_paths)),
        scope='Complete owned practical training/search execution and separately counted program overhead. Data acquisition/engineering qualification, observer publication/operator overhead and future protected/deployment stages are separate, never zero-imputed. No whole-pipeline or efficiency claim.')


def _publication_location(output,*,payload):
    from .native_qualification_registry_v2 import load_native_qualification_registry,native_registry_storage_roots
    from .primary_program_v2 import _check_program_location
    q=payload['qualification_reference'];registry=load_native_qualification_registry(q)
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=q)
    path=_check_program_location(output,working_roots=roots['working_roots'],proof_roots=[*roots['proof_roots'],*payload['proof_roots']],
        input_files=[payload[k]['path'] for k in ('models_reference','global_barrier_reference')])
    if path.exists() or path.is_symlink():raise ValueError('Fresh pre-T cost seal required')
    return path


@dataclass(frozen=True)
class PracticalPreTCosts:
    reference:dict
    _payload_json:str

    @property
    def payload(self):return json.loads(self._payload_json)


def seal_practical_preT_costs(*,models_reference,global_barrier_reference,qualification_reference,program_bundles,output):
    models=_actual_models(models_reference,global_barrier_reference,qualification_reference)
    payload=_assemble(models=models,models_reference=models_reference,global_barrier_reference=global_barrier_reference,
        qualification_reference=qualification_reference,program_bundles=program_bundles,created_utc_ns=time.time_ns())
    from treeneuron.config import canonical_hash,canonical_json
    from .ledger import write_new_json
    if _actual_models(models_reference,global_barrier_reference,qualification_reference).payload!=models.payload:
        raise ValueError('Actual model/native evidence changed before cost publication')
    path=_publication_location(output,payload=payload)
    write_new_json(path,dict(schema='research/v2-practical-preT-costs',payload=payload,payload_hash=canonical_hash(payload)))
    return PracticalPreTCosts(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),canonical_json(payload))


def load_practical_preT_costs(reference,*,models_reference,global_barrier_reference,qualification_reference):
    models=_actual_models(models_reference,global_barrier_reference,qualification_reference)
    from treeneuron.config import canonical_hash,canonical_json
    path,_,value=_read_reference(reference)
    if (type(value) is not dict or set(value)!={'schema','payload','payload_hash'} or value['schema']!='research/v2-practical-preT-costs'
            or type(value['payload']) is not dict or canonical_hash(value['payload'])!=value['payload_hash']):
        raise ValueError('Immutable pre-T cost envelope changed')
    saved=value['payload']
    try:actual=_assemble(models=models,models_reference=models_reference,global_barrier_reference=global_barrier_reference,
        qualification_reference=qualification_reference,program_bundles=saved['program_bundles'],created_utc_ns=saved['created_utc_ns'])
    except (KeyError,TypeError) as error:raise ValueError('Malformed pre-T cost payload') from error
    if canonical_json(saved)!=canonical_json(actual):raise ValueError('Actual pre-T cost evidence changed')
    _read_reference(reference)
    return PracticalPreTCosts(copy.deepcopy(reference),canonical_json(actual))
