"""Serial primary program with immutable episodes; no secondary/T authority."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_program_location(directory,*,working_roots,proof_roots,input_files):
    from .primary_search_worker_v2 import _path,_ROOT
    root=_path(str(directory))
    protected=[*(Path(p) for p in working_roots+proof_roots+input_files),Path(sys.prefix).resolve(),
        Path(sys.executable).resolve().parent,*((_ROOT/p).resolve() for p in ('src','tests','configs','locks','protocol','environment','.git','.agents','.codex','.aws'))]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError('Program outputs must preserve all working/proof/executable/input paths')
    return root


def build_primary_program_plan(*,practical_plan,practical_plan_reference,expected_practical_plan_hash,
        support_reference,procedure_reference,qualification_reference,resource_forecast_reference,
        directory,rss_limit_bytes=4*1024**3):
    """Bind metadata only; supplied qualification references grant no authority."""
    from treeneuron.config import canonical_hash,canonical_json
    from .primary_selection_v2 import _primary,_ref
    from .practical_procedure_v2 import load_practical_procedure
    from .primary_search_worker_v2 import _path,_reference,_digest
    value,jobs=_primary(practical_plan,expected_practical_plan_hash)
    if value.get('budget_policy')!='native_quality':raise ValueError('Native-quality primary program required')
    stored=_ref(practical_plan_reference,{},decode=True)
    if canonical_json(stored)!=canonical_json(value):raise ValueError('Exact practical plan bytes required')
    load_practical_procedure(procedure_reference,practical_plan=practical_plan,
        expected_practical_plan_hash=expected_practical_plan_hash,support_reference=support_reference)
    _reference(resource_forecast_reference)
    if (type(qualification_reference) is not dict or set(qualification_reference)!={'root','inventory_sha256'}
            or not _digest(qualification_reference['inventory_sha256']) or type(rss_limit_bytes) is not int
            or not 0<rss_limit_bytes<=16*1024**3):raise ValueError('Exact registry and bounded root RSS required')
    _path(qualification_reference['root']);root=_path(str(directory))
    result=dict(schema='research/v2-primary-program-plan',source_sha256=value['source_sha256'],
        practical_plan_reference=copy.deepcopy(practical_plan_reference),expected_practical_plan_hash=expected_practical_plan_hash,
        support_reference=copy.deepcopy(support_reference),procedure_reference=copy.deepcopy(procedure_reference),
        qualification_reference=copy.deepcopy(qualification_reference),resource_forecast_reference=copy.deepcopy(resource_forecast_reference),
        directory=str(root),rss_limit_bytes=rss_limit_bytes,jobs=copy.deepcopy(jobs),worker_concurrency=1,
        execution_authorized=False,native_workload_qualified=False,secondary_release_authorized=False,
        protected_access_authorized=False,T_accessed=False,
        scope='Immutable primary-only metadata queue; actual qualification, admission, native terminal audits and final model/global freeze remain required')
    result['program_hash']=canonical_hash(result)
    return result


def _request_binding(program,job):
    secondary=program.get('schema')=='research/v2-secondary-program-plan'
    value=dict(schema='research/v2-secondary-program-episode-request' if secondary else 'research/v2-primary-program-episode-request',source_sha256=program['source_sha256'],
        **{k:copy.deepcopy(program[k]) for k in ('practical_plan_reference','expected_practical_plan_hash','support_reference',
            'qualification_reference','resource_forecast_reference','procedure_reference','rss_limit_bytes')},
        job_id=job['job_id'],search_directory=str(Path(program['directory'])/'searches'/job['job_id']))
    if secondary:value['global_barrier_reference']=copy.deepcopy(program['global_barrier_reference'])
    return value


def _closed_episode(program,job,episode_root,previous,unstarted,completion_reference):
    from .primary_caller_v2 import inspect_caller_boundary
    from .primary_search_worker_v2 import (_reference,_json,_checked_request,_job_binding,
        _unstarted_episode_state,_previous_episode_state,_check_episode_invocation,_check_episode_journal_prefix,_body_schema)
    complete_path=_reference(completion_reference)
    if complete_path!=episode_root/'caller-complete.json' or _sha(complete_path)!=completion_reference['sha256']:
        raise ValueError('Exact local caller completion required')
    complete=_json(complete_path.read_bytes());base=complete['process_reference']
    boundary=inspect_caller_boundary(completion_reference,process_reference=base)
    request=_checked_request(episode_root/'request.json',complete['request_reference']['sha256'])
    if (_job_binding(request)!=_request_binding(program,job) or request['previous_episodes']!=previous
            or request.get('previous_unstarted_episodes')!=unstarted):
        raise ValueError('Queue episode changed program/job or exact operational history')
    _previous_episode_state(request)
    episode=_json((episode_root/'episode.json').read_bytes())
    if complete['status']=='closed_unstarted_episode':
        reference=dict(base,caller_reference=completion_reference)
        probe=dict(request,previous_unstarted_episodes=[*unstarted,reference]);_unstarted_episode_state(probe)
        return dict(kind='unstarted',reference=reference,elapsed_ns=boundary['elapsed_ns'],stop_reason='resource_pause')
    if complete['status']!='closed_primary_body':
        raise PermissionError('Authorized failed/ambiguous caller cannot be automatically rerun')
    body_reference=episode['closed_body_reference'];control=Path(base['process_root'])
    if any(body_reference.get(k)!=v for k,v in base.items()) or _sha(control/'body-outcome.json')!=body_reference['body_outcome_sha256']:
        raise ValueError('Exact closed body reference required')
    body=_json((control/'body-outcome.json').read_bytes());start=_json((control/'started.json').read_bytes())
    if (body.get('schema')!=_body_schema(request)
            or body.get('status')!='closed_search_body' or body.get('job_binding')!=_job_binding(request)
            or body.get('request_sha256')!=complete['request_reference']['sha256']
            or body.get('previous_episodes')!=previous or any(body.get(k) is not False for k in
                ('quality_eligible','protected_access_authorized','T_accessed','whole_training_procedure_cost_authenticated'))):
        raise ValueError('Queue closed body contradicts its exact request or authority')
    _check_episode_invocation(start,body,previous)
    path=_reference(body['search_result_reference'])
    if _sha(path)!=body['search_result_reference']['sha256']:raise ValueError('Closed search result bytes changed')
    result=_json(path.read_bytes());_check_episode_journal_prefix(request['search_directory'],body,result)
    stop=result.get('stop_reason')
    if stop not in ('resource_pause','operator_pause','interrupted_recoverable','attempt_limit','support_exhausted'):
        raise ValueError('Unknown native-quality search disposition')
    return dict(kind='terminal' if stop in ('attempt_limit','support_exhausted') else 'paused',
        reference=dict(body_reference,caller_reference=completion_reference),elapsed_ns=boundary['elapsed_ns'],
        stop_reason=stop,body=body,result=result,job=job)


def _append(root,kind,job,index,episode_root,completion_reference):
    from treeneuron.config import canonical_hash,canonical_json
    raw=(root/'events.jsonl').read_bytes();previous=json.loads(raw.splitlines()[-1])['event_hash'] if raw else '0'*64
    event=dict(kind=kind,job_id=job['job_id'],episode_index=index,episode_root=str(episode_root),
        completion_reference=completion_reference,previous_hash=previous)
    event['event_hash']=canonical_hash(event)
    import os
    with (root/'events.jsonl').open('ab') as stream:stream.write((canonical_json(event)+'\n').encode());stream.flush();os.fsync(stream.fileno())


def _progress(program,audit_terminal):
    from treeneuron.config import canonical_hash
    from .primary_search_worker_v2 import _json
    root=Path(program['directory']);raw=(root/'events.jsonl').read_bytes()
    if raw and not raw.endswith(b'\n'):raise ValueError('Partial queue journal cannot be resumed')
    completed=0;previous=[];unstarted=[];index=0;pending=None;digest='0'*64;known=0
    fields={'kind','job_id','episode_index','episode_root','completion_reference','previous_hash','event_hash'}
    for line in raw.splitlines():
        event=_json(line);body={k:v for k,v in event.items() if k!='event_hash'}
        if (set(event)!=fields or event['previous_hash']!=digest or canonical_hash(body)!=event['event_hash']
                or completed>=len(program['jobs'])):raise ValueError('Queue hash chain or terminal inventory changed')
        job=program['jobs'][completed];path=root/'episodes'/job['job_id']/f'episode-{index:04}'
        if (event['job_id']!=job['job_id'] or type(event['episode_index']) is not int
                or event['episode_index']!=index or event['episode_root']!=str(path)):
            raise ValueError('Queue cannot reorder jobs or replace an episode')
        if event['kind']=='start_episode':
            if pending is not None or event['completion_reference'] is not None:raise ValueError('Ambiguous or duplicate started episode')
            pending=path
        elif event['kind']=='close_episode':
            if pending!=path:raise ValueError('Queue close has no matching start')
            closed=_closed_episode(program,job,path,previous,unstarted,event['completion_reference']);known+=closed['elapsed_ns']
            if closed['kind']=='terminal':
                if audit_terminal is not None:audit_terminal(**closed)
                completed+=1;previous=[];unstarted=[];index=0
            else:
                (unstarted if closed['kind']=='unstarted' else previous).append(closed['reference']);index+=1
            pending=None
        else:raise ValueError('Unknown queue event')
        digest=event['event_hash']
    return dict(completed_jobs=completed,previous_episodes=previous,previous_unstarted_episodes=unstarted,
        episode_index=index,pending=pending,known_observed_caller_ns=known)


def _run_program_queue(program,*,execute_episode,audit_terminal,expected_journal_sha256=None,
        max_episode_starts=168,invocation_origin_ns=None):
    """Private queue composition; injected fixtures confer no native authority."""
    from treeneuron.config import canonical_hash,canonical_json
    from .deployment_worker_v2 import _source
    from .ledger import StudyLease,write_new_json
    from .primary_search_worker_v2 import _path,_json
    started=time.perf_counter_ns() if invocation_origin_ns is None else invocation_origin_ns
    secondary=program.get('schema')=='research/v2-secondary-program-plan';limit=672 if secondary else 168
    if (type(max_episode_starts) is not int or not 1<=max_episode_starts<=limit
            or type(started) is not int or not 0<started<=time.perf_counter_ns()
            or program.get('source_sha256')!=_source()
            or canonical_hash({k:v for k,v in program.items() if k!='program_hash'})!=program.get('program_hash')):
        raise ValueError('Unchanged program/source and bounded administrative invocation required')
    root=_path(program['directory']);saved=root/'plan.json';journal=root/'events.jsonl'
    if root.exists():
        if (not expected_journal_sha256 or _sha(journal)!=expected_journal_sha256
                or canonical_json(_json(saved.read_bytes()))!=canonical_json(program)):
            raise ValueError('Exact program and externally pinned resume journal required')
    else:
        if expected_journal_sha256 is not None:raise ValueError('Cannot resume an absent program')
        root.mkdir(parents=True,exist_ok=False);write_new_json(saved,program);journal.touch(exist_ok=False)
    status='operator_pause';error=None;value=None
    with StudyLease(root):
        try:
            progress=_progress(program,audit_terminal);starts=0
            while progress['completed_jobs']<len(program['jobs']):
                job=program['jobs'][progress['completed_jobs']];index=progress['episode_index']
                path=root/'episodes'/job['job_id']/f'episode-{index:04}'
                if progress['pending'] is None:
                    if starts>=max_episode_starts:break
                    if path.exists() or path.is_symlink():raise ValueError('Uncommitted episode output cannot be overwritten')
                    _append(root,'start_episode',job,index,path,None);starts+=1
                    execute_episode(program=program,job=copy.deepcopy(job),episode_root=path,
                        previous_episodes=copy.deepcopy(progress['previous_episodes']),
                        previous_unstarted_episodes=copy.deepcopy(progress['previous_unstarted_episodes']))
                completion=path/'caller-complete.json'
                if not completion.is_file():raise PermissionError('Started episode lacks closed caller evidence; no automatic refit')
                reference=dict(path=str(completion),sha256=_sha(completion))
                closed=_closed_episode(program,job,path,progress['previous_episodes'],progress['previous_unstarted_episodes'],reference)
                if closed['kind']=='terminal':audit_terminal(**closed)
                _append(root,'close_episode',job,index,path,reference)
                # Entry audited every saved terminal job; the current terminal
                # was audited immediately above. Reopen all journal/caller
                # commitments, without repeating expensive native audits of
                # that same prefix on every new close. A new invocation repeats
                # them, and the strong model barrier still revalidates models.
                progress=_progress(program,None)
                if closed['kind']!='terminal':status=closed['stop_reason'];break
            if progress['completed_jobs']==len(program['jobs']):status='secondary_searches_terminal' if secondary else 'primary_searches_terminal'
            if _source()!=program['source_sha256']:raise ValueError('Program source changed during invocation')
            value=dict(schema='research/v2-secondary-program-progress' if secondary else 'research/v2-primary-program-progress',status=status,
                program_hash=program['program_hash'],journal_sha256=_sha(journal),completed_jobs=progress['completed_jobs'],
                known_observed_caller_ns=progress['known_observed_caller_ns'],
                native_model_barrier_authenticated=False,whole_training_procedure_cost_authenticated=False,
                secondary_release_authorized=False,protected_access_authorized=False,T_accessed=False,
                scope='Closed serial episode queue with known caller clocks; actual model/global freeze and physical whole-cost remain separate')
        except BaseException as caught:error=caught
        finally:
            receipt=dict(schema='research/v2-secondary-program-invocation' if secondary else 'research/v2-primary-program-invocation',source_sha256=program['source_sha256'],
                program_hash=program['program_hash'],journal_sha256=_sha(journal),elapsed_ns=time.perf_counter_ns()-started,
                result=value,error=None if error is None else dict(type=type(error).__name__,reason=str(error)),
                whole_training_procedure_cost_authenticated=False,protected_access_authorized=False,T_accessed=False,
                clock_scope='Actual invocation entry through preparation/queue/readback; own final receipt publication/return excluded')
            write_new_json(root/'invocations'/(uuid.uuid4().hex+'.json'),receipt)
        if error is not None:raise error
    return value


def run_primary_program(*,practical_plan_reference,expected_practical_plan_hash,support_reference,
        procedure_reference,qualification_reference,resource_forecast_reference,directory,rss_limit_bytes=4*1024**3,
        expected_journal_sha256=None,max_episode_starts=168):
    """Execute actual qualified primary jobs serially, or stop at a safe pause."""
    started=time.perf_counter_ns()
    from .native_qualification_registry_v2 import load_native_qualification_registry,native_registry_storage_roots
    registry=load_native_qualification_registry(qualification_reference)
    from treeneuron.config import canonical_json
    from .primary_selection_v2 import _ref
    from .workload import PracticalPlan
    from .native_resource_forecast_v2 import load_native_practical_resource_forecast
    from .primary_caller_v2 import run_primary_search_episode
    from .qualified_search_audit_v2 import audit_primary_qualified_search
    practical=PracticalPlan(canonical_json(_ref(practical_plan_reference,{},decode=True)))
    program=build_primary_program_plan(practical_plan=practical,practical_plan_reference=practical_plan_reference,
        expected_practical_plan_hash=expected_practical_plan_hash,support_reference=support_reference,
        procedure_reference=procedure_reference,qualification_reference=qualification_reference,
        resource_forecast_reference=resource_forecast_reference,directory=directory,rss_limit_bytes=rss_limit_bytes)
    load_native_practical_resource_forecast(resource_forecast_reference,practical_plan=practical,
        expected_practical_plan_hash=expected_practical_plan_hash,support_reference=support_reference,qualification_reference=qualification_reference)
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=qualification_reference)
    _check_program_location(directory,working_roots=roots['working_roots'],proof_roots=roots['proof_roots'],
        input_files=[r['path'] for r in (practical_plan_reference,support_reference,procedure_reference,resource_forecast_reference)])
    def execute(*,program,job,episode_root,previous_episodes,previous_unstarted_episodes):
        return run_primary_search_episode(practical_plan_reference=practical_plan_reference,
            expected_practical_plan_hash=expected_practical_plan_hash,job_id=job['job_id'],support_reference=support_reference,
            qualification_reference=qualification_reference,resource_forecast_reference=resource_forecast_reference,
            procedure_reference=procedure_reference,episode_root=episode_root,
            search_directory=Path(program['directory'])/'searches'/job['job_id'],previous_episodes=previous_episodes,
            previous_unstarted_episodes=previous_unstarted_episodes,rss_limit_bytes=rss_limit_bytes)
    def audit(**closed):
        body=closed['body'];job=closed['job']
        result=audit_primary_qualified_search(practical_plan=practical,expected_practical_plan_hash=expected_practical_plan_hash,
            job_id=job['job_id'],support_reference=support_reference,qualification_reference=qualification_reference,
            search_directory=Path(program['directory'])/'searches'/job['job_id'],expected_search_plan_sha256=body['search_plan_sha256'],
            result_path=Path(body['search_result_reference']['path']),expected_result_sha256=body['search_result_reference']['sha256'],
            expected_journal_sha256=body['journal_sha256'],official_schedule_reference=body['official_schedule_reference'])
        if result.get('completed_search') is not True or any(result.get(k) is not True for k in
                ('HPO_journal_authenticated','all_attempt_native_evidence_authenticated','all_eligible_V_scores_authenticated','native_workload_qualified')):
            raise ValueError('Every terminal primary search needs the genuine native/HPO/V audit')
    return _run_program_queue(program,execute_episode=execute,audit_terminal=audit,
        expected_journal_sha256=expected_journal_sha256,max_episode_starts=max_episode_starts,invocation_origin_ns=started)
