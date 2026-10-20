"""Cold primary-search body; production admission and protected access are separate.

Only standard-library imports precede the cooperative ready/grant handshake.
Actual complete qualification is required before plan/role staging or fitting.
This worker is a building block, not a production admission or freeze command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

_ROOT=Path(__file__).resolve().parents[3]
_THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _json(raw):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Duplicate primary-worker field')
            result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite primary-worker field')))


def _path(value):
    if (type(value) is not str or not value or not Path(value).is_absolute()
            or Path(value).resolve()!=Path(value) or Path(value).is_symlink()):
        raise ValueError('Canonical absolute primary-worker path required')
    return Path(value)


def _reference(value):
    if type(value) is not dict or set(value)!={'path','sha256'} or not _digest(value['sha256']):
        raise ValueError('Explicit primary-worker file reference required')
    return _path(value['path'])


def _checked_request(path,expected):
    path=_path(str(path))
    if not _digest(expected) or not path.is_file() or _sha(path)!=expected:
        raise ValueError('Primary-worker request commitment changed')
    value=_json(path.read_bytes())
    fields={'schema','source_sha256','practical_plan_reference','expected_practical_plan_hash',
        'job_id','support_reference','qualification_reference','search_directory','rss_limit_bytes','previous_episodes'}
    if (type(value) is not dict or set(value)!=fields
            or value['schema']!='research/v2-primary-search-episode-request'
            or not _digest(value['source_sha256']) or not _digest(value['expected_practical_plan_hash'])
            or type(value['job_id']) is not str or not value['job_id'] or len(value['job_id'])>230
            or type(value['rss_limit_bytes']) is not int or not 0<value['rss_limit_bytes']<=16*1024**3
            or type(value['previous_episodes']) is not list or len(value['previous_episodes'])>32):
        raise ValueError('Strict primary-worker request required')
    _reference(value['practical_plan_reference']);_reference(value['support_reference']);_path(value['search_directory'])
    q=value['qualification_reference']
    if type(q) is not dict or set(q)!={'root','inventory_sha256'} or not _digest(q['inventory_sha256']):
        raise ValueError('Actual native qualification inventory reference required')
    _path(q['root'])
    for episode in value['previous_episodes']:
        if (type(episode) is not dict
                or set(episode)!={'process_root','process_receipt_sha256','process_started_sha256','body_outcome_sha256'}
                or any(not _digest(episode[k]) for k in episode if k!='process_root')):
            raise ValueError('Exact closed prior episode commitments required')
        _path(episode['process_root'])
    return value


def primary_worker_identity(request_path,expected_request_sha256):
    """Read only the strict command request, without imports, plans or arrays."""
    value=_checked_request(request_path,expected_request_sha256)
    return dict(purpose='primary_search_episode',request_sha256=expected_request_sha256,
        source_sha256=value['source_sha256'])


def _job_binding(request):
    return {k:v for k,v in request.items() if k!='previous_episodes'}


def _episode_environment():
    value={k:'1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}
    return value|dict(PYTHONPATH=str(_ROOT/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0',KMP_DUPLICATE_LIB_OK=None)


def _check_episode_invocation(start,body,previous_prefix):
    """A real script/module invocation is required in addition to success flags."""
    argv=start.get('argv')
    if (type(argv) is not list or len(argv)!=11 or any(type(v) is not str for v in argv)
            or not Path(argv[0]).is_absolute() or Path(argv[0]).resolve()!=Path(sys.executable).resolve()
            or argv[1:6]!=['-P','-B','-m','treeneuron.research.primary_search_worker_v2','--request']
            or argv[7]!='--expected-request-sha256' or argv[9]!='--output'
            or argv[8]!=body.get('request_sha256') or start.get('cwd')!=str(_ROOT)
            or start.get('environment_overrides')!=_episode_environment()):
        raise ValueError('Prior episode must be the exact owned primary worker invocation')
    request=_checked_request(argv[6],argv[8])
    if (request['previous_episodes']!=previous_prefix or _job_binding(request)!=body.get('job_binding')
            or start.get('identity')!=primary_worker_identity(argv[6],argv[8])):
        raise ValueError('Prior command request/job/lineage differs')
    result=body.get('search_result_reference')
    if (type(result) is not dict or _reference(result)!=_path(argv[10])/'body/search-result.json'):
        raise ValueError('Prior immutable search result differs from its actual output argument')


def _check_episode_journal_prefix(search,body,result):
    """Match a past endpoint after later sessions append; no native authority."""
    raw=(Path(search)/'events.jsonl').read_bytes();prefix=b'';records=[]
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):raise ValueError('Complete journal prefix lines required')
        prefix+=line;event=_json(line)
        if event.get('kind')=='attempt_finished':records.append(event['payload']['record'])
        if event.get('event_hash')!=body.get('journal_hash'):continue
        if (event.get('kind')!='session_finished' or hashlib.sha256(prefix).hexdigest()!=body.get('journal_sha256')
                or type(body.get('total_charged_ns')) is not int
                or event.get('total_charged_ns')!=body['total_charged_ns']
                or result.get('total_charged_ns')!=body['total_charged_ns']
                or result.get('journal_hash')!=body['journal_hash']
                or json.dumps(result.get('records'),sort_keys=True)!=json.dumps(records,sort_keys=True)):
            raise ValueError('Prior outcome/result differs from the exact paid journal prefix')
        return len(prefix)
    raise ValueError('Prior closed episode endpoint is absent from the current journal')


def _previous_episode_state(request):
    """Authenticate closed root costs/lineage; native journal audit is separate."""
    from .cooperative_process_v2 import inspect_cooperative_process
    previous=request['previous_episodes'];total=0;last=None;seen=set();last_prefix=0
    for index,reference in enumerate(previous):
        root=_path(reference['process_root'])
        if root in seen:raise ValueError('Duplicate prior root cannot be charged twice')
        seen.add(root)
        proof=inspect_cooperative_process(root=root,
            expected_receipt_sha256=reference['process_receipt_sha256'],
            expected_started_sha256=reference['process_started_sha256'])
        path=root/'body-outcome.json'
        if not path.is_file() or _sha(path)!=reference['body_outcome_sha256']:
            raise ValueError('Prior body outcome commitment changed')
        body=_json(path.read_bytes())
        start=_json((root/'started.json').read_bytes())
        closed_fields={'schema','status','request_sha256','job_binding','previous_episodes','quality_eligible',
            'protected_access_authorized','T_accessed','whole_training_procedure_cost_authenticated',
            'search_result_reference','search_plan_sha256','journal_sha256','journal_hash','total_charged_ns',
            'official_schedule_reference'}
        if (proof['returncode']!=0 or not proof['worker_protocol_closed']
                or not proof['shared_monotonic_clock_observed']
                or type(body) is not dict or set(body)!=closed_fields
                or body.get('schema')!='research/v2-primary-search-body-outcome'
                or body.get('status')!='closed_search_body'
                or body.get('job_binding')!=_job_binding(request)
                or body.get('previous_episodes')!=previous[:index]
                or start['identity']!=dict(purpose='primary_search_episode',
                    request_sha256=body.get('request_sha256'),source_sha256=request['source_sha256'])
                or any(not _digest(body.get(k)) for k in ('request_sha256','search_plan_sha256','journal_sha256','journal_hash'))
                or any(body.get(k) is not False for k in ('quality_eligible','protected_access_authorized',
                    'T_accessed','whole_training_procedure_cost_authenticated'))):
            raise ValueError('Prior closed episode source/job/lineage/ownership differs')
        _check_episode_invocation(start,body,previous[:index])
        result_path=_reference(body['search_result_reference'])
        if not result_path.is_file() or _sha(result_path)!=body['search_result_reference']['sha256']:
            raise ValueError('Prior immutable search result changed')
        result=_json(result_path.read_bytes())
        if (result.get('stop_reason') not in ('operator_pause','interrupted_recoverable')
                or result.get('journal_hash')!=body.get('journal_hash')
                or type(result.get('total_charged_ns')) is not int or result['total_charged_ns']<0
                or last is not None and result['total_charged_ns']<last['result']['total_charged_ns']):
            raise ValueError('Prior search is terminal/unclosed or paid time moved backwards')
        prefix=_check_episode_journal_prefix(request['search_directory'],body,result)
        if prefix<=last_prefix:raise ValueError('Prior episodes must advance the exact closed journal')
        last_prefix=prefix
        total+=proof['elapsed_ns']
        if result['total_charged_ns']>total:
            raise ValueError('Controller cost exceeds its complete closed outer episodes')
        last=dict(body=body,result=result)
    return dict(previous_outer_tail_ns=0 if last is None else total-last['result']['total_charged_ns'],
        last=last,total_prior_outer_ns=total,
        official_schedule_reference=None if last is None else last['body']['official_schedule_reference'])


def _sampler_runtime(registry,job):
    from .native_anchor_v2 import load_native_anchor_execution
    from .method_worker import _ROOT
    prepared=Path(registry['inputs']['prepared_roots'][job['dataset']])
    manifest=registry['preparation']['datasets'][job['dataset']]['manifest_file_sha256']
    anchor=next(a for a in registry['anchors'] if a['dataset']==job['dataset'] and a['family_id']=='P06')
    proof=load_native_anchor_execution(anchor['reference'],prepared_root=prepared,
        expected_manifest_sha256=manifest,expected_family='P06',expected_anchor_id=anchor['anchor_id'],expected_dataset=job['dataset'])
    process=_json((Path(proof['generation']['native_inventory']['root'])/'process/process.json').read_bytes())
    paths=process['environment_overrides']['PYTHONPATH'].split(os.pathsep);interpreter=process['argv'][0]
    if (Path(interpreter).resolve()!=Path(sys.executable).resolve() or paths[0]!=str(_ROOT/'src')
            or any(not Path(p).is_absolute() or not Path(p).is_dir()
                or not Path(p).resolve().is_relative_to((_ROOT/'environment').resolve()) for p in paths[1:])):
        raise ValueError('Official sampler runtime differs from its actual qualified anchor')
    return interpreter,paths[1:]


def _prepare_qualified_episode(*,request,output):
    """Actual registry first; no supplied successful flag can reach staging."""
    from .native_qualification_registry_v2 import load_native_qualification_registry,_all_references
    registry=load_native_qualification_registry(request['qualification_reference'])
    from treeneuron.config import canonical_json
    from .deployment_worker_v2 import _source
    from .primary_native_preparation_v2 import prepare_primary_native_callback,_check_preparation_locations
    from .primary_selection_v2 import _primary
    from .workload import PracticalPlan
    if request['source_sha256']!=_source():raise ValueError('Primary worker source changed')
    output=_path(str(output));search=_path(request['search_directory'])
    _check_preparation_locations(output=output,search=search,
        working_roots=registry['inputs']['prepared_roots'].values(),
        proof_roots=[r['root'] for r in _all_references(registry['inputs'])],rss_limit_bytes=request['rss_limit_bytes'])
    path=_reference(request['practical_plan_reference'])
    if not path.is_file() or _sha(path)!=request['practical_plan_reference']['sha256']:
        raise ValueError('Practical plan file commitment changed')
    plan=PracticalPlan(canonical_json(_json(path.read_bytes())))
    _,jobs=_primary(plan,request['expected_practical_plan_hash'])
    matches=[job for job in jobs if job['job_id']==request['job_id']]
    if len(matches)!=1:raise ValueError('Only the exact declared primary job may run')
    job=matches[0];prior=_previous_episode_state(request)
    output.mkdir(parents=True,exist_ok=False)
    official=prior['official_schedule_reference']
    if job['family_id']=='P06' and not request['previous_episodes']:
        from .native_schedule_v2 import prepare_realmlp_native_schedule
        from .transfer import TransferStore
        store=TransferStore(registry['inputs']['prepared_roots'][job['dataset']])
        interpreter,overlays=_sampler_runtime(registry,job)
        generated=prepare_realmlp_native_schedule(feature_schema=dict(columns=store.manifest['feature_names'],
            categorical=store.manifest['categorical_features']),hpo_seed=job['hpo_seed'],
            output=output/'native-sampler',interpreter=interpreter,extra_pythonpath=overlays,
            rss_limit_bytes=request['rss_limit_bytes'])
        if generated['status']!='completed_native_schedule':
            raise ValueError('Official native sampler did not complete; paid evidence retained')
        official=generated['reference']
    if prior['last'] is not None:
        from .qualified_search_audit_v2 import audit_primary_qualified_search
        body=prior['last']['body']
        audit_primary_qualified_search(practical_plan=plan,expected_practical_plan_hash=request['expected_practical_plan_hash'],
            job_id=job['job_id'],support_reference=request['support_reference'],qualification_reference=request['qualification_reference'],
            search_directory=search,expected_search_plan_sha256=body['search_plan_sha256'],
            result_path=Path(body['search_result_reference']['path']),expected_result_sha256=body['search_result_reference']['sha256'],
            expected_journal_sha256=body['journal_sha256'],official_schedule_reference=official)
    prepared=prepare_primary_native_callback(practical_plan=plan,
        expected_practical_plan_hash=request['expected_practical_plan_hash'],job_id=job['job_id'],
        support_reference=request['support_reference'],qualification_reference=request['qualification_reference'],
        output=output/'preparation',search_directory=search,official_schedule_reference=official,
        rss_limit_bytes=request['rss_limit_bytes'])
    return prepared,prior,official


def _run_prepared_search(*,plan,callback,search_directory,output,previous_outer_tail_ns,
        resume,expected_journal_hash,max_attempts_this_invocation=32):
    """Compose a generic paid controller; the public worker supplies qualification."""
    from .cooperative_process_v2 import _read,_grant_expected,_canonical
    from .search_controller_v2 import run_search
    from .ledger import write_new_json
    control=Path(os.environ['ATN_SEARCH_CONTROL_ROOT'])
    start,raw=_read(control/'started.json');ready,ready_raw=_read(control/'ready.json');grant,_=_read(control/'grant.json')
    if (_sha(control/'started.json')!=os.environ['ATN_SEARCH_CONTROL_START_SHA256']
            or ready.get('child_pid')!=os.getpid() or ready.get('child_pgid')!=os.getpgid(0)
            or _canonical(grant)!=_canonical(_grant_expected(hashlib.sha256(ready_raw).hexdigest(),start['nonce']))
            or type(start.get('supervisor_started_monotonic_ns')) is not int
            or not 0<start['supervisor_started_monotonic_ns']<=ready.get('worker_ready_monotonic_ns',0)<=time.perf_counter_ns()):
        raise ValueError('Paid controller requires its current granted cold-worker clock')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=run_search(search_directory,plan=plan,evaluate_trial=callback,resume=resume,
        expected_journal_hash=expected_journal_hash,invocation_preparation_ns=previous_outer_tail_ns,
        invocation_origin_ns=start['supervisor_started_monotonic_ns'],
        max_attempts_this_invocation=max_attempts_this_invocation)
    path=output/'search-result.json';write_new_json(path,result)
    return dict(search_result_reference=dict(path=str(path.resolve()),sha256=_sha(path)),
        search_plan_sha256=_sha(Path(search_directory)/'plan.json'),
        journal_sha256=_sha(Path(search_directory)/'events.jsonl'),journal_hash=result['journal_hash'],
        total_charged_ns=result['total_charged_ns'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--expected-request-sha256',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    identity=primary_worker_identity(args.request,args.expected_request_sha256)
    from .cooperative_process_v2 import cooperative_worker,_write
    with cooperative_worker(identity=identity):
        request=_checked_request(args.request,args.expected_request_sha256)
        outcome=dict(schema='research/v2-primary-search-body-outcome',status='body_failed',
            request_sha256=args.expected_request_sha256,job_binding=_job_binding(request),
            previous_episodes=request['previous_episodes'],quality_eligible=False,
            protected_access_authorized=False,T_accessed=False,whole_training_procedure_cost_authenticated=False)
        code=1
        try:
            if not sys.dont_write_bytecode or any(os.environ.get(k)!=v for k,v in _episode_environment().items()):
                raise ValueError('Cold primary worker requires its declared core-only environment')
            prepared,prior,official=_prepare_qualified_episode(request=request,output=Path(args.output))
            saved=_run_prepared_search(plan=prepared.plan,callback=prepared.callback,
                search_directory=Path(request['search_directory']),output=Path(args.output)/'body',
                previous_outer_tail_ns=prior['previous_outer_tail_ns'],resume=bool(request['previous_episodes']),
                expected_journal_hash=None if prior['last'] is None else prior['last']['body']['journal_hash'])
            if _checked_request(args.request,args.expected_request_sha256)!=request:
                raise ValueError('Primary episode request changed during execution')
            outcome.update(saved,status='closed_search_body',official_schedule_reference=official)
            code=0
        except BaseException as error:
            # A KeyboardInterrupt outside a managed native attempt can leave an
            # unclosed journal. It is NOT fabricated into a resumable model.
            outcome.update(error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        finally:
            _write(Path(os.environ['ATN_SEARCH_CONTROL_ROOT'])/'body-outcome.json',outcome)
    return code


if __name__=='__main__':
    raise SystemExit(main())
