"""Actual protocol admission is separate from private queue fixture execution."""
import importlib.util
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.protected_prediction_queue_v2'),'Missing genuine protected prediction coordinator'
    from treeneuron.research import protected_prediction_queue_v2
    return protected_prediction_queue_v2


def test_public_queue_requires_actual_protocol_before_study_or_attempt_output(tmp_path):
    module=api()
    kwargs=dict(models_reference=dict(path=str(tmp_path/'forbidden-models'),sha256='a'*64),
        global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='b'*64),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64),
        preT_costs_reference=dict(path=str(tmp_path/'forbidden-cost'),sha256='d'*64),
        readiness_reference=dict(path=str(tmp_path/'forbidden-readiness'),sha256='e'*64))
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.run_protected_predictions(dict(path=str(tmp_path/'forbidden-protocol'),sha256='f'*64),
            directory=tmp_path/'protected-study',**kwargs)
    assert all(name not in str(error.value) for name in ('forbidden-models','forbidden-protocol','forbidden-cost','forbidden-readiness'))
    assert list(tmp_path.iterdir())==[]


def test_public_queue_does_not_accept_supplied_context_or_true_native_flags(tmp_path):
    module=api()
    kwargs=dict(models_reference=dict(path=str(tmp_path/'models'),sha256='a'*64),
        global_barrier_reference=dict(path=str(tmp_path/'global'),sha256='b'*64),
        qualification_reference=dict(root=str(tmp_path/'registry'),inventory_sha256='c'*64),
        preT_costs_reference=dict(path=str(tmp_path/'cost'),sha256='d'*64),
        readiness_reference=dict(path=str(tmp_path/'readiness'),sha256='e'*64))
    with pytest.raises(TypeError,match="unexpected keyword argument 'context'"):
        module.run_protected_predictions(dict(path=str(tmp_path/'protocol'),sha256='a'*64),
            directory=tmp_path/'study',context=dict(native_qualification_authenticated=True),**kwargs)
    assert list(tmp_path.iterdir())==[]


def local_protocol(tmp_path):
    import copy
    import json
    from test_protected_query_v2 import query_fixture,rebind,canonical_hash
    from test_protected_query_worker_v2 import ref
    from treeneuron.research.deployment_worker_v2 import _source
    _,_,core=query_fixture(tmp_path)
    first=core['query'];second=copy.deepcopy(core)
    second['query']['job_id']='unqualified-second-logical-job';second['query']['fit_seed']+=1;rebind(second)
    queries=sorted([first,second['query']],key=lambda q:q['query_id'])
    data_root=Path(first['feature_reference']['path']).parent.parent
    manifest=data_root/'known-fixture-manifest.json'
    manifest.write_text(json.dumps(dict(raw_sha256='d'*64,semantic_schema=dict(target='known-fixture-target'),raw_identity_verified=False))+'\n')
    payload=dict(source_sha256=_source(),data={'toy':dict(root=str(data_root),manifest_reference=ref(manifest),manifest_hash=first['manifest_hash'],
        feature_names=core['feature_schema']['columns'],categorical_features=core['feature_schema']['categorical'],
        folds=[dict(outer_fold=0,split_hash=first['split_hash'],roles=dict(T_features=dict(reference=first['feature_reference'])))])},
        prediction_manifest=dict(queries=queries,manifest_hash=canonical_hash(dict(queries=queries))))
    protocol=tmp_path/'known-fixture-protocol.json';protocol.write_text(json.dumps(payload,sort_keys=True)+'\n')
    return payload,ref(protocol)


def test_private_queue_keeps_exact_frozen_order_and_never_grants_label_authority(tmp_path):
    module=api();payload,reference=local_protocol(tmp_path);seen=[]
    def no_worker(*args,**kwargs):
        seen.append(kwargs.get('query_id'))
        raise RuntimeError('Intentional fixture failure before a worker exists')
    result=module._run_prediction_queue(protocol_payload=payload,protocol_reference=reference,directory=tmp_path/'study',
        owner_root=tmp_path/'owners',execute_episode=no_worker,guard=lambda:None,max_queries_this_invocation=1,rss_limit_bytes=1024**3)
    assert seen==[payload['prediction_manifest']['queries'][0]['query_id']]
    assert result['status']!='all_predictions_committed'
    assert result['native_qualification_authenticated'] is result['protected_access_authorized'] is result['T_accessed'] is False
    assert result['completed_queries']==0 and result['labels_read'] is False


def test_changed_private_source_blocks_before_creating_owner_or_study(tmp_path):
    module=api();payload,reference=local_protocol(tmp_path);payload['source_sha256']='0'*64
    with pytest.raises(ValueError):
        module._run_prediction_queue(protocol_payload=payload,protocol_reference=reference,directory=tmp_path/'study',
            owner_root=tmp_path/'owners',execute_episode=lambda **kw:None,guard=lambda:None,max_queries_this_invocation=1,rss_limit_bytes=1024**3)
    assert not (tmp_path/'study').exists() and not (tmp_path/'owners').exists()


def owned_fixture_executor(seen,decisions=None,admission_error=None):
    def execute(**kwargs):
        import hashlib
        import sys
        from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process
        from treeneuron.research.protected_query_worker_v2 import protected_query_worker_identity
        project=Path(__file__).resolve().parents[1];request=kwargs['request_path'];control=kwargs['control_output']
        seen.append(kwargs['query_id'])
        overrides={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}
        overrides.update(PYTHONPATH=str(project/'src'),KMP_DUPLICATE_LIB_OK=None)
        def admission():
            if admission_error is not None:raise RuntimeError(admission_error)
            return dict(admitted=decisions.pop(0) if decisions else True,reason='private fixture only; no native/protected qualification')
        supervise_cooperative_process(argv=[sys.executable,'-P','-B','-m','treeneuron.research.protected_query_worker_v2',
            '--request',str(request),'--expected-request-sha256',kwargs['expected_request_sha256'],'--output',str(kwargs['worker_output'])],
            cwd=project,env_overrides=overrides,output=control,
            identity=protected_query_worker_identity(request,kwargs['expected_request_sha256']),
            admission_identity=dict(purpose='unqualified_local_queue_fixture'),
            admission_check=admission,
            rss_pause_bytes=kwargs['rss_limit_bytes'],admit_before_spawn=True)
        return dict(process_reference=dict(process_root=str(control),process_receipt_sha256=hashlib.sha256((control/'process.json').read_bytes()).hexdigest(),
            process_started_sha256=hashlib.sha256((control/'started.json').read_bytes()).hexdigest()))
    return execute


def test_private_real_owned_queue_resumes_without_predicting_a_committed_model_again(tmp_path):
    module=api();payload,reference=local_protocol(tmp_path);seen=[];guards=[]
    common=dict(protocol_payload=payload,protocol_reference=reference,directory=tmp_path/'study',owner_root=tmp_path/'owners',
        execute_episode=owned_fixture_executor(seen),guard=lambda:guards.append('guard'),rss_limit_bytes=1024**3)
    first=module._run_prediction_queue(**common,max_queries_this_invocation=1)
    assert first['completed_queries']==1 and first['status']=='operator_pause',first
    second=module._run_prediction_queue(**common)
    assert second['completed_queries']==2 and second['status']=='all_predictions_committed',second
    third=module._run_prediction_queue(**common)
    assert third['attempted_queries']==0 and third['status']=='all_predictions_committed'
    assert seen==[q['query_id'] for q in payload['prediction_manifest']['queries']]
    from treeneuron.research.protected_journal_v2 import _read_metadata_study
    state=_read_metadata_study(tmp_path/'study')
    assert state['phase']=='all_predictions_committed' and state['known_paid_ns']>0
    assert state['execution_cost_complete'] is True and len(guards)>=6
    assert all(v['protected_access_authorized'] is v['native_qualification_authenticated'] is v['T_accessed'] is False for v in (first,second,third))


def test_failed_final_context_guard_never_publishes_the_all_predictions_barrier(tmp_path):
    module=api();payload,reference=local_protocol(tmp_path);seen=[];calls=[]
    def guard():
        calls.append('guard')
        if len(calls)==2:raise ValueError('Simulated changed frozen global context; fixture only')
    result=module._run_prediction_queue(protocol_payload=payload,protocol_reference=reference,directory=tmp_path/'study',
        owner_root=tmp_path/'owners',execute_episode=owned_fixture_executor(seen),guard=guard,rss_limit_bytes=1024**3)
    assert result['completed_queries']==2 and result['status']=='blocked_coordinator_failure',result
    from treeneuron.research.protected_journal_v2 import _read_metadata_study
    assert _read_metadata_study(tmp_path/'study')['phase']=='owners_ready'
    assert result['labels_read'] is result['protected_access_authorized'] is False


def test_resource_denial_resumes_same_frozen_query_and_retains_both_costs(tmp_path):
    module=api();payload,reference=local_protocol(tmp_path);seen=[];decisions=[False]
    common=dict(protocol_payload=payload,protocol_reference=reference,directory=tmp_path/'study',owner_root=tmp_path/'owners',
        execute_episode=owned_fixture_executor(seen,decisions),guard=lambda:None,rss_limit_bytes=1024**3)
    paused=module._run_prediction_queue(**common)
    assert paused['status']=='resource_pause' and paused['completed_queries']==0
    done=module._run_prediction_queue(**common)
    assert done['status']=='all_predictions_committed' and done['completed_queries']==2
    ids=[q['query_id'] for q in payload['prediction_manifest']['queries']]
    assert seen==[ids[0],ids[0],ids[1]]
    from treeneuron.research.protected_journal_v2 import _read_metadata_study
    state=_read_metadata_study(tmp_path/'study')
    assert state['attempts']=={ids[0]:2,ids[1]:1} and state['known_paid_ns']>0 and state['execution_cost_complete']


def test_self_consistent_journal_with_wrong_failed_attempt_cost_blocks_before_retry(tmp_path):
    import json
    from treeneuron.research import protected_journal_v2 as journal
    module=api();payload,reference=local_protocol(tmp_path);seen=[]
    common=dict(protocol_payload=payload,protocol_reference=reference,directory=tmp_path/'study',owner_root=tmp_path/'owners',
        guard=lambda:None,rss_limit_bytes=1024**3)
    paused=module._run_prediction_queue(**common,execute_episode=owned_fixture_executor(seen,[False]))
    assert paused['status']=='resource_pause'
    root=tmp_path/'study';state=journal._read_metadata_study(root);events=(root/'events.jsonl').read_bytes().splitlines()
    row=json.loads(events[-1]);assert row['kind']=='query_failed_with_evidence'
    row['details']['elapsed_ns']+=1
    row['event_hash']=journal._hash({k:v for k,v in row.items() if k!='event_hash'})
    (root/'events.jsonl').write_bytes(b'\n'.join(events[:-1])+b'\n'+(journal._canonical(row)+'\n').encode())
    witness=tmp_path/'owners/histories'/state['binding']['genesis_sha256']/f"{row['sequence']:08}.json"
    witness.write_text(journal._canonical(row)+'\n')
    # Structural checks alone accept the modified cost. Actual call/process evidence must reject it.
    assert journal._read_metadata_study(root)['known_paid_ns']==state['known_paid_ns']+1
    retried=[]
    def forbidden_retry(**kwargs):
        retried.append(kwargs['query_id']);raise RuntimeError('Must not restart a cost-inconsistent attempt')
    result=module._run_prediction_queue(**common,execute_episode=forbidden_retry)
    assert retried==[],result
    assert result['status']=='blocked_coordinator_failure' and result['labels_read'] is False


def test_input_failure_before_spawn_is_not_reported_as_resource_pressure(tmp_path):
    module=api();payload,reference=local_protocol(tmp_path);seen=[]
    result=module._run_prediction_queue(protocol_payload=payload,protocol_reference=reference,directory=tmp_path/'study',
        owner_root=tmp_path/'owners',execute_episode=owned_fixture_executor(seen,admission_error='Intentional fixture dependency error'),
        guard=lambda:None,rss_limit_bytes=1024**3)
    from treeneuron.research.protected_journal_v2 import _read_metadata_study
    state=_read_metadata_study(tmp_path/'study');qid=payload['prediction_manifest']['queries'][0]['query_id']
    assert state['last_failures'][qid]=='input_or_dependency_failure'
    assert result['status']=='blocked_query_failure' and result['completed_queries']==0


def test_ambiguous_started_query_is_not_automatically_restarted(tmp_path):
    module=api();payload,reference=local_protocol(tmp_path)
    from treeneuron.research import protected_journal_v2 as journal
    root=tmp_path/'study';owners=tmp_path/'owners'
    journal._prepare_metadata_study(directory=root,owner_root=owners,**module._admitted_metadata(payload,reference));journal._claim_metadata_owners(root)
    qid=payload['prediction_manifest']['queries'][0]['query_id']
    journal._append_metadata_event(root,kind='query_started',details=dict(query_id=qid,attempt_index=0,resume_kind='initial'))
    seen=[]
    def forbidden(**kw):seen.append(kw['query_id']);raise AssertionError('Unknown active state must not rerun')
    result=module._run_prediction_queue(protocol_payload=payload,protocol_reference=reference,directory=root,owner_root=owners,
        execute_episode=forbidden,guard=lambda:None,rss_limit_bytes=1024**3)
    assert not seen and result['status'].startswith('blocked')
    assert journal._read_metadata_study(root)['active']==(qid,0)


def test_changed_committed_prediction_blocks_before_the_next_model(tmp_path):
    module=api();payload,reference=local_protocol(tmp_path);seen=[]
    common=dict(protocol_payload=payload,protocol_reference=reference,directory=tmp_path/'study',owner_root=tmp_path/'owners',
        guard=lambda:None,rss_limit_bytes=1024**3)
    first=module._run_prediction_queue(**common,execute_episode=owned_fixture_executor(seen),max_queries_this_invocation=1)
    assert first['completed_queries']==1
    qid=payload['prediction_manifest']['queries'][0]['query_id']
    pred=tmp_path/'study/queries'/qid/'episode-0000/worker/prediction.npz'
    with pred.open('ab') as stream:stream.write(b'changed')
    called=[]
    def forbidden(**kw):called.append(kw['query_id']);raise AssertionError('Damaged evidence must not advance')
    result=module._run_prediction_queue(**common,execute_episode=forbidden)
    assert called==[] and result['status']=='blocked_coordinator_failure' and result['labels_read'] is False


def test_dependency_failure_has_one_paid_same_seed_rerun_then_stops(tmp_path):
    module=api();payload,reference=local_protocol(tmp_path);seen=[]
    common=dict(protocol_payload=payload,protocol_reference=reference,directory=tmp_path/'study',owner_root=tmp_path/'owners',
        execute_episode=owned_fixture_executor(seen,admission_error='Intentional persistent fixture failure'),
        guard=lambda:None,rss_limit_bytes=1024**3)
    first=module._run_prediction_queue(**common);second=module._run_prediction_queue(**common);third=module._run_prediction_queue(**common)
    assert first['completed_queries']==second['completed_queries']==third['completed_queries']==0
    assert third['status']=='blocked_terminal_query_failure' and len(seen)==2
    from treeneuron.research.protected_journal_v2 import _read_metadata_study
    state=_read_metadata_study(tmp_path/'study');qid=payload['prediction_manifest']['queries'][0]['query_id']
    assert state['attempts'][qid]==2 and state['reruns'][qid]==1 and state['known_paid_ns']>0
