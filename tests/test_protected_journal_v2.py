"""Ownership and append-only metadata do not themselves authorize test labels."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.protected_journal_v2'), 'Missing protected ownership/journal component'
    from treeneuron.research import protected_journal_v2
    return protected_journal_v2


def ref(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def metadata(tmp_path):
    from treeneuron.research.deployment_worker_v2 import _source
    source=_source();queries=[dict(query_id=c*64) for c in ('a','b')]
    protocol=tmp_path/'protocol.json';protocol.write_text('{}\n')
    return dict(protocol_reference=ref(protocol),source_sha256=source,
        prediction_manifest_sha256='c'*64,queries=queries,
        assets=[dict(dataset='toy',raw_sha256='d'*64,target='target')])


def create(tmp_path,metadata,name='study'):
    m=api();root=tmp_path/name;owners=tmp_path/'owners'
    binding=m._prepare_metadata_study(directory=root,owner_root=owners,**metadata)
    return m,root,owners,binding


def event(m,root,kind,**details):
    return m._append_metadata_event(root,kind=kind,details=details)


def ready(m,root):
    m._claim_metadata_owners(root)
    return m._read_metadata_study(root)


def start(m,root,q='a',index=0,mode='initial'):
    return event(m,root,'query_started',query_id=q*64,attempt_index=index,resume_kind=mode)


def commit(m,root,tmp_path,q='a',index=0):
    p=tmp_path/f'prediction-{q}-{index}.json';p.write_text('{}\n')
    return event(m,root,'query_committed',query_id=q*64,attempt_index=index,
        prediction_reference=ref(p),elapsed_ns=7)


def test_metadata_creation_and_read_never_grant_label_or_native_authority(tmp_path,metadata):
    m,root,owners,binding=create(tmp_path,metadata)
    assert binding['protocol_sha256']==metadata['protocol_reference']['sha256']
    result=m._read_metadata_study(root)
    assert result['phase']=='prepared' and result['event_count']==1
    assert all(result[k] is False for k in ('protected_access_authorized','native_qualification_authenticated','T_accessed'))
    assert owners.is_dir() and binding['study_directory']==str(root)
    assert len(binding['genesis_sha256'])==64


def test_owner_key_ignores_study_source_timestamp_and_preserves_target_identity():
    m=api();asset=dict(dataset='toy',raw_sha256='d'*64,target='target')
    assert m._asset_key(asset)==m._asset_key(copy.deepcopy(asset))
    for field,value in [('dataset','different'),('raw_sha256','e'*64),('target','other')]:
        assert m._asset_key(dict(asset,**{field:value}))!=m._asset_key(asset)


def test_same_protocol_copy_resumes_original_genesis_but_not_new_study(tmp_path,metadata):
    m,root,owners,binding=create(tmp_path,metadata);ready(m,root)
    copied=tmp_path/'copied-protocol.json';copied.write_bytes(Path(metadata['protocol_reference']['path']).read_bytes())
    changed=dict(metadata,protocol_reference=ref(copied))
    again=m._prepare_metadata_study(directory=root,owner_root=owners,**changed)
    assert again==binding
    second=m._prepare_metadata_study(directory=tmp_path/'new-study',owner_root=owners,**changed)
    assert second['genesis_sha256']!=binding['genesis_sha256']
    with pytest.raises(PermissionError,match='owner|claimed'):m._claim_metadata_owners(tmp_path/'new-study')
    assert m._read_metadata_study(tmp_path/'new-study')['phase']=='prepared'


def test_all_asset_conflicts_checked_before_any_new_owner_is_published(tmp_path,metadata):
    m,root,owners,_=create(tmp_path,metadata);ready(m,root)
    other=dict(metadata,assets=[dict(dataset='new',raw_sha256='e'*64,target='target'),*metadata['assets']])
    root2=tmp_path/'second';m._prepare_metadata_study(directory=root2,owner_root=owners,**other)
    with pytest.raises(PermissionError):m._claim_metadata_owners(root2)
    assert not (owners/'claims'/f"{m._asset_key(other['assets'][0])}.json").exists()


def test_partial_owner_publication_retained_and_resumes_exact_binding(tmp_path,metadata,monkeypatch):
    metadata['assets'].append(dict(dataset='second',raw_sha256='e'*64,target='target'))
    m,root,owners,binding=create(tmp_path,metadata)
    original=m._write_new;count=0
    def disk_full(path,value):
        nonlocal count
        if Path(path).parent==owners/'claims':
            count+=1
            if count==2:raise OSError(28,'No space left on device')
        return original(path,value)
    monkeypatch.setattr(m,'_write_new',disk_full)
    with pytest.raises(OSError):m._claim_metadata_owners(root)
    assert len(list((owners/'claims').glob('*.json')))==1
    assert m._read_metadata_study(root)['phase']=='prepared'
    monkeypatch.setattr(m,'_write_new',original);ready(m,root)
    assert len(list((owners/'claims').glob('*.json')))==2
    assert m._read_metadata_study(root)['binding']==binding


@pytest.mark.parametrize('fault',['missing','partial','truncate','hash','extra_field','witness_missing'])
def test_journal_damage_is_not_treated_as_fresh_or_resumable(tmp_path,metadata,fault):
    m,root,owners,binding=create(tmp_path,metadata);ready(m,root)
    p=root/'events.jsonl';raw=p.read_bytes();lines=raw.splitlines(keepends=True)
    if fault=='missing':p.unlink()
    elif fault=='partial':p.write_bytes(raw[:-1])
    elif fault=='truncate':p.write_bytes(lines[0])
    elif fault=='hash':p.write_bytes(raw.replace(b'owners_ready',b'owners_weird'))
    elif fault=='extra_field':
        row=json.loads(lines[-1]);row['unexpected']=True
        p.write_bytes(lines[0]+(json.dumps(row)+'\n').encode())
    else:
        witness=owners/'histories'/binding['genesis_sha256']/'00000001.json';witness.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        m._prepare_metadata_study(directory=root,owner_root=owners,**metadata)
    assert len(list((owners/'claims').glob('*.json')))==1


def test_lost_entire_study_cannot_reset_owner_history(tmp_path,metadata):
    m,root,owners,binding=create(tmp_path,metadata);ready(m,root)
    # Simulate lost study files, retaining the workspace-wide owner archive.
    for p in root.iterdir():p.unlink()
    root.rmdir()
    replacement=m._prepare_metadata_study(directory=root,owner_root=owners,**metadata)
    assert replacement['genesis_sha256']!=binding['genesis_sha256']
    with pytest.raises(PermissionError):m._claim_metadata_owners(root)


def test_all_predictions_before_label_stage_and_no_repeat_prediction(tmp_path,metadata):
    m,root,_,_=create(tmp_path,metadata)
    with pytest.raises(ValueError):start(m,root)
    ready(m,root)
    with pytest.raises(ValueError):event(m,root,'labels_started')
    start(m,root);commit(m,root,tmp_path)
    with pytest.raises(ValueError):start(m,root)
    with pytest.raises(ValueError):event(m,root,'all_predictions_committed')
    start(m,root,'b');commit(m,root,tmp_path,'b')
    event(m,root,'all_predictions_committed');event(m,root,'labels_started')
    p=tmp_path/'metrics.json';p.write_text('{}\n')
    event(m,root,'metrics_committed',metrics_reference=ref(p),elapsed_ns=11)
    out=m._read_metadata_study(root)
    assert out['phase']=='metrics_committed' and out['committed_queries']==['a'*64,'b'*64]
    assert out['known_paid_ns']==25 and out['execution_cost_complete'] is True
    assert out['protected_access_authorized'] is out['T_accessed'] is False
    with pytest.raises(ValueError):event(m,root,'labels_started')
    with pytest.raises(ValueError):event(m,root,'query_started',query_id='b'*64,attempt_index=1,resume_kind='initial')


@pytest.mark.parametrize('status',['resource_pause','interrupted_recoverable'])
def test_resource_resume_keeps_all_attempts_and_cost(tmp_path,metadata,status):
    m,root,_,_=create(tmp_path,metadata);ready(m,root);start(m,root)
    checkpoint=tmp_path/'query-checkpoint.json';checkpoint.write_text('{}\n')
    event(m,root,'query_failed',query_id='a'*64,attempt_index=0,status=status,
        error='shared machine pressure',elapsed_ns=5,recovery_reference=ref(checkpoint))
    start(m,root,index=1,mode='same_state_resume');commit(m,root,tmp_path,index=1)
    out=m._read_metadata_study(root)
    assert out['known_paid_ns']==12 and out['event_count']==6
    assert out['attempts']['a'*64]==2


def test_unknown_failure_cost_not_zero_and_numerical_failure_blocks_labels(tmp_path,metadata):
    m,root,_,_=create(tmp_path,metadata);ready(m,root);start(m,root)
    event(m,root,'query_failed',query_id='a'*64,attempt_index=0,status='numerical_failure',
        error='nonfinite predictions',elapsed_ns=None,recovery_reference=None)
    out=m._read_metadata_study(root)
    assert out['known_paid_ns']==0 and out['execution_cost_complete'] is False
    with pytest.raises(ValueError):start(m,root,index=1,mode='same_seed_rerun')
    with pytest.raises(ValueError):event(m,root,'all_predictions_committed')


def test_dependency_failure_has_one_same_seed_rerun_not_unbounded_retries(tmp_path,metadata):
    m,root,_,_=create(tmp_path,metadata);ready(m,root);start(m,root)
    detail=dict(query_id='a'*64,status='input_or_dependency_failure',error='load failed',elapsed_ns=3,recovery_reference=None)
    event(m,root,'query_failed',attempt_index=0,**detail)
    start(m,root,index=1,mode='same_seed_rerun');event(m,root,'query_failed',attempt_index=1,**detail)
    with pytest.raises(ValueError):start(m,root,index=2,mode='same_seed_rerun')


def test_committed_file_mutation_unknown_query_and_actor_rejected(tmp_path,metadata):
    m,root,_,_=create(tmp_path,metadata);ready(m,root)
    with pytest.raises(ValueError):start(m,root,'e')
    with pytest.raises(ValueError):m._append_metadata_event(root,kind='query_started',actor='evaluator',
        details=dict(query_id='a'*64,attempt_index=0,resume_kind='initial'))
    start(m,root);commit(m,root,tmp_path)
    (tmp_path/'prediction-a-0.json').write_text('{"changed":true}\n')
    with pytest.raises(ValueError):m._read_metadata_study(root)


def test_storage_roots_protect_data_models_source_runtime_and_each_other(tmp_path):
    m=api();data=tmp_path/'data';data.mkdir();study=tmp_path/'study';owners=tmp_path/'owners'
    assert m._check_storage(study,owners,protected_paths=[data])==(study,owners)
    for a,b in [(data/'study',owners),(study,data/'owners'),(study,study/'owners')]:
        with pytest.raises(ValueError):m._check_storage(a,b,protected_paths=[data])
    assert not study.exists() and not owners.exists()


@pytest.mark.parametrize('entry',['prepare','inspect'])
def test_public_gate_rejects_false_qualification_before_study_or_owner_io(tmp_path,entry):
    m=api();kwargs=dict(models_reference=dict(path=str(tmp_path/'forbidden-model'),sha256='a'*64),
        global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='b'*64),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64),
        preT_costs_reference=dict(path=str(tmp_path/'forbidden-cost'),sha256='d'*64),
        readiness_reference=dict(path=str(tmp_path/'forbidden-readiness'),sha256='e'*64))
    protocol=dict(path=str(tmp_path/'forbidden-protocol'),sha256='f'*64)
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        if entry=='prepare':m.prepare_protected_study(protocol,directory=tmp_path/'study',**kwargs)
        else:m.inspect_protected_study(protocol,directory=tmp_path/'study',**kwargs)
    assert all(s not in str(error.value) for s in ('forbidden-model','forbidden-protocol','forbidden-cost'))
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('item',['binding.json','events.jsonl','journal.lock','witness','claim'])
def test_symlinked_history_owner_or_lock_is_rejected_before_write(tmp_path,metadata,item):
    m,root,owners,binding=create(tmp_path,metadata);ready(m,root)
    if item=='witness':path=owners/'histories'/binding['genesis_sha256']/'00000001.json'
    elif item=='claim':path=owners/'claims'/f"{m._asset_key(metadata['assets'][0])}.json"
    else:path=root/item
    external=tmp_path/'redirected';external.write_bytes(path.read_bytes());path.unlink();path.symlink_to(external)
    before=external.read_bytes()
    with pytest.raises(ValueError):start(m,root)
    assert external.read_bytes()==before


@pytest.mark.parametrize('field',['source_sha256','protocol_sha256','prediction_manifest_sha256','genesis_sha256','study_directory','query_ids','owner_root'])
def test_changed_binding_cannot_resume_or_claim_again(tmp_path,metadata,field):
    m,root,owners,binding=create(tmp_path,metadata);ready(m,root)
    p=root/'binding.json';changed=copy.deepcopy(binding)
    if field=='query_ids':changed[field]=['e'*64]
    elif field in ('study_directory','owner_root'):changed[field]=str(tmp_path/'changed')
    else:changed[field]='e'*64
    p.write_text(json.dumps(changed)+'\n')
    with pytest.raises((ValueError,FileNotFoundError)):m._prepare_metadata_study(directory=root,owner_root=owners,**metadata)


def test_started_query_with_lost_outcome_remains_ambiguous_not_automatically_retried(tmp_path,metadata):
    m,root,_,_=create(tmp_path,metadata);ready(m,root);start(m,root)
    out=m._read_metadata_study(root)
    assert out['active']==('a'*64,0)
    assert out['execution_cost_complete'] is False
    with pytest.raises(ValueError):start(m,root,index=1,mode='same_state_resume')
    with pytest.raises(ValueError):event(m,root,'all_predictions_committed')


def test_witness_written_but_journal_append_lost_blocks_resume(tmp_path,metadata):
    m,root,owners,binding=create(tmp_path,metadata);ready(m,root)
    old=(root/'events.jsonl').read_bytes();start(m,root)
    (root/'events.jsonl').write_bytes(old)
    with pytest.raises(ValueError,match='witness|truncat|transaction'):m._read_metadata_study(root)


@pytest.mark.parametrize('field,value',[('attempt_index',True),('elapsed_ns',-1),('elapsed_ns',True),('extra',0)])
def test_invalid_query_outcome_cannot_publish_witness(tmp_path,metadata,field,value):
    m,root,owners,binding=create(tmp_path,metadata);ready(m,root);start(m,root)
    p=tmp_path/'prediction.json';p.write_text('{}\n')
    details=dict(query_id='a'*64,attempt_index=0,prediction_reference=ref(p),elapsed_ns=3);details[field]=value
    before=m._read_metadata_study(root)['journal_sha256']
    with pytest.raises(ValueError):event(m,root,'query_committed',**details)
    assert m._read_metadata_study(root)['journal_sha256']==before
    assert len(list((owners/'histories'/binding['genesis_sha256']).glob('*.json')))==3


def test_protected_storage_preserves_all_model_searches_and_execution_proofs(tmp_path):
    m=api();program=tmp_path/'training-program';search=program/'searches/job';snapshot=search/'snapshots/best'
    data=tmp_path/'sealed-data';protocol=tmp_path/'protocol.json';owners=tmp_path/'owners'
    payload=dict(data={'toy':dict(root=str(data))},prediction_manifest=dict(queries=[dict(model_binding=dict(snapshot_root=str(snapshot)))]),
        models_reference=dict(path=str(tmp_path/'models.json'),sha256='a'*64),
        preT_costs_reference=dict(path=str(tmp_path/'costs.json'),sha256='b'*64),
        readiness_reference=dict(path=str(tmp_path/'readiness.json'),sha256='c'*64),
        global_barrier_reference=dict(path=str(tmp_path/'global.json'),sha256='d'*64),
        practical_plan_reference=dict(path=str(tmp_path/'plan.json'),sha256='e'*64),
        procedure_reference=dict(path=str(tmp_path/'procedure.json'),sha256='f'*64))
    protected=m._protected_storage_paths(payload,proof_roots=[str(program)],protocol_path=str(protocol))
    for location in (program/'hidden-study',data/'study',snapshot/'study'):
        with pytest.raises(ValueError):m._check_storage(location,owners,protected_paths=protected)
    assert m._check_storage(tmp_path/'protected-study',owners,protected_paths=protected)==(tmp_path/'protected-study',owners)


def test_journal_lock_serializes_actual_competing_writer(tmp_path,metadata):
    import fcntl
    import os
    import select
    import subprocess
    import sys
    m,root,_,_=create(tmp_path,metadata);ready(m,root)
    code='''import sys
from treeneuron.research.protected_journal_v2 import _append_metadata_event
print("ready",flush=True)
_append_metadata_event(sys.argv[1],kind="query_started",details=dict(query_id="a"*64,attempt_index=0,resume_kind="initial"))
print("committed",flush=True)
'''
    with (root/'journal.lock').open('a+b') as locked:
        fcntl.flock(locked.fileno(),fcntl.LOCK_EX)
        worker=subprocess.Popen([sys.executable,'-P','-B','-c',code,str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src')),text=True)
        try:
            assert select.select([worker.stdout],[],[],10)[0]
            assert worker.stdout.readline().strip()=='ready'
            assert not select.select([worker.stdout],[],[],.15)[0]
            fcntl.flock(locked.fileno(),fcntl.LOCK_UN)
            out,err=worker.communicate(timeout=10)
            assert worker.returncode==0 and out=='committed\n',err
        finally:
            if worker.poll() is None:worker.kill();worker.communicate(timeout=10)
    assert m._read_metadata_study(root)['active']==('a'*64,0)


@pytest.mark.parametrize('field,value',[('query_ids',['e'*64]),('prediction_manifest_sha256','e'*64)])
def test_witnesses_commit_entire_binding_not_just_protocol_and_genesis(tmp_path,metadata,field,value):
    m,root,_,binding=create(tmp_path,metadata);ready(m,root)
    changed=dict(binding,**{field:value});(root/'binding.json').write_text(json.dumps(changed)+'\n')
    with pytest.raises(ValueError,match='binding|journal|witness|owner'):m._read_metadata_study(root)


def test_evidenced_failure_retains_immutable_attempt_proof(tmp_path,metadata):
    m,root,_,_=create(tmp_path,metadata);ready(m,root);start(m,root)
    checkpoint=tmp_path/'checkpoint.json';checkpoint.write_text('{}\n')
    evidence=tmp_path/'attempt-clock.json';evidence.write_text('{"fixture":true}\n')
    event(m,root,'query_failed_with_evidence',query_id='a'*64,attempt_index=0,status='resource_pause',
        error='shared machine pressure',elapsed_ns=5,recovery_reference=ref(checkpoint),evidence_reference=ref(evidence))
    assert m._read_metadata_study(root)['known_paid_ns']==5
    evidence.write_text('{"changed":true}\n')
    with pytest.raises(ValueError):m._read_metadata_study(root)
