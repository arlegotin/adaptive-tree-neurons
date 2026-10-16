"""Budget, selection, immutable replay and worker leakage contracts."""
import json
from pathlib import Path
import numpy as np
import pytest
from treeneuron.artifacts import create_run
from treeneuron.config import resolve_experiment
from treeneuron.data.types import DevelopmentData, ObservationSet


def api():
    from treeneuron.evaluation import search
    return search


@pytest.fixture
def case(tmp_path):
    rng = np.random.default_rng(9)
    def rows(start, n):
        x = rng.normal(size=(n, 2)); ids = np.arange(start, start+n)
        return ObservationSet(x, 4+x[:, 0], ids, ids)
    data = DevelopmentData(rows(0, 12), rows(20, 5), rows(30, 7), 'search-test')
    config = resolve_experiment({'study_id':'search-test','model':{'d':2,'h':4},
        'data':{'n_fit':12,'n_growth':5,'n_validation':7},
        'training':{'batch_size':4,'base_updates':2,'continuation_updates':2,'validation_interval':1,'patience':3}})
    return data, create_run(tmp_path, config.to_dict())


def test_official_anchor_first_retained_v_winner_and_failed_costs(case, monkeypatch):
    s = api(); data, paths = case; clock = [0]; seen = []
    monkeypatch.setattr(s.time, 'perf_counter_ns', lambda: clock[0])
    def worker(request, directory, timeout):
        assert 'y_validation' not in request and set(request['arrays']) == {'F_X','F_y','G_X','G_y','V_X','F_ids','G_ids','V_ids'}
        seen.append(request['config'])
        clock[0] += 2_000_000_000
        if len(seen) == 2: raise RuntimeError('failed trial')
        np.save(directory/'prediction.npy', data.V.y + (3 if len(seen)==1 else 1), allow_pickle=False)
        return {'quality_selection_eligible':True,'prediction_sha256':s.file_hash(directory/'prediction.npy'), 'resource_summary':{}}
    monkeypatch.setattr(s, '_run_worker', worker)
    result = s.search('Ridge', data, s.SearchBudget(seconds=7, max_trials=3), 12, paths)
    assert seen[0] == {'alpha':1.0}
    assert [t['status'] for t in result.trials] == ['completed','failed','completed']
    assert result.trials[0]['official_anchor'] and result.winner == 2
    assert [t['charged_ns'] for t in result.trials] == [2_000_000_000]*3
    assert result.charged_ns == 6_000_000_000
    assert json.loads((paths.directory/'manifest.json').read_text())['identity']['status']=='completed'


def test_timeout_official_default_never_wins(case, monkeypatch):
    s=api(); data,paths=case; clock=[0]
    monkeypatch.setattr(s.time,'perf_counter_ns',lambda:clock[0])
    def worker(*args):
        clock[0]+=1_100_000_000
        raise TimeoutError('deadline')
    monkeypatch.setattr(s,'_run_worker',worker)
    result=s.search('B7',data,s.SearchBudget(seconds=1,max_trials=1,max_updates=3),7,paths)
    assert result.winner is None
    assert result.trials[0]['status']=='unavailable_at_budget'
    assert result.charged_ns==1_100_000_000
    assert result.trials[0]['error']['type']=='TimeoutError'


def test_sampler_resume_exact_immutable_and_rejects_changed_identity(tmp_path):
    s=api(); support={'alpha':{'kind':'float','low':1e-6,'high':1e6,'log':True}}
    sampler=s.ReplayTPE(5,support,{'source':'abc'})
    for i in range(14):
        params=sampler.ask({'alpha':1.} if i==0 else None)
        sampler.tell(float((np.log10(params['alpha'])-2)**2) if i!=3 else None)
    path=tmp_path/'sampler.json'; sampler.save(path); original=path.read_bytes()
    loaded=s.ReplayTPE.load(path,identity={'source':'abc'})
    assert loaded.ask()==sampler.ask()
    assert path.read_bytes()==original
    loaded.tell(1.)
    with pytest.raises(FileExistsError): loaded.save(path)
    with pytest.raises(ValueError,match='identity'): s.ReplayTPE.load(path,identity={'source':'different'})
    payload=json.loads(path.read_text()); payload['state']['history'][4]['params']['alpha']=.2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='hash'):s.ReplayTPE.load(path,identity={'source':'abc'})


def test_real_isolated_ridge_worker_and_strict_budget(case):
    s=api(); data,paths=case
    result=s.search('Ridge',data,s.SearchBudget(seconds=30,max_trials=1),9,paths)
    assert result.winner==0 and result.trials[0]['validation_mse']<.02
    assert result.trials[0]['resource_summary']['threads']=={'cpu':1,'interop':1}
    assert result.trials[0]['resource_summary']['fit_label_partitions']==['F','G']
    for kwargs in ({'seconds':float('nan')},{'seconds':0},{'seconds':1,'max_trials':True}):
        with pytest.raises(ValueError):s.SearchBudget(**kwargs)
    with pytest.raises(TypeError):s.SearchBudget(seconds=2,unknown=4)


def test_unsupported_method_finishes_failed_with_original_reason(case):
    s=api();data,paths=case
    with pytest.raises(ValueError,match='unsupported'):s.search('B8',data,s.SearchBudget(seconds=1),1,paths)
    metrics=json.loads((paths.directory/'metrics.json').read_text())
    assert metrics['error']['type']=='ValueError'
    assert json.loads((paths.directory/'manifest.json').read_text())['identity']['status']=='failed'


@pytest.mark.parametrize('method,recipe',[('B1',{}),('B3',{'variant':'partial','count':1,'coordinates':'centered'}),('B4',{}),('B5',{'variant':'adam'}),('B6',{'probe_updates':1})])
def test_actual_causal_worker_preserves_fg_provenance_and_closes_runs(case,search_backend,method,recipe):
    s=api();data,paths=case
    result=s.search(method,data,s.SearchBudget(seconds=30,max_trials=1,recipe=recipe,backend_report=str(search_backend)),8,paths)
    assert result.winner==0,result.trials
    root=paths.directory/'trial-0000'/'runs'
    manifests=[json.loads(p.read_text()) for p in root.glob('*/manifest.json')]
    assert manifests and all(m['identity']['status']=='completed' for m in manifests)
    assert all(m['data']['split_hash']==data.manifest.assignment_hash for m in manifests)
    assert all(m['data']['test_labels_accessed'] is False for m in manifests)
    assert result.trials[0]['resource_summary']['fit_label_partitions']==['F','G']


def test_budget_expiry_after_validation_rejects_trial(case,monkeypatch):
    s=api();data,paths=case;clock=[0]
    monkeypatch.setattr(s.time,'perf_counter_ns',lambda:clock[0])
    def worker(request,directory,timeout):
        np.save(directory/'prediction.npy',data.V.y,allow_pickle=False)
        clock[0]=900_000_000
        return {'quality_selection_eligible':True,'prediction_sha256':s.file_hash(directory/'prediction.npy'),'resource_summary':{}}
    monkeypatch.setattr(s,'_run_worker',worker)
    original=np.mean
    def mean(*args,**kwargs):
        clock[0]=1_200_000_000
        return original(*args,**kwargs)
    monkeypatch.setattr(s.np,'mean',mean)
    result=s.search('Mean',data,s.SearchBudget(seconds=1),1,paths)
    assert result.winner is None and result.trials[0]['status']=='unavailable_at_budget'


from test_train import cpu_training_evidence


@pytest.fixture(scope='module')
def search_backend(tmp_path_factory):
    s=api();path=tmp_path_factory.mktemp('search-backend')/'report.json'
    s.produce_search_backend_receipt(path)
    report=json.loads(path.read_text())
    assert report['backend_probes']['probes']['cpu']['environment']['interop_threads']==1
    return path


def test_ineligible_official_checkpoint_retained_without_v_scoring(case,monkeypatch):
    s=api();data,paths=case
    def worker(request,directory,timeout):
        return {'quality_selection_eligible':False,'resource_summary':{'selection_status':'unavailable_at_budget'}}
    monkeypatch.setattr(s,'_run_worker',worker)
    result=s.search('B7',data,s.SearchBudget(seconds=30,max_updates=3),3,paths)
    assert result.winner is None and result.trials[0]['official_anchor']
    assert result.trials[0]['validation_mse'] is None
    assert result.trials[0]['resource_summary']['selection_status']=='unavailable_at_budget'


@pytest.mark.parametrize('method,recipe',[('B2',{'variant':'upper','target_parameters':25}),('B2',{'variant':'widen','added_units':2}),('B3',{'variant':'full','coordinates':'direct_leaf'})])
def test_alternate_causal_recipes_have_only_their_own_keys(case,search_backend,method,recipe):
    s=api();data,paths=case
    result=s.search(method,data,s.SearchBudget(seconds=30,recipe=recipe,backend_report=str(search_backend)),7,paths)
    assert result.winner==0,result.trials


def test_resume_keeps_old_artifact_and_rejects_historical_model_winner(case,monkeypatch):
    s=api();data,paths=case
    def worker(request,directory,timeout):
        np.save(directory/'prediction.npy',data.V.y,allow_pickle=False)
        return {'quality_selection_eligible':True,'prediction_sha256':s.file_hash(directory/'prediction.npy'),'resource_summary':{}}
    monkeypatch.setattr(s,'_run_worker',worker)
    first=s.search('Ridge',data,s.SearchBudget(seconds=30),11,paths)
    old=Path(first.sampler_state).read_bytes()
    new=create_run(paths.directory.parent,json.loads((paths.directory/'config.resolved.json').read_text()))
    second=s.search('Ridge',data,s.SearchBudget(seconds=30,resume_sampler=first.sampler_state),11,new)
    assert Path(first.sampler_state).read_bytes()==old
    assert not second.trials[0]['official_anchor'] and second.trials[0]['sampler_number']==1
    assert second.winner==0


def test_nonfinite_or_tampered_prediction_is_charged_and_cannot_win(case,monkeypatch):
    s=api();data,paths=case
    def worker(request,directory,timeout):
        np.save(directory/'prediction.npy',np.full(len(data.V.y),np.nan),allow_pickle=False)
        return {'quality_selection_eligible':True,'prediction_sha256':s.file_hash(directory/'prediction.npy'),'resource_summary':{}}
    monkeypatch.setattr(s,'_run_worker',worker)
    result=s.search('Ridge',data,s.SearchBudget(seconds=30),1,paths)
    assert result.winner is None and result.trials[0]['status']=='failed'
    assert result.trials[0]['charged_ns']>0
    assert result.trials[0]['error']['type']=='ValueError'


def test_boundaries_and_improving_ceiling_require_expansion():
    s=api()
    trials=[{'status':'completed','config':{'alpha':1e6},'validation_mse':.5}]
    status=s.expansion_diagnostics(trials,0,{'alpha':{'kind':'float','low':1e-6,'high':1e6,'log':True}},[{'seconds':2.,'best_validation_mse':1.},{'seconds':4.,'best_validation_mse':.5}])
    assert status['boundary_winners']==['alpha:high'] and status['improving_ceiling']
    assert not status['freeze_ready']


def test_worker_timeout_closes_created_children_without_losing_error(case,monkeypatch,tmp_path):
    s=api();data,paths=case;directory=tmp_path/'worker';directory.mkdir()
    child=create_run(directory/'runs',json.loads((paths.directory/'config.resolved.json').read_text()))
    def timeout(*args,**kwargs):raise s.subprocess.TimeoutExpired('worker',.1)
    monkeypatch.setattr(s.subprocess,'run',timeout)
    with pytest.raises(TimeoutError,match='exceeded'):s._run_worker({},directory,.1)
    manifest=json.loads((child.directory/'manifest.json').read_text())
    assert manifest['identity']['status']=='interrupted'
    assert json.loads((child.directory/'metrics.json').read_text())['error']['type']=='TimeoutError'


def test_changed_backend_receipt_between_coordinator_and_worker_fails(case,search_backend,monkeypatch,tmp_path):
    s=api();data,paths=case;receipt=tmp_path/'receipt.json';receipt.write_bytes(search_backend.read_bytes())
    original=s._run_worker
    def mutate_then_run(request,directory,timeout):
        receipt.write_text(receipt.read_text()+'\n')
        return original(request,directory,timeout)
    monkeypatch.setattr(s,'_run_worker',mutate_then_run)
    result=s.search('B1',data,s.SearchBudget(seconds=30,backend_report=str(receipt)),5,paths)
    assert result.winner is None
    assert 'receipt hash mismatch' in result.trials[0]['error']['message']


def test_costed_resume_retains_failed_anchor_and_cumulative_lineage(case,monkeypatch):
    s=api();data,paths=case;clock=[0];calls=[0]
    monkeypatch.setattr(s.time,'perf_counter_ns',lambda:clock[0])
    def worker(request,directory,timeout):
        calls[0]+=1;clock[0]+=2_000_000_000
        if calls[0]==1:raise RuntimeError('paid anchor failure')
        np.save(directory/'prediction.npy',data.V.y,allow_pickle=False)
        return {'quality_selection_eligible':True,'prediction_sha256':s.file_hash(directory/'prediction.npy'),'resource_summary':{}}
    monkeypatch.setattr(s,'_run_worker',worker)
    first=s.search('Ridge',data,s.SearchBudget(seconds=10,max_trials=2),11,paths)
    new=create_run(paths.directory.parent,json.loads((paths.directory/'config.resolved.json').read_text()))
    second=s.search('Ridge',data,s.SearchBudget(seconds=10,resume_sampler=first.sampler_state),11,new)
    assert second.attempt_charged_ns==second.charged_ns==2_000_000_000
    assert second.cumulative_charged_ns==6_000_000_000
    assert len(second.prior_trials)==2 and len(second.all_trials)==3
    assert second.prior_trials[0]['official_anchor'] and second.prior_trials[0]['error']['message']=='paid anchor failure'
    assert second.prior_trials[0]['charged_ns']==2_000_000_000
    assert second.winner==0 and not second.trials[0]['official_anchor']
    # Both missing full ledger ancestry and changed retained trial artifacts fail closed.
    raw=paths.directory/'sampler.final.json'
    attempt=create_run(paths.directory.parent,json.loads((paths.directory/'config.resolved.json').read_text()))
    with pytest.raises(ValueError,match='lineage'):s.search('Ridge',data,s.SearchBudget(seconds=10,resume_sampler=str(raw)),11,attempt)
    trial=paths.directory/'trial-0000'/'trial.json';original=trial.read_bytes();trial.write_bytes(original+b'\n')
    attempt=create_run(paths.directory.parent,json.loads((paths.directory/'config.resolved.json').read_text()))
    with pytest.raises(ValueError,match='hash'):s.search('Ridge',data,s.SearchBudget(seconds=10,resume_sampler=second.sampler_state),11,attempt)

    trial.write_bytes(original)
    descriptor=json.loads(Path(first.sampler_state).read_text())['payload']
    Path(descriptor['ledger']['path']).unlink()
    attempt=create_run(paths.directory.parent,json.loads((paths.directory/'config.resolved.json').read_text()))
    with pytest.raises(ValueError,match='Missing cost lineage'):
        s.search('Ridge',data,s.SearchBudget(seconds=10,resume_sampler=second.sampler_state),11,attempt)
