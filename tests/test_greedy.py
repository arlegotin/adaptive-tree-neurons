"""B6 must charge exhaustive independent short continuations, without V access."""
import copy
import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_train import cpu_training_evidence, training_case, assert_state_equal, events
from treeneuron.artifacts import create_run
from treeneuron.config import resolve_experiment
from treeneuron.train import fit_parent, pack_state, unpack_state, capture_rng
from treeneuron.growth.score import score_units


@pytest.fixture
def probe_case(training_case):
    data, config, original = training_case
    cfg = config.to_dict()
    cfg['model']['h'] = 2
    cfg['growth']['m_requested'] = 1
    config = resolve_experiment(cfg)
    parent = fit_parent(data, config, create_run(original.directory, cfg))
    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError('V/T labels must not enter probes')
    object.__setattr__(data, 'V', Forbidden())
    table = score_units(copy.deepcopy(parent.model).double().eval(),
        torch.tensor(parent.preprocessing.transform_X(data.F.X)),
        torch.tensor(parent.preprocessing.transform_y(data.F.y)), epsilon=.05, tau=1.,
        adam_epsilon=1e-8, batch_size=4, unit_block_size=2)
    assert table.eligible.tolist() == [True, True]
    return data, parent, table, create_run(original.directory, cfg)


def test_probes_restart_whole_parent_charge_losers_and_joint_restart(probe_case, monkeypatch):
    from treeneuron.baselines import greedy
    from treeneuron.baselines.causal import build_causal, CausalPredictor
    from treeneuron.baselines.api import FitRequest
    data, parent, table, paths = probe_case
    before = copy.deepcopy(parent)
    starts = []
    real = greedy.continue_fit
    def observe(state, data, policy, paths):
        starts.append(copy.deepcopy(state))
        return real(state, data, policy, paths)
    monkeypatch.setattr(greedy, 'continue_fit', observe)
    allocation = greedy.run_greedy_probes(parent, table, data, probe_updates=2, seed=7, paths=paths)
    assert allocation.policy == 'greedy' and allocation.effective_count == 1
    assert allocation.table_hash == table.table_hash
    assert_state_equal(parent, before)
    report = json.loads((paths.directory/'greedy.json').read_text())
    assert report['complete'] and len(report['trials']) == 3
    assert {r['candidate'] for r in report['trials']} == {'keep','unit_0','unit_1'}
    assert all(r['nanoseconds'] > 0 and r['executed_updates'] == 2 for r in report['trials'])
    streams = []
    for trial, start in zip(report['trials'], starts):
        assert (start.update,start.cursor,start.epoch) == (parent.update,parent.cursor,parent.epoch)
        assert start.scheduler.position == parent.scheduler.position
        assert torch.equal(start.order,parent.order)
        assert torch.equal(start.sampler_generator.get_state(),parent.sampler_generator.get_state())
        for name,p in parent.model.named_parameters():
            q=dict(start.model.named_parameters())[name]
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            for key in ('step','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(parent.optimizer.state[p][key],start.optimizer.state[q][key],rtol=0,atol=0)
        child = paths.directory/trial['path']
        ev = [json.loads(v) for v in (child/'events.jsonl').read_text().splitlines()]
        streams.append([e['row_ids'] for e in ev if e['type']=='update'])
        assert [e['stage_update'] for e in ev if e['type']=='evaluation']==[0,2]
        manifest=json.loads((child/'manifest.json').read_text())
        assert manifest['identity']['coordinator_run_id']==paths.run_id
        assert manifest['identity']['score_hash']==table.table_hash
    assert streams[0] == streams[1] == streams[2]
    assert len([e for e in events(paths) if e['type']=='probe_registered']) == 3
    adapter=build_causal('B6',{},parent=parent,allocation=allocation)
    assert adapter.state.update == parent.update and adapter.state.cursor == parent.cursor
    np.testing.assert_array_equal(adapter.predict(data.G.X),parent.predict(data.G.X))
    adapter.fit(FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,2,{'development_data':data},
                          {'max_updates':6},paths.directory/'joint'))
    adapter.save(paths.directory/'joint.json')
    restored=CausalPredictor.load(paths.directory/'joint.json',expected_hashes=adapter.state.identity_hashes)
    assert_state_equal(adapter.state,restored.state)


def test_negative_and_tied_improvements_still_allocate_fixed_count(probe_case):
    from treeneuron.baselines.greedy import _rank_allocation
    _, parent, table, _ = probe_case
    # Both probes lose to continued keep; the fixed count still picks lower unit ID.
    allocation = _rank_allocation(table,1,7,keep_mse=1.,unit_mse={1:2.,0:2.})
    assert allocation.ids.tolist() == [0] and allocation.effective_count == 1
    with pytest.raises(ValueError,match='complete'):
        _rank_allocation(table,1,7,keep_mse=1.,unit_mse={0:2.})


def test_zero_eligible_retains_keep_and_zero_growth(probe_case):
    from treeneuron.baselines.greedy import run_greedy_probes
    data,parent,table,paths=probe_case
    table=replace(table,eligible=torch.zeros(2,dtype=torch.bool))
    allocation=run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=paths)
    assert allocation.requested_count==1 and allocation.effective_count==0
    report=json.loads((paths.directory/'greedy.json').read_text())
    assert [r['candidate'] for r in report['trials']]==['keep']
    assert report['complete']


def test_probe_failure_retains_registration_and_cost_without_partial_ranking(probe_case,monkeypatch):
    from treeneuron.baselines.greedy import run_greedy_probes
    data,parent,table,paths=probe_case
    real=torch.optim.AdamW.step
    def fail_grown(optimizer,*args,**kwargs):
        if len(optimizer.param_groups)>1:
            raise FloatingPointError('injected probe update failure')
        return real(optimizer,*args,**kwargs)
    monkeypatch.setattr(torch.optim.AdamW,'step',fail_grown)
    with pytest.raises(FloatingPointError,match='injected'):
        run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=paths)
    report=json.loads((paths.directory/'greedy.json').read_text())
    assert not report['complete'] and report['allocation'] is None
    assert len(report['trials'])==2
    assert all(r['nanoseconds']>0 for r in report['trials'])
    failed=report['trials'][-1]
    child=paths.directory/failed['path']
    manifest=json.loads((child/'manifest.json').read_text())
    assert manifest['identity']['status']=='failed'
    assert manifest['identity']['coordinator_run_id']==paths.run_id
    assert manifest['identity']['score_hash']==table.table_hash
    assert failed['selected_update'] is None and failed['unavailable_reason']
    assert report['not_attempted']==['unit_1']


@pytest.mark.parametrize('mutation',['unfinished','selected_snapshot','allowance','data'])
def test_rejects_noncanonical_parent_and_invalid_probe_contract(probe_case,mutation):
    from treeneuron.baselines.greedy import run_greedy_probes
    data,parent,table,paths=probe_case
    allowance=2
    if mutation=='unfinished': parent.stage['complete']=False
    if mutation=='selected_snapshot': parent.best_snapshot=pack_state(parent)
    if mutation=='allowance': allowance=7
    if mutation=='data': parent.data_hashes={}
    with pytest.raises(ValueError):
        run_greedy_probes(parent,table,data,probe_updates=allowance,seed=7,paths=paths)


def test_setup_failure_is_registered_and_costed_before_training(probe_case,monkeypatch):
    from treeneuron.baselines import greedy
    data,parent,table,paths=probe_case
    def fail_insertion(*args):
        raise RuntimeError('injected insertion setup failure')
    monkeypatch.setattr(greedy,'insert_growth',fail_insertion)
    with pytest.raises(RuntimeError,match='setup failure'):
        greedy.run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=paths)
    report=json.loads((paths.directory/'greedy.json').read_text())
    trial=report['trials'][-1]
    manifest=json.loads((paths.directory/trial['path']/'manifest.json').read_text())
    assert manifest['identity']['status']=='failed'
    assert manifest['identity']['coordinator_run_id']==paths.run_id
    assert manifest['costs']['probe']['nanoseconds']>0
    registration=[e for e in events(paths) if e['type']=='probe_registered'][-1]
    assert registration['method_id']=='B6'
    assert registration['stage']=='single_unit_probe'
    assert registration['parent_identity_hashes']==parent.identity_hashes
    assert registration['V_role']=='not_accessed'
    assert manifest['data']['counts']['V']['groups']==5


def test_rng_and_table_evidence_survive_probes(probe_case):
    from treeneuron.baselines.greedy import run_greedy_probes
    import hashlib
    data,parent,table,paths=probe_case
    before=capture_rng()
    run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=paths)
    after=capture_rng()
    state_before,state_after=copy.copy(parent),copy.copy(parent)
    state_before.rng_states,state_after.rng_states=before,after
    assert_state_equal(state_before,state_after)
    report=json.loads((paths.directory/'greedy.json').read_text())
    record=report['score_artifact']
    artifact=paths.directory/record['path']
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==record['sha256']
    with np.load(artifact,allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['contrasts'],table.contrasts.numpy())
        np.testing.assert_array_equal(stored['eligible'],table.eligible.numpy())


def test_greedy_preserves_noncontiguous_original_row_contrasts(probe_case):
    from treeneuron.baselines.greedy import _rank_allocation
    _,_,table,_=probe_case
    table=replace(table,eligible=torch.tensor([False,True]))
    allocation=_rank_allocation(table,2,7,keep_mse=1.,unit_mse={1:3.})
    assert allocation.ids.tolist()==[1] and allocation.requested_count==2
    torch.testing.assert_close(allocation.contrasts,table.contrasts[1:2],rtol=0,atol=0)


def test_probe_patience_does_not_silently_rank_shortened_trials(probe_case,monkeypatch):
    from treeneuron.baselines import greedy
    data,parent,table,paths=probe_case
    real=greedy.continue_fit
    def shortened(*args,**kwargs):
        state=real(*args,**kwargs)
        state.stage['updates_completed']=1
        return state
    monkeypatch.setattr(greedy,'continue_fit',shortened)
    with pytest.raises(ValueError,match='Incomplete probe allowance'):
        greedy.run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=paths)
    report=json.loads((paths.directory/'greedy.json').read_text())
    assert not report['complete'] and report['allocation'] is None


@pytest.mark.parametrize('publication',['probe_result','greedy_report'])
def test_publication_failure_preserves_primary_failure_and_rng(probe_case,monkeypatch,publication):
    from treeneuron.baselines import greedy
    data,parent,table,paths=probe_case
    before=copy.copy(parent); before.rng_states=capture_rng()
    def fail_insertion(*args): raise RuntimeError('primary insertion failure')
    monkeypatch.setattr(greedy,'insert_growth',fail_insertion)
    if publication=='probe_result':
        original=greedy.append_event
        def fail_result(run,event):
            if event['type']=='probe_result' and event['candidate']=='unit_0':
                raise OSError('secondary publication failure')
            return original(run,event)
        monkeypatch.setattr(greedy,'append_event',fail_result)
    else:
        def fail_report(*args,**kwargs): raise OSError('secondary publication failure')
        monkeypatch.setattr(greedy,'_atomic_json',fail_report)
    with pytest.raises(RuntimeError,match='primary insertion failure') as caught:
        greedy.run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=paths)
    assert any('secondary publication failure' in note for note in caught.value.__notes__)
    after=copy.copy(parent); after.rng_states=capture_rng()
    assert_state_equal(before,after)
    manifest=json.loads((paths.directory/'manifest.json').read_text())
    assert manifest['costs']['probe']['nanoseconds']>0


def test_b6_adapter_rejects_mislabeled_adam_allocation(probe_case):
    from treeneuron.baselines.causal import build_causal
    from treeneuron.growth.allocate import allocate
    _,parent,table,_=probe_case
    with pytest.raises(ValueError,match='policy'):
        build_causal('B6',{},parent=parent,allocation=allocate(table,1,policy='adam',seed=7))


def test_b5_cannot_mislabel_greedy_as_its_score_variant(probe_case):
    from treeneuron.baselines.causal import build_causal
    from treeneuron.baselines.greedy import _rank_allocation
    _,parent,table,_=probe_case
    allocation=_rank_allocation(table,1,7,keep_mse=1.,unit_mse={0:2.,1:3.})
    with pytest.raises(ValueError,match='policy'):
        build_causal('B5',{'variant':'greedy'},parent=parent,allocation=allocation)
