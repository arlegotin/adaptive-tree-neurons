"""Bounded diagnostic machinery fixtures; no production cohort or receipt."""
import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from treeneuron.config import canonical_hash
from treeneuron.experiments.paired import file_hash, write_json
from test_pilot_calibrate import config


def module():
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')


def spec(tmp_path):
    c=config(tmp_path).to_dict()
    return dict(schema_version=1,base_config=c,teachers=3,repetitions=2,
        short_probe_updates=1,long_probe_updates=1,evaluation_rows=5,
        cost_seeds=dict(teacher=261030100,data=261030101,optimizer=261030102,
            allocation=261030103,contrast=261030104,reference=261030105),
        study_seeds=dict(teacher=261031100,data=261031101,optimizer=261031102,
            allocation=261031103,contrast=261031104,reference=261031105),
        bootstrap=dict(iterations=512000,seed=261031106,confidence=.95),
        resources=dict(safety_factor=1.5,disk_reserve_bytes=10*1024**3,
            rss_cap_bytes=6*1024**3,severe_available_percent=10,automatic_retries=0))


def test_spec_strict_and_cost_study_streams_disjoint(tmp_path):
    m=module(); value=spec(tmp_path)
    path=tmp_path/'spec.json'; path.write_text(json.dumps(value))
    loaded=m.load_spec(path)
    a,b=[m._config(loaded,phase) for phase in ('cost','study')]
    from treeneuron.pilot.calibrate import teacher_inputs,repetition_config
    _,ad,ae,ac=teacher_inputs(a,0); _,bd,be,bc=teacher_inputs(b,0)
    assert ad.teacher_id!=bd.teacher_id and not set(ae.row_id)&set(be.row_id)
    assert repetition_config(ac,0).seeds.optimizer!=repetition_config(bc,0).seeds.optimizer
    for edit in ({'unknown':1},{'teachers':True},{'long_probe_updates':0},
                 {'cost_seeds':value['study_seeds']},{'bootstrap':dict(value['bootstrap'],unknown=1)}):
        broken=copy.deepcopy(value); broken.update(edit); path.write_text(json.dumps(broken))
        with pytest.raises(ValueError): m.load_spec(path)
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError,match='Duplicate'): m.load_spec(path)


def fake_execution(monkeypatch,tmp_path,*,fail_at=None):
    m=module(); calls=[]
    monkeypatch.setattr(m,'_runtime_preflight',lambda c: {'receipt_sha256':'receipt'})
    monkeypatch.setattr(m,'_provenance',lambda s: dict(source='source',lock='lock',runtime='runtime',receipt='receipt',commit='commit',execution=canonical_hash(s)))
    monkeypatch.setattr(m,'_resource_snapshot',lambda path: dict(free_disk_bytes=10**14,peak_rss_bytes=100,available_percent=80,swapouts=0))
    def fit(c,data,e,root,recipe_hash,*,reference_probe_updates,resource_guard=None):
        root.mkdir()
        calls.append((data.teacher_id,c.seeds.optimizer))
        if fail_at==len(calls): raise RuntimeError('deliberate atomic failure')
        predictions={name:e.y+index for index,name in enumerate(m.METHODS)}
        np.savez_compressed(root/'E-predictions.npz',**predictions,y=e.y,row_id=e.row_id)
        result=dict(status='completed',teacher_id=data.teacher_id,methods={name:{} for name in m.METHODS},
            E_predictions_sha256=file_hash(root/'E-predictions.npz'),enclosing_ns=20,
            costs=dict(probe_ns=2,long_probe_ns=5),source_sha256='source',recipe_hash=recipe_hash,
            data_hash=data.manifest.assignment_hash,optimizer_seed=c.seeds.optimizer)
        write_json(root/'pair.json',result)
        return result
    monkeypatch.setattr(m,'run_repetition',fit)
    return m,calls


def test_measurement_lock_and_ordered_failure_membership(tmp_path,monkeypatch):
    m,calls=fake_execution(monkeypatch,tmp_path)
    value=spec(tmp_path)
    cost=m.measure(value,tmp_path/'cost')
    assert len(calls)==1 and cost['phase']=='cost' and cost['status']=='completed'
    decision=tmp_path/'cost'/'resource-decision.json'
    assert decision.is_file()
    cohort=m.run(value,decision,tmp_path/'study')
    assert cohort['status']=='completed' and len(calls)==7
    assert [(v['teacher_index'],v['repetition']) for v in cohort['members']]==[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
    assert all('record' not in v and 'pair_sha256' in v for v in cohort['members'])
    assert (tmp_path/'study'/'study-lock.json').is_file()
    assert cohort['members'][0]['teacher_id']!=cost['members'][0]['teacher_id']
    # The resource decision is hash bound and cannot silently change the recipe.
    changed=copy.deepcopy(value); changed['teachers']=4
    with pytest.raises(ValueError,match='spec|decision'):
        m.run(changed,decision,tmp_path/'drift')
    m,calls=fake_execution(monkeypatch,tmp_path,fail_at=3)
    with pytest.raises(RuntimeError,match='atomic failure'):
        m.run(value,decision,tmp_path/'failed')
    ledger=json.loads((tmp_path/'failed'/'members.json').read_text())
    assert ledger['status']=='failed'
    assert [v['status'] for v in ledger['members']]==['completed','completed','failed']
    assert len(ledger['expected_members'])==6 and all('enclosing_ns' in v for v in ledger['members'])


def test_resource_failure_retains_attempt_without_silent_retry(tmp_path,monkeypatch):
    m,calls=fake_execution(monkeypatch,tmp_path)
    monkeypatch.setattr(m,'_resource_snapshot',lambda path: dict(free_disk_bytes=1,peak_rss_bytes=100,available_percent=80,swapouts=0))
    with pytest.raises(RuntimeError,match='disk'):
        m.measure(spec(tmp_path),tmp_path/'blocked')
    ledger=json.loads((tmp_path/'blocked'/'members.json').read_text())
    assert ledger['status']=='failed' and not calls
    assert ledger['resource_observations'] and ledger['failures']


def test_severe_pressure_requires_both_low_availability_and_new_swapouts(tmp_path,monkeypatch):
    m=module(); policy=spec(tmp_path)['resources']
    ledger=dict(resource_observations=[])
    snapshots=iter([dict(free_disk_bytes=10**14,peak_rss_bytes=1,available_percent=9,swapouts=4),
        dict(free_disk_bytes=10**14,peak_rss_bytes=1,available_percent=9,swapouts=4),
        dict(free_disk_bytes=10**14,peak_rss_bytes=1,available_percent=9,swapouts=5)])
    monkeypatch.setattr(m,'_resource_snapshot',lambda p:next(snapshots))
    guard=m._resource_guard(tmp_path,policy,ledger)
    guard(); guard()
    with pytest.raises(RuntimeError,match='pressure'): guard()


def analysis_fixture(tmp_path,monkeypatch):
    m,calls=fake_execution(monkeypatch,tmp_path)
    value=spec(tmp_path)
    m.measure(value,tmp_path/'cost'); m.run(value,tmp_path/'cost'/'resource-decision.json',tmp_path/'study')
    # Unequal retained repetitions: teacher 0 two, teacher 1 one, teacher 2 two.
    # Analyze must keep teacher weights equal and retain zero/negative effects.
    root=tmp_path/'study'; ledger=json.loads((root/'members.json').read_text())
    ledger['status']='failed'; ledger['members'][3]['status']='failed'
    for entry in ledger['members']:
        if entry['status']!='completed': continue
        pair_root=root/entry['pair_path']; pair=json.loads((pair_root/'pair.json').read_text())
        with np.load(root/entry['observations_path'],allow_pickle=False) as obs:
            y=obs['E_y']; rows=obs['E_rows']; scale=np.var(obs['F_y'])
        effect={0:2.,1:0.,2:-1.}[entry['teacher_index']]
        pred={name:y.copy() for name in m.METHODS}
        # long risk is 1; random risk is 1+effect.
        pred['greedy_long']=y+np.sqrt(scale)
        pred['random']=y+np.sqrt(scale*(1+effect))
        np.savez_compressed(pair_root/'E-predictions.npz',**pred,y=y,row_id=rows)
        pair['E_predictions_sha256']=file_hash(pair_root/'E-predictions.npz')
        (pair_root/'pair.json').write_text(json.dumps(pair))
        entry['pair_sha256']=file_hash(pair_root/'pair.json')
        entry['predictions_sha256']=pair['E_predictions_sha256']
        artifacts=json.loads((pair_root/'artifacts.json').read_text())
        artifacts.update({'pair.json':entry['pair_sha256'],'E-predictions.npz':entry['predictions_sha256']})
        (pair_root/'artifacts.json').write_text(json.dumps(artifacts))
        entry['artifacts_sha256']=file_hash(pair_root/'artifacts.json')
    (root/'members.json').write_text(json.dumps(ledger))
    (root/'members-seal.json').write_text(json.dumps({'sha256':file_hash(root/'members.json')}))
    monkeypatch.setattr(m,'_audit_pair',lambda root,pair: {})
    return m,root


def test_analysis_teacher_weighting_original_units_and_retained_effects(tmp_path,monkeypatch):
    m,root=analysis_fixture(tmp_path,monkeypatch)
    result=m.analyze(root)
    assert not result['complete_cohort'] and result['independent_teachers']==3 and result['completed_repetitions']==5
    primary=result['differences']['random_minus_greedy_long']
    assert primary['teacher_effects']==pytest.approx([2.,0.,-1.])
    assert primary['mean']==pytest.approx(1/3) and primary['median']==pytest.approx(0.)
    assert primary['zero_teachers']==1 and primary['negative_teachers']==1
    assert result['methods']['greedy_long']['normalized_mse']==pytest.approx(1.)
    assert result['bootstrap']['iterations']==512000
    first=tmp_path/'report-a.md'; second=tmp_path/'report-b.md'
    m.report(root,first); m.report(root,second)
    assert first.read_bytes()==second.read_bytes()
    assert 'single-unit' in first.read_text() and 'Incomplete' in first.read_text()
    assert '```json' not in first.read_text()


@pytest.mark.parametrize('tamper',['prediction','targets','rows','fit_variance','pair','source'])
def test_analysis_rejects_missing_or_altered_evidence(tmp_path,monkeypatch,tamper):
    m,root=analysis_fixture(tmp_path,monkeypatch)
    ledger=json.loads((root/'members.json').read_text()); entry=ledger['members'][0]
    if tamper=='source':
        monkeypatch.setattr(m,'_provenance',lambda s: dict(source='changed',lock='lock',runtime='runtime',receipt='receipt',commit='other',execution=canonical_hash(s)))
    elif tamper=='pair': (root/entry['pair_path']/'pair.json').write_text('{}')
    elif tamper=='prediction': (root/entry['pair_path']/'E-predictions.npz').unlink()
    else:
        path=root/entry['observations_path']
        with np.load(path,allow_pickle=False) as stored: arrays={k:stored[k] for k in stored.files}
        key={'targets':'E_y','rows':'E_rows','fit_variance':'F_y'}[tamper]
        arrays[key]=arrays[key]+1
        np.savez_compressed(path,**arrays)
    with pytest.raises((ValueError,FileNotFoundError)): m.analyze(root)


from test_train import cpu_training_evidence


def test_bounded_real_diagnostic_graph_is_auditable_and_cli_report_is_deterministic(tmp_path,cpu_training_evidence,monkeypatch):
    m=module(); value=spec(tmp_path)
    value.update(teachers=1,repetitions=1)
    value['base_config']['execution']['backend_report']=str(cpu_training_evidence)
    monkeypatch.setattr(m,'_resource_snapshot',lambda path: dict(free_disk_bytes=10**14,peak_rss_bytes=100,available_percent=80,swapouts=0))
    m.measure(value,tmp_path/'cost-real')
    m.run(value,tmp_path/'cost-real'/'resource-decision.json',tmp_path/'study-real')
    result=m.analyze(tmp_path/'study-real')
    assert result['complete_cohort'] and result['independent_teachers']==1
    assert result['records'][0]['diagnostics']['pairing']['passed']
    assert len(result['records'][0]['diagnostics']['searches']['long']['trials'])==3
    assert all(np.isfinite(v) for metrics in result['methods'].values() for v in metrics.values())
    assert m.main(['report','--output',str(tmp_path/'study-real'),'--destination',str(tmp_path/'one.md')])==0
    m.report(tmp_path/'study-real',tmp_path/'two.md')
    assert (tmp_path/'one.md').read_bytes()==(tmp_path/'two.md').read_bytes()


def test_docs_only_commit_is_recorded_without_invalidating_execution_identity():
    m=module()
    before=dict(source='s',lock='l',runtime='r',receipt='p',execution='x',commit='first')
    m._same_execution(before,dict(before,commit='docs-only'))
    with pytest.raises(ValueError,match='drift'): m._same_execution(before,dict(before,source='changed'))


def test_resource_projection_decreases_only_for_completed_members(tmp_path,monkeypatch):
    m=module(); policy=spec(tmp_path)['resources']; reserve=policy['disk_reserve_bytes']
    ledger=dict(resource_observations=[],expected_members=[{},{}],members=[])
    snapshots=iter([dict(free_disk_bytes=reserve+100,peak_rss_bytes=1,available_percent=80,swapouts=0),
        dict(free_disk_bytes=reserve+50,peak_rss_bytes=1,available_percent=80,swapouts=0)])
    monkeypatch.setattr(m,'_resource_snapshot',lambda p:next(snapshots))
    guard=m._resource_guard(tmp_path,policy,ledger,100)
    guard(); ledger['members'].append({'status':'completed'}); guard()


def test_published_spec_resolves_exact_capacity_and_distinct_namespaces():
    m=module()
    value=m.load_spec(Path('protocol/diagnostics/2026-10-18-long-probes.json'))
    c=m._config(value,'study')
    from treeneuron.baselines.causal import ordinary_width_brackets
    target=(c.model.d+2)*c.model.h+1+2*c.growth.m_requested*(c.model.d+1)
    lower,upper=ordinary_width_brackets(c.model.d,target)
    assert (target,(c.model.d+2)*lower+1,(c.model.d+2)*upper+1)==(1697,1693,1711)
    assert value['teachers']*value['repetitions']==16
    assert value['short_probe_updates']==320 and value['long_probe_updates']==8000


def test_observation_retention_does_not_access_V_labels(tmp_path):
    m=module()
    from treeneuron.pilot.calibrate import teacher_inputs
    teacher,data,e,_=teacher_inputs(m._config(spec(tmp_path),'cost'),0)
    class Forbidden:
        def __getattribute__(self,name): raise AssertionError('V must remain unused')
    object.__setattr__(data,'V',Forbidden())
    path=m._observations(tmp_path,0,teacher,data,e)
    with np.load(path,allow_pickle=False) as stored:
        assert not any(key.startswith('V_') for key in stored.files)
        np.testing.assert_array_equal(stored['E_y'],e.y)
