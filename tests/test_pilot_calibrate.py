"""Bounded fixtures for coordinator boundaries; not a scientific pilot."""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from treeneuron.config import resolve_experiment, load_experiment


def config(tmp_path, **pilot):
    return resolve_experiment(dict(study_id='engineering-pilot',model=dict(d=2,h=2),
        data=dict(n_fit=8,n_growth=4,n_validation=4),
        training=dict(base_updates=1,continuation_updates=1,validation_interval=1,patience=1,batch_size=2),
        backend=dict(cpu_threads=1,interop_threads=1),
        execution=dict(artifacts_root=str(tmp_path)),
        pilot=dict(purpose='engineering',initial_teachers=2,max_teachers=4,repetitions=2,
            evaluation_rows=5,bootstrap_iterations=200,bootstrap_seed=123,
            endpoint_tolerance=.1,probe_updates=1,max_seconds=30.,**pilot)))


def test_pilot_schema_strict_and_no_scientific_execution_before_gates(tmp_path):
    from treeneuron.pilot.calibrate import calibrate
    c=config(tmp_path)
    assert c.pilot.repetitions==2
    for value in ({'unknown':1},{'repetitions':True},{'max_teachers':3},{'purpose':'scientific'}):
        d=c.to_dict(); d['pilot'].update(value)
        with pytest.raises(ValueError): resolve_experiment(d)
    d=c.to_dict(); d['data']['n_test']=2
    with pytest.raises(ValueError,match='protected'): resolve_experiment(d)
    path=tmp_path/'duplicate.yaml'; path.write_text('study_id: a\npilot:\n  repetitions: 2\n  repetitions: 3\n')
    with pytest.raises(ValueError,match='duplicate'): load_experiment(path)


def test_teacher_data_fixed_reps_fresh_E_and_no_confirmation_namespace(tmp_path):
    from treeneuron.pilot.calibrate import teacher_inputs, repetition_config
    c=config(tmp_path)
    teacher,data,e,tc=teacher_inputs(c,0)
    _,other,other_e,_=teacher_inputs(c,1)
    assert data.teacher_id!=other.teacher_id
    assert not set(e.row_id)&set(data.manifest.row_id)
    assert not set(e.row_id)&set(other_e.row_id)
    a,b=[repetition_config(tc,j) for j in (0,1)]
    assert a.seeds.teacher==b.seeds.teacher and a.seeds.data==b.seeds.data
    assert a.seeds.optimizer!=b.seeds.optimizer and a.seeds.allocation!=b.seeds.allocation
    assert a.seeds.namespace=='exploratory' and a.data.n_test is None
    assert a.training==b.training


def test_coordinator_doubling_ledger_and_every_call_fresh_attempt(tmp_path,monkeypatch):
    module=importlib.import_module('treeneuron.pilot.calibrate')
    calls=[]
    def run(c,data,e,root,recipe_hash):
        calls.append((data.teacher_id,data.manifest.assignment_hash,tuple(e.row_id),c.seeds.optimizer))
        return dict(teacher_id=data.teacher_id,repetition=0,data_hash=data.manifest.assignment_hash,
            evaluation_hash=str(tuple(e.row_id)),recipe_hash=recipe_hash,
            h1_rows=[0.,.01,.02,.03,.04],static_rows=[0.,.01,.02,.03,.04],
            controls=['B1','B2','B3','B4','B5','B6'],status='completed',enclosing_ns=100)
    monkeypatch.setattr(module,'run_repetition',run)
    monkeypatch.setattr(module,'_runtime_preflight',lambda c: {'fixture':'explicitly mocked'})
    c=config(tmp_path)
    first=module.calibrate(c); second=module.calibrate(c)
    assert first.independent_teachers==second.independent_teachers==4
    assert len(calls)==16
    assert calls[0][:3]==calls[1][:3] and calls[0][3]!=calls[1][3]
    assert calls[0][0]!=calls[2][0]
    attempts=list(tmp_path.glob('pilot-*'))
    assert len(attempts)==2
    for attempt in attempts:
        raw=json.loads((attempt/'raw.json').read_text())
        assert raw['batch_completed_counts']==[2,4]
        assert raw['status']=='completed' and raw['purpose']=='engineering'
        decision=json.loads((attempt/'design_decision.json').read_text())
        assert decision['branch']=='blocked' and not decision['freeze_eligible']
        assert (attempt/'design_decision.md').is_file()


def test_failed_attempt_retained_and_not_relabelled_as_complete(tmp_path,monkeypatch):
    module=importlib.import_module('treeneuron.pilot.calibrate')
    monkeypatch.setattr(module,'_runtime_preflight',lambda c: {})
    def fail(*args): raise RuntimeError('fixture failure')
    monkeypatch.setattr(module,'run_repetition',fail)
    with pytest.raises(RuntimeError,match='fixture failure'): module.calibrate(config(tmp_path))
    raw=json.loads(next(tmp_path.glob('pilot-*/raw.json')).read_text())
    assert raw['status']=='failed' and raw['attempts'][0]['status']=='failed'
    assert raw['enclosing_ns']>0 and raw['attempts'][0]['enclosing_ns']>0


def test_cli_calibrate_dispatches_exact_config(tmp_path,monkeypatch,capsys):
    from treeneuron.cli import main
    module=importlib.import_module('treeneuron.pilot.calibrate')
    from treeneuron.pilot.design import summarize_pilot
    from test_design import records
    c=config(tmp_path); path=tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(c.to_dict()))
    seen=[]
    def fake(value):
        seen.append(value)
        return summarize_pilot(records(),batch_runtimes=(1.,1.),bootstrap_iterations=200,seed=1)
    monkeypatch.setattr(module,'calibrate',fake)
    assert main(['calibrate','--config',str(path)])==0
    assert seen==[c] and json.loads(capsys.readouterr().out)['purpose']=='engineering'

from test_train import cpu_training_evidence


@pytest.mark.parametrize('growth_count',[0,1])
def test_real_minimal_pair_controls_lineage_and_E_losses(tmp_path,cpu_training_evidence,monkeypatch,growth_count):
    import torch
    from treeneuron import train
    from treeneuron.pilot.calibrate import teacher_inputs,repetition_config,run_repetition
    from treeneuron.baselines.causal import CausalPredictor
    c=config(tmp_path); raw=c.to_dict()
    raw['execution']['backend_report']=str(cpu_training_evidence)
    raw['growth']['m_requested']=growth_count
    raw['backend']['interop_threads']=torch.get_num_interop_threads()
    c=resolve_experiment(raw)
    teacher,data,e,tc=teacher_inputs(c,0)
    tc=repetition_config(tc,0)
    monkeypatch.setattr(train,'BACKEND_REPORT_PATH',cpu_training_evidence)
    original=CausalPredictor.predict
    def predict(self,X,**kw):
        assert not np.array_equal(X,data.V.X), 'V must not be scored by pilot paired driver'
        return original(self,X,**kw)
    monkeypatch.setattr(CausalPredictor,'predict',predict)
    result=run_repetition(tc,data,e,tmp_path/'pair','common')
    assert result['status']=='completed'
    assert set(result['methods'])=={'keep','top','random','ordinary_lower','ordinary_upper','widen_upper','static_partial','greedy'}
    assert result['pairing_audit']['passed']
    assert result['methods']['top']['parent_checkpoint_sha256']==result['methods']['random']['parent_checkpoint_sha256']
    assert result['methods']['top']['allocation']['effective_count']==result['methods']['random']['allocation']['effective_count']
    preds=np.load(tmp_path/'pair'/'E-predictions.npz')
    expected=((preds['random']-e.y)**2-(preds['top']-e.y)**2)/np.var(data.F.y)
    assert result['h1_rows']==pytest.approx(expected)
    if growth_count==0: assert result['h1_rows']==[0.]*len(e.y)
    fallback=((preds['ordinary_upper']-e.y)**2-(preds['static_partial']-e.y)**2)/np.var(data.F.y)
    assert result['static_rows']==pytest.approx(fallback)
    assert result['enclosing_ns']>result['costs']['probe_ns']>0
    assert result['costs']['aggregation']=='enclosing only; all component clocks are nested'
    assert json.loads(Path(result['greedy_report']).read_text())['complete']
    assert result['source_sha256']==train.source_hash()
    assert not result['V_accessed'] and not result['T_accessed']


def test_regeneration_rejects_source_drift_and_raw_measurement_tampering(tmp_path,monkeypatch):
    module=importlib.import_module('treeneuron.pilot.calibrate')
    monkeypatch.setattr(module,'_runtime_preflight',lambda c: {})
    def run(c,data,e,root,recipe_hash):
        return dict(teacher_id=data.teacher_id,repetition=0,data_hash=data.manifest.assignment_hash,
            evaluation_hash=str(tuple(e.row_id)),recipe_hash=recipe_hash,h1_rows=[0.,1.,2.,3.,4.],
            static_rows=[0.,1.,2.,3.,4.],controls=['B1','B2','B3','B4','B5','B6'],status='completed')
    monkeypatch.setattr(module,'run_repetition',run)
    original=module.calibrate(config(tmp_path))
    directory=Path(original.diagnostics['attempt_directory'])
    assert module.regenerate_design(directory)==original
    raw=json.loads((directory/'raw.json').read_text())
    raw['accepted_records'][0]['h1_rows'][0]=99.
    (directory/'raw.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='raw.*hash'): module.regenerate_design(directory)
    # An independently retained hash protects raw bytes; old-source reinterpretation
    # must be explicitly versioned rather than silently rewriting the design.
    monkeypatch.setattr(module.train,'source_hash',lambda:'0'*64)
    with pytest.raises(ValueError,match='source'): module.regenerate_design(directory)


def test_pair_publication_failure_restores_global_receipt(tmp_path,cpu_training_evidence,monkeypatch):
    module=importlib.import_module('treeneuron.pilot.calibrate')
    from treeneuron import train
    import torch
    c=config(tmp_path); raw=c.to_dict()
    raw['backend']['interop_threads']=torch.get_num_interop_threads()
    raw['execution']['backend_report']=str(cpu_training_evidence)
    c=resolve_experiment(raw)
    _,data,e,tc=module.teacher_inputs(c,0)
    original=train.BACKEND_REPORT_PATH
    def fail(*args,**kw): raise OSError('publication fixture')
    monkeypatch.setattr(module,'_atomic_json',fail)
    with pytest.raises(OSError,match='publication fixture'):
        module.run_repetition(module.repetition_config(tc,0),data,e,tmp_path/'failed-pair','recipe')
    assert train.BACKEND_REPORT_PATH==original
