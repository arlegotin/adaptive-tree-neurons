"""Bounded preparation/training fixtures; no genuine confirmation evidence."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from test_protocol import repository, prepared, candidate, write, ref
from test_train import cpu_training_evidence


def test_seed_mapping_uses_committed_optimizer_and_distinct_allocation_streams(tmp_path):
    from test_pilot_calibrate import config
    from treeneuron.protocol import confirmation_seeds
    from treeneuron.evaluation.confirm import repetition_config
    seeds=confirmation_seeds('toy','fixture',7,0,2)
    a,b=[repetition_config(config(tmp_path),seeds,j) for j in range(2)]
    assert a.seeds.optimizer==seeds['optimizers'][0] % 2**32
    assert b.seeds.optimizer==seeds['optimizers'][1] % 2**32
    assert a.seeds.allocation!=b.seeds.allocation and a.seeds.contrast!=b.seeds.contrast
    assert a.seeds.allocation!=a.seeds.contrast
    assert a.seeds.teacher==b.seeds.teacher==seeds['teacher']
    assert a.data.n_test is None and a.seeds.namespace=='exploratory'


def test_prepare_creates_only_development_and_committed_T_ids(prepared,monkeypatch):
    from treeneuron.evaluation.confirm import prepare_cohort
    from treeneuron.evaluation import protected
    from treeneuron.protocol import eligible_design, _scientific_sources
    root,config,sources=prepared
    def forbidden(*a,**kw): raise AssertionError('T generation before freeze')
    monkeypatch.setattr(protected,'_rng',forbidden)
    monkeypatch.setattr(protected,'_test_labels',forbidden)
    decision=eligible_design(sources)
    result=prepare_cohort(decision,sources,root/'prepared',cohort_id='fixture',root_seed=72)
    assert len(result['cohort']['units'])==decision.K
    assert sources['cohort'] is None
    _scientific_sources(result,decision)
    receipt=json.loads((root/result['cohort']['preparation']['path']).read_text())
    assert receipt['seconds']>0 and receipt['retained_bytes']>0
    for unit in result['cohort']['units']:
        directory=(root/unit['teacher']['path']).parent
        arrays=np.load(directory/'development.npz')
        assert set(arrays.files)=={f'{part}_{name}' for part in 'FGV' for name in ('X','y','row_id','group_id')}
        assert len(unit['run_ids'])==decision.repetitions
        assert unit['manifest']['assignments'].count('T')==decision.n_T
    with pytest.raises(FileExistsError):
        prepare_cohort(decision,sources,root/'prepared',cohort_id='fixture',root_seed=72)


@pytest.mark.parametrize('growth_count',[0,1])
def test_label_free_fitting_saves_reloadable_all_eight_models(tmp_path,cpu_training_evidence,monkeypatch,growth_count):
    import torch
    from test_pilot_calibrate import config
    from treeneuron.config import resolve_experiment
    from treeneuron.pilot.calibrate import teacher_inputs, fit_repetition
    from treeneuron.evaluation.confirm import load_predictors, audit_models
    c=config(tmp_path); value=c.to_dict()
    value['execution']['backend_report']=str(cpu_training_evidence)
    value['backend']['interop_threads']=torch.get_num_interop_threads()
    value['growth']['m_requested']=growth_count
    c=resolve_experiment(value)
    _,data,e,tc=teacher_inputs(c,0)
    root=tmp_path/'fit'
    report=fit_repetition(tc,data,root,'fixture')
    assert report['status']=='completed' and report['pairing_audit']['passed']
    assert not list(root.glob('*predictions*'))
    audit_models(tmp_path,report,root)
    import copy
    swapped=copy.deepcopy(report); swapped['models']['top']=report['models']['ordinary_upper']
    with pytest.raises(ValueError,match='model|checkpoint'): audit_models(tmp_path,swapped,root)
    callbacks=load_predictors(tmp_path,dict(reversed(list(report['models'].items()))))
    from treeneuron.protocol import EVALUATION_METHODS
    assert tuple(callbacks)==EVALUATION_METHODS
    assert len(callbacks)==8
    values={k:f(e.X) for k,f in callbacks.items()}
    assert all(v.shape==(5,) for v in values.values())
    if growth_count==0: np.testing.assert_array_equal(values['top'],values['random'])
    from treeneuron.protocol import _validate_pilot_graph
    _validate_pilot_graph(report,root,tc)


@pytest.fixture
def mock_cohort(candidate,monkeypatch):
    from treeneuron.protocol import freeze, EVALUATION_METHODS
    module=importlib.import_module('treeneuron.evaluation.confirm')
    root,decision,sources,manifests=candidate
    frozen=freeze(decision,sources,root/'protocol/frozen.yaml')
    # The original freeze is real. Cache the invariant expensive pilot/source
    # validation only here; other tests exercise mutation rejection fully.
    from treeneuron.evaluation import protected
    payload=json.loads(frozen.path.read_text())['payload']
    byrun={rid:u for u in payload['sources']['cohort']['units'] for rid in u['run_ids']}
    def validate(p,manifest,rid):
        assert p==frozen and json.loads(json.dumps(manifest.to_dict()))==byrun[rid]['manifest']
        return root,payload,byrun[rid],frozen
    monkeypatch.setattr(protected,'_validate',validate)
    model=write(root,'evidence/toy-model.json',{'fixture':True})
    monkeypatch.setattr(module,'_runtime_preflight',lambda c: {})
    calls=[]
    def fit(protocol,payload,unit,repetition,directory):
        calls.append((unit['unit_id'],repetition))
        directory.mkdir(parents=True)
        report=dict(status='completed',models={k:ref(root,model) for k in EVALUATION_METHODS},capacity={'effective_count':0})
        write(directory,'pair.json',report)
        return report
    monkeypatch.setattr(module,'_fit_run',fit)
    monkeypatch.setattr(module,'load_predictors',lambda root,models:
        {k:(lambda X: np.zeros(len(X))) for k in EVALUATION_METHODS})
    return module,root,frozen,decision,calls,fit


def test_fixed_cohort_completes_zero_growth_all_units_and_resume_once(mock_cohort,monkeypatch):
    module,root,frozen,decision,calls,fit=mock_cohort
    directory=module.run_confirmation(frozen,'synthetic')
    ledger=json.loads((directory/'confirmation.json').read_text())
    assert ledger['status']=='completed'
    assert len(calls)==decision.K*decision.repetitions
    assert len(ledger['runs'])==len(calls)
    assert all(r['status']=='completed' for r in ledger['runs'])
    assert module.run_confirmation(frozen,'synthetic')==directory
    assert len(calls)==decision.K*decision.repetitions
    assert ledger['costs']['seconds']>=0 and ledger['costs']['forecast_seconds']>0
    assert ledger['costs']['preparation_seconds'] is None
    assert ledger['costs']['combined_seconds'] is None
    ledger['status']='running'; ledger['runs'][0]['status']='pending'
    ledger['runs'][0]['attempts'][-1].update(status='running',seconds=None)
    (directory/'confirmation.json').write_text(json.dumps(ledger))
    module.run_confirmation(frozen,'synthetic')
    recovered=json.loads((directory/'confirmation.json').read_text())
    assert recovered['status']=='completed' and recovered['costs']['hard_interrupt_cost_unknown']
    assert recovered['costs']['combined_seconds'] is None
    assert len(calls)==decision.K*decision.repetitions


def test_failed_unit_retries_finite_same_seed_and_rest_of_cohort_runs(mock_cohort,monkeypatch):
    module,root,frozen,decision,calls,fit=mock_cohort
    def fail(protocol,payload,unit,repetition,directory):
        if unit['unit_id']=='unit-0':
            calls.append((unit['unit_id'],repetition)); raise RuntimeError('fixture exhausted')
        return fit(protocol,payload,unit,repetition,directory)
    monkeypatch.setattr(module,'_fit_run',fail)
    directory=module.run_confirmation(frozen,'synthetic')
    ledger=json.loads((directory/'confirmation.json').read_text())
    assert ledger['status']=='incomplete'
    assert len(ledger['runs'])==decision.K*decision.repetitions
    failed=[r for r in ledger['runs'] if r['status']=='failed']
    assert len(failed)==decision.repetitions
    assert all(len(r['attempts'])==decision.statistical_policy['retries']+1 for r in failed)
    assert all('fixture exhausted' in a['error'] for r in failed for a in r['attempts'])
    module.run_confirmation(frozen,'synthetic')
    assert len(calls)==decision.K*decision.repetitions+decision.repetitions*decision.statistical_policy['retries']


def test_interrupt_resumes_declared_unit_without_replacing_teacher(mock_cohort,monkeypatch):
    module,root,frozen,decision,calls,fit=mock_cohort
    first=True
    def interrupt(*args):
        nonlocal first
        if first:
            first=False
            raise KeyboardInterrupt()
        return fit(*args)
    monkeypatch.setattr(module,'_fit_run',interrupt)
    with pytest.raises(KeyboardInterrupt): module.run_confirmation(frozen,'synthetic')
    directory=module.run_confirmation(frozen,'synthetic')
    ledger=json.loads((directory/'confirmation.json').read_text())
    assert ledger['status']=='completed' and len(calls)==decision.K*decision.repetitions
    assert [a['status'] for a in ledger['runs'][0]['attempts']]==['interrupted','completed']


def test_cli_confirmation_and_preparation_expose_distinct_gates(monkeypatch,tmp_path,capsys):
    from treeneuron.cli import main
    from treeneuron import protocol
    module=importlib.import_module('treeneuron.evaluation.confirm')
    frozen_path=write(tmp_path,'frozen.json',{'payload':{'sources':{'root':str(tmp_path)}}})
    frozen=protocol.FrozenProtocol(frozen_path,'hash','pin','checkout')
    monkeypatch.setattr(protocol,'validate_frozen',lambda path,current: frozen)
    seen=[]
    monkeypatch.setattr(module,'run_confirmation',lambda p,s: seen.append((p,s)) or tmp_path)
    assert main(['confirm','--protocol',str(frozen_path),'--study','synthetic'])==0
    assert seen==[(frozen,'synthetic')]
    monkeypatch.setattr(protocol,'load_design',lambda path: ('numeric',{'source':'fixture'}))
    monkeypatch.setattr(module,'prepare_cohort',lambda d,s,o,**kw: seen.append((d,s,o,kw)) or {'prepared':True})
    monkeypatch.setattr(protocol,'publish_design',lambda d,s,p: seen.append((d,s,p)))
    assert main(['prepare-cohort','--design','design.md','--output',str(tmp_path/'cohort'),
        '--cohort-id','fixture','--root-seed','7','--publish',str(tmp_path/'fresh.md')])==0
    assert seen[-1]==('numeric',{'prepared':True},tmp_path/'fresh.md')


def test_real_prepared_fits_protected_rows_and_saved_analysis(prepared,cpu_training_evidence,monkeypatch):
    """One real toy teacher, two repetitions; other declared units retained failed.

    Only the numerical receipt path/thread fixture is substituted for pytest's
    runtime; genuine CPU1/1 receipts and scientific runs are not produced here.
    """
    import torch
    from treeneuron.protocol import eligible_design, freeze
    from treeneuron.analysis.primary import analyze_primary
    from treeneuron.config import resolve_experiment
    from treeneuron.evaluation import protected
    module=importlib.import_module('treeneuron.evaluation.confirm')
    root,config,sources=prepared; decision=eligible_design(sources)
    sources=module.prepare_cohort(decision,sources,root/'real-toy',cohort_id='tiny-real-path',root_seed=17)
    frozen=freeze(decision,sources,root/'protocol/frozen.yaml')
    original_resolve=module.resolve_experiment
    def fixture_runtime(value):
        value=dict(value); value['execution']=dict(value['execution'],backend_report=str(cpu_training_evidence))
        value['backend']=dict(value['backend'],interop_threads=torch.get_num_interop_threads())
        return original_resolve(value)
    monkeypatch.setattr(module,'resolve_experiment',fixture_runtime)
    monkeypatch.setattr(module,'_runtime_preflight',lambda c: {})
    original_fit=module._fit_run
    def bounded_fit(p,payload,unit,rep,directory):
        if unit['unit_id']!='unit-0': raise RuntimeError('Declared toy fixture omitted numerical work')
        return original_fit(p,payload,unit,rep,directory)
    monkeypatch.setattr(module,'_fit_run',bounded_fit)
    directory=module.run_confirmation(frozen,'synthetic')
    result=analyze_primary(directory,frozen)
    assert result['status']=='incomplete' and result['valid_runs']==2
    assert result['costs']['preparation_seconds']>0
    assert result['costs']['combined_seconds']>=result['costs']['seconds']
    assert result['observed_teachers']==decision.K and result['valid_teachers']==1
    teacher=result['teachers'][0]
    assert len(teacher['regimes'])==8 and teacher['repetitions']==2
    values=[]
    for rid in sources['cohort']['units'][0]['run_ids']:
        saved=protected.load_evaluation(frozen,run_id=rid)
        y=np.asarray(saved['observations']['y']); p=saved['predictions']
        scale=np.var(np.load(root/sources['cohort']['units'][0]['fit_targets']['path']))
        values.append(np.mean((np.asarray(p['random'])-y)**2-(np.asarray(p['top'])-y)**2)/scale)
    assert teacher['D']==pytest.approx(np.mean(values))
    assert not result['primary']['superiority']
    assert analyze_primary(directory,frozen)['report_sha256']==result['report_sha256']
    # A self-consistent local ledger edit must not detach fitting from the
    # committed optimizer stream, even when saved predictions remain sealed.
    ledger=json.loads((directory/'confirmation.json').read_text())
    attempt=ledger['runs'][0]['attempts'][-1]
    pair_path=root/attempt['pair']['path']; pair=json.loads(pair_path.read_text())
    pair['confirmation']['effective_seeds']['optimizer']+=1
    pair_path.write_text(json.dumps(pair)); attempt['pair']=ref(root,pair_path)
    (directory/'confirmation.json').write_text(json.dumps(ledger))
    invalid=analyze_primary(directory,frozen)
    assert invalid['valid_runs']==1 and invalid['valid_teachers']==0


def test_prediction_retry_keeps_registered_models_without_refitting(mock_cohort,monkeypatch):
    module,root,frozen,decision,calls,fit=mock_cohort
    original=module.load_predictors
    first=True
    def predictors(root,models):
        nonlocal first
        callbacks=original(root,models)
        if first:
            first=False
            def fail(X): raise RuntimeError('transient prediction fixture')
            callbacks['top']=fail
        return callbacks
    monkeypatch.setattr(module,'load_predictors',predictors)
    directory=module.run_confirmation(frozen,'synthetic')
    ledger=json.loads((directory/'confirmation.json').read_text())
    assert ledger['status']=='completed'
    assert len(calls)==decision.K*decision.repetitions
    assert [a['status'] for a in ledger['runs'][0]['attempts']]==['failed','completed']


def test_sealed_evidence_failure_does_not_retry_immutable_evidence(mock_cohort,monkeypatch):
    from treeneuron.evaluation import protected
    module,root,frozen,decision,calls,fit=mock_cohort
    directory=module.run_confirmation(frozen,'synthetic')
    ledger=json.loads((directory/'confirmation.json').read_text())
    ledger['status']='running'; ledger['runs'][0]['status']='pending'
    (directory/'confirmation.json').write_text(json.dumps(ledger))
    original=protected.load_evaluation; reads=[]
    def fail_first(protocol,*,run_id):
        reads.append(run_id)
        if len(reads)==1: raise ValueError('sealed evidence drift fixture')
        return original(protocol,run_id=run_id)
    monkeypatch.setattr(protected,'load_evaluation',fail_first)
    module.run_confirmation(frozen,'synthetic')
    result=json.loads((directory/'confirmation.json').read_text())
    assert result['status']=='incomplete' and len(reads)==1
    assert len(calls)==decision.K*decision.repetitions
