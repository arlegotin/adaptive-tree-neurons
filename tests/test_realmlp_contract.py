"""Native RealMLP configuration and role boundaries; no full quality anchors."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.realmlp_native') is not None, \
        'Versioned official RealMLP adapter is missing'
    return importlib.import_module('treeneuron.research.realmlp_native')


def config(mode='native_anchor', **changes):
    value = dict(schema='research/v2-realmlp-config', family='P06', mode=mode,
        feature_schema=dict(columns=['x', 'kind'], categorical=['kind']), parameters={})
    value.update(changes)
    return value


def test_native_recipe_cannot_be_silently_shortened_or_reparameterized():
    module = api()
    cfg = module.RealMLPConfig.parse(config())
    assert cfg.max_epochs == 256
    for changes in (dict(max_epochs=3), dict(parameters={'lr': .2}),
            dict(mode='search', parameters={'opt': 'adamw'}),
            dict(mode='search', parameters={'use_early_stopping': True}),
            dict(mode='engineering_fixture', max_epochs=4)):
        with pytest.raises(ValueError):
            module.RealMLPConfig.parse(config(**changes))


@pytest.mark.parametrize('params', [dict(lr=.001), dict(p_drop=.2), dict(hidden_sizes=[128]*3),
    dict(wd=.01), dict(num_emb_type='ple'), dict(plr_sigma=.001),
    dict(add_front_scale=1), dict(act='gelu'), dict(n_epochs=100)])
def test_unknown_or_out_of_frozen_native_support_settings_reject(params):
    with pytest.raises(ValueError):
        api().RealMLPConfig.parse(config('search', parameters=params))


def test_declared_engineering_mode_and_native_search_support_are_distinct():
    module = api()
    tiny = module.RealMLPConfig.parse(config('engineering_fixture', max_epochs=3))
    assert tiny.max_epochs == 3
    candidate = module.RealMLPConfig.parse(config('search', parameters=dict(hidden_sizes=[64]*5,
        lr=.02, p_drop=.3, wd=0., plr_sigma=.05, num_emb_type='plr', act='relu', add_front_scale=False)))
    assert candidate.to_state()['parameters']['hidden_sizes'] == [64]*5


def test_adapter_module_import_is_lazy():
    api()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONDONTWRITEBYTECODE='1')
    program = ('import sys; import treeneuron.research.realmlp_native; '
        'assert not set(("torch","sklearn","pytabkit","pytorch_lightning","torchmetrics")) & set(sys.modules)')
    result = subprocess.run([sys.executable, '-c', program], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    assert result.returncode == 0, result.stdout


def test_realmlp_snapshot_has_no_unreviewed_pickle_or_state_dict_shortcut(tmp_path):
    module = api()
    assert hasattr(module, 'RealMLPPredictor'), 'Strict native RealMLP predictor is missing'
    with pytest.raises(RuntimeError, match='fitted'):
        module.RealMLPPredictor().save(tmp_path/'snapshot')


def test_bytecode_policy_rejects_before_optional_native_imports():
    module = api()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root/'src'),
        str(root/'environment/research-v2-realmlp-site')])); env.pop('KMP_DUPLICATE_LIB_OK', None)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    code = '''import sys, importlib
from treeneuron.research import realmlp_native, ordinary_neural
def reject(name): raise AssertionError('Unexpected native import: '+name)
importlib.import_module=reject
sys.dont_write_bytecode=False
try: realmlp_native._dependencies()
except ValueError as error: assert 'bytecode' in str(error), str(error)
else: raise AssertionError('Bytecode-enabled worker admitted')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_raw_G_nonfinite_evaluation_rejects_before_native_metric_sanitization(value):
    import torch
    module=api()
    assert hasattr(module,'check_raw_G'), 'Raw G evaluation guard is missing'
    with pytest.raises(FloatingPointError,match='raw G'):
        module.check_raw_G([torch.tensor([[[value]]],dtype=torch.float32)])


def test_portable_loader_requires_trusted_external_manifest_before_import(tmp_path):
    module=api()
    assert hasattr(module.RealMLPPredictor,'load'), 'Closed RealMLP inference loader is missing'
    with pytest.raises(ValueError,match='trusted'):
        module.RealMLPPredictor.load(tmp_path,expected_manifest_sha256=None)


def summary_fixture():
    cfg=api().RealMLPConfig.parse(config('engineering_fixture',max_epochs=3))
    history=[dict(epoch=i,G_mse=1.,G_native_rmse=1.,applied_updates=i,F_online_native_loss=1.) for i in range(1,4)]
    value=dict(schema='research/v2-realmlp-resources',family='P06',config=cfg.to_state(),mode=cfg.mode,
        quality_eligible=False,checkpoint_eligible=False,status='completed_ineligible',fit_rows=2,
        native_epochs_executed=3,accessed_roles=['F','G'],history=history,selected_epoch=3,applied_updates=3,
        admitted_row_ids=[[0,1]]*3,omitted_row_ids=[[]]*3,unvisited_F_row_ids=[],cumulative_visited_rows=2)
    return cfg,value


@pytest.mark.parametrize('change',['negative_mse','nonfinite_rmse','negative_online_loss','first_tie',
    'promoted_quality','short_horizon','wrong_config','row_outside_F','incomplete_exposure'])
def test_incoherent_native_selection_risk_and_exposure_summary_rejects(change):
    cfg,value=summary_fixture()
    if change=='negative_mse':value['history'][0]['G_mse']=-1.
    elif change=='nonfinite_rmse':value['history'][0]['G_native_rmse']=float('nan')
    elif change=='negative_online_loss':value['history'][0]['F_online_native_loss']=-1.
    elif change=='first_tie':value['selected_epoch']=1
    elif change=='promoted_quality':value['quality_eligible']=True
    elif change=='short_horizon':value['native_epochs_executed']=2
    elif change=='wrong_config':value['config']=config('native_anchor')
    elif change=='row_outside_F':value['admitted_row_ids']=[[0,2]]*3
    else:value['unvisited_F_row_ids']=[1]
    with pytest.raises(ValueError):api()._validate_summary(cfg,value)


def test_native_latest_tie_and_nondivisible_exposure_are_accepted_as_declared():
    cfg,value=summary_fixture()
    api()._validate_summary(cfg,value)


@pytest.mark.parametrize('failure',['closed_graph_binding','success_receipt_publication'])
def test_late_failed_fit_never_exposes_new_prediction_or_save_state(failure,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import torch
    from treeneuron.baselines.api import FitRequest
    module=api()
    class State:
        def state_dict(self):return {'value':torch.ones((1,),dtype=torch.float32)}
    interface=SimpleNamespace(model=SimpleNamespace(model=State(),creator=SimpleNamespace(static_model=State())),
        fit_params=[{'stop_epoch':{'rmse':3}}])
    history=summary_fixture()[1]['history']
    observed=SimpleNamespace(history=history,batches=[[0,1]]*3,updates=3,initialization={})
    monkeypatch.setattr(module,'_source',lambda:{})
    monkeypatch.setattr(module,'_dependencies',lambda:{})
    monkeypatch.setattr(module,'native_config',lambda *args:{})
    monkeypatch.setattr(module,'prepare_native_data',lambda *args:dict(
        split=SimpleNamespace(n_train=2,n_val=1),native_subseed=1,cardinalities=[],converter=object(),
        dataset=SimpleNamespace(tensors={'x_cont':torch.arange(3,dtype=torch.float32)[:,None]})))
    monkeypatch.setattr(module,'_fit_native',lambda *args:(interface,observed))
    monkeypatch.setattr(module,'_runtime',lambda:{})
    monkeypatch.setattr(module,'_graph_inspection',lambda *args:[])
    model=module.RealMLPPredictor()
    if failure=='closed_graph_binding':
        def reject(**unused):raise ValueError('injected closed graph binding failure')
        monkeypatch.setattr(model,'_snapshot_parts',reject)
        expected=ValueError
    else:
        monkeypatch.setattr(model,'_snapshot_parts',lambda **unused:({},{}))
        original=module._write
        def writer(path,value):
            if path.name=='attempt.json' and value.get('status')=='completed_ineligible':
                raise OSError('injected success receipt publication failure')
            return original(path,value)
        monkeypatch.setattr(module,'_write',writer)
        expected=OSError
    request=FitRequest(np.zeros((2,2)),np.zeros(2),np.zeros((1,2)),np.zeros(1),17,
        config('engineering_fixture',max_epochs=3),{},tmp_path/'fit')
    with pytest.raises(expected,match='injected'):model.fit(request)
    assert model.resource_summary()['checkpoint_eligible'] is False
    assert model._interface is None, 'Failed new fit leaked a usable native interface'
    with pytest.raises(RuntimeError,match='fitted'):model.predict(np.zeros((1,2)))
    with pytest.raises(RuntimeError,match='fitted'):model.save(tmp_path/'snapshot')
    with pytest.raises(RuntimeError,match='fresh'):model.fit(request)
    assert (tmp_path/'fit/attempt.json').exists()


def test_rejected_refit_preserves_prior_successful_prediction_state():
    module=api();model=module.RealMLPPredictor();prior=object()
    model._interface=prior;model._summary={'retained':'prior successful state'}
    with pytest.raises(RuntimeError,match='fresh'):model.fit(None)
    assert model._interface is prior and model.resource_summary()=={'retained':'prior successful state'}
