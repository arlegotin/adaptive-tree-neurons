"""Actual complete native epoch procedures retain every admitted F row."""
import numpy as np
import pytest

from native_workload_cases import actual_native_family


@pytest.mark.parametrize('family,variant',[('P03',None),('P04',None),('P07','plain'),('P07','piecewise')])
def test_actual_native_epoch_workload_checks_complete_orders_gradients_and_native_selection_without_refitting(tmp_path,monkeypatch,family,variant):
    from treeneuron.research.native_workload_v2 import inspect_native_workload
    args,attempt=actual_native_family(tmp_path,family,variant=variant)
    if family=='P07':
        from treeneuron.research import tabm_native as native
        predictor=native.TabMPredictor;transform=native.TabMTransform
    else:
        from treeneuron.research import ordinary_neural as native
        predictor=native.OrdinaryNeuralPredictor;transform=native.NeuralTransform
    def forbidden(*a,**kw):raise AssertionError('Workload reader must not fit, predict or refit numerical transforms')
    monkeypatch.setattr(predictor,'fit',forbidden);monkeypatch.setattr(predictor,'predict',forbidden)
    monkeypatch.setattr(transform,'fit',forbidden)
    result=inspect_native_workload(**args)
    with np.load(args['request_path'].parent/'F.npz',allow_pickle=False) as saved:nF=len(saved['y'])
    training=result['training']
    assert training['procedure']=='native_epoch_adamw' and training['F_context_rows']==nF
    assert training['full_F_coverage_each_epoch_verified'] is True
    assert training['finite_recorded_gradients_checked'] is True
    origin=0 if family=='P07' else 1
    assert training['executed_epochs']>=1
    assert origin<=training['selected_epoch']<training['executed_epochs']+origin
    assert training['native_epoch_index_origin']==origin
    assert training['training_objective']==('mean_member_mse' if family=='P07' else 'mse')
    assert result['native_workload_qualified'] is result['quality_eligible'] is result['T_accessed'] is False


@pytest.mark.parametrize('family,fault',[('P03','duplicate_row'),('P03','negative_gradient'),
    ('P07','boolean_epoch'),('P07','false_selected_epoch')])
def test_self_recommitted_epoch_content_must_match_native_rows_gradients_and_selection(tmp_path,family,fault):
    import hashlib,json
    from treeneuron.research.native_generation_v2 import audit_native_generation
    from treeneuron.research.native_workload_v2 import inspect_native_workload
    from test_native_workload_observation import recommit_closed_records
    args,attempt=actual_native_family(tmp_path,family);root=attempt/'native/worker'
    path=root/'snapshot/manifest.json';state=json.loads(path.read_text());summary=state['resources']
    if fault=='duplicate_row':
        orders=root/'fit/F-row-orders.npz'
        with np.load(orders,allow_pickle=False) as saved:rows=saved['row_indices'].copy()
        rows[0,0]=rows[0,1];np.savez(orders,row_indices=rows)
        summary['row_orders_sha256']=hashlib.sha256(orders.read_bytes()).hexdigest()
        summary['history'][0]['row_order_sha256']=hashlib.sha256(rows[0].tobytes()).hexdigest()
    elif fault=='negative_gradient':summary['history'][0]['gradient_norm_min']=-1.
    elif fault=='boolean_epoch':summary['history'][0]['epoch']=False
    else:summary['selected_epoch']=summary['executed_epochs']+1
    path.write_text(json.dumps(state));(root/'fit/attempt.json').write_text(json.dumps(summary))
    terminal=json.loads((root/'terminal.json').read_text());terminal['adapter_resources']=summary
    (root/'terminal.json').write_text(json.dumps(terminal));recommit_closed_records(args,attempt)
    assert audit_native_generation(**args)['model_generation_chain_authenticated'] is True
    with pytest.raises(ValueError,match='epoch|gradient|permutation|stopping|selected'):
        inspect_native_workload(**args)
