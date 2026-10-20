"""Inspect actual native procedures without inventing unexposed observations."""
import numpy as np
import pytest

from native_workload_cases import actual_native_family


@pytest.mark.parametrize('family',['P06','P08','P09','P10','P11'])
def test_actual_full_native_external_workload_has_method_specific_exposure_and_selection(tmp_path,monkeypatch,family):
    from treeneuron.research.native_workload_v2 import inspect_native_workload
    args,attempt=actual_native_family(tmp_path,family)
    if family=='P06':
        from treeneuron.research.realmlp_native import RealMLPPredictor as predictor
    else:
        from treeneuron.research.native_trees import NativeTreePredictor as predictor
    def forbidden(*a,**kw):raise AssertionError('Recorded workload observation cannot fit or make new predictions')
    monkeypatch.setattr(predictor,'fit',forbidden);monkeypatch.setattr(predictor,'predict',forbidden)
    result=inspect_native_workload(**args);training=result['training']
    with np.load(args['request_path'].parent/'F.npz',allow_pickle=False) as saved:nF=len(saved['y'])
    assert nF==325 and training['F_context_rows']==nF
    if family=='P06':
        assert training['procedure']=='native_realmlp_fixed_epochs'
        assert training['executed_epochs']==256 and training['rows_per_epoch']==256
        assert training['omitted_rows_per_epoch']==69 and training['cumulative_visited_F_rows']==nF
        assert training['native_drop_last_preserved'] is True
        assert training['gradient_norm_trace_checked'] is False
    else:
        assert training['procedure']=='native_boosting'
        assert 1<=training['selected_rounds']<=training['executed_rounds']
        assert training['gradient_trace']=='not_exposed_by_native_adapter'
        assert training['training_row_visit_trace']=='not_exposed_by_native_adapter'
    assert result['native_workload_qualified'] is result['quality_eligible'] is result['T_accessed'] is False


@pytest.mark.parametrize('family,fault',[('P06','omitted_row'),('P06','initialization_root'),
    ('P11','input_features'),('P09','stopping_history')])
def test_self_recommitted_external_content_cannot_supply_native_workload_evidence(tmp_path,family,fault):
    import json,hashlib
    from treeneuron.research.native_generation_v2 import audit_native_generation
    from treeneuron.research.native_workload_v2 import inspect_native_workload
    from treeneuron.research.native_trees import _canonical
    from test_native_workload_observation import recommit_closed_records
    args,attempt=actual_native_family(tmp_path,family);root=attempt/'native/worker'
    path=root/'snapshot'/('manifest.json' if family=='P06' else 'state.json')
    envelope=json.loads(path.read_text());state=envelope if family=='P06' else envelope['payload'];summary=state['resources']
    if fault=='omitted_row':summary['omitted_row_ids'][0][0]=summary['admitted_row_ids'][0][0]
    elif fault=='initialization_root':summary['initialization_observation']['root_F_positions'][0]=1
    elif fault=='stopping_history':summary['G_rmse_history'][0]=-1.
    else:
        inputs=root/'fit/manifest.json';value=json.loads(inputs.read_text());value['F_features_sha256']='0'*64
        inputs.write_text(json.dumps(value))
    if family!='P06':envelope['sha256']=_canonical(state)
    path.write_text(json.dumps(envelope));(root/'fit/attempt.json').write_text(json.dumps(summary))
    terminal=json.loads((root/'terminal.json').read_text());terminal['adapter_resources']=summary
    (root/'terminal.json').write_text(json.dumps(terminal));recommit_closed_records(args,attempt)
    assert audit_native_generation(**args)['model_generation_chain_authenticated'] is True
    with pytest.raises(ValueError,match='exposure|initialization|input|history'):
        inspect_native_workload(**args)
