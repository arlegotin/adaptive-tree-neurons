"""Full native two-stage records must agree with the actual admitted rows."""
import numpy as np
import pytest

from native_workload_cases import actual_native_family


@pytest.mark.parametrize('family',['P00','P01','P02','P05'])
def test_actual_two_stage_native_observation_checks_rows_gradients_stopping_and_geometry(tmp_path,family):
    from treeneuron.research.native_workload_v2 import inspect_native_workload
    args,attempt=actual_native_family(tmp_path,family)
    result=inspect_native_workload(**args)
    with np.load(args['request_path'].parent/'F.npz',allow_pickle=False) as saved:nF=len(saved['y'])
    assert nF>128, 'Exercise a partial final batch and multiple sampler epochs'
    training=result['training']
    assert training['procedure']=='two_stage_adamw' and training['F_context_rows']==nF
    assert training['finite_recorded_gradients_checked'] is True
    assert len(training['stages'])==2
    assert all(s['visited_F_rows']==nF and s['executed_updates']>0 for s in training['stages'])
    assert result['family_id']==family
    assert result['native_workload_qualified'] is result['quality_eligible'] is result['T_accessed'] is False


@pytest.mark.parametrize('fault',['duplicate_row','false_G_best','negative_gradient','wrong_tensor_dtype','boolean_initial_update'])
def test_self_recommitted_inconsistent_gated_training_cannot_supply_a_workload_observation(tmp_path,fault):
    import hashlib,json
    from treeneuron.research.gated_config import GatedConfig
    from treeneuron.research.gated_native import _prediction_binding
    from treeneuron.research.native_generation_v2 import audit_native_generation
    from treeneuron.research.native_workload_v2 import inspect_native_workload
    from treeneuron.research.p13_simple import _array_identity,_hash
    from test_native_workload_observation import recommit_closed_records
    args,attempt=actual_native_family(tmp_path,'P00');root=attempt/'native/worker'
    path=root/'snapshot/manifest.json';state=json.loads(path.read_text());summary=state['resources']
    if fault=='wrong_tensor_dtype':
        with np.load(root/'snapshot/weights.npz',allow_pickle=False) as saved:values={k:saved[k].copy() for k in saved.files}
        values['head.bias']=values['head.bias'].astype(np.float64)
        np.savez(root/'snapshot/weights.npz',**values);np.savez(root/'fit/selected-state.npz',**values)
        state['weights_sha256']=hashlib.sha256((root/'snapshot/weights.npz').read_bytes()).hexdigest()
        state['payload']['head.bias']['dtype']='float64'
        summary['selected_native_state_sha256']=_hash({k:_array_identity(v) for k,v in values.items()})
    else:
        events=root/'fit/events.jsonl';rows=[json.loads(line) for line in events.read_text().splitlines()]
        row=next(r for r in rows if r['type']==('evaluation' if fault in ('false_G_best','boolean_initial_update') else 'update'))
        if fault=='duplicate_row':row['row_indices'][0]=row['row_indices'][1]
        elif fault=='false_G_best':row['improved']=False
        elif fault=='boolean_initial_update':row['update']=False
        else:row['gradient_norms'][0]=-1.
        events.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        summary['event_stream_sha256']=hashlib.sha256(events.read_bytes()).hexdigest()
    summary['selected_prediction_state_sha256']=_prediction_binding(GatedConfig.parse('P00',args['expected_configuration']),summary)
    path.write_text(json.dumps(state));(root/'fit/attempt.json').write_text(json.dumps(summary))
    terminal=json.loads((root/'terminal.json').read_text());terminal['adapter_resources']=summary
    (root/'terminal.json').write_text(json.dumps(terminal));recommit_closed_records(args,attempt)
    # The opaque managed byte chain is consistent; recorded content still is not.
    assert audit_native_generation(**args)['model_generation_chain_authenticated'] is True
    with pytest.raises(ValueError,match='sampler|strict-best|gradient|tensor|grid'):
        inspect_native_workload(**args)
