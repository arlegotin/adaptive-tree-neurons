"""A declared search geometry must complete NODE's actual native patience."""
import numpy as np
import pytest

from native_workload_cases import actual_native_family


@pytest.fixture(scope='module')
def native_node_case(tmp_path_factory):
    return actual_native_family(tmp_path_factory.mktemp('native-NODE-workload'),'P12')


def test_actual_full_native_NODE_observation_preserves_update_rows_and_native_window_selection(native_node_case,monkeypatch):
    from treeneuron.research.native_workload_v2 import inspect_native_workload
    args,attempt=native_node_case
    from treeneuron.research.node_v2 import NodeV2Predictor,NodeTransform
    def forbidden(*a,**kw):raise AssertionError('NODE workload reader cannot fit, predict or refit its quantiles')
    monkeypatch.setattr(NodeV2Predictor,'fit',forbidden);monkeypatch.setattr(NodeV2Predictor,'predict',forbidden)
    monkeypatch.setattr(NodeTransform,'fit',forbidden)
    result=inspect_native_workload(**args);training=result['training']
    with np.load(args['request_path'].parent/'F.npz',allow_pickle=False) as saved:nF=len(saved['y'])
    assert nF==10 and training['procedure']=='native_NODE_qhadam'
    assert training['executed_updates']>5000 and training['selected_step']>=100
    assert training['native_stop']=='G_patience' and training['visited_F_rows']==nF
    assert training['finite_recorded_gradients_checked'] is True
    assert training['native_batch_size']==512 and training['trees']==512 and training['depth']==4
    assert training['rolling_checkpoints']==5 and training['actual_optimization_independently_reexecuted'] is False
    assert result['native_workload_qualified'] is result['quality_eligible'] is result['T_accessed'] is False


@pytest.mark.parametrize('fault',['sampler_order','duplicate_row','negative_second_moment','false_selected_step',
    'unknown_sampler_field','receipt_counter'])
def test_self_recommitted_NODE_content_cannot_supply_a_workload_observation(native_node_case,fault):
    import copy,hashlib,json
    from treeneuron.research.native_generation_v2 import audit_native_generation
    from treeneuron.research.native_node_workload_v2 import _Array,_metadata
    from treeneuron.research.native_workload_v2 import inspect_native_workload
    from treeneuron.research.node_v2 import _array_identity
    from test_native_workload_observation import recommit_closed_records
    original,attempt=native_node_case;args=copy.deepcopy(original);root=attempt/'native';worker=root/'worker'
    paths=[worker/'snapshot/manifest.json',worker/'snapshot/arrays.npz',worker/'terminal.json',
        worker/'fit/fit-receipt.json',root/'process/process.json',root/'result.json']
    before={p:p.read_bytes() for p in paths}
    def pack(value):
        if isinstance(value,_Array):return dict(array=value.name,tensor=value.tensor)
        if type(value) is dict:return {'mapping':[[pack(k),pack(value[k])] for k in sorted(value,key=lambda k:(type(k).__name__,k))]}
        if type(value) in (list,tuple):return {'tuple' if type(value) is tuple else 'list':[pack(v) for v in value]}
        return value
    try:
        path=worker/'snapshot/manifest.json';manifest=json.loads(path.read_text());state=_metadata(manifest['payload'],manifest['arrays'],set())
        receipt=json.loads((worker/'fit/fit-receipt.json').read_text())
        if fault in ('sampler_order','negative_second_moment'):
            target=(state['sampler']['permutation'] if fault=='sampler_order'
                else state['raw_optimizer']['state'][1]['exp_avg_sq'])
            with np.load(worker/'snapshot/arrays.npz',allow_pickle=False) as saved:values={k:saved[k].copy() for k in saved.files}
            if fault=='sampler_order':values[target.name]=np.roll(values[target.name],1)
            else:values[target.name].flat[0]=-1.
            np.savez(worker/'snapshot/arrays.npz',**values)
            manifest['arrays'][target.name]=_array_identity(values[target.name])
            manifest['array_file']['sha256']=hashlib.sha256((worker/'snapshot/arrays.npz').read_bytes()).hexdigest()
        elif fault=='duplicate_row':
            state['trace'][0]['rows'][0]=state['trace'][0]['rows'][1];receipt['trace']=state['trace']
        elif fault=='unknown_sampler_field':state['sampler']['undeclared_phase']=0
        elif fault=='receipt_counter':receipt['executed_updates']=1
        else:
            state['summary']['selected_step']+=100;receipt['summary']=state['summary']
            terminal=json.loads((worker/'terminal.json').read_text());terminal['adapter_resources']['selected_step']=state['summary']['selected_step']
            (worker/'terminal.json').write_text(json.dumps(terminal))
        manifest['payload']=pack(state);path.write_text(json.dumps(manifest))
        (worker/'fit/fit-receipt.json').write_text(json.dumps(receipt));recommit_closed_records(args,attempt)
        assert audit_native_generation(**args)['model_generation_chain_authenticated'] is True
        with pytest.raises(ValueError,match='sampler|trace|second moment|patience|selected|receipt'):
            inspect_native_workload(**args)
    finally:
        for path,raw in before.items():path.write_bytes(raw)
        # Each controlled mutation restores the exact original closed generation.
        from treeneuron.config import canonical_hash
        from treeneuron.research.workqueue import _inventory
        assert canonical_hash(_inventory(root))==original['native_reference']['inventory_sha256']
