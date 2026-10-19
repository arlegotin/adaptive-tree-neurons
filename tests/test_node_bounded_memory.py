"""Equivalent native NODE arithmetic with explicit smaller resident batches."""
import copy
import hashlib
import json

import numpy as np
import pytest

from test_node_v2 import config, request


@pytest.fixture(scope='module')
def torch():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    return torch


def bounded(mode='engineering_fixture',**parameters):
    return dict(config(mode,**parameters),execution='bounded_memory_v1')


def test_policy_is_explicit_fixed_and_preserves_native_recipe():
    from treeneuron.research.node_v2 import NodeV2Config
    legacy=NodeV2Config.parse(config('native_anchor')).resolved()
    parsed=NodeV2Config.parse(bounded('native_anchor'))
    actual=parsed.resolved();policy=actual.pop('execution')
    assert policy==dict(schema='research/v2-node-execution',microbatch_rows=32,
        prediction_rows=32,initialization='full_prefix_no_output')
    assert actual.pop('validation_chunk')==32
    legacy.pop('validation_chunk')
    assert actual==legacy
    assert parsed.to_state()['execution']=='bounded_memory_v1'
    for value in ('automatic',None,True,{'microbatch_rows':32}):
        with pytest.raises(ValueError):NodeV2Config.parse(dict(config(),execution=value))


def test_full_prefix_initialization_is_exact_including_rng_without_routing_output(torch,monkeypatch):
    from treeneuron.research import node_v2 as node
    legacy=node.NodeV2Config.parse(config(trees=32,depth=4)).resolved()
    small=node.NodeV2Config.parse(bounded(trees=32,depth=4)).resolved()
    torch.manual_seed(19);first=node._build_model(3,legacy);second=copy.deepcopy(first)
    F=np.random.RandomState(3).normal(size=(97,3)).astype(np.float32)
    np.random.seed(29)
    with torch.no_grad():first(torch.from_numpy(F))
    expected=node._model_state(first);numpy_after=np.random.get_state();torch_after=torch.get_rng_state().clone()
    np.random.seed(29)
    # The avoided operation is the ODST output, not its full-prefix initializer.
    def forbidden(*a,**kw):raise AssertionError('Discarded full-prefix output allocated')
    monkeypatch.setattr(second[0][0],'forward',forbidden)
    node._initialize_model(second,F,small)
    assert node._same_model(expected,node._model_state(second))
    now=np.random.get_state()
    assert now[0]==numpy_after[0] and now[2:]==numpy_after[2:]
    np.testing.assert_array_equal(now[1],numpy_after[1]);assert torch.equal(torch.get_rng_state(),torch_after)
    assert second[0][0]._is_initialized_bool is True


def test_float64_accumulation_uses_original_batch_denominator_for_uneven_chunks(torch,tmp_path):
    from treeneuron.research import node_v2 as node
    torch.manual_seed(71)
    full=torch.nn.Linear(3,1).double();small=copy.deepcopy(full)
    x=torch.randn(65,3,dtype=torch.float64);y=torch.randn(65,dtype=torch.float64)
    class Scalar(torch.nn.Module):
        def __init__(self,layer):super().__init__();self.layer=layer
        def forward(self,x):return self.layer(x).squeeze(-1)
    full=Scalar(full);small=Scalar(small)
    expected=(full(x)-y).square().mean();expected.backward()
    resolved=node.NodeV2Config.parse(bounded()).resolved()
    actual=node._loss_backward(small,x,y,resolved)
    torch.testing.assert_close(actual,expected.detach(),rtol=1e-10,atol=1e-12)
    for a,b in zip(full.parameters(),small.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-10,atol=1e-12)
    (tmp_path/'maximum-errors.json').write_text(json.dumps(dict(dtype='float64',
        loss=float(abs(actual-expected.detach())),gradient=max(float((a.grad-b.grad).abs().max()) for a,b in zip(full.parameters(),small.parameters())))))


def test_float32_vendor_gradients_predictions_and_qhadam_agree(torch,tmp_path):
    from treeneuron.research import node_v2 as node
    resolved=node.NodeV2Config.parse(bounded(trees=32,depth=4)).resolved()
    rng=np.random.RandomState(41);F=rng.normal(size=(97,3)).astype(np.float32)
    torch.manual_seed(47);full=node._build_model(3,resolved)
    np.random.seed(43);node._initialize_model(full,F,resolved);small=copy.deepcopy(full)
    x=torch.from_numpy(F[:65]);y=torch.from_numpy(rng.normal(size=65).astype(np.float32))
    full_opt=node._optimizer(full,resolved);small_opt=node._optimizer(small,resolved)
    expected=(full(x)-y).square().mean();expected.backward()
    actual=node._loss_backward(small,x,y,resolved)
    torch.testing.assert_close(actual,expected.detach(),rtol=1e-5,atol=1e-6)
    for a,b in zip(full.parameters(),small.parameters()):
        if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)
    errors=dict(dtype='float32',loss=float(abs(actual-expected.detach())),
        gradient=max(float((a.grad-b.grad).abs().max()) for a,b in zip(full.parameters(),small.parameters()) if a.requires_grad))
    full_opt.step();small_opt.step()
    for name,a in full.state_dict().items():torch.testing.assert_close(a,small.state_dict()[name],rtol=1e-5,atol=1e-6)
    for key,slot in full_opt.state_dict()['state'].items():
        other=small_opt.state_dict()['state'][key]
        for field,value in slot.items():
            if isinstance(value,torch.Tensor):torch.testing.assert_close(value,other[field],rtol=1e-5,atol=1e-6)
            else:assert value==other[field]
    torch.testing.assert_close(full(x),small(x),rtol=1e-5,atol=1e-6)
    errors.update(parameter=max(float((a-small.state_dict()[name]).abs().max()) for name,a in full.state_dict().items() if a.is_floating_point()),
        prediction=float((full(x)-small(x)).abs().max().detach()),
        moment=max(float((value-small_opt.state_dict()['state'][key][field]).abs().max())
            for key,slot in full_opt.state_dict()['state'].items() for field,value in slot.items() if isinstance(value,torch.Tensor)))
    (tmp_path/'maximum-errors.json').write_text(json.dumps(errors))


def test_accumulated_updates_keep_one_step_rows_and_short_trajectory(torch):
    from treeneuron.research import node_v2 as node
    legacy=node.NodeV2Config.parse(config(trees=32,depth=4)).resolved()
    small=node.NodeV2Config.parse(bounded(trees=32,depth=4)).resolved()
    legacy['batch_size']=small['batch_size']=65
    rng=np.random.RandomState(83);F=rng.normal(size=(101,3)).astype(np.float32)
    G=rng.normal(size=(37,3)).astype(np.float32);y=rng.normal(size=101).astype(np.float32)
    gy=rng.normal(size=37).astype(np.float32)
    torch.manual_seed(89);a=node._build_model(3,legacy)
    np.random.seed(97);node._initialize_model(a,F,small);b=copy.deepcopy(a)
    np.random.seed(101);first=node._train(a,node._optimizer(a,legacy),F,y,G,gy,legacy,12,[])
    np.random.seed(101);second=node._train(b,node._optimizer(b,small),F,y,G,gy,small,12,[])
    assert [r['rows'] for r in first['trace']]==[r['rows'] for r in second['trace']]
    assert all(r['applied'] for r in second['trace'])
    assert first['summary']==second['summary']
    for name,tensor in first['raw_model'].items():
        torch.testing.assert_close(tensor,second['raw_model'][name],rtol=1e-5,atol=1e-6)
    for left,right in zip(first['reports'],second['reports']):
        assert left['step']==right['step'] and left['snapshot_steps']==right['snapshot_steps']
        assert left['g_mse']==pytest.approx(right['g_mse'],rel=1e-5,abs=1e-6)


def test_deadline_mid_batch_does_not_enter_optimizer(torch,monkeypatch):
    from treeneuron.research import node_v2 as node
    resolved=node.NodeV2Config.parse(bounded()).resolved();resolved['batch_size']=65
    F=np.ones((65,2),dtype=np.float32);y=np.ones(65,dtype=np.float32)
    torch.manual_seed(109);model=node._build_model(2,resolved)
    np.random.seed(113);node._initialize_model(model,F,resolved)
    before=node._model_state(model);calls=[]
    def check(deadline):
        calls.append(deadline)
        if len(calls)==3:raise TimeoutError('after first microbatch')
    monkeypatch.setattr(node,'_check_deadline',check)
    def forbidden(opt):raise AssertionError('Partial gradients must not update parameters')
    monkeypatch.setattr(node,'_optimizer_step',forbidden)
    trace=[]
    with pytest.raises(TimeoutError):node._train(model,node._optimizer(model,resolved),F,y,F[:2],y[:2],resolved,1,trace,deadline_ns=99)
    assert trace[0]['optimizer_entered'] is False and trace[0]['applied'] is False
    assert node._same_model(before,node._model_state(model))


def test_fitted_policy_bounds_query_override_and_closed_reload(tmp_path,torch):
    from treeneuron.research import node_v2 as node
    model=node.NodeV2Predictor().fit(request(tmp_path/'fit',4,bounded(batch_size=32)))
    sizes=[]
    hook=model._model.register_forward_pre_hook(lambda m,args:sizes.append(len(args[0])))
    query=np.random.RandomState(127).normal(size=(101,2)).astype(np.float32)
    prediction=model.predict(query,batch_size=1024);hook.remove()
    assert max(sizes)<=32 and sum(sizes)==101
    model.save(tmp_path/'model');manifest=tmp_path/'model/manifest.json'
    loaded=node.NodeV2Predictor.load(tmp_path/'model',expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert loaded._config.to_state()['execution']=='bounded_memory_v1'
    np.testing.assert_array_equal(loaded.predict(query),prediction)
    receipt=json.loads((tmp_path/'fit/fit-receipt.json').read_text())
    assert receipt['resolved']['execution']['microbatch_rows']==32
