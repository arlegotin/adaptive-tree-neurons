"""NODE fidelity: changing selector axis/functions/head/init/coverage must fail."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from treeneuron.baselines.api import FitRequest

TOL = dict(rtol=1e-6, atol=1e-7)  # declared before adapter execution; CPU float32


def adapter():
    assert importlib.util.find_spec('treeneuron.baselines.node') is not None, 'NODE adapter is absent'
    from treeneuron.baselines.node import NodePredictor
    return NodePredictor


def request(tmp_path, n=37, seed=1337):
    rng=np.random.RandomState(9);x=rng.normal(size=(n,4)).astype('float32')
    g=rng.normal(size=(11,4)).astype('float32')
    return FitRequest(x,100+3*x[:,0],g,100+3*g[:,0],seed,{}, {'max_updates':12},tmp_path)


def fixture_config(**kw):
    return dict(recipe='engineering_fixture', trees=5, depth=2, batch_size=8,
                report_frequency=2, **kw)


def test_adapter_exists():
    adapter()


@pytest.mark.parametrize('shape,dim', [((4,5,3),0), ((3,4,5),-1)])
def test_sparse_choice_independent_cpu_backward(shape, dim):
    cls=adapter();from node_reference import nn
    model=cls(fixture_config()).build_model(4)
    f=model[0][0].choice_function
    # Noncontiguous actual-selector layout, ties and exact support boundaries.
    x=torch.tensor(np.random.RandomState(71).normal(size=tuple(reversed(shape)))*1.5,dtype=torch.float32).permute(2,1,0)
    x=x.clone().requires_grad_();z=x.detach().clone().requires_grad_()
    weights=torch.sin(torch.arange(x.numel()).reshape(x.shape))
    a=f(x,dim=dim);b=nn.entmax15(z,dim=dim)
    torch.testing.assert_close(a,b,**TOL)
    (a*weights).sum().backward();(b*weights).sum().backward()
    torch.testing.assert_close(x.grad,z.grad,**TOL)
    tied=torch.tensor([[0.,0.,0.],[2.,0.,-2.]],requires_grad=True)
    torch.testing.assert_close(f(tied,dim=-1),nn.entmax15(tied,dim=-1),**TOL)
    assert a.eq(0).any() and torch.isfinite(x.grad).all() and x.grad.abs().gt(1e-5).any()


def test_bin_support_backward_and_sigmoid_counterexample():
    cls=adapter();from node_reference import nn
    f=cls(fixture_config()).build_model(4)[0][0].bin_function
    x=torch.tensor([-3.,-2.,-1.999,-1.,0.,1.,1.999,2.,3.],requires_grad=True)
    z=x.detach().clone().requires_grad_();a=f(x);b=nn.entmoid15(z)
    torch.testing.assert_close(a,b,**TOL);a.sum().backward();b.sum().backward()
    torch.testing.assert_close(x.grad,z.grad,**TOL)
    assert a[0]==0 and a[-1]==1 and x.grad[0]==0 and x.grad[-1]==0
    with pytest.raises(AssertionError):torch.testing.assert_close(torch.sigmoid(x),b,**TOL)


def test_model_vendor_head_axis_init_and_backward(tmp_path):
    cls=adapter();from node_reference import arch,nn
    x=torch.tensor(request(tmp_path).X_fit)
    np.random.seed(41);torch.manual_seed(41)
    ours=cls(fixture_config()).build_model(4)
    with torch.no_grad():actual=ours(x)
    np.random.seed(41);torch.manual_seed(41)
    vendor=arch.DenseBlock(4,5,num_layers=1,tree_dim=3,depth=2,flatten_output=False,
                          choice_function=nn.entmax15,bin_function=nn.entmoid15)
    with torch.no_grad():expected=vendor(x)[...,0].mean(-1)
    torch.testing.assert_close(actual,expected,**TOL)
    for k,v in vendor.state_dict().items():torch.testing.assert_close(ours[0].state_dict()[k],v,**TOL)
    ours(x).square().mean().backward();vendor(x)[...,0].mean(-1).square().mean().backward()
    for p,q in zip(ours[0].parameters(),vendor.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,**TOL)
    assert ours[0][0].response.shape==(5,3,4)


def test_f_only_initialization_full_fit_and_original_unit_reload(tmp_path):
    cls=adapter();r=request(tmp_path);a=cls(fixture_config());a.fit(r)
    altered=FitRequest(r.X_fit,r.y_fit,r.X_select*10000,r.y_select+9000,r.seed,{},r.budget,tmp_path)
    b=cls(fixture_config());b.fit(altered)
    for k in a.initial_state:torch.testing.assert_close(a.initial_state[k],b.initial_state[k],rtol=0,atol=0)
    np.testing.assert_array_equal(a.quantile.quantiles_,b.quantile.quantiles_)
    assert set(a.training_row_indices)==set(range(37)) and len(a.training_row_indices)==90
    assert a.resource_summary()['initialization_rows']==37
    p=a.predict(r.X_select);assert p.shape==(11,) and np.isfinite(p).all() and np.mean(p)>90
    path=tmp_path/'node.json';a.save(path);restored=cls.load(path)
    np.testing.assert_array_equal(restored.predict(r.X_select,batch_size=11),p)
    before=copy.deepcopy(restored.model.state_dict());restored.predict(r.X_select*1e5)
    for k,v in before.items():torch.testing.assert_close(restored.model.state_dict()[k],v,rtol=0,atol=0)
    assert all(bool(m._is_initialized_tensor) for m in restored.model[0])


def test_predict_before_fit_and_nonfaithful_anchor_rejected(tmp_path):
    cls=adapter()
    with pytest.raises(RuntimeError):cls({}).predict(np.ones((3,4)))
    with pytest.raises(ValueError):cls({'trees':4})
    with pytest.raises(ValueError):cls({'device':'mps'})
    with pytest.raises(ValueError):cls({'optimizer':'Adam'})


def test_parameter_average_latest_five_and_raw_restore(tmp_path):
    cls=adapter();r=request(tmp_path);a=cls(fixture_config());a.fit(r)
    assert len(a.report_history)==6
    latest=a.report_history[-1]
    assert latest['snapshot_steps']==[4,6,8,10,12]
    for event in a.report_history:
        assert event['raw_restore_max_abs']==0 and event['optimizer_restore_max_abs']==0
    # Selected snapshot is the parameter mean, not the last raw state.
    for key,value in a.model.state_dict().items():
        torch.testing.assert_close(value,a.best_state[key],rtol=0,atol=0)
    assert a.selected_step==min(a.report_history,key=lambda h:h['g_mse'])['step']


def test_qhadam_multistep_and_state_reload(tmp_path):
    cls=adapter();from node_reference import QHAdam
    p=torch.nn.Parameter(torch.tensor([1.,-2.]));q=torch.nn.Parameter(p.detach().clone())
    a=cls(fixture_config()).build_optimizer([p]);b=QHAdam([q],nus=(.7,1.),betas=(.95,.998))
    for i in range(4):
        p.grad=q.grad=torch.tensor([.2+i,-.8+i]);a.step();b.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    c=cls(fixture_config()).build_optimizer([p]);c.load_state_dict(copy.deepcopy(a.state_dict()))
    p.grad=q.grad=torch.tensor([.3,-.2]);c.step();b.step();torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert a.param_groups[0]['lr']==.001


def test_training_snapshots_match_independent_unaveraged_vendor_updates(tmp_path):
    cls=adapter();from node_reference import arch,nn,QHAdam
    r=request(tmp_path);a=cls(fixture_config());a.fit(r)
    v=torch.nn.Sequential(arch.DenseBlock(4,5,1,tree_dim=3,depth=2,flatten_output=False,
        choice_function=nn.entmax15,bin_function=nn.entmoid15),nn.Lambda(lambda x:x[...,0].mean(-1)))
    v.load_state_dict(a.initial_state);o=QHAdam(v.parameters(),nus=(.7,1.),betas=(.95,.998))
    x=torch.tensor(a.quantile.transform(r.X_fit));y=torch.tensor(((r.y_fit-r.y_fit.mean())/r.y_fit.std()).astype('float32'))
    states=[];cursor=0
    for step in range(1,13):
        count=min(8,37-((step-1)%5)*8)
        rows=a.training_row_indices[cursor:cursor+count];cursor+=count
        o.zero_grad();(v(x[rows])-y[rows]).square().mean().backward();o.step()
        if step%2==0:states.append({k:t.clone() for k,t in v.state_dict().items()})
    for k,t in v.state_dict().items():torch.testing.assert_close(a.raw_state[k],t,**TOL)
    index=a.selected_step//2
    selected=states[max(0,index-5):index]
    for k,t in a.best_state.items():
        expected=torch.stack([s[k] for s in selected]).mean(0) if t.is_floating_point() else selected[-1][k]
        torch.testing.assert_close(t,expected,**TOL)


def test_snapshot_rejects_source_change_and_uninitialized_state(tmp_path,monkeypatch):
    cls=adapter();a=cls(fixture_config());a.fit(request(tmp_path));path=tmp_path/'node.json';a.save(path)
    import treeneuron.baselines.node as module
    identity=module.node_identity();identity['package_source']='changed'
    with monkeypatch.context() as m:
        m.setattr(module,'node_identity',lambda:identity)
        with pytest.raises(ValueError,match='identity'):cls.load(path)
    a.model[0][0]._is_initialized_tensor.data.zero_();a.save(path)
    with pytest.raises(ValueError,match='uninitialized'):cls.load(path)


def test_reload_retains_training_counts_and_next_raw_optimizer_update(tmp_path):
    cls=adapter();r=request(tmp_path);a=cls(fixture_config());a.fit(r);p=tmp_path/'state.json';a.save(p);b=cls.load(p)
    assert b.resource_summary()['training_rows_seen']==a.resource_summary()['training_rows_seen']
    for model in (a,b):
        model.model.load_state_dict(model.raw_state);model.optimizer.load_state_dict(model.raw_optimizer_state);model.optimizer.zero_grad()
        model.model(torch.tensor(model.quantile.transform(r.X_fit))).square().mean().backward();model.optimizer.step()
    for k,v in a.model.state_dict().items():torch.testing.assert_close(v,b.model.state_dict()[k],rtol=0,atol=0)


def test_checkpoint_metadata_integrity(tmp_path):
    cls=adapter();a=cls(fixture_config());a.fit(request(tmp_path));path=tmp_path/'node.json';a.save(path)
    data=json.loads(path.read_text());data['state']['mapping'].append(['unexpected','tampered'])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='integrity'):cls.load(path)


def test_noisy_quantile_transform_matches_independent_formula(tmp_path):
    from sklearn.preprocessing import QuantileTransformer
    cls=adapter();r=request(tmp_path);a=cls(fixture_config());a.fit(r)
    rng=np.random.RandomState(r.seed);noisy=r.X_fit.copy()
    noisy+=(.001/np.maximum(np.std(noisy,axis=0,keepdims=True),.001))*rng.randn(*noisy.shape)
    expected=QuantileTransformer(n_quantiles=1000,output_distribution='normal',ignore_implicit_zeros=False,
        subsample=10000,random_state=r.seed,copy=True).fit(noisy)
    np.testing.assert_array_equal(a.quantile.quantiles_,expected.quantiles_)
    assert a.y_mean==float(r.y_fit.mean()) and a.y_std==float(r.y_fit.std())


def test_checkpoint_average_matches_unmodified_vendor_reduction_order(tmp_path):
    # Execute the archived method body independently, avoiding tensorboard/UI imports.
    # Vendor takes filesystem-newest first and uses sequential sum / count.
    import ast
    from copy import deepcopy
    from treeneuron.baselines.node import average_states
    source=Path('configs/vendor/node/upstream/lib/trainer.py').read_text()
    tree=ast.parse(source)
    method=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Trainer'
                for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='average_checkpoints')
    module=ast.Module(body=[method],type_ignores=[]);namespace={'torch':torch,'deepcopy':deepcopy}
    exec(compile(module,'archived-trainer-average','exec'),namespace)
    chronological=[{'value':torch.tensor([v]),'init':torch.tensor(1,dtype=torch.uint8)} for v in (1e8,-1e8,3.)]
    paths=[]
    for i,state in enumerate(chronological):
        path=tmp_path/f'{i}.pt';torch.save({'model':state,'opt':{},'step':i},path);paths.append(str(path))
    output=tmp_path/'average.pt'
    namespace['average_checkpoints'](None,paths=list(reversed(paths)),out_path=str(output))
    vendor=torch.load(output,weights_only=True)['model']
    actual=average_states(chronological)
    torch.testing.assert_close(actual['value'],vendor['value'],rtol=0,atol=0)
    assert actual['init'].dtype==torch.uint8 and actual['init']==1


def test_native_environment_mismatch_rejected(monkeypatch):
    import importlib.metadata
    from treeneuron.baselines.node import node_identity
    original=importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda key:'0.0' if key=='numpy' else original(key))
    with pytest.raises(RuntimeError,match='environment'):node_identity()


def test_best_checkpoint_restores_its_optimizer_and_unselected_budget_is_explicit(tmp_path):
    cls=adapter();r=request(tmp_path)
    opposite=FitRequest(r.X_fit,r.y_fit,r.X_select,100-100*(r.y_select-100),r.seed,{},r.budget,tmp_path)
    a=cls(fixture_config());a.fit(opposite)
    assert a.selected_step<a.executed_updates, 'fixture must select before last update'
    expected=sum(.95**i for i in range(a.selected_step))
    for state in a.optimizer.state.values():assert state['beta1_weight']==pytest.approx(expected)
    assert a.resource_summary()['selection_checkpoint_available']
    assert a.resource_summary()['quality_selection_eligible'] is False  # fixture is not the full anchor
    short=FitRequest(r.X_fit,r.y_fit,r.X_select,r.y_select,r.seed,{},dict(max_updates=1),tmp_path)
    b=cls(fixture_config());b.fit(short)
    assert b.resource_summary()['selection_status']=='unavailable_at_budget'
    assert b.resource_summary()['quality_selection_eligible'] is False


def test_strict_native_patience_and_dense_parameter_gradients(tmp_path):
    cls=adapter();r=request(tmp_path)
    opposite=FitRequest(r.X_fit,r.y_fit,r.X_select,100-100*(r.y_select-100),r.seed,{},r.budget,tmp_path)
    a=cls(fixture_config(patience=2));a.fit(opposite)
    assert a.stop_reason=='G_patience' and a.executed_updates==a.selected_step+3
    assert all(not p.grad.is_sparse for p in a.model.parameters() if p.grad is not None)


def test_entmoid_cpu64_known_values_and_derivatives():
    cls=adapter();f=cls(fixture_config()).build_model(4)[0][0].bin_function
    x=torch.tensor([-2.,0.,2.],dtype=torch.float64,requires_grad=True);y=f(x);y.sum().backward()
    torch.testing.assert_close(y,torch.tensor([0.,.5,1.],dtype=torch.float64),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(x.grad,torch.tensor([0.,2**.5/4,0.],dtype=torch.float64),rtol=1e-12,atol=1e-12)
