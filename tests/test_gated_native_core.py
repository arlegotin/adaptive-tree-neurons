"""Own-method tabular support must not weaken legacy shape/state contracts."""
import copy
import importlib.util

import numpy as np
import pytest
import torch


def config_api():
    assert importlib.util.find_spec('treeneuron.research.gated_config') is not None, 'Missing gated v2 configuration'
    from treeneuron.research import gated_config
    return gated_config


def model_api():
    assert importlib.util.find_spec('treeneuron.research.gated_models') is not None, 'Missing gated v2 model core'
    from treeneuron.research import gated_models
    return gated_models


def config(family='P00',mode='native_anchor',**changes):
    value=dict(schema='research/v2-gated-native-config',family=family,mode=mode,
        feature_schema=dict(columns=['x','kind'],categorical=['kind']),parameters={})
    if family=='P02':value['variant']='upper'
    if family=='P05':value['variant']='centered_partial'
    value.update(changes)
    return value


def test_native_selected_random_policies_share_every_resolved_training_choice():
    a=config_api().GatedConfig.parse('P00',config()).resolved()
    b=config_api().GatedConfig.parse('P01',config('P01')).resolved()
    assert a.pop('placement')=='legacy_adam' and b.pop('placement')=='random'
    a.pop('family');b.pop('family');assert a==b
    assert a['width']==64 and a['requested_count']==16
    assert a['base_updates']==a['continuation_updates']==8000
    assert a['lr']==.005 and a['weight_decay']==0 and a['batch_size']==128
    assert a['validation_interval']==25 and a['patience']==20 and a['schedule_horizon']==16000
    assert a['epsilon']==.05 and a['tau']==1.


@pytest.mark.parametrize('variant,count',[('centered_partial',16),('centered_full',64),
    ('direct_partial',16),('direct_full',64)])
def test_static_anchor_variants_preserve_their_distinct_capacity(variant,count):
    value=config_api().GatedConfig.parse('P05',config('P05',variant=variant)).resolved()
    assert value['requested_count']==count and value['dropout']==0.
    assert value['base_updates']==value['continuation_updates']==8000


@pytest.mark.parametrize('change',[
    {'selection':'joint'}, {'treatment':'restart'}, {'family':'P03'}, {'parameters':{'width':True}},
    {'parameters':{'width':32}}, {'parameters':{'lr':True}}, {'parameters':{'lr':float('nan')}},
    {'parameters':{'lr':.1}}, {'parameters':{'weight_decay':1e-9}},
    {'parameters':{'batch_size':512}}, {'parameters':{'dropout':.1}},
    {'parameters':{'growth_fraction':.3}}, {'parameters':{'scheduler':'restart'}},
    {'feature_schema':{'columns':['x','x'],'categorical':[]}},
])
def test_search_configuration_rejects_undeclared_choices(change):
    value=config(mode='search',**change)
    with pytest.raises(ValueError):config_api().GatedConfig.parse('P00',value)


def test_an_extreme_numeric_request_is_rejected_as_input_before_model_execution():
    with pytest.raises(ValueError):
        config_api().GatedConfig.parse('P00',config(mode='search',parameters={'lr':10**1000}))


def test_anchor_overrides_and_mutable_config_aliases_cannot_change_a_saved_recipe():
    with pytest.raises(ValueError):config_api().GatedConfig.parse('P00',config(parameters={'lr':.001}))
    value=config(mode='search',parameters={'width':128,'growth_fraction':.5,'scheduler':'cosine','lr':.001})
    parsed=config_api().GatedConfig.parse('P00',value);original=copy.deepcopy(value)
    value['parameters']['width']=64;parsed.to_state()['parameters']['width']=64
    assert parsed.to_state()==original
    resolved=parsed.resolved();assert resolved['width']==128 and resolved['requested_count']==64
    assert resolved['scheduler']=='cosine';resolved['width']=1
    assert parsed.resolved()['width']==128


def test_small_engineering_recipe_is_explicit_and_never_changes_native_limits():
    tiny=config(mode='engineering_fixture',parameters={'width':8},
        base_updates=4,continuation_updates=4,validation_interval=1,patience=2)
    r=config_api().GatedConfig.parse('P00',tiny).resolved()
    assert r['width']==8 and r['requested_count']==2 and r['schedule_horizon']==8
    for change in ({'base_updates':26},{'base_updates':True},{'patience':0}):
        with pytest.raises(ValueError):config_api().GatedConfig.parse('P00',tiny|change)
    with pytest.raises(ValueError):config_api().GatedConfig.parse('P00',config(base_updates=4))


@pytest.mark.parametrize('d,h,count,expected',[(16,64,16,(94,95)),(2,8,2,(11,11)),
    (1,8,2,(10,11)),(128,128,64,(255,256)),(21,64,0,(64,64))])
def test_ordinary_parameter_brackets_follow_actual_transformed_dimension(d,h,count,expected):
    assert model_api().ordinary_brackets(d,h,count)==expected


def optimizer(model):
    names,parameters=zip(*model.named_parameters())
    return torch.optim.AdamW([dict(params=list(parameters),param_names=list(names))],lr=.003,foreach=False)


def exercised_parent(d=21):
    torch.manual_seed(17);model=model_api().parent_model(d,8,1.)
    X=torch.linspace(.1,1.,12*d).reshape(12,d)
    with torch.no_grad():model.layer.base.weight.zero_();model.layer.base.bias.fill_(1.);model.head.weight.fill_(.1)
    opt=optimizer(model)
    for _ in range(3):
        opt.zero_grad();((model(X)-.5)**2).mean().backward();opt.step()
    return model,opt,X


def test_gate_birth_preserves_parent_objects_moments_and_function_above_legacy_dimension():
    model,opt,X=exercised_parent();old=dict(model.named_parameters())
    moments={name:{k:v.clone() for k,v in opt.state[p].items()} for name,p in old.items()}
    before=model(X).detach().clone();c=torch.full((2,22),.05)
    manifest=model_api().insert_gates(model,opt,torch.tensor([0,2]),c)
    assert manifest.added_parameter_count==88
    torch.testing.assert_close(model(X),before,rtol=0,atol=0)
    for name,p in old.items():
        assert dict(model.named_parameters())[name] is p
        for key,v in moments[name].items():assert torch.equal(opt.state[p][key],v)
    newborn=[p for name,p in model.named_parameters() if name not in old]
    assert newborn and len(opt.param_groups)==2
    assert all(float(opt.state[p]['step'])==0 and torch.count_nonzero(opt.state[p]['exp_avg_sq'])==0 for p in newborn)
    opt.zero_grad();((model(X)-.5)**2).mean().backward()
    assert model.layer.gate.weight.grad.abs().sum()>0 and model.layer.contrast.weight.grad.abs().sum()==0
    opt.step();opt.zero_grad();((model(X)-.5)**2).mean().backward()
    assert model.layer.contrast.weight.grad.abs().sum()>0


def test_failed_gate_optimizer_registration_rolls_back_model_rng_and_old_adam(tmp_path,monkeypatch):
    model,opt,X=exercised_parent();before=model(X).detach().clone()
    tensors={k:v.clone() for k,v in model.state_dict().items()};state=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
    def fail(group):raise RuntimeError('controlled newborn registration failure')
    monkeypatch.setattr(opt,'add_param_group',fail)
    with pytest.raises(RuntimeError,match='controlled newborn'):model_api().insert_gates(model,opt,torch.tensor([0,2]),torch.full((2,22),.05))
    assert not model.layer.growth_applied and model.layer.gate is None and len(opt.param_groups)==1
    assert set(model.state_dict())==set(tensors) and torch.equal(torch.get_rng_state(),rng)
    for name,tensor in tensors.items():assert torch.equal(model.state_dict()[name],tensor)
    assert opt.state_dict()['param_groups']==state['param_groups']
    for index,fields in state['state'].items():
        for key,value in fields.items():assert torch.equal(opt.state_dict()['state'][index][key],value)
    torch.testing.assert_close(model(X),before,rtol=0,atol=0)


def test_warm_ordinary_growth_preserves_old_parameters_and_adam_ages():
    model,opt,X=exercised_parent();old=dict(model.named_parameters());before=model(X).detach().clone()
    moments={name:{k:v.clone() for k,v in opt.state[p].items()} for name,p in old.items()};rng=torch.get_rng_state().clone()
    widened=model_api().widen_ordinary(model,opt,11,seed=23)
    torch.testing.assert_close(widened(X),before,rtol=0,atol=0)
    assert sum(p.numel() for p in widened.parameters())==11*23+1
    assert torch.equal(torch.get_rng_state(),rng)
    for name,p in old.items():
        assert dict(widened.named_parameters())['incumbent.'+name] is p
        for key,value in moments[name].items():assert torch.equal(opt.state[p][key],value)
    assert opt.param_groups[0]['param_names']==['incumbent.'+name for name in old]
    # Ensure the ordinary newborn is active; a dead ReLU is not a learning bug.
    with torch.no_grad():widened.added_affine.weight.zero_();widened.added_affine.bias.fill_(1.)
    opt.zero_grad();((widened(X)-.5)**2).mean().backward()
    assert widened.added_head.weight.grad.abs().sum()>0
    assert widened.added_affine.weight.grad.abs().sum()==0
    opt.step();opt.zero_grad();((widened(X)-.5)**2).mean().backward()
    assert widened.added_affine.weight.grad.abs().sum()>0


@pytest.mark.parametrize('corruption',['negative_second_moment','age_disagreement','implicit_dtype'])
def test_new_growth_rejects_corrupt_parent_adam_before_changing_topology(corruption):
    model,opt,X=exercised_parent();p=next(model.parameters())
    if corruption=='negative_second_moment':opt.state[p]['exp_avg_sq'].flatten()[0]=-1
    elif corruption=='age_disagreement':opt.state[p]['step'].zero_()
    else:opt.state[p]['exp_avg']=opt.state[p]['exp_avg'].double()
    with pytest.raises(ValueError):model_api().insert_gates(model,opt,torch.tensor([0,2]),torch.full((2,22),.05))
    assert not model.layer.growth_applied and len(opt.param_groups)==1


@pytest.mark.parametrize('coordinates',['centered','direct_leaf'])
def test_static_forward_mixes_affine_branches_before_relu_in_encoded_dimension(coordinates):
    torch.manual_seed(23);model=model_api().StaticModel(21,4,2,coordinates,tau=1.,dropout=0.).double()
    X=np.linspace(-1,1,63).reshape(3,21)
    def affine(layer):return X@layer.weight.detach().numpy().T+layer.bias.detach().numpy()
    gate=affine(model.gate)
    if coordinates=='centered':z=affine(model.center)+np.tanh(gate/2)*affine(model.contrast)
    else:
        q=1/(1+np.exp(-gate));z=q*affine(model.left)+(1-q)*affine(model.right)
    z=np.concatenate((z,affine(model.ordinary)),axis=1)
    expected=np.maximum(z,0)@model.head.weight.detach().numpy().T+model.head.bias.detach().numpy()
    np.testing.assert_allclose(model(torch.from_numpy(X)).detach().numpy(),expected.ravel(),rtol=1e-12,atol=1e-12)
    model(torch.from_numpy(X)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters())==4*23+1+2*2*22


def test_legacy_static_and_widening_envelopes_remain_closed():
    from treeneuron.models.static import StaticRegressor
    from treeneuron.models.widened import WidenedRegressor
    with pytest.raises(ValueError):StaticRegressor(21,64,16,'centered')
    with pytest.raises(ValueError):WidenedRegressor(21,95,64)
    # The v2 mathematical core is separate; execution still needs its own receipt.
    assert model_api().StaticModel(21,128,32,'centered',tau=1.,dropout=0.).head.in_features==128
