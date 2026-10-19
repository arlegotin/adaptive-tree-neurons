"""Prediction derivatives of the actual first-growth affine-stump model."""
import copy
import importlib.util
import itertools
import json

import numpy as np
import pytest
import torch

from treeneuron.train import SelectiveRegressor


def api():
    assert importlib.util.find_spec('treeneuron.research.jacobian') is not None, 'Missing full-parent/gate prediction Jacobian builder'
    from treeneuron.research import jacobian
    return jacobian


def model_fixture():
    model=SelectiveRegressor(2,3,tau=.7).double()
    with torch.no_grad():
        model.layer.base.weight.copy_(torch.tensor([[1.,0.],[0.,1.],[-1.,.5]],dtype=torch.float64))
        model.layer.base.bias.copy_(torch.tensor([0.,.25,-.2],dtype=torch.float64))
        model.head.weight.copy_(torch.tensor([[.8,-.3,.5]],dtype=torch.float64));model.head.bias.fill_(.15)
    X=np.array([[0.,0.],[1.,-.25],[-1.,2.],[.3,.4],[-.7,-.8]],dtype=np.float64)
    contrasts={2:np.array([.4,-.2,.1]),0:np.array([-.1,.6,-.3])}
    return model,X,contrasts


def autograd_parent(model,X):
    parameters=[p for p in model.parameters() if p.requires_grad];rows=[]
    prediction=model(torch.from_numpy(X))
    for scalar in prediction:
        gradient=torch.autograd.grad(scalar,parameters,retain_graph=True)
        rows.append(torch.cat([g.flatten() for g in gradient]).detach().numpy())
    return np.stack(rows)


@pytest.mark.parametrize('freeze',['none','head','base_weight'])
def test_full_currently_trainable_parent_jacobian_matches_autograd_in_parameter_order(freeze):
    module=api();model,X,contrasts=model_fixture()
    if freeze=='head':
        for p in model.head.parameters():p.requires_grad_(False)
    elif freeze=='base_weight':model.layer.base.weight.requires_grad_(False)
    bundle=module.prediction_jacobians(model,X,contrasts)
    np.testing.assert_allclose(bundle.parent,autograd_parent(model,X),rtol=1e-12,atol=1e-13)
    np.testing.assert_array_equal(bundle.prediction,model(torch.from_numpy(X)).detach().numpy())
    assert [r['name'] for r in bundle.parameter_columns]==[n for n,p in model.named_parameters() if p.requires_grad]
    assert bundle.parent.dtype==np.float64 and bundle.parent.shape[0]==len(X)


def test_analytic_gate_blocks_match_actual_zero_gate_insertion_with_bias_and_relu_boundary():
    module=api();model,X,contrasts=model_fixture();bundle=module.prediction_jacobians(model,X,contrasts)
    grown=copy.deepcopy(model);ids=sorted(contrasts)
    grown.layer.grow(torch.tensor(ids,dtype=torch.int64),torch.from_numpy(np.stack([contrasts[i] for i in ids])))
    np.testing.assert_array_equal(grown(torch.from_numpy(X)).detach().numpy(),bundle.prediction)
    prediction=grown(torch.from_numpy(X));blocks={unit:[] for unit in ids}
    for scalar in prediction:
        weight,bias=torch.autograd.grad(scalar,[grown.layer.gate.weight,grown.layer.gate.bias],retain_graph=True)
        for row,unit in enumerate(ids):blocks[unit].append(torch.cat([weight[row],bias[row:row+1]]).numpy())
    for unit in ids:np.testing.assert_allclose(bundle.gate_blocks[unit],np.stack(blocks[unit]),rtol=1e-12,atol=1e-13)
    # Unit0's base response at X[0] is exactly zero: PyTorch's derivative is zero.
    assert np.array_equal(bundle.gate_blocks[0][0],np.zeros(3))


def test_jacobian_construction_preserves_model_parameters_gradients_flags_and_RNG():
    module=api();model,X,contrasts=model_fixture();model.train()
    for p in model.parameters():p.grad=torch.ones_like(p)*7.
    saved=[p.detach().clone() for p in model.parameters()];grads=[p.grad.clone() for p in model.parameters()]
    state=torch.random.get_rng_state().clone();bundle=module.prediction_jacobians(model,X,contrasts)
    assert model.training and torch.equal(torch.random.get_rng_state(),state) and model.layer.growth_applied is False
    for p,old,grad in zip(model.parameters(),saved,grads):assert torch.equal(p,old) and torch.equal(p.grad,grad)
    X[:]=0;contrasts[0][:]=0
    assert not np.array_equal(bundle.gate_blocks[0],np.zeros_like(bundle.gate_blocks[0]))


@pytest.mark.parametrize('change',['float32_model','float32_X','grown','unknown_parameter','nonfinite','wrong_contrast','unknown_id','replaced_layer'])
def test_unsupported_or_corrupt_model_geometry_cannot_silently_condition_on_a_partial_parent(change):
    module=api();model,X,contrasts=model_fixture()
    if change=='float32_model':model=model.float()
    elif change=='float32_X':X=X.astype(np.float32)
    elif change=='grown':model.layer.grow(torch.tensor([0]),torch.from_numpy(contrasts[0][None]))
    elif change=='unknown_parameter':model.extra=torch.nn.Parameter(torch.ones(1,dtype=torch.float64))
    elif change=='nonfinite':
        with torch.no_grad():model.head.bias.fill_(float('nan'))
    elif change=='wrong_contrast':contrasts[0]=np.zeros(2)
    elif change=='replaced_layer':model.layer=torch.nn.ReLU()
    else:contrasts[9]=np.zeros(3)
    with pytest.raises(ValueError):module.prediction_jacobians(model,X,contrasts)


def test_actual_stationary_parent_gate_jacobians_exhibit_singleton_pair_failure(tmp_path):
    module=api();model=SelectiveRegressor(1,3).double();X=np.linspace(-1.,1.,801)[:,None]
    with torch.no_grad():
        model.layer.base.weight.copy_(torch.tensor([[1.],[1.],[-1.]],dtype=torch.float64))
        model.layer.base.bias.copy_(torch.tensor([2.,2.,0.],dtype=torch.float64))
        model.head.weight.fill_(1.);model.head.bias.zero_()
    bundle=module.prediction_jacobians(model,X,{i:np.array([1.,0.]) for i in range(3)})
    raw=X[:,0]**2*np.where(X[:,0]>=0,1.,.1)
    U,s,_=np.linalg.svd(bundle.parent,full_matrices=False);basis=U[:,s>1e-10*s[0]]
    residual=raw-basis@(basis.T@raw)
    assert np.max(np.abs(bundle.parent.T@residual/len(X)))<1e-13
    def unpenalized(ids):
        design=np.column_stack([bundle.parent,*[bundle.gate_blocks[i] for i in ids]])
        fit=design@np.linalg.lstsq(design,residual,rcond=1e-10)[0]
        return float(np.mean((residual-fit)**2))
    risks={name:unpenalized(ids) for name,ids in [('singleton_duplicate',[0]),('singleton_complement',[2]),('duplicate_pair',[0,1]),('complementary_pair',[0,2])]}
    assert risks['singleton_duplicate']<risks['singleton_complement']
    assert risks['complementary_pair']<1e-25 and risks['duplicate_pair']>1e-5
    from treeneuron.research.joint_selector import prepare_joint_problem,JointSelectionConfig
    problem=prepare_joint_problem(bundle.parent,residual,bundle.gate_blocks,JointSelectionConfig(count=2,parent_ridge=0.))
    assert problem.gain([0])>problem.gain([2]) and problem.gain([0,2])>problem.gain([0,1])
    assert problem.greedy([0,1,2])['selected_ids']==[0,2]
    # Positive parent ridge is a different objective; verify its own dense
    # reference instead of imposing the unpenalized singleton ordering.
    from test_joint_selector import dense_objective
    regularized=prepare_joint_problem(bundle.parent,residual,bundle.gate_blocks,JointSelectionConfig(count=2))
    errors=[]
    for count in range(4):
        for ids in itertools.combinations(range(3),count):
            exact=dense_objective(bundle.parent,residual,bundle.gate_blocks,ids,.01,.01)
            errors.append(abs(regularized.objective(ids)-exact))
    assert max(errors)<1e-10
    (tmp_path/'counterexample.json').write_text(json.dumps(dict(unpenalized_MSE=risks,
        parent_stationarity_max=float(np.max(np.abs(bundle.parent.T@residual/len(X)))),
        zero_parent_ridge_selected_ids=problem.greedy([0,1,2])['selected_ids'],
        default_ridge_maximum_dense_objective_error=max(errors)),indent=2)+'\n')
