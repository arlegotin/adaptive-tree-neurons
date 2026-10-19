"""Exact local objectives before any predictive joint-selector experiment."""
import importlib.util
import itertools

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.joint_selector') is not None, 'Missing exact joint ridge core'
    from treeneuron.research import joint_selector
    return joint_selector


def reference_columns(matrix):
    # Independent dense reference uses norm/sqrt(n), not core scaling output.
    rms=np.linalg.norm(matrix,axis=0)/np.sqrt(len(matrix))
    cutoff=max(1e-15,1e-12*max(rms,default=0.))
    return matrix[:,rms>cutoff]/rms[rms>cutoff]


def dense_objective(parent,residual,blocks,ids,parent_ridge,gate_ridge,condition=True):
    n=len(residual);parent=reference_columns(parent) if condition else np.empty((n,0))
    if parent_ridge==0 and parent.shape[1]:
        U,s,_=np.linalg.svd(parent,full_matrices=False)
        parent=U[:,s>1e-10*s[0]]
    gate=np.concatenate([reference_columns(blocks[i]) for i in sorted(ids)],axis=1) if ids else np.empty((n,0))
    design=np.concatenate([parent,gate],axis=1)
    penalties=np.sqrt(np.r_[np.full(parent.shape[1],parent_ridge),np.full(gate.shape[1],gate_ridge)])
    augmented=np.vstack([design/np.sqrt(n),np.diag(penalties)])
    target=np.r_[residual/np.sqrt(n),np.zeros(design.shape[1])]
    coefficients=np.linalg.lstsq(augmented,target,rcond=1e-12)[0]
    error=augmented@coefficients-target
    return float(error@error)


@pytest.mark.parametrize('parent_ridge,condition',[(.01,True),(0.,True),(.01,False)])
def test_every_small_subset_matches_independent_augmented_least_squares(parent_ridge,condition):
    module=api();rng=np.random.default_rng(18404);parent=rng.normal(size=(19,4));r=rng.normal(size=19)*3.
    blocks={7:rng.normal(size=(19,3)),2:rng.normal(size=(19,2)),11:rng.normal(size=(19,3))}
    blocks[11][:,0]=parent[:,0];blocks[11][:,1]=blocks[7][:,0];blocks[11][:,2]=0.
    for gate_ridge in (1e-4,.01,1.):
        config=module.JointSelectionConfig(count=2,parent_ridge=parent_ridge,gate_ridge=gate_ridge,condition_on_parent=condition)
        problem=module.prepare_joint_problem(parent,r,blocks,config)
        empty=dense_objective(parent,r,blocks,[],parent_ridge,gate_ridge,condition)
        for count in range(4):
            for subset in itertools.combinations(blocks,count):
                objective=dense_objective(parent,r,blocks,subset,parent_ridge,gate_ridge,condition)
                assert problem.objective(subset)==pytest.approx(objective,rel=1e-8,abs=1e-10)
                assert problem.gain(subset)==pytest.approx(empty-objective,rel=1e-8,abs=1e-10)
                for candidate in set(blocks)-set(subset):
                    direct=dense_objective(parent,r,blocks,[*subset,candidate],parent_ridge,gate_ridge,condition)
                    assert problem.marginal_gain(subset,candidate)==pytest.approx(objective-direct,rel=1e-8,abs=1e-10)


def test_one_row_duplicate_parent_shares_penalty_but_has_no_additional_span():
    module=api();one=np.ones((1,1));r=np.ones(1)
    positive=module.prepare_joint_problem(one,r,{3:one},module.JointSelectionConfig(count=1,parent_ridge=1.,gate_ridge=1.))
    assert positive.objective(())==pytest.approx(.5)
    assert positive.objective((3,))==pytest.approx(1/3)
    assert positive.gain((3,))==pytest.approx(1/6)
    span=module.prepare_joint_problem(one,r,{3:one},module.JointSelectionConfig(count=1,parent_ridge=0.,gate_ridge=1.))
    assert span.parent_rank==1 and span.gain((3,))==0.


def test_original_gate_scales_are_not_changed_after_parent_residualization():
    module=api();parent=np.arange(1.,8.)[:,None];extra=np.array([1.,-2.,1.,0.,0.,0.,0.])
    extra-=parent[:,0]*np.dot(parent[:,0],extra)/np.dot(parent[:,0],parent[:,0])
    blocks={0:parent+1e-6*extra[:,None]};r=extra.copy()
    problem=module.prepare_joint_problem(parent,r,blocks,module.JointSelectionConfig(parent_ridge=0.))
    expected=dense_objective(parent,r,blocks,[],0.,.01)-dense_objective(parent,r,blocks,[0],0.,.01)
    assert 0<problem.gain([0])<1e-7
    assert problem.gain([0])==pytest.approx(expected,abs=1e-10)


def test_conditional_greedy_selects_complement_over_duplicate_singleton():
    module=api();parent=np.zeros((3,0));r=np.array([1.,.5,0.]);a=np.array([[1.],[0.],[0.]])
    blocks={5:a.copy(),2:a.copy(),9:np.array([[0.],[1.],[0.]])}
    problem=module.prepare_joint_problem(parent,r,blocks,module.JointSelectionConfig(count=2,parent_ridge=0.))
    assert problem.gain([2])>problem.gain([9])
    assert problem.gain([2,9])>problem.gain([2,5])
    trace=problem.greedy([9,5,2]);assert trace['selected_ids']==[2,9]
    assert trace['requested_count']==trace['effective_count']==2
    assert len(trace['slots'])==2 and len(trace['slots'][0]['candidate_gains'])==3


def test_zero_columns_keep_fixed_count_and_original_input_arrays_are_unchanged():
    module=api();parent=np.ones((5,1));r=np.arange(5,dtype=np.float64);blocks={4:np.zeros((5,2)),1:np.zeros((5,2))}
    saved=[x.copy() for x in (parent,r,*blocks.values())]
    problem=module.prepare_joint_problem(parent,r,blocks,module.JointSelectionConfig(count=4))
    trace=problem.greedy([4,1]);assert trace['selected_ids']==[1,4] and trace['effective_count']==2 and trace['requested_count']==4
    assert problem.gain([1,4])==0.
    for original,before in zip((parent,r,*blocks.values()),saved):assert np.array_equal(original,before)
    assert problem.greedy([])['selected_ids']==[]


def test_positive_gate_ridge_can_reward_duplicate_gate_penalty_sharing():
    module=api();parent=np.zeros((3,0));r=np.ones(3);one=np.ones((3,1))
    problem=module.prepare_joint_problem(parent,r,{0:one,1:one.copy()},module.JointSelectionConfig(parent_ridge=0.))
    assert problem.marginal_gain([0],1)>0.


@pytest.mark.parametrize('bad',[dict(count=True),dict(parent_ridge=-1.),dict(gate_ridge=0.),dict(parent_ridge=float('nan')),
    dict(condition_on_parent=1),dict(column_scaling='after_residualization'),dict(solver='inverse')])
def test_invalid_numerical_policy_is_rejected(bad):
    with pytest.raises(ValueError):api().JointSelectionConfig(**bad)


@pytest.mark.parametrize('kind,rank',[('duplicate',1),('zero',0),('full_row',7)])
@pytest.mark.parametrize('ridge',[0.,.01])
def test_rank_deficient_and_full_row_parent_use_the_declared_original_span(kind,rank,ridge):
    module=api();rng=np.random.default_rng(7129);r=rng.normal(size=7)
    parent=np.column_stack([np.ones(7),np.ones(7)*3,np.zeros(7)]) if kind=='duplicate' else np.zeros((7,3)) if kind=='zero' else np.eye(7)
    blocks={0:rng.normal(size=(7,2)),1:np.zeros((7,2))}
    problem=module.prepare_joint_problem(parent,r,blocks,module.JointSelectionConfig(parent_ridge=ridge))
    assert problem.parent_rank==rank
    for ids in ((),(0,),(1,),(0,1)):
        assert problem.objective(ids)==pytest.approx(dense_objective(parent,r,blocks,ids,ridge,.01),abs=1e-10)
    if kind=='full_row' and ridge==0:assert problem.gain([0,1])==0.


def test_non_tied_candidate_relabeling_and_row_permutation_preserve_selected_functional_blocks():
    module=api();rng=np.random.default_rng(518);parent=rng.normal(size=(21,3));r=rng.normal(size=21)
    blocks={unit:rng.normal(size=(21,2)) for unit in (2,6,11,14)};config=module.JointSelectionConfig(count=2)
    original=module.prepare_joint_problem(parent,r,blocks,config);chosen=original.greedy(list(blocks))
    order=rng.permutation(21);mapping={2:19,6:3,11:8,14:5}
    permuted=module.prepare_joint_problem(parent[order],r[order],{mapping[u]:b[order] for u,b in blocks.items()},config)
    second=permuted.greedy(list(reversed(list(mapping.values()))))
    assert set(second['selected_ids'])=={mapping[u] for u in chosen['selected_ids']}
    assert second['predicted_gain']==pytest.approx(chosen['predicted_gain'],abs=1e-10)


def test_prepared_problem_is_independent_of_later_caller_array_mutation():
    module=api();rng=np.random.default_rng(583);parent=rng.normal(size=(9,2));r=rng.normal(size=9);block=rng.normal(size=(9,2))
    problem=module.prepare_joint_problem(parent,r,{0:block},module.JointSelectionConfig(count=1));before=problem.gain([0])
    parent[:]=0;r[:]=0;block[:]=0
    assert problem.gain([0])==before


@pytest.mark.parametrize('change',['parent_dtype','residual_dtype','block_rows','nonfinite','boolean_id','duplicate_subset'])
def test_malformed_array_or_candidate_geometry_fails_without_a_fallback(change):
    module=api();parent=np.ones((5,1));r=np.arange(5,dtype=np.float64);blocks={0:np.ones((5,1))}
    if change=='parent_dtype':parent=parent.astype(np.float32)
    elif change=='residual_dtype':r=r.astype(np.float32)
    elif change=='block_rows':blocks[0]=np.ones((4,1))
    elif change=='nonfinite':blocks[0][0,0]=np.inf
    elif change=='boolean_id':blocks={True:blocks[0]}
    if change=='duplicate_subset':
        problem=module.prepare_joint_problem(parent,r,blocks,module.JointSelectionConfig())
        with pytest.raises(ValueError):problem.gain([0,0])
    else:
        with pytest.raises(ValueError):module.prepare_joint_problem(parent,r,blocks,module.JointSelectionConfig())
