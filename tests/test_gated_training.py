"""Complete-state selection, age validation and sampler replay for v2 growth.

The breaks exercised here are weight-only restoration, implicit casting, stale
newborn ages, sampler restart and schedule restart. No predictive claim follows.
"""
import copy
import importlib.util
import random

import numpy as np
import pytest
import torch

from treeneuron.research.gated_models import parent_model, insert_gates, widen_ordinary


def api():
    assert importlib.util.find_spec('treeneuron.research.gated_training') is not None, 'Missing complete v2 training session'
    from treeneuron.research import gated_training
    return gated_training


def tensors():
    X=torch.linspace(-.4,.9,17*3,dtype=torch.float32).reshape(17,3)
    return X,torch.linspace(-.2,.6,17)


def session(*,scheduler='constant',static=False,weight_decay=.0001):
    torch.manual_seed(23);np.random.seed(23);random.seed(23)
    if static:
        from treeneuron.research.gated_models import StaticModel
        model=StaticModel(3,8,2,'centered',tau=1.,dropout=.1)
    else:model=parent_model(3,8,1.)
    return api().TrainingSession.create(model,n_rows=17,sampler_seed=47,
        lr=.003,weight_decay=weight_decay,scheduler=scheduler,horizon=20,batch_size=6)


def same(a,b):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


@pytest.mark.parametrize('scheduler',['constant','cosine'])
def test_step_matches_independent_adam_and_sampler_without_schedule_restart(scheduler):
    state=session(scheduler=scheduler);X,y=tensors();model=copy.deepcopy(state.model)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001,foreach=False)
    generator=torch.Generator().manual_seed(47);order=torch.randperm(17,generator=generator)
    cursor=0;epoch=0
    for u in range(4):
        if cursor==17:order=torch.randperm(17,generator=generator);cursor=0;epoch+=1
        rows=order[cursor:cursor+6];cursor+=len(rows)
        lr=.003 if scheduler=='constant' else .003*(1+np.cos(np.pi*u/20))/2
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss=(model(X[rows])-y[rows]).square().mean();loss.backward();opt.step()
        event=state.step(X,y)
        assert event['row_indices']==rows.tolist() and event['update']==u+1
        assert event['learning_rates']==pytest.approx([lr],rel=1e-14)
        assert event['F_online_mse']==float(loss.detach())
        for name,p in state.model.named_parameters():
            torch.testing.assert_close(p,dict(model.named_parameters())[name],rtol=0,atol=0)
    assert state.cursor==6 and state.epoch==1 and state.update==4
    assert state.scheduler.position==4


def test_snapshot_at_partial_epoch_replays_dropout_rng_optimizer_rows_and_loss_exactly():
    state=session(scheduler='cosine',static=True);X,y=tensors()
    state.step(X,y);state.step(X,y);saved=state.snapshot()
    assert saved['cursor']==12 and saved['update']==2
    event=state.step(X,y);after=state.snapshot()
    random.random();np.random.random(5);torch.rand(11)
    state.restore(saved)
    same(state.snapshot(),saved)
    replay=state.step(X,y)
    assert replay==event;same(state.snapshot(),after)
    saved['model'][next(iter(saved['model']))].zero_()
    assert not torch.equal(saved['model'][next(iter(saved['model']))],state.snapshot()['model'][next(iter(saved['model']))])


def test_patience_restores_update_zero_complete_state_and_keeps_paid_events():
    state=session(scheduler='cosine');X,y=tensors()
    with torch.no_grad():G_y=state.model(X).clone()
    # G equals the exact initial model; F differs, so every nonzero update loses.
    saved=state.snapshot();events=[]
    result=state.fit_stage(X,y,X,G_y,max_updates=8,validation_interval=1,patience=2,event_sink=events.append)
    assert result['selected_stage_update']==0 and result['executed_stage_updates']==2
    assert result['stop']=='patience' and result['selected_absolute_update']==0
    assert state.update==0 and state.cursor==0 and state.scheduler.position==0
    restored=state.snapshot()
    for key in saved:
        if key not in ('stage','training'):same(restored[key],saved[key])
    assert len([e for e in events if e['type']=='update'])==2
    assert [e['stage_update'] for e in events if e['type']=='evaluation']==[0,1,2]
    assert state.stage['complete'] is True and state.stage['updates_completed']==0
    again=[];state.fit_stage(X,y,X,G_y,max_updates=8,validation_interval=1,patience=2,event_sink=again.append)
    assert [e for e in again if e['type']=='update']==[e for e in events if e['type']=='update']


def test_stage_uses_nested_grid_and_endpoint_with_strict_earliest_ties():
    state=session(weight_decay=0.);X,y=tensors()
    # A zero residual and WD0 makes every complete update a true tie.
    with torch.no_grad():y=state.model(X).clone()
    events=[];r=state.fit_stage(X,y,X,y,max_updates=5,validation_interval=2,patience=20,event_sink=events.append)
    assert [e['stage_update'] for e in events if e['type']=='evaluation']==[0,2,4,5]
    assert r['selected_stage_update']==0 and r['executed_stage_updates']==5 and r['stop']=='max_updates'


@pytest.mark.parametrize('kind',['gate','ordinary'])
def test_growth_keeps_absolute_schedule_and_maps_original_and_newborn_ages(kind):
    state=session(scheduler='cosine');X,y=tensors()
    for _ in range(3):state.step(X,y)
    before=state.model(X).detach().clone();old={id(p) for p in state.model.parameters()}
    if kind=='gate':insert_gates(state.model,state.optimizer,torch.tensor([0,2]),torch.full((2,4),.05));model=state.model
    else:model=widen_ordinary(state.model,state.optimizer,10,seed=71)
    state.adopt_growth(model)
    torch.testing.assert_close(state.model(X),before,rtol=0,atol=0)
    assert state.scheduler.position==3 and len(state.scheduler.base_lrs)==2
    assert state.optimizer.param_groups[0]['lr']==state.optimizer.param_groups[1]['lr']
    assert state.optimizer.param_groups[0]['lr']==pytest.approx(.003*(1+np.cos(np.pi*3/20))/2)
    snapshot=state.snapshot()
    for name,p in state.model.named_parameters():assert snapshot['births'][name]==(0 if id(p) in old else 3)
    state.step(X,y);saved=state.snapshot();state.step(X,y);state.restore(saved)
    for name,p in state.model.named_parameters():assert float(state.optimizer.state[p]['step'])==(4 if id(p) in old else 1)


@pytest.mark.parametrize('kind',['dtype','negative_moment','age','model_dtype','bad_order','bad_cursor','schedule','rng'])
def test_corrupt_snapshot_rejects_before_mutating_live_state(kind):
    state=session();X,y=tensors();state.step(X,y);before=state.snapshot();bad=copy.deepcopy(before)
    if kind=='dtype':bad['optimizer']['state'][0]['exp_avg']=bad['optimizer']['state'][0]['exp_avg'].double()
    elif kind=='negative_moment':bad['optimizer']['state'][0]['exp_avg_sq'].flatten()[0]=-1
    elif kind=='age':bad['optimizer']['state'][0]['step'].zero_()
    elif kind=='model_dtype':
        name=next(iter(bad['model']));bad['model'][name]=bad['model'][name].double()
    elif kind=='bad_order':bad['order'][0]=bad['order'][1]
    elif kind=='bad_cursor':bad['cursor']=18
    elif kind=='schedule':bad['scheduler']['position']=0
    else:bad['rng']['torch_cpu']=torch.tensor([1,2],dtype=torch.uint8)
    with pytest.raises(ValueError):state.restore(bad)
    same(state.snapshot(),before)


@pytest.mark.parametrize('change',['target_shape','nonfinite','wrong_F_count'])
def test_training_geometry_is_checked_before_sampler_or_parameter_motion(change):
    state=session();X,y=tensors();before=state.snapshot()
    if change=='target_shape':y=y[:,None]
    elif change=='nonfinite':X[0,0]=float('inf')
    else:X=X[:-1];y=y[:-1]
    with pytest.raises(ValueError):state.step(X,y)
    same(state.snapshot(),before)


def test_wrong_feature_width_at_epoch_rollover_cannot_advance_sampler():
    state=session();X,y=tensors()
    for _ in range(3):state.step(X,y)
    before=state.snapshot();assert before['cursor']==17
    with pytest.raises(ValueError):state.step(X[:,:2],y)
    same(state.snapshot(),before)


@pytest.mark.parametrize('kind',['cursor_phase','epoch_phase','stage_progress','stage_nonfinite'])
def test_internally_inconsistent_complete_state_is_rejected_before_restoration(kind):
    state=session();X,y=tensors();state.step(X,y);before=state.snapshot();bad=copy.deepcopy(before)
    if kind=='cursor_phase':bad['cursor']=5
    elif kind=='epoch_phase':bad['epoch']=1
    elif kind=='stage_progress':bad['stage']=dict(start_update=0,updates_completed=0,bad_evaluations=0,best_mse=.1,complete=False)
    else:bad['stage']=dict(start_update=0,updates_completed=1,bad_evaluations=0,best_mse=float('nan'),complete=True)
    with pytest.raises(ValueError):state.restore(bad)
    same(state.snapshot(),before)


def test_full_stage_restores_the_actual_later_best_including_adam_and_rows():
    state=session(scheduler='cosine');X,y=tensors();origin=state.snapshot()
    # Set G to a future point on this exact trajectory; update2 must be best.
    state.step(X,y);state.step(X,y)
    state.model.eval()
    with torch.no_grad():future=state.model(X).clone()
    selected=state.snapshot();state.restore(origin);events=[]
    result=state.fit_stage(X,y,X,future,max_updates=5,validation_interval=1,patience=2,event_sink=events.append)
    assert result['selected_stage_update']==2 and result['executed_stage_updates']==4
    for key in selected:
        if key not in ('stage','training'):same(state.snapshot()[key],selected[key])
    assert len([e for e in events if e['type']=='update'])==4


def test_G_selection_uses_original_unit_float64_targets_without_broadcast_or_rounding():
    state=session();X,y=tensors();state.model.eval()
    with torch.no_grad():original=state.model(X).double()*3.25+17.125
    target=original+torch.full_like(original,1e-6)
    expected=float((original-target).square().mean());events=[]
    result=state.fit_stage(X,y,X,target,max_updates=3,validation_interval=1,patience=1,
        event_sink=events.append,target_mean=17.125,target_scale=3.25)
    assert result['selected_stage_update']==0
    assert result['G_mse']==expected and result['G_mse']>0
    assert events[0]['G_mse']==expected


@pytest.mark.parametrize('kind',['python_gauss','numpy_gauss'])
def test_nonfinite_cached_random_draw_cannot_enter_a_restored_training_state(kind):
    state=session();before=state.snapshot();bad=copy.deepcopy(before)
    if kind=='python_gauss':
        version,keys,_=bad['rng']['python'];bad['rng']['python']=(version,keys,float('nan'))
    else:bad['rng']['numpy']['has_gauss']=1;bad['rng']['numpy']['cached_gaussian']=float('nan')
    with pytest.raises(ValueError):state.restore(bad)
    same(state.snapshot(),before)
