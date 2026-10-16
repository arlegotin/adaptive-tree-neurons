"""Trained Adam history cannot silently become lazy during save or recovery."""
import hashlib
import json

import numpy as np
import pytest
import torch

from test_train import training_case, cpu_training_evidence, assert_state_equal
from test_checkpoint import assert_nested_equal
from test_optimizer_growth import allocation
from treeneuron import train, checkpoint
from treeneuron.artifacts import create_run
from treeneuron.baselines.causal import widen_parent
from treeneuron.config import resolve_experiment
from treeneuron.data.transforms import FittedTransform
from treeneuron.data.types import DevelopmentData, ObservationSet
from treeneuron.growth.insert import insert_growth
from treeneuron.models.static import StaticRegressor


def make_state(training_case, family, *, zero=False):
    data,config,paths=training_case
    if family in ('centered','direct_leaf'):
        value=config.to_dict()
        value['model'].update(coordinates=family,static_count=2)
        config=resolve_experiment(value)
        paths=create_run(paths.directory.parent,config.to_dict())
    if zero:
        torch.manual_seed(config.seeds.optimizer)
        initial=(StaticRegressor(2,4,2,family) if family in ('centered','direct_leaf')
                 else train.SelectiveRegressor(2,4)).eval()
        transform=FittedTransform.fit(data.F.X,data.F.y)
        with torch.no_grad():
            y=transform.inverse_y(initial(torch.tensor(transform.transform_X(data.G.X),dtype=torch.float32)).numpy())
        data=DevelopmentData(data.F,ObservationSet(data.G.X,y,data.G.row_id,data.G.group_id),data.V,data.teacher_id)
    state=train.fit_parent(data,config,paths)
    assert state.update==0 if zero else state.update>0
    if family.startswith('widened'): widen_parent(state,2,42)
    if family.startswith('grown'): insert_growth(state,allocation())
    return state,data,paths


def target_name(state,family):
    return ('layer.added.weight' if family=='widened_new' else
            'layer.gate.weight' if family=='grown_new' else next(iter(dict(state.model.named_parameters()))))


def saved_identifier(snapshot,name):
    return next(identifier for group in snapshot['tensors']['optimizer']['param_groups']
                for n,identifier in zip(group['param_names'],group['params']) if n==name)


def damage(records,key,defect):
    if defect=='deleted': del records[key]
    else: records[key]={}


@pytest.mark.parametrize('family',['ordinary','centered','direct_leaf','widened_old','widened_new','grown_old','grown_new'])
@pytest.mark.parametrize('defect',['deleted','empty'])
def test_missing_history_rejected_before_reconstruction(training_case,monkeypatch,family,defect):
    state,_,_=make_state(training_case,family)
    snapshot=train.pack_state(state)
    identifier=saved_identifier(snapshot,target_name(state,family))
    damage(snapshot['tensors']['optimizer']['state'],identifier,defect)
    calls=[]
    original=torch.optim.AdamW.load_state_dict
    def observed(self,*args,**kwargs):
        calls.append(True)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(torch.optim.AdamW,'load_state_dict',observed)
    before=train.capture_rng()
    with pytest.raises(ValueError,match='Adam'):
        train.unpack_state(snapshot,expected_hashes=state.identity_hashes,device='cpu')
    assert not calls
    assert_nested_equal(train.capture_rng(),before)


@pytest.mark.parametrize('family',['ordinary','centered','direct_leaf','widened_old','widened_new','grown_old','grown_new'])
@pytest.mark.parametrize('defect',['deleted','empty'])
def test_public_save_rejects_missing_history_without_replacing_pointer(training_case,family,defect):
    state,_,paths=make_state(training_case,family)
    pointer=paths.directory/'history.json'
    checkpoint.save_checkpoint(state,pointer)
    previous=pointer.read_bytes()
    parameter=dict(state.model.named_parameters())[target_name(state,family)]
    damage(state.optimizer.state,parameter,defect)
    with pytest.raises(ValueError,match='Adam'): checkpoint.save_checkpoint(state,pointer)
    assert pointer.read_bytes()==previous


@pytest.mark.parametrize('operation',['widen','insert','continue'])
@pytest.mark.parametrize('defect',['deleted','empty'])
def test_public_preflight_rejects_reset_parent_history(training_case,operation,defect):
    state,data,paths=make_state(training_case,'ordinary')
    parameter=next(state.model.parameters())
    damage(state.optimizer.state,parameter,defect)
    original=state.model
    before=train.capture_rng()
    with pytest.raises(ValueError,match='Adam'):
        if operation=='widen':widen_parent(state,2,42)
        elif operation=='insert':insert_growth(state,allocation())
        else:
            train.continue_fit(state,data,train.TrainingPolicy.from_config(state.config,continuation=True),
                               create_run(paths.directory.parent,state.config.to_dict()))
    assert state.model is original
    assert_nested_equal(train.capture_rng(),before)


@pytest.mark.parametrize('family',['ordinary','centered','direct_leaf','widened_new','grown_new'])
def test_selected_zero_lazy_old_state_and_explicit_new_state_roundtrip(training_case,family):
    state,data,paths=make_state(training_case,family,zero=True)
    added=('layer.added.','head.added.') if family=='widened_new' else ('layer.gate.','layer.contrast.') if family=='grown_new' else ()
    for name,p in state.model.named_parameters():
        if name.startswith(added):assert state.optimizer.state[p]['step'].item()==0
        else:assert p not in state.optimizer.state
    pointer=paths.directory/'zero-history.json'
    checkpoint.save_checkpoint(state,pointer)
    restored=checkpoint.load_checkpoint(pointer,expected_hashes=state.identity_hashes,device='cpu')
    assert_state_equal(state,restored)
    x=torch.tensor(state.preprocessing.transform_X(data.F.X),dtype=torch.float32)
    for current in (state,restored):
        current.optimizer.zero_grad();current.model(x).square().mean().backward();current.optimizer.step()
        assert all(record['step'].item()==1 for record in current.optimizer.state.values())
    assert_state_equal(state,restored)


@pytest.mark.parametrize('family',['widened_new','grown_new'])
@pytest.mark.parametrize('defect',['deleted','empty'])
def test_explicit_new_step_zero_record_cannot_be_omitted(training_case,family,defect):
    state,_,paths=make_state(training_case,family,zero=True)
    snapshot=train.pack_state(state)
    name=target_name(state,family)
    damage(snapshot['tensors']['optimizer']['state'],saved_identifier(snapshot,name),defect)
    with pytest.raises(ValueError,match='Adam'):
        train.unpack_state(snapshot,expected_hashes=state.identity_hashes,device='cpu')
    damage(state.optimizer.state,dict(state.model.named_parameters())[name],defect)
    with pytest.raises(ValueError,match='Adam'):
        checkpoint.save_checkpoint(state,paths.directory/'bad-zero.json')


@pytest.mark.parametrize('family',['widened_new','grown_new'])
def test_fresh_and_inherited_adam_ages_remain_distinct(training_case,family):
    state,data,_=make_state(training_case,family)
    steps={name:state.optimizer.state[p]['step'].item() for name,p in state.model.named_parameters()}
    assert steps[target_name(state,family)]==0
    assert max(steps.values())==state.update>0
    restored=train.unpack_state(train.pack_state(state),expected_hashes=state.identity_hashes,device='cpu')
    x=torch.tensor(restored.preprocessing.transform_X(data.F.X),dtype=torch.float32)
    restored.optimizer.zero_grad();restored.model(x).square().mean().backward();restored.optimizer.step()
    assert all(restored.optimizer.state[p]['step'].item()==steps[name]+1 for name,p in restored.model.named_parameters())


@pytest.mark.parametrize('defect',['deleted','empty'])
@pytest.mark.parametrize('boundary',['save','load'])
def test_embedded_best_history_checked_before_publication_or_optimizer_load(training_case,monkeypatch,defect,boundary):
    state,_,paths=make_state(training_case,'ordinary')
    state.best_snapshot=train.pack_state(state)
    pointer=paths.directory/'embedded.json'
    checkpoint.save_checkpoint(state,pointer)
    previous=pointer.read_bytes()
    if boundary=='save':
        damage(state.best_snapshot['tensors']['optimizer']['state'],0,defect)
        with pytest.raises(ValueError,match='Adam'):checkpoint.save_checkpoint(state,pointer)
        assert pointer.read_bytes()==previous
    else:
        # Recompute hashes to exercise content validation, not checksum failure.
        descriptor_path=pointer.parent/json.loads(previous)['metadata']
        descriptor=json.loads(descriptor_path.read_text());payload=pointer.parent/descriptor['payload']
        tensors=torch.load(payload,weights_only=True)
        damage(tensors['best_state']['optimizer']['state'],0,defect)
        torch.save(tensors,payload)
        descriptor['sha256']=hashlib.sha256(payload.read_bytes()).hexdigest()
        descriptor_path.write_text(json.dumps(descriptor))
        root=json.loads(previous);root['sha256']=hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
        pointer.write_text(json.dumps(root))
        calls=[]
        original=torch.optim.AdamW.load_state_dict
        def observed(self,*args,**kwargs):calls.append(True);return original(self,*args,**kwargs)
        monkeypatch.setattr(torch.optim.AdamW,'load_state_dict',observed)
        before=train.capture_rng()
        with pytest.raises(ValueError,match='Adam'):
            checkpoint.load_checkpoint(pointer,expected_hashes=state.identity_hashes,device='cpu')
        assert not calls
        assert_nested_equal(train.capture_rng(),before)
