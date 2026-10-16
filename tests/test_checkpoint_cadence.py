"""Snapshot cadence changes persistence work, never trajectories or G selection."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from test_train import cpu_training_evidence, training_case, events, assert_state_equal
from treeneuron import train
from treeneuron.artifacts import create_run
from treeneuron.checkpoint import load_checkpoint, save_checkpoint
from treeneuron.config import resolve_experiment


def cadence_case(training_case, interval):
    data, config, paths=training_case
    raw=config.to_dict();raw['training']['checkpoint_interval']=interval
    config=resolve_experiment(raw)
    return data,config,create_run(paths.directory.parent,config.to_dict())


def test_cadence_retains_all_updates_and_every_G_boundary(training_case):
    data,legacy,legacy_paths=training_case
    reference=train.fit_parent(data,legacy,legacy_paths)
    data,config,paths=cadence_case(training_case,3)
    actual=train.fit_parent(data,config,paths)
    assert_state_equal(reference,actual)
    assert [e for e in events(paths) if e['type']=='evaluation']==[e for e in events(legacy_paths) if e['type']=='evaluation']
    assert [e for e in events(paths) if e['type']=='update']==[e for e in events(legacy_paths) if e['type']=='update']
    assert [e['update'] for e in events(paths) if e['type']=='checkpoint']==[0,2,3,4,6]
    assert config.training.checkpoint_interval==3
    assert actual.stage['policy']['checkpoint_interval']==3


def test_interruption_between_snapshots_publishes_complete_update_and_resumes(training_case,monkeypatch):
    data,config,baseline=cadence_case(training_case,3)
    reference=train.fit_parent(data,config,baseline)
    attempt=create_run(baseline.directory.parent,config.to_dict())
    append=train.append_event
    def stop(paths,event):
        append(paths,event)
        if event['type']=='update' and event['update']==1:
            raise KeyboardInterrupt('between durable snapshot intervals')
    monkeypatch.setattr(train,'append_event',stop)
    with pytest.raises(KeyboardInterrupt):train.fit_parent(data,config,attempt)
    monkeypatch.setattr(train,'append_event',append)
    restored=load_checkpoint(attempt.directory/'checkpoints/last.json',expected_hashes=reference.identity_hashes,device='cpu')
    assert restored.update==1
    assert restored.stage['policy']['checkpoint_interval']==3
    retry=create_run(baseline.directory.parent,config.to_dict())
    actual=train.continue_fit(restored,data,train.TrainingPolicy.from_config(config),retry)
    assert_state_equal(reference,actual)
    np.testing.assert_array_equal(reference.predict(data.G.X),actual.predict(data.G.X))
    assert [e['row_ids'] for e in events(retry) if e['type']=='update']==[e['row_ids'] for e in events(baseline) if e['type']=='update'][1:]


def test_mid_optimizer_interruption_keeps_previous_durable_state(training_case,monkeypatch):
    data,config,baseline=cadence_case(training_case,3)
    reference=train.fit_parent(data,config,baseline)
    attempt=create_run(baseline.directory.parent,config.to_dict())
    step=torch.optim.AdamW.step
    def stop(optimizer,*args,**kwargs):
        step(optimizer,*args,**kwargs)
        raise KeyboardInterrupt('optimizer partially completed bookkeeping')
    monkeypatch.setattr(torch.optim.AdamW,'step',stop)
    with pytest.raises(KeyboardInterrupt):train.fit_parent(data,config,attempt)
    monkeypatch.setattr(torch.optim.AdamW,'step',step)
    restored=load_checkpoint(attempt.directory/'checkpoints/last.json',expected_hashes=reference.identity_hashes,device='cpu')
    assert restored.update==0
    retry=create_run(baseline.directory.parent,config.to_dict())
    actual=train.continue_fit(restored,data,train.TrainingPolicy.from_config(config),retry)
    assert_state_equal(reference,actual)


def test_cadence_mismatch_rejects_save_and_continuation(training_case):
    data,config,paths=cadence_case(training_case,3)
    state=train.fit_parent(data,config,paths)
    policy=replace(train.TrainingPolicy.from_config(config,continuation=True),checkpoint_interval=1)
    with pytest.raises(ValueError,match='paired'):
        train.continue_fit(state,data,policy,create_run(paths.directory.parent,config.to_dict()))
    state.stage['policy']['checkpoint_interval']=1
    with pytest.raises(ValueError,match='cadence'):
        save_checkpoint(state,paths.directory/'bad.json')


@pytest.mark.parametrize('bad',[0,-1,True,1.5])
def test_checkpoint_interval_is_strict_positive_integer(training_case,bad):
    with pytest.raises(ValueError):cadence_case(training_case,bad)
