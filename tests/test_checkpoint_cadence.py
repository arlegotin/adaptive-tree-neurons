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
    import sys
    def stop(frame,event,arg):
        if event=='line' and frame.f_code is train._run.__code__:
            state=frame.f_locals['state']
            if state.update==1 and frame.f_locals.get('checkpoint_safe') is True:
                raise KeyboardInterrupt('completed bookkeeping between durable intervals')
        return stop
    sys.settrace(stop)
    try:
        with pytest.raises(KeyboardInterrupt):train.fit_parent(data,config,attempt)
    finally:
        sys.settrace(None)
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


@pytest.mark.parametrize('event_type,update', [('update',1),('update',2),('evaluation',2)])
@pytest.mark.parametrize('boundary', ['before_history','before_append','after_append'])
def test_incomplete_event_bookkeeping_replays_previous_durable_state(training_case,monkeypatch,event_type,update,boundary):
    """No partial update/G ledger publication may advance recovery past its rows."""
    import sys
    data,config,baseline=cadence_case(training_case,3)
    reference=train.fit_parent(data,config,baseline)
    attempt=create_run(baseline.directory.parent,config.to_dict())
    append=train.append_event
    fired=[]
    def matches(event):
        return event.get('type')==event_type and event.get('update')==update
    def trace(frame,kind,arg):
        if kind=='call' and frame.f_code.co_name=='record' and frame.f_globals.get('__name__')==train.__name__ and matches(frame.f_locals['event']):
            fired.append(True)
            raise KeyboardInterrupt('before history bookkeeping')
        return trace
    def stop(paths,event):
        if matches(event) and boundary=='before_append':
            fired.append(True)
            raise KeyboardInterrupt('before event append')
        append(paths,event)
        if matches(event) and boundary=='after_append':
            fired.append(True)
            raise KeyboardInterrupt('append returned before bookkeeping completed')
    if boundary=='before_history': sys.settrace(trace)
    else: monkeypatch.setattr(train,'append_event',stop)
    try:
        with pytest.raises(KeyboardInterrupt):train.fit_parent(data,config,attempt)
    finally:
        sys.settrace(None)
        monkeypatch.setattr(train,'append_event',append)
    assert fired
    restored=load_checkpoint(attempt.directory/'checkpoints/last.json',expected_hashes=reference.identity_hashes,device='cpu')
    assert restored.update==0
    assert not [e for e in restored.history if e['type']=='update']
    retry=create_run(baseline.directory.parent,config.to_dict())
    actual=train.continue_fit(restored,data,train.TrainingPolicy.from_config(config),retry)
    assert_state_equal(reference,actual)
    np.testing.assert_array_equal(reference.predict(data.G.X),actual.predict(data.G.X))
    for event in ('update','evaluation'):
        expected=[e for e in events(baseline) if e['type']==event]
        if event=='evaluation':expected=expected[1:]  # update-zero selection is durable
        assert [e for e in events(retry) if e['type']==event]==expected
