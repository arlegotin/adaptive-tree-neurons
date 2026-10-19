"""Changing only disk publication cadence preserves every scientific update."""
import numpy as np
import torch

from test_train import training_case,cpu_training_evidence,assert_state_equal


def test_G_grid_checkpoint_cadence_preserves_parent_and_grown_selected_states(training_case,tmp_path):
    from treeneuron import train
    from treeneuron.artifacts import create_run
    from treeneuron.config import resolve_experiment
    from treeneuron.growth.allocate import allocate
    from treeneuron.growth.insert import insert_growth
    from treeneuron.research.diagnostics import score_parent
    from treeneuron.research.roles import fitting_view
    data,base,_=training_case;view=fitting_view(data);parents=[];grown=[];curves=[]
    for cadence in (1,25):
        raw=base.to_dict();raw['training']['checkpoint_interval']=cadence
        config=resolve_experiment(raw);parent=train.fit_parent(view,config,create_run(tmp_path/f'parent-{cadence}',config.to_dict()))
        parents.append(train.pack_state(parent))
        scored=score_parent(parent,view.F)
        insert_growth(parent,allocate(scored.table,2,policy='random',seed=71))
        policy=train.TrainingPolicy.from_config(config,continuation=True)
        from dataclasses import replace
        state=train.continue_fit(parent,view,replace(policy,stop_by_patience=False),create_run(tmp_path/f'grown-{cadence}',config.to_dict()))
        grown.append(state)
        curves.append([(r['update'],r['metrics'],r['improved']) for r in state.history if r['type']=='evaluation'])
    from test_checkpoint import assert_nested_equal
    assert_nested_equal(parents[0]['tensors'],parents[1]['tensors'])
    assert_state_equal(grown[0],grown[1])
    np.testing.assert_array_equal(grown[0].predict(data.G.X),grown[1].predict(data.G.X))
    assert curves[0]==curves[1]
    assert grown[0].stage['updates_completed']==grown[1].stage['updates_completed']==6
