"""Retained F statistics and finite diagnostic candidate bindings."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from test_train import training_case,cpu_training_evidence


def test_scored_parent_roundtrip_binds_all_arrays_and_rejects_tampering(training_case,tmp_path):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.diagnostics import score_parent
    from treeneuron.research.opportunity_inputs import save_scores,load_scores
    data,config,paths=training_case
    view=fitting_view(data);parent=train.fit_parent(view,config,paths)
    scored=score_parent(parent,view.F)
    root=tmp_path/'scores';save_scores(root,scored)
    restored=load_scores(root,parent)
    assert restored.table.table_hash==scored.table.table_hash
    assert restored.parent_sha256==scored.parent_sha256 and restored.elapsed_ns==scored.elapsed_ns
    npz=root/'arrays.npz';npz.write_bytes(npz.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):
        load_scores(root,parent)


def test_horizon_slots_use_common_eligibility_and_score_rank_without_rng_motion(training_case,tmp_path):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.diagnostics import score_parent
    from treeneuron.research.opportunity_inputs import horizon_slots
    from test_checkpoint import assert_nested_equal
    data,config,paths=training_case
    view=fitting_view(data);parent=train.fit_parent(view,config,paths)
    table=score_parent(parent,view.F).table
    h=len(table.ids)
    table=replace(table,adam_scores=torch.arange(1,h+1,dtype=torch.float64),
        sgd_scores=torch.ones(h,dtype=torch.float64),eligible=torch.ones(h,dtype=torch.bool))
    table.eligible[0]=False
    rng=train.capture_rng()
    slots=horizon_slots(table,881)
    assert_nested_equal(train.capture_rng(),rng)
    ranked=list(range(h-1,0,-1))
    assert [s['unit_ids'] for s in slots[1:1+min(4,h-1)]]==[[i] for i in ranked[:4]]
    admitted=[s['unit_ids'][0] for s in slots[1:] if s['status']=='bound']
    assert len(set(admitted))==min(8,h-1) and 0 not in admitted
    assert len(slots)==9 and slots[0]['role']=='keep'
    assert slots==horizon_slots(table,881)
    assert sum(s['status']=='completed_ineligible' for s in slots)==8-min(8,h-1)


def test_pair_oracle_selects_only_eligible_pairs_with_declared_ties():
    from treeneuron.research.opportunity_inputs import pair_oracle
    rows=[dict(unit_ids=[0,1],status='completed_eligible',selected_g_mse=1.+5e-13),
          dict(unit_ids=[0,2],status='completed_eligible',selected_g_mse=1.),
          dict(unit_ids=[1,2],status='completed_ineligible'),
          dict(unit_ids=[3],status='completed_eligible',selected_g_mse=0.)]
    assert pair_oracle(rows)['unit_ids']==[0,1]
    assert pair_oracle(rows)['attempted_pair_count']==3
    assert pair_oracle(rows[2:])['status']=='completed_ineligible'
    with pytest.raises(ValueError,match='finite'):
        pair_oracle([dict(unit_ids=[0,1],status='completed_eligible',selected_g_mse=float('nan'))])
    failed=pair_oracle([rows[0],dict(unit_ids=[0,2],status='numerical_failure')])
    assert failed['status']=='irrecoverable_missing' and failed['unit_ids'] is None


def test_horizon_slots_sample_four_from_the_remaining_pool():
    from treeneuron.growth.score import ScoreTable
    from treeneuron.research.opportunity_inputs import horizon_slots
    h,p=12,3
    column=torch.ones(h,dtype=torch.float64)
    table=ScoreTable(torch.arange(h),torch.ones((h,p),dtype=torch.float64),
        torch.ones((h,p),dtype=torch.float64),torch.arange(1,h+1,dtype=torch.float64),
        column,torch.ones((h,p),dtype=torch.float64),column,column,column,
        torch.ones(h,dtype=torch.bool),('eligible',)*h,torch.zeros(h,dtype=torch.bool),
        torch.eye(p,dtype=torch.float64),.05,1.,1e-8,20)
    before=torch.get_rng_state().clone();slots=horizon_slots(table,881)
    assert torch.equal(torch.get_rng_state(),before)
    assert [s['unit_ids'][0] for s in slots[1:5]]==[11,10,9,8]
    sampled=[s['unit_ids'][0] for s in slots[5:]]
    assert len(set(sampled))==4 and set(sampled).issubset(set(range(8)))
    assert slots==horizon_slots(table,881)
