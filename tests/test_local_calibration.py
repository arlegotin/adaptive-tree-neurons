"""The score's limited gate-only first-step target is measured independently."""
import copy

import numpy as np
import pytest
import torch

from test_train import training_case,cpu_training_evidence
from test_checkpoint import assert_nested_equal


def test_F_only_local_calibration_matches_autograd_and_preserves_parent(training_case):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.diagnostics import score_parent
    from treeneuron.research.local_calibration import calibrate_local_score
    data,config,paths=training_case;view=fitting_view(data)
    parent=train.fit_parent(view,config,paths);scored=score_parent(parent,view.F)
    before,rng=train.pack_state(parent),train.capture_rng()
    result=calibrate_local_score(parent,view.F,scored,seed=123,minibatches=4)
    assert result.report['selection_labels']=='F_only'
    assert result.report['gate_gradient_max_absolute_error']<1e-12
    assert result.report['birth_contrast_gradient_max_absolute'] ==0.
    assert len(result.report['units'])==int(scored.table.eligible.sum())
    for row in result.report['units']:
        assert row['predicted_F_decrease']>0
        assert abs(row['observed_F_decrease']-row['predicted_F_decrease'])<1e-10+1e-3*row['predicted_F_decrease']
    assert result.arrays['minibatch_row_indices'].shape==(4,config.training.batch_size)
    with torch.no_grad():
        again=calibrate_local_score(parent,view.F,scored,seed=123,minibatches=4)
    for key in result.arrays:np.testing.assert_array_equal(result.arrays[key],again.arrays[key])
    assert_nested_equal(train.pack_state(parent),before)
    assert_nested_equal(train.capture_rng(),rng)


def test_no_eligible_score_is_retained_as_insensitive_not_success(training_case):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.diagnostics import score_parent
    from treeneuron.research.local_calibration import calibrate_local_score
    data,config,paths=training_case;view=fitting_view(data);parent=train.fit_parent(view,config,paths)
    with torch.no_grad():parent.model.head.weight.zero_()
    scored=score_parent(parent,view.F)
    result=calibrate_local_score(parent,view.F,scored,seed=123,minibatches=4)
    assert result.report['status']=='completed_ineligible'
    assert result.report['units']==[] and len(result.arrays['nominal_ids'])==config.model.h
    assert result.report['gate_gradient_max_absolute_error'] is None


def test_score_or_F_drift_rejects_before_diagnostic(training_case):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.diagnostics import score_parent
    from treeneuron.research.local_calibration import calibrate_local_score
    data,config,paths=training_case;view=fitting_view(data);parent=train.fit_parent(view,config,paths)
    scored=score_parent(parent,view.F)
    with pytest.raises(ValueError,match='F-only'):
        calibrate_local_score(parent,view.G,scored,seed=123,minibatches=4)
    scored.table.gradients[0,0]+=1
    with pytest.raises(ValueError,match='binding'):
        calibrate_local_score(parent,view.F,scored,seed=123,minibatches=4)
