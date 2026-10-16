"""Finite F-only mechanistic diagnostics, with independently known gate/leaf values."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from treeneuron.config import canonical_json
from treeneuron.data.controls import make_control
from treeneuron.data.synthetic import make_teacher
from treeneuron.experiments.paired import _diagnostics
from treeneuron.models.static import StaticRegressor


def diagnostic_fixture(coordinates='centered', count=2):
    model=StaticRegressor(2,2,count,coordinates)
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
        if count:
            if coordinates=='centered': model.layer.contrast.bias.fill_(2.)
            else:
                model.layer.left.bias.fill_(2.)
                model.layer.right.bias.fill_(-2.)
    state=SimpleNamespace(model=model,config=SimpleNamespace(model=SimpleNamespace(tau=1.)),
        preprocessing=SimpleNamespace(transform_X=lambda x:x,transform_y=lambda y:y))
    data=SimpleNamespace(F=SimpleNamespace(X=np.array([[3.,4.],[0.,0.],[-3.,-4.]]),y=np.zeros(3)))
    return state,data


@pytest.mark.parametrize('coordinates',['centered','direct_leaf'])
def test_gate_moments_quantiles_and_coordinate_equivalent_leaf_disagreement(coordinates):
    state,data=diagnostic_fixture(coordinates)
    record=_diagnostics(state,data)
    gate=record['gate']
    assert gate['status']=='observed'
    assert gate['variance']==0.
    assert gate['quantile_probabilities']==[0.,.25,.5,.75,1.]
    assert gate['quantiles']==[.5]*5
    assert gate['mean_g_one_minus_g']==.25
    assert record['leaf_disagreement']['rms']==4.
    assert record['leaf_disagreement']['coordinates']==coordinates
    assert record['input_norms']['raw']['max']==5.
    assert record['input_norms']['raw']['mean']==pytest.approx(10/3)
    assert record['input_norms']['standardized']['rms']==pytest.approx(np.sqrt(50/3))
    assert record['population']=='F' and record['role']=='diagnostic_only'
    canonical_json(record)


def test_varying_gate_variance_and_declared_quantiles():
    state,data=diagnostic_fixture()
    with torch.no_grad(): state.model.layer.gate.weight[:,0]=1.
    record=_diagnostics(state,data)
    values=1/(1+np.exp(-np.array([3.,0.,-3.])))
    assert record['gate']['variance']==pytest.approx(float(np.var(values)),abs=1e-7)
    # Two identical gates per F row: quantiles are over the declared pooled entries.
    assert record['gate']['quantiles']==pytest.approx(np.quantile(np.repeat(values,2),[0,.25,.5,.75,1]),abs=1e-7)
    assert record['gate']['mean_g_one_minus_g']==pytest.approx(np.mean(values*(1-values)),abs=1e-7)


@pytest.mark.parametrize('model_present',[False,True])
def test_absent_gates_and_regimes_are_explicitly_inapplicable(model_present):
    state,data=diagnostic_fixture(count=0)
    teacher=make_control('globally_affine',3,d=2,reference_seed=7)
    record=_diagnostics(state if model_present else None,data,teacher)
    for key in ('gate','leaf_disagreement','synthetic_regime_occupancy'):
        assert record[key]['status']=='not_applicable'
        assert record[key]['value'] is None
        assert record[key]['reason']
    canonical_json(record)


def test_synthetic_conditional_regime_counts_use_raw_F_only():
    state,data=diagnostic_fixture()
    teacher=make_control('conditional_slope_2d',3,d=2,reference_seed=7)
    record=_diagnostics(state,data,teacher)
    occupancy=record['synthetic_regime_occupancy']
    assert occupancy['status']=='observed'
    assert occupancy['rows']==3
    assert occupancy['counts']=={'left':1,'right':2}
    assert occupancy['fractions']=={'left':1/3,'right':2/3}
    assert occupancy['role']=='oracle_diagnostic_only'


def test_synthetic_teacher_component_occupancy_is_finite_and_diagnostic_only():
    state,data=diagnostic_fixture()
    teacher=make_teacher(3,d=2,reference_seed=7)
    record=_diagnostics(state,data,teacher)
    occupancy=record['synthetic_regime_occupancy']
    assert occupancy['role']=='oracle_diagnostic_only'
    assert occupancy['rows']==3
    assert len(occupancy['mean_gate_probability'])==4
    assert all(0<=v<=1 for v in occupancy['fraction_at_or_above_half'])
    canonical_json(record)
