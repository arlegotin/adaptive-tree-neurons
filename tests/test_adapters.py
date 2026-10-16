"""Additive spline and truthful adapter capability boundaries."""
import json
import numpy as np
import pandas as pd
import pytest
from test_simple_baselines import request


def test_spline_f_only_knots_additivity_original_units_and_reload(tmp_path):
    from treeneuron.baselines.simple import SplineRidgePredictor
    x=np.linspace(-2,2,31)
    X=np.column_stack([x,np.sin(x)])
    model=SplineRidgePredictor(.01,n_knots=3).fit(request(tmp_path,X,100+3*x*x+np.sin(x)))
    knots=[k.copy() for k in model.knots]
    prediction=model.predict(X,batch_size=2)
    assert prediction.shape==(31,) and np.isfinite(prediction).all()
    assert np.mean((prediction-(100+3*x*x+np.sin(x)))**2)<.1
    p=model.predict(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]))
    assert abs(p[0]+p[3]-p[1]-p[2])<1e-10
    model.predict(np.array([[1e2,-1e2]]))
    for a,b in zip(model.knots,knots): np.testing.assert_array_equal(a,b)
    model.save(tmp_path/'spline.json')
    restored=SplineRidgePredictor.load(tmp_path/'spline.json')
    np.testing.assert_array_equal(restored.predict(X),prediction)
    summary=model.resource_summary()
    assert summary['capabilities']['internal_validation'] is False
    assert summary['capabilities']['categories']=='unsupported; explicit error'
    assert summary['rss_bytes']['value'] is None and summary['rss_bytes']['reason']
    assert summary['alpha']==.01


@pytest.mark.parametrize('mutation',['nan','knots','width','basis','category'])
def test_spline_snapshot_strict_validation(tmp_path,mutation):
    from treeneuron.baselines.simple import SplineRidgePredictor
    path=tmp_path/'spline.json'
    SplineRidgePredictor(1.,n_knots=2).fit(request(tmp_path)).save(path)
    state=json.loads(path.read_text())
    if mutation=='nan': state['knots'][0]=[float('nan')]
    if mutation=='knots': state['knots'][0]=[2.,1.]
    if mutation=='width': state['coefficients']=[1.]
    if mutation=='basis': state['basis']='multivariable'
    if mutation=='category': state['transform']['columns'][0]={'name':'x0','kind':'categorical','categories':[['str','a']]}
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError): SplineRidgePredictor.load(path)


def test_spline_missingness_constant_and_unsupported_categories(tmp_path):
    from treeneuron.baselines.simple import SplineRidgePredictor
    model=SplineRidgePredictor(1.).fit(request(tmp_path,np.array([[1.],[np.nan],[1.]]),np.array([7.,7.,7.])))
    np.testing.assert_allclose(model.predict(np.array([[np.nan],[2.]])),[7.,7.])
    with pytest.raises(ValueError,match='numeric'):
        SplineRidgePredictor(1.).fit(request(tmp_path,pd.DataFrame({'kind':['a','b','c']})))
    for count in (True,0,-1,1.5):
        with pytest.raises(ValueError): SplineRidgePredictor(1.,n_knots=count)
    for alpha in (True,-1,np.inf):
        with pytest.raises(ValueError): SplineRidgePredictor(alpha)
    for batch in (0,True,-1):
        with pytest.raises(ValueError): model.predict(np.array([[1.]]),batch_size=batch)


def test_capabilities_do_not_claim_pending_adapters_verified():
    from treeneuron.baselines.registry import capabilities, AdapterCapability
    rows=capabilities()
    assert all(isinstance(row,AdapterCapability) for row in rows)
    by_method={r.method:r for r in rows}
    assert by_method['B6'].status=='implemented'
    assert 'restricted' in by_method['B6'].recipe
    assert by_method['B7'].status=='implemented' and by_method['B7'].unavailable_reason is None
    assert by_method['SplineRidge'].internal_selection is False
    assert by_method['SplineRidge'].fallback is False
    for row in rows:
        assert row.version and row.source
        if row.status in ('pending','unavailable'): assert row.unavailable_reason


def test_practical_cli_reports_pending_required_adapters_nonzero(tmp_path):
    from treeneuron.cli import main
    path=tmp_path/'matrix.json'
    assert main(['verify-baselines','--track','practical','--output',str(path)])==1
    report=json.loads(path.read_text())
    assert not report['implemented_checks_passed'] and report['adapter_matrix']
    assert all(row['unavailable_reason'] for row in report['adapter_matrix'] if row['status']!='verified')


def test_practical_matrix_retains_every_required_family():
    from treeneuron.baselines.registry import capabilities
    rows={r.method:r for r in capabilities()}
    assert {name for name,row in rows.items() if row.track=='practical' and row.required}=={'B8','B9','B10','B11','B14'}
    assert rows['B12'].status=='unavailable' and rows['B13'].status=='unavailable'


def test_spline_knots_are_f_quantiles_and_invalid_output_is_rejected(tmp_path):
    from treeneuron.baselines.simple import SplineRidgePredictor
    X=np.array([[0.],[1.],[2.],[3.],[4.]])
    model=SplineRidgePredictor(.1,n_knots=3).fit(request(tmp_path,X,np.array([10.,11.,14.,19.,26.])))
    np.testing.assert_allclose(model.knots[0],[-1/np.sqrt(2),0.,1/np.sqrt(2)])
    with pytest.raises(ValueError): model.predict(np.array([[np.inf]]))
    with pytest.raises(ValueError): model.predict(np.array([[1e200]]))
    assert model.predict(np.empty((0,1))).shape==(0,)


def test_causal_cli_invalid_required_adapter_returns_nonzero(tmp_path):
    from treeneuron.cli import main
    path=tmp_path/'invalid.json'
    assert main(['verify-baselines','--track','causal','--backend-report',str(tmp_path/'missing.json'),
                 '--artifacts-dir',str(tmp_path/'runs'),'--output',str(path)])==1
    report=json.loads(path.read_text())
    assert not report['implemented_checks_passed']
    assert any(row['required'] and row['status']=='invalid' and row['unavailable_reason'] for row in report['adapter_matrix'])
