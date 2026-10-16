"""Teacher-level inference from bounded saved-row fixtures."""
import copy
import numpy as np
import pytest


def rows(effects=(1.,2.,3.), repetitions=2):
    predictions={}; runs=[]
    for i,effect in enumerate(effects):
        for j in range(repetitions):
            rid=f'{i}-{j}'
            predictions[rid]=dict(run_id=rid,row_id=[1,2],observations=dict(y=[0.,0.],
                noise_free_target=[0.,0.],group_id=[1,2]),
                predictions={'top':[0.,0.],'random':[np.sqrt(abs(effect))]*2})
            if effect<0: predictions[rid]['predictions']['top'],predictions[rid]['predictions']['random']=predictions[rid]['predictions']['random'],predictions[rid]['predictions']['top']
            runs.append(dict(run_id=rid,teacher_id=f't{i}',unit_id=f'u{i}',status='completed',f_target_variance=1.))
    return predictions,runs


def policy(**kw):
    return dict(bootstrap_seed=12,bootstrap_iterations=1000,endpoint_tolerance=1.,alpha=.05,inference_supported=True,**kw)


def test_repetitions_are_not_independent_teachers():
    from treeneuron.analysis.primary import aggregate_teachers, summarize_effects
    predictions,runs=rows()
    teachers=aggregate_teachers(predictions,runs)
    assert len(teachers)==3
    assert [r['D'] for r in teachers]==pytest.approx([1.,2.,3.])
    duplicated=aggregate_teachers(predictions,runs+runs)
    assert duplicated==teachers
    summary=summarize_effects([r['D'] for r in teachers],policy())
    assert summary['independent_count']==summary['bootstrap_unit_count']==3
    assert summary['mean']==pytest.approx(2.)
    assert summary['median']==pytest.approx(2.)


def test_missing_invalid_pair_counts_teacher_without_zero_fill():
    from treeneuron.analysis.primary import aggregate_teachers
    predictions,runs=rows(); del predictions['0-1']; predictions['1-0']['predictions']['top']=[float('nan')]*2
    teachers=aggregate_teachers(predictions,runs)
    assert len(teachers)==3
    assert teachers[0]['D'] is None and teachers[1]['D'] is None
    assert teachers[0]['status']=='incomplete' and teachers[1]['status']=='invalid'
    assert teachers[2]['D']==pytest.approx(3.)


@pytest.mark.parametrize('effects',[(1.,),(1.,1.,1.),(0.,0.,0.)])
def test_degenerate_intervals_cannot_establish_superiority(effects):
    from treeneuron.analysis.primary import summarize_effects
    result=summarize_effects(effects,policy())
    assert not result['superiority'] and result['status']=='degenerate'
    assert result['interval'] is None


def test_positive_sign_favors_top_and_distinguishes_negative_inconclusive():
    from treeneuron.analysis.primary import summarize_effects
    assert summarize_effects([1.,2.,3.,4.,5.],policy())['interpretation']=='positive'
    assert summarize_effects([-1.,-2.,-3.,-4.,-5.],policy())['interpretation']=='negative'
    assert summarize_effects([-2.,-1.,0.,1.,2.],policy())['interpretation']=='inconclusive'
    assert not summarize_effects([1.,2.,3.],dict(policy(),inference_supported=False))['superiority']
    assert not summarize_effects([1.,2.,3.],dict(policy(),endpoint_tolerance=1e-30))['superiority']


def test_exact_metric_reconstruction_and_degenerate_scale():
    from treeneuron.analysis.primary import aggregate_teachers
    predictions,runs=rows((0.,),1)
    predictions['0-0']['observations']['y']=[1.,3.]
    predictions['0-0']['predictions']={'top':[1.,1.],'random':[0.,0.]}
    runs[0]['f_target_variance']=2.
    teacher=aggregate_teachers(predictions,runs)[0]
    assert teacher['metrics']['top']['mse']==2.
    assert teacher['metrics']['random']['mse']==5.
    assert teacher['D']==1.5 and teacher['D_original']==3.
    runs[0]['f_target_variance']=0.
    teacher=aggregate_teachers(predictions,runs)[0]
    assert teacher['D'] is None and teacher['D_original']==3.
    assert teacher['status']=='degenerate_risk'


def test_conflicting_duplicate_identity_rejected():
    from treeneuron.analysis.primary import aggregate_teachers
    predictions,runs=rows(); wrong=copy.deepcopy(runs[0]); wrong['teacher_id']='other'
    with pytest.raises(ValueError,match='duplicate'): aggregate_teachers(predictions,runs+[wrong])


def test_only_declared_holm_family_is_adjusted():
    from treeneuron.analysis.primary import holm_adjust
    assert holm_adjust({'H1':.01,'static':.03,'other':.0001},[['H1','static']])=={'H1':.02,'static':.03}
    assert holm_adjust({'H1':.01},[])=={}


def test_policy_rejects_undeclared_methods_and_overlapping_families():
    from treeneuron.protocol import confirmation_policy
    policy=confirmation_policy({})
    assert policy['holm_families']==[] and policy['gate_threshold']==.5
    for families in ([['mystery:top']],[['random:top'],['random:top']]):
        with pytest.raises(ValueError): confirmation_policy({'confirmation_policy':dict(policy,holm_families=families)})

from test_protocol import repository, prepared, candidate


def test_saved_primary_counts_entire_unobserved_cohort(candidate):
    from treeneuron.protocol import freeze
    from treeneuron.analysis.primary import analyze_primary
    import json
    root,decision,sources,manifests=candidate
    frozen=freeze(decision,sources,root/'protocol/frozen.yaml')
    directory=frozen.path.parent/'confirmation'/frozen.protocol_hash; directory.mkdir(parents=True)
    ledger=dict(protocol_hash=frozen.protocol_hash,status='incomplete',costs={},runs=[
        dict(unit_id=u['unit_id'],repetition=j,run_id=rid,status='failed',attempts=[])
        for u in sources['cohort']['units'] for j,rid in enumerate(u['run_ids'])])
    (directory/'confirmation.json').write_text(json.dumps(ledger))
    report=analyze_primary(directory,frozen)
    assert report['status']=='incomplete' and not report['primary']['superiority']
    assert report['declared_teachers']==report['observed_teachers']==decision.K
    assert report['valid_teachers']==0 and report['valid_runs']==0
    assert {'ordinary_upper:static_partial','ordinary_upper:widen_upper'}<=set(report['controls'])
    assert all(not c['superiority'] and c['claim']=='descriptive' for c in report['controls'].values())
    assert all(r['D'] is None for r in report['teachers'])
    assert len(list(directory.glob('primary-*.json')))==1
    import json
    receipt=json.loads(next((directory/'analysis-work').glob('*.json')).read_text())
    assert receipt['seconds']>0 and receipt['total_study_work_seconds'] is None
