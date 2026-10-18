"""Opportunity screening uses fixed arms, paired repeats and declared references."""
import copy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def screen():
    from treeneuron.research.config import expand,load_research_config
    study=expand(load_research_config(Path('configs/research/v2/M2-opportunity.json'))).to_dict()
    cases=[c for c in study['attempts']+study['conditional_attempts'] if c['teacher_index']<4
        and c['matrix'] in ('M2a','M2e_neural','M2e_head')]
    outcomes={c['attempt_id']:dict(status='completed_ineligible') if c['admission']=='missing_snapshot_fallback'
        else dict(status='completed_eligible',V_NMSE=1.) for c in cases}
    return study,cases,outcomes


def change(cases,outcomes,predicate,**value):
    for c in cases:
        if predicate(c):outcomes[c['attempt_id']]=dict(value)


def analyze(screen):
    from treeneuron.research.opportunity_analysis import opportunity_summary
    study,cases,outcomes=screen
    return opportunity_summary(cases,outcomes,study['opportunity_decision_contract'],teachers=range(4))


def test_decision_contract_fixes_registry_bootstrap_and_limited_recipe_scope(screen):
    study,_,_=screen;c=study['opportunity_decision_contract']
    assert c['bootstrap_iterations']==20000 and len(c['contrasts'])==24
    assert c['minimum_relative_reduction']==.02 and c['minimum_qualifying_teachers']==2
    assert [r['role'] for r in c['contrasts'][:4]]==['top','random','warm_lower','warm_upper']
    late=[r for r in c['contrasts'] if r['snapshot']=='late_G_best' and r['treatment']=='continued_stream']
    assert all(len(r['ordinary_recipes'])==5 for r in late)
    assert all(not r['ordinary_recipes'] for r in c['contrasts'] if r not in late)


def test_mean_repeats_precedes_reference_selection_and_never_selects_best_seed(screen):
    study,cases,outcomes=screen
    change(cases,outcomes,lambda c:c['matrix']=='M2e_neural' and c['recipe']=='constant_001'
        and c['repetition']==0,status='completed_eligible',V_NMSE=.4)
    change(cases,outcomes,lambda c:c['matrix']=='M2e_neural' and c['recipe']=='constant_001'
        and c['repetition']==1,status='completed_eligible',V_NMSE=1.6)
    result=analyze(screen)
    assert result['status']=='completed_eligible'
    assert all(r['mean_relative_benefit']==0 for r in result['contrasts'])
    assert result['extension_required'] is True
    assert all(t['reference']=='matched_keep' for r in result['contrasts'] for t in r['teachers'])


def test_one_fixed_arm_must_qualify_no_union_of_different_teacher_winners(screen):
    study,cases,outcomes=screen
    change(cases,outcomes,lambda c:c['matrix']=='M2a' and c['admission']=='initial'
        and c['role'] in ('top','warm_lower'),status='completed_eligible',V_NMSE=1.1)
    change(cases,outcomes,lambda c:c['matrix']=='M2a' and c['admission']=='initial'
        and ((c['role']=='top' and c['teacher_index']==0) or (c['role']=='warm_lower' and c['teacher_index']==1)),
        status='completed_eligible',V_NMSE=.9)
    result=analyze(screen)
    assert result['maximum_qualifying_teachers']==1 and result['extension_required']
    assert result['gated_growth_maximum_qualifying_teachers']==1
    assert result['ordinary_capacity_maximum_qualifying_teachers']==1


def test_warm_only_opportunity_is_capacity_and_bootstrap_keeps_one_fixed_arm(screen):
    study,cases,outcomes=screen;effects=np.array([.01,.02,.04,.07])
    change(cases,outcomes,lambda c:c['matrix']=='M2a' and c['snapshot']=='actual_250'
        and c['treatment']=='continued_stream' and c['role']=='warm_lower' and c['admission']=='initial',
        status='completed_eligible',V_NMSE=.99)
    for c in cases:
        if c['matrix']=='M2a' and c['snapshot']=='actual_250' and c['treatment']=='continued_stream' and c['role']=='warm_lower' and c['admission']=='initial':
            outcomes[c['attempt_id']]['V_NMSE']=float(1-effects[c['teacher_index']])
    result=analyze(screen)
    assert result['best_contrast_id']=='actual_250/continued_stream/warm_lower'
    assert result['gated_growth_maximum_qualifying_teachers']==0
    assert result['ordinary_capacity_maximum_qualifying_teachers']==3
    rng=np.random.default_rng(study['opportunity_decision_contract']['bootstrap_seed'])
    samples=(1-(1-effects))[rng.integers(4,size=(20000,4))].mean(axis=1)
    assert result['best_fixed_interval']==list(np.quantile(samples,[.025,.975]))
    assert result==analyze(screen)


def test_missing_snapshot_uses_required_fallback_by_availability_not_risk(screen):
    study,cases,outcomes=screen
    target=lambda c:c['matrix']=='M2a' and c['snapshot']=='actual_1000' and c['teacher_index']==0 and c['repetition']==0
    change(cases,outcomes,lambda c:target(c) and c['admission']=='initial',status='completed_ineligible')
    change(cases,outcomes,lambda c:target(c) and c['admission']=='missing_snapshot_fallback',status='completed_eligible',V_NMSE=1.)
    result=analyze(screen)
    assert result['status']=='completed_eligible'
    records=[r for r in result['contrasts'] if r['snapshot']=='actual_1000']
    assert all(r['teachers'][0]['origins']==['fixed_parent_budget_fallback','original'] for r in records)
    # Both origins available is a semantic error, not permission to choose one.
    change(cases,outcomes,lambda c:target(c) and c['admission']=='initial',status='completed_eligible',V_NMSE=.1)
    assert analyze(screen)['status']=='irrecoverable_missing'


def test_missing_or_failed_evidence_blocks_decision_without_dropping_teacher(screen):
    study,cases,outcomes=screen
    del outcomes[cases[0]['attempt_id']]
    assert analyze(screen)['status']=='irrecoverable_missing'
    outcomes[cases[0]['attempt_id']]=dict(status='numerical_failure')
    assert analyze(screen)['status']=='irrecoverable_missing'


def test_strong_late_recipe_does_not_become_an_early_or_new_stream_reference(screen):
    study,cases,outcomes=screen
    change(cases,outcomes,lambda c:c['matrix']=='M2e_neural' and c['recipe']=='constant_001',
        status='completed_eligible',V_NMSE=.1)
    result=analyze(screen)
    for row in result['contrasts']:
        late=row['snapshot']=='late_G_best' and row['treatment']=='continued_stream'
        assert row['mean_relative_benefit']==pytest.approx(-9. if late else 0.)


def test_extension_worker_reads_only_declared_V_and_freezes_its_single_decision(screen,tmp_path):
    from treeneuron.research.opportunity_graph import compile_bounded_graph
    from treeneuron.research.opportunity_workers import OpportunityWorkers
    study,cases,outcomes=screen;graph=compile_bounded_graph(study)
    job=next(j for j in graph['attempts'] if j['kind']=='extension_decision')
    dependencies={}
    for case in cases:
        value=copy.deepcopy(outcomes[case['attempt_id']])
        if value['status']=='completed_eligible':
            risk=value.pop('V_NMSE');value['metrics']={'selected':{'V':{'normalized_mse':risk},
                'E':{'normalized_mse':float('nan')}}}
        dependencies['evaluate:'+case['attempt_id']]=dict(value=value)
    result=OpportunityWorkers(graph)(job,tmp_path,dependencies,0)
    assert result['status']=='completed_eligible' and result['summary']['extension_required']
    assert result['summary']['maximum_qualifying_teachers']==0
    assert (tmp_path/'decision.json').exists()


def test_inclusive_two_percent_boundary_uses_risk_inequality(screen):
    study,cases,outcomes=screen;reference=11/997
    for value in outcomes.values():
        if value['status']=='completed_eligible':value['V_NMSE']=reference
    for c in cases:
        if c['matrix']=='M2a' and c['snapshot']=='actual_250' and c['treatment']=='continued_stream' and c['role']=='top' and c['admission']=='initial':
            outcomes[c['attempt_id']]['V_NMSE']=reference*(.98 if c['teacher_index']<2 else .99)
    result=analyze(screen)
    assert result['maximum_qualifying_teachers']==2
    assert result['extension_required'] is False


def test_exhausted_eight_teacher_extension_cannot_be_requested_again(screen):
    from treeneuron.research.opportunity_analysis import opportunity_summary
    study,_,_=screen
    cases=[c for c in study['attempts']+study['conditional_attempts']
        if c['matrix'] in ('M2a','M2e_neural','M2e_head')]
    outcomes={c['attempt_id']:dict(status='completed_ineligible') if 'fallback' in c['admission']
        else dict(status='completed_eligible',V_NMSE=1.) for c in cases}
    result=opportunity_summary(cases,outcomes,study['opportunity_decision_contract'],teachers=range(8))
    assert result['extension_required'] is False
    assert result['extension_rule_triggered'] is True and result['extension_exhausted'] is True


def test_bootstrap_margin_endpoint_uses_same_inclusive_risk_boundary(screen):
    study,cases,outcomes=screen;reference=11/997
    for value in outcomes.values():
        if value['status']=='completed_eligible':value['V_NMSE']=reference
    ratios=[.98,.98,1.,1.01]
    for c in cases:
        if c['matrix']=='M2a' and c['snapshot']=='actual_250' and c['treatment']=='continued_stream' and c['role']=='top' and c['admission']=='initial':
            outcomes[c['attempt_id']]['V_NMSE']=reference*ratios[c['teacher_index']]
    result=analyze(screen)
    assert result['maximum_qualifying_teachers']==2
    assert result['extension_required'] is True
    assert result['extension_triggers']['interval_covers_zero_and_margin'] is True
    assert result['best_fixed_risk_centered_intervals']['margin'][1]==0.
