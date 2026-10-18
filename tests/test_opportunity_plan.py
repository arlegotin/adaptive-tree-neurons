"""M2 freezes nominal and conditional work before eligibility or outcomes exist."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import pytest


def m2_config():
    from treeneuron.research.config import load_research_config
    return load_research_config(Path('configs/research/v2/M2-opportunity.json'))


def test_m2_nominal_counts_and_disclosed_contingencies():
    from treeneuron.research.config import expand
    plan = expand(m2_config()).to_dict()
    counts = Counter(a['matrix'] for a in plan['attempts'])
    assert counts == {'M2a': 240, 'M2b': 520, 'M2c': 72, 'M2d': 296,
                      'M2e_neural': 40, 'M2e_head': 8}
    assert len({a['attempt_id'] for a in plan['attempts']}) == 1176
    conditional = Counter(a['admission'] for a in plan['conditional_attempts'])
    assert conditional == {'primary_extension': 288, 'missing_snapshot_fallback': 160,
                           'extension_missing_snapshot_fallback': 160}
    all_cases = plan['attempts']+plan['conditional_attempts']
    assert len({a['attempt_id'] for a in all_cases}) == len(all_cases)
    assert plan['cost_inventory']['initial_ridge_solves'] == 48
    assert plan['cost_inventory']['initial_ordinary_neural_stages'] == 80
    assert plan['cost_inventory']['fallback_parent_ceiling'] == 16
    assert plan['protected_labels_accessed'] is False


def test_m2_pairing_slots_oracle_and_horizons_are_fixed():
    from treeneuron.research.config import expand
    p = expand(m2_config()).to_dict()
    cases = [a for a in p['attempts'] if a['teacher_index'] == 0 and a['repetition'] == 0]
    primary = [a for a in cases if a['population'] == 'primary']
    small = [a for a in cases if a['population'] == 'positive_small']
    assert len({a['teacher_seed'] for a in primary}) == 1
    assert len({a['data_seed'] for a in primary}) == 1
    assert len({a['optimizer_seed'] for a in primary}) == 1
    assert small[0]['teacher_seed'] != primary[0]['teacher_seed']
    assert small[0]['data_seed'] != primary[0]['data_seed']
    pairs = [a for a in small if len(a['unit_ids']) == 2]
    assert len(pairs) == 28
    assert len([a for a in pairs if 7 in a['unit_ids']]) == 7
    assert all(a['eligibility_policy'] == 'common_current_scorer_retain_nominal_skips' for a in pairs)
    horizons = [a for a in cases if a['matrix'] == 'M2c']
    assert len(horizons) == 9
    assert {a['slot'] for a in horizons} == {'keep',*(f'top-{i}' for i in range(4)),*(f'sampled-{i}' for i in range(4))}
    assert all(a['max_updates'] == 8000 and a['stop_by_patience'] is False
               and a['selection_prefixes'] == [0,300,1200,8000] for a in horizons)
    assert all('unit_ids' not in a for a in horizons if a['slot'] != 'keep')
    neural = [a for a in cases if a['matrix'] == 'M2e_neural']
    assert len(neural) == 5 and all(a['stop_by_patience'] for a in neural)
    cosine = next(a for a in neural if a['recipe'] == 'cosine_005')
    assert cosine['absolute_schedule_horizon'] == 16000
    heads = [a for a in cases if a['matrix'] == 'M2e_head']
    assert heads[0]['alphas'] == [0,1e-6,1e-4,1e-2,1,100]
    assert heads[0]['parent_recipe'] == 'legacy_005'
    recipes = {p['parent_recipe']:p['training'] for p in p['prerequisites']
               if p['population']=='primary' and p['admission']=='initial'}
    assert recipes['constant_001']['learning_rate'] == .001
    assert recipes['wd_1e4']['weight_decay'] == 1e-4
    assert recipes['wd_1e3']['weight_decay'] == 1e-3
    assert recipes['cosine_005']['scheduler'] == 'cosine'
    assert recipes['cosine_005']['min_learning_rate'] == 0
    assert all(a['snapshot']=='late_G_best' for a in primary if a['matrix']!='M2a')
    assert all(a['requested_count']==1 for a in horizons if a['slot']!='keep')


def test_m2_plan_rejects_changed_primary_design():
    from treeneuron.research.config import ResearchConfig, expand
    original = m2_config().to_dict()
    for key, value in (('teachers', 5), ('repetitions', 3)):
        raw = json.loads(json.dumps(original)); raw[key] = value
        with pytest.raises(ValueError):
            expand(ResearchConfig.from_dict(raw))
    raw = json.loads(json.dumps(original));raw['base_experiment']['growth']['m_requested'] = 8
    with pytest.raises(ValueError):
        expand(ResearchConfig.from_dict(raw))
    for section, key, value in (('training','adam_beta2',.95),('data','population','pure_noise'),
                                ('data','regime','correlated'),('model','coordinates','static_centered')):
        raw = json.loads(json.dumps(original));raw['base_experiment'][section][key]=value
        with pytest.raises(ValueError):
            expand(ResearchConfig.from_dict(raw))
    raw = json.loads(json.dumps(original)); raw['evaluation_rows']=16384
    with pytest.raises(ValueError):
        expand(ResearchConfig.from_dict(raw))


def test_M2_contract_binds_admissions_and_does_not_claim_paid_costs():
    from treeneuron.research.config import expand
    p = expand(m2_config()).to_dict()
    assert p['design_bindings']['binding_spec_sha256']
    assert p['design_bindings']['M2_execution_sha256']
    assert 'all_costs_paid' not in p['cost_inventory']
    extended_fallback=[a for a in p['conditional_attempts'] if a['admission']=='extension_missing_snapshot_fallback']
    assert all(a['requires_admissions']==['primary_extension','original_snapshot_unavailable'] for a in extended_fallback)
    small=[a for a in p['attempts'] if a['matrix']=='M2d']
    assert all(a['evaluation_rows']==16384 for a in small)
    assert p['selection_contract']['pair_oracle']['select_on']=='G'


def test_public_M2_plan_succeeds_and_run_rejects_before_creating_output(tmp_path):
    output = tmp_path/'plan.json'
    command = [sys.executable,'-m','treeneuron.research.cli','plan','--config',
               'configs/research/v2/M2-opportunity.json','--output',str(output)]
    result = subprocess.run(command,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['execution_implemented'] is True
    command[3] = 'run'
    command[-1] = str(tmp_path/'unimplemented')
    result = subprocess.run(command,capture_output=True,text=True)
    assert result.returncode == 2 and 'admission' in result.stderr
    assert not (tmp_path/'unimplemented').exists()
