"""Planned paired model seeds must be legal for every actual native adapter."""
import hashlib
import json

import pytest


def plan():
    from test_practical_workload import fixtures
    from treeneuron.research.workload import expand_practical
    from treeneuron.research.method_worker import scientific_source_sha256
    return expand_practical(fixtures(),support_lock_sha256='a'*64,source_sha256=scientific_source_sha256(),
        runtime_sha256='b'*64,lock_sha256='c'*64,native_quality=True)


def test_every_planned_tree_coordinate_passes_the_unchanged_actual_native_seed_guard():
    from treeneuron.data.tabular import DATASETS
    from treeneuron.research.search_supports_v2 import family_search_spec
    from treeneuron.research.native_trees import NativeTreeConfig,native_parameters
    rows=[r for r in plan().to_dict()['jobs'] if r['family_id'] in ('P08','P09','P10','P11')]
    assert len(rows)==240
    for row in rows:
        schema=DATASETS[row['dataset']];family=row['family_id']
        config=family_search_spec(family,dict(columns=list(schema.features),categorical=list(schema.categorical)))['anchors'][0]['config']
        actual=native_parameters(NativeTreeConfig.parse(family,config),row['fit_seed'])
        assert actual['random_seed' if family=='P08' else 'seed']==row['fit_seed']
        assert 0<=row['fit_seed']<2**31 and row['fit_seed_bits']==31


def test_common_model_seed31_allocator_resolves_collisions_without_changing_library32():
    from treeneuron.research import seeds
    assert hasattr(seeds,'seed31'),'Missing declared collision-managed library31 namespace'
    used=set();first,counter=seeds.seed31(2026101802,'fit',used=used)
    assert 0<=first<2**31 and counter==0 and used=={first}
    other,second=seeds.seed31(2026101802,'fit',used=used)
    assert other!=first and second>=1 and used=={first,other}


def test_HPO_and_fit_root64_namespaces_remain_byte_identical_after_seed_range_repair():
    from treeneuron.research.workload import _paired_search_seeds
    records=[]
    for (dataset,fold,repeat),value in _paired_search_seeds('P2-search',2026101802,native_quality=True).items():
        records.append(dict(dataset=dataset,outer_fold=fold,repeat=repeat,**{k:value[k] for k in
            ('hpo_seed','hpo_seed32','seed32_collision_counter','fit_seed64')}))
    raw=json.dumps(dict(records=records),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    assert hashlib.sha256(raw).hexdigest()=='a4227587309f5cf2735f47b393b3dbda3ed7623bab57dd37c76b4839b329fa79'


def test_self_rehashed_unknown_model_seed_policy_cannot_enter_primary_sealing():
    from treeneuron.config import canonical_hash,canonical_json
    from treeneuron.research.workload import PracticalPlan
    from treeneuron.research.primary_selection_v2 import _primary
    value=plan().to_dict();value['native_fit_seed_policy']='unknown'
    value['plan_hash']=canonical_hash({k:v for k,v in value.items() if k!='plan_hash'})
    with pytest.raises(ValueError,match='seed|policy'):
        _primary(PracticalPlan(canonical_json(value)),value['plan_hash'])
