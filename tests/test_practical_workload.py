"""The fixed practical plan preserves pairing and the primary-data barrier."""
import copy

import pytest

from treeneuron.config import canonical_hash


def fixtures():
    manifests={}
    for dataset in ('airfoil','concrete','ccpp','parkinsons','bike','energy'):
        splits=[]
        for fold in range(5):
            split=dict(schema='research/v2-transfer-split',dataset=dataset,outer_fold=fold,
                primary=fold==(4 if dataset=='bike' else 0),roles=['F','G','V','T'],
                actual_rows=dict(F=1,G=1,V=1,T=1))
            split['split_hash']=canonical_hash(split);splits.append(split)
        manifest=dict(schema='research/v2-sealed-transfer',dataset=dataset,splits=splits,
            raw_identity_verified=True,files={},protected_labels_read_for_modeling=False)
        manifest['manifest_hash']=canonical_hash(manifest);manifests[dataset]=manifest
    return manifests


def expand(manifests=None,**overrides):
    from treeneuron.research.workload import expand_practical
    pins=dict(support_lock_sha256='a'*64,source_sha256='b'*64,runtime_sha256='c'*64,lock_sha256='d'*64)
    return expand_practical(fixtures() if manifests is None else manifests,**(pins|overrides))


def test_all_840_search_jobs_keep_exact_primary_secondary_partition_and_caps():
    plan=expand().to_dict();jobs=plan['jobs']
    assert len(jobs)==840 and len({r['job_id'] for r in jobs})==840
    primary={r['job_id'] for r in jobs if r['tranche']=='primary'}
    secondary={r['job_id'] for r in jobs if r['tranche']=='secondary'}
    assert len(primary)==168 and len(secondary)==672 and primary.isdisjoint(secondary)
    assert plan['limits']['conservative_attempt_ceiling']==26880
    assert plan['limits']['finite_unique_attempt_ceiling']==26700
    assert plan['limits']['dedicated_worker_hours_ceiling']==1680
    assert plan['limits']['primary_worker_hours_ceiling']==336
    assert all(r['budget']['max_attempts']==32 and r['budget']['deadline_ns']==7200000000000 for r in jobs)
    assert all(r['budget']['descriptive_prefixes_ns']==[600000000000,1800000000000] for r in jobs)
    assert all(r['budget']['unique_configuration_limit']==29 for r in jobs if r['family_id']=='P13')
    assert plan['execution_implemented'] is False and plan['protected_access_authorized'] is False


def test_native_quality_keeps_the_finite_workload_without_inventing_a_time_ceiling():
    plan=expand(native_quality=True).to_dict();legacy=expand().to_dict()
    assert len(plan['jobs'])==840 and plan['tranches']==dict(primary_jobs=168,secondary_jobs=672)
    assert plan['budget_policy']=='native_quality' and plan['plan_hash']!=legacy['plan_hash']
    assert plan['limits']['dedicated_worker_hours_ceiling'] is None
    assert plan['limits']['primary_worker_hours_ceiling'] is None
    assert plan['limits']['conservative_attempt_ceiling']==26880
    assert all(j['budget']['deadline_ns'] is None and j['budget']['max_attempts']==32
        and j['budget']['descriptive_prefixes_ns']==[600000000000,1800000000000,7200000000000] for j in plan['jobs'])
    assert [(j['job_id'],j['hpo_seed']) for j in plan['jobs']]==[(j['job_id'],j['hpo_seed']) for j in legacy['jobs']]
    assert plan['execution_implemented'] is False and plan['protected_access_authorized'] is False


def test_native_quality_fit_randomization_is_fixed_independent_and_paired_across_every_family():
    import hashlib,json
    rows=expand(native_quality=True).to_dict()['jobs'];paired={}
    for row in rows:
        key=(row['dataset'],row['outer_fold'],row['search_repeat'])
        paired.setdefault(key,set()).add((row['fit_seed64'],row['fit_seed'],row['fit_seed32_collision_counter']))
    assert len(paired)==60 and all(len(v)==1 for v in paired.values())
    assert len({next(iter(v))[1] for v in paired.values()})==60
    assert {next(iter(v))[1] for v in paired.values()}.isdisjoint({r['hpo_seed32'] for r in rows})
    labels=[2026101802,'P2-search','practical','airfoil','outer','0','repeat','0','fit']
    raw=json.dumps(labels,separators=(',',':'),ensure_ascii=True).encode('ascii')
    first=rows[0]
    assert first['fit_seed64']==int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')&((1<<63)-1)
    library=json.dumps(labels+['library32',str(first['fit_seed32_collision_counter'])],separators=(',',':')).encode('ascii')
    assert first['fit_seed']==int.from_bytes(hashlib.sha256(library).digest()[:8],'big')&((1<<32)-1)
    assert all('fit_seed' not in r for r in expand().to_dict()['jobs'])


def test_bike_uses_last_block_and_secondary_requires_all_primary_hashes():
    rows=expand().to_dict()['jobs'];primary=[r for r in rows if r['tranche']=='primary']
    assert {r['outer_fold'] for r in primary if r['dataset']=='bike'}=={4}
    assert {r['outer_fold'] for r in primary if r['dataset']!='bike'}=={0}
    for row in rows:
        if row['tranche']=='secondary':
            assert row['execution_prerequisites']==['frozen_global_procedure','all_168_primary_selected_model_hashes']
        else:assert row['execution_prerequisites']==['qualified_native_anchor','measured_resource_admission','frozen_search_support']


def test_hpo_randomization_is_paired_across_families_and_repeats_never_candidates():
    rows=expand().to_dict()['jobs'];labels={}
    for row in rows:
        label=(row['dataset'],row['outer_fold'],row['search_repeat'])
        labels.setdefault(label,set()).add((row['hpo_seed'],row['hpo_seed32'],row['seed32_collision_counter']))
    assert len(labels)==60 and all(len(s)==1 for s in labels.values())
    assert len({next(iter(s))[1] for s in labels.values()})==60
    assert all(row['search_repeat'] in (0,1) and row['seed_selection_allowed'] is False for row in rows)
    # Hand-checked canonical hash namespace, independent of the planner.
    import hashlib,json
    payload=json.dumps([2026101802,'P2-search','practical','airfoil','outer','0','repeat','0','HPO'],separators=(',',':')).encode()
    expected=int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')&((1<<63)-1)
    first=next(r for r in rows if r['dataset']=='airfoil' and r['outer_fold']==0 and r['search_repeat']==0)
    assert first['hpo_seed']==expected


def test_plan_is_immutable_and_content_identity_binds_data_source_support_and_budget():
    manifests=fixtures();plan=expand(manifests);baseline=plan.to_dict()
    manifests['airfoil']['splits'][0]['roles'][0]='T'
    changed=plan.to_dict();changed['jobs'][0]['budget']['max_attempts']=100
    assert plan.to_dict()==baseline
    reversed_inputs=dict(reversed(list(fixtures().items())))
    assert expand(reversed_inputs).to_dict()==baseline
    assert expand(source_sha256='e'*64).to_dict()['plan_hash']!=baseline['plan_hash']
    assert expand(support_lock_sha256='e'*64).to_dict()['plan_hash']!=baseline['plan_hash']


@pytest.mark.parametrize('change',['missing_dataset','wrong_dataset','duplicate_fold','wrong_primary','changed_split','changed_manifest','unverified_raw','inspected_T'])
def test_missing_or_changed_roles_cannot_be_planned_as_the_fixed_benchmark(change):
    value=fixtures()
    if change=='missing_dataset':del value['energy']
    elif change=='wrong_dataset':value['extra']=value.pop('energy')
    else:
        manifest=value['bike']
        if change=='duplicate_fold':manifest['splits'][1]=copy.deepcopy(manifest['splits'][0])
        elif change=='wrong_primary':
            for split in manifest['splits']:
                split['primary']=split['outer_fold']==0
                split['split_hash']=canonical_hash({k:v for k,v in split.items() if k!='split_hash'})
        elif change=='changed_split':manifest['splits'][0]['roles'][0]='T'
        elif change=='changed_manifest':manifest['schema']='changed'
        elif change=='unverified_raw':manifest['raw_identity_verified']=False
        else:manifest['protected_labels_read_for_modeling']=True
        if change not in ('changed_manifest',):
            manifest['manifest_hash']=canonical_hash({k:v for k,v in manifest.items() if k!='manifest_hash'})
    with pytest.raises(ValueError):expand(value)


@pytest.mark.parametrize('pin',[None,'main','A'*64,'g'*64,'a'*63])
def test_floating_or_malformed_implementation_pin_is_not_admitted(pin):
    with pytest.raises(ValueError):expand(support_lock_sha256=pin)
