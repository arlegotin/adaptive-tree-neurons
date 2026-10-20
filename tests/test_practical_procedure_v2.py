"""Actual procedure semantics bind metadata without granting access or fitting."""
import hashlib
import importlib.util
import json

import pytest

from test_primary_search_declaration_v2 import study


def api():
    assert importlib.util.find_spec('treeneuron.research.practical_procedure_v2'),'Missing strict practical procedure declaration'
    from treeneuron.research import practical_procedure_v2
    return practical_procedure_v2


def reference(path):return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_exact_presearch_procedure_covers_native_quality_and_primary_barrier_without_authority(study,tmp_path):
    module=api();plan,support=study;path=tmp_path/'procedure.json'
    ref=module.seal_practical_procedure(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],support_reference=support,output=path)
    value=module.load_practical_procedure(ref,practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],support_reference=support)
    assert ref==reference(path)
    assert value['plan_hash']==plan.to_dict()['plan_hash']
    policy=value['policy'];assert policy['program']['primary_jobs']==168 and policy['program']['secondary_jobs']==672
    assert policy['method']['P00_placement']=='legacy_cheap_adam'
    assert policy['method']['P01_placement']=='informed_direction_random'
    assert policy['search']['training_wall_time_limit_ns'] is None
    assert policy['search']['max_attempts']==32
    assert policy['analysis']['individual_family_ids']==[f'P{i:02}' for i in range(1,14)]
    assert policy['analysis']['champion_family_ids']==[f'P{i:02}' for i in range(2,14)]
    assert policy['analysis']['bootstrap_iterations']==20000 and policy['analysis']['bike_primary_block_days']==7
    assert value['execution_authorized'] is value['protected_access_authorized'] is value['secondary_release_authorized'] is False
    assert not policy['mechanism_confirmation']['admitted']
    assert not value['global_postsearch_freeze_authenticated']


@pytest.mark.parametrize('fault',['native_deadline','joint_lead','family','refit','bootstrap','teacher_analysis','seed','champion','secondary_barrier','missingness','authority','extra','closure'])
def test_rehashing_changed_semantic_policy_is_not_a_valid_declaration(study,tmp_path,fault):
    module=api();plan,support=study;path=tmp_path/'procedure.json'
    module.seal_practical_procedure(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],support_reference=support,output=path)
    value=json.loads(path.read_bytes());policy=value['policy']
    if fault=='native_deadline':policy['search']['training_wall_time_limit_ns']=7200_000_000_000
    elif fault=='joint_lead':policy['method']['P00_placement']='joint_ridge'
    elif fault=='family':policy['analysis']['individual_family_ids'].pop()
    elif fault=='refit':policy['information']['refit_F_G_V']=True
    elif fault=='bootstrap':policy['analysis']['bootstrap_iterations']=1000
    elif fault=='teacher_analysis':policy['analysis']['primary_interval']='teacher_t'
    elif fault=='seed':policy['analysis']['bootstrap_seed']+=1
    elif fault=='champion':policy['analysis']['champion_selection']='best_T'
    elif fault=='secondary_barrier':policy['program']['primary_models_before_secondary']=1
    elif fault=='missingness':policy['analysis']['missing_predictions']='complete_case'
    elif fault=='authority':value['protected_access_authorized']=True
    elif fault=='extra':value['best_task']='energy'
    else:value['development_closures']['M3']['sha256']='0'*64
    # No external commitment is trusted to validate a changed policy.
    from treeneuron.config import canonical_hash
    value['declaration_hash']=canonical_hash({k:v for k,v in value.items() if k!='declaration_hash'})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):module.load_practical_procedure(reference(path),practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],support_reference=support)


def test_wrong_plan_support_or_external_hash_cannot_load_and_existing_declaration_is_immutable(study,tmp_path):
    module=api();plan,support=study;path=tmp_path/'procedure.json';args=dict(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],support_reference=support)
    ref=module.seal_practical_procedure(**args,output=path);before=path.read_bytes()
    with pytest.raises(ValueError):module.seal_practical_procedure(**args,output=path)
    assert path.read_bytes()==before
    with pytest.raises(ValueError):module.load_practical_procedure(dict(ref,sha256='0'*64),**args)
    with pytest.raises(ValueError):module.load_practical_procedure(ref,**dict(args,expected_practical_plan_hash='0'*64))
    with pytest.raises(ValueError):module.load_practical_procedure(ref,**dict(args,support_reference=dict(support,sha256='0'*64)))


def test_declaration_does_not_open_roles_or_start_processes(study,tmp_path,monkeypatch):
    module=api();plan,support=study
    import subprocess
    from treeneuron.research.transfer import TransferStore
    def forbidden(*args,**kwargs):raise AssertionError('A metadata declaration cannot open roles or spawn')
    monkeypatch.setattr(subprocess,'Popen',forbidden);monkeypatch.setattr(TransferStore,'__init__',forbidden)
    ref=module.seal_practical_procedure(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],support_reference=support,output=tmp_path/'procedure.json')
    assert module.load_practical_procedure(ref,practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],support_reference=support)


def test_strong_barrier_does_not_accept_an_arbitrary_global_procedure_json(tmp_path,monkeypatch):
    # Only a negative private assembler call: the metadata inventory is real,
    # but its artificial models/procedure carry no native authority. Do not
    # stub the actual public qualification loader into reporting success.
    from test_primary_selection_inventory_v2 import prepare,seal,ref
    from treeneuron.research import qualified_primary_models_v2 as module
    plan,procedure,selections=prepare(tmp_path,native_quality=True)
    inventory=seal(tmp_path,plan,procedure,selections)
    def forbidden(*args,**kwargs):raise AssertionError('Bad semantic procedure must fail before any native search audit')
    monkeypatch.setattr(module,'audit_primary_qualified_search',forbidden)
    proofs={key:dict(journal_sha256='a'*64,official_schedule_reference=None) for key in selections}
    with pytest.raises((ValueError,FileNotFoundError)):
        module._assemble(registry={},practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],
            support_reference=dict(path=str(tmp_path/'absent-support.json'),sha256='a'*64),
            qualification_reference=dict(root=str(tmp_path/'unqualified'),inventory_sha256='b'*64),
            primary_inventory_reference=ref(inventory.path),search_proofs=proofs,created_utc_ns=1)
