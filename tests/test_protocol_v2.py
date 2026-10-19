"""Temporary source/metadata fixtures; no calibration or protected observations."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from test_protocol import repository,git,write
from treeneuron.config import canonical_hash


def api():
    from treeneuron.research import protocol_v2
    return protocol_v2


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(repository):
    project=Path(__file__).resolve().parents[1]
    for name in ('src/treeneuron/research/protocol_v2.py','src/treeneuron/analysis/design_v2.py',
                 'src/treeneuron/research/seeds.py','src/treeneuron/analysis/primary_v2.py'):
        p=repository/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/name,p)
    git(repository,'add','.');git(repository,'commit','-qm','toy committed v2 decision source')
    refs={}
    for name in ('decision','runtime','development_registry','search_support'):
        p=write(repository,f'artifacts/{name}.json',dict(kind=name,fixture_only=True))
        refs[name]=dict(path=str(p.relative_to(repository)),sha256=sha(p))
    return git(repository,'rev-parse','HEAD'),refs


def method(claim='placement'):
    roles=(dict(gated_residual='gated16',ordinary_lower='residual_lower',ordinary_upper='residual_upper')
           if claim=='residual_capacity' else dict(selected='chosen',random='random'))
    return dict(procedure_id='explicitly_unqualified_fixture_method',claim=claim,
        configuration=dict(selector='fixture',count=2,epsilon=.05),primary_roles=roles,
        ordinary_control_roles={} if claim=='residual_capacity' else dict(keep='keep',warm='warm',fresh='fresh'))


def create(repository,tmp_path,**changes):
    commit,refs=prepare(repository)
    kwargs=dict(repository_root=repository,pinned_commit=commit,study_id='fixture-method-lock',
        method=method(),references=refs,output=tmp_path/'method-lock.json')
    kwargs.update(changes)
    return api().lock_provisional_method(**kwargs),kwargs


def test_provisional_lock_roundtrip_is_immutable_and_grants_no_data_authority(repository,tmp_path):
    assert hasattr(api(),'lock_provisional_method')
    locked,args=create(repository,tmp_path)
    payload=locked.payload
    assert payload['phase']=='provisional_method' and payload['method']==args['method']
    assert payload['calibration_execution_authorized'] is False and payload['protected_access_authorized'] is False
    assert payload['configuration_semantics_qualified'] is False
    analysis=payload['analysis_policy']
    assert analysis['repetitions']==2 and analysis['calibration_teacher_counts']==[16,32]
    assert analysis['teacher_grid']==list(range(32,129,8))
    assert analysis['useful_relative_risk_reduction']==.02 and analysis['design_alternative_relative_risk_reduction']==.05
    assert analysis['bootstrap_SD_resamples']==10000 and analysis['power_simulation_cohorts']==100000
    assert analysis['primary_CI']=='two_sided_95_percent_teacher_Student_t'
    payload['method']['configuration']['epsilon']=999
    assert locked.payload['method']['configuration']['epsilon']==.05
    loaded=api().load_provisional_method(locked.path,expected_file_sha256=locked.file_sha256,repository_root=repository)
    assert loaded.payload==locked.payload and loaded.lock_hash==locked.lock_hash
    previous=locked.path.read_bytes()
    with pytest.raises((ValueError,FileExistsError)):api().lock_provisional_method(**args)
    assert locked.path.read_bytes()==previous


@pytest.mark.parametrize('claim',['placement','pruning_selection','residual_capacity'])
def test_claim_kind_keeps_capacity_and_placement_role_scopes_separate(claim,repository,tmp_path):
    assert hasattr(api(),'lock_provisional_method')
    locked,_=create(repository,tmp_path,method=method(claim))
    assert locked.payload['method']['claim']==claim
    assert locked.payload['analysis_policy']['primary_estimand']==(
        'V_selected_ordinary_residual_over_gated_residual_teacher_log_risk' if claim=='residual_capacity'
        else 'matched_random_over_selected_teacher_log_risk')


def test_calibration_randomness_is_fixed_by_the_method_lock_before_data(repository,tmp_path):
    assert hasattr(api(),'lock_provisional_method')
    from treeneuron.research.seeds import seed64
    locked,_=create(repository,tmp_path);policy=locked.payload['analysis_policy']
    assert policy['variance_seed_root']==2026101804
    assert policy['bootstrap_seeds']=={stage:seed64(2026101804,'C0','fixture-method-lock','variance-bootstrap',stage)
        for stage in ('initial16','extended32')}
    assert policy['simulation_seed']==seed64(2026101804,'C0','fixture-method-lock','power-validation')


def test_self_rehashed_analysis_or_authority_cannot_promote_a_provisional_lock(repository,tmp_path):
    assert hasattr(api(),'lock_provisional_method')
    locked,_=create(repository,tmp_path)
    for field in ('useful_relative_risk_reduction','protected_access_authorized'):
        value=json.loads(locked.path.read_text());p=value['payload']
        if field=='protected_access_authorized':p[field]=True
        else:
            p['analysis_policy'][field]=.000001;p['analysis_sha256']=canonical_hash(p['analysis_policy'])
        value['lock_hash']=canonical_hash(p);file=tmp_path/(field+'.json');file.write_text(json.dumps(value))
        with pytest.raises(ValueError):api().load_provisional_method(file,expected_file_sha256=sha(file),repository_root=repository)
    from treeneuron.evaluation.protected import _validate
    with pytest.raises(ValueError,match='frozen protocol'):_validate(locked,None,'cannot_open_T')


@pytest.mark.parametrize('change',['unknown_method_field','wrong_role_map','duplicate_primary_role','unknown_reference',
    'reference_bytes','untracked_source','dirty_source','ignored_source'])
def test_invalid_provisional_input_fails_before_publication(change,repository,tmp_path):
    assert hasattr(api(),'lock_provisional_method')
    commit,refs=prepare(repository);m=method()
    if change=='unknown_method_field':m['margin']=.000001
    elif change=='wrong_role_map':m['claim']='residual_capacity'
    elif change=='duplicate_primary_role':m['primary_roles']['random']='chosen'
    elif change=='unknown_reference':refs['test_results']=copy.deepcopy(refs['runtime'])
    elif change=='reference_bytes':(repository/refs['runtime']['path']).write_text('changed')
    elif change=='dirty_source':(repository/'src/model.py').write_text('# changed after pin\n')
    elif change=='untracked_source':(repository/'src/override.py').write_text('# untracked\n')
    else:
        (repository/'.gitignore').write_text('src/hidden.py\n');(repository/'src/hidden.py').write_text('# ignored override\n')
    output=tmp_path/'lock.json'
    with pytest.raises(ValueError):api().lock_provisional_method(repository_root=repository,pinned_commit=commit,
        study_id='fixture-method-lock',method=m,references=refs,output=output)
    assert not output.exists()


def test_doc_only_commit_preserves_lock_but_changed_bound_file_or_payload_rejects(repository,tmp_path):
    assert hasattr(api(),'lock_provisional_method')
    locked,args=create(repository,tmp_path)
    write(repository,'docs/note.json',dict(documentation_only=True));git(repository,'add','.');git(repository,'commit','-qm','docs only')
    # Toy artifact references also became tracked in this commit, but are excluded
    # from the execution snapshot and remain separately byte-authenticated.
    assert api().load_provisional_method(locked.path,expected_file_sha256=locked.file_sha256,
        repository_root=repository).lock_hash==locked.lock_hash
    ref=repository/args['references']['runtime']['path'];original=ref.read_bytes();ref.write_text('changed')
    with pytest.raises(ValueError):api().load_provisional_method(locked.path,expected_file_sha256=locked.file_sha256,repository_root=repository)
    ref.write_bytes(original)
    value=json.loads(locked.path.read_text());value['payload']['method']['configuration']['epsilon']=.2
    value['lock_hash']=canonical_hash(value['payload']);altered=tmp_path/'altered.json';altered.write_text(json.dumps(value))
    with pytest.raises(ValueError):api().load_provisional_method(altered,expected_file_sha256=sha(altered),repository_root=repository)
    with pytest.raises(ValueError):api().load_provisional_method(locked.path,expected_file_sha256='0'*64,repository_root=repository)


@pytest.mark.parametrize('checkout',['nonexistent','different_execution_tree'])
def test_checkout_provenance_must_name_real_execution_equivalent_bytes(checkout,repository,tmp_path):
    assert hasattr(api(),'lock_provisional_method')
    original_commit=git(repository,'rev-parse','HEAD')
    locked,_=create(repository,tmp_path)
    value=json.loads(locked.path.read_text())
    value['payload']['source']['checkout_commit']='0'*40 if checkout=='nonexistent' else original_commit
    value['lock_hash']=canonical_hash(value['payload']);path=tmp_path/'forged-checkout.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError):api().load_provisional_method(path,expected_file_sha256=sha(path),repository_root=repository)


def test_provisional_policy_fixes_secondary_construction_and_endpoint_convention(repository,tmp_path):
    locked,_=create(repository,tmp_path);policy=locked.payload['analysis_policy']
    assert policy['compatible_adjusted_lower_bounds']==dict(
        construction='teacher_t_guilbaud2008_choice2_strict',alpha=.025,
        marginal_lower_endpoint='included',rejected_margin_endpoint='excluded',
        partial_rejection='no_numeric_gap_for_rejected_effect',all_rejected='common_minimum_marginal_lower')
    value=json.loads(locked.path.read_text());p=value['payload']
    p['analysis_policy']['compatible_adjusted_lower_bounds']['marginal_lower_endpoint']='excluded'
    p['analysis_sha256']=canonical_hash(p['analysis_policy']);value['lock_hash']=canonical_hash(p)
    changed=tmp_path/'reinterpreted-endpoint.json';changed.write_text(json.dumps(value))
    with pytest.raises(ValueError):api().load_provisional_method(changed,expected_file_sha256=sha(changed),repository_root=repository)


def test_secondary_interval_code_is_bound_by_bytes_before_calibration(repository,tmp_path):
    locked,_=create(repository,tmp_path)
    relative='src/treeneuron/analysis/primary_v2.py'
    assert locked.payload['code_bindings']['secondary_bounds']==dict(path=relative,sha256=sha(repository/relative))
    (repository/relative).write_text('# replaced secondary inference\n')
    with pytest.raises(ValueError):api().load_provisional_method(locked.path,expected_file_sha256=locked.file_sha256,repository_root=repository)
