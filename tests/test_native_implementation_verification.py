"""Software parity and full real native workloads are separate qualifications."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.native_verification_v2') is not None,'Missing source-bound native implementation verification'
    from treeneuron.research import native_verification_v2
    return native_verification_v2


def test_fixed_profiles_cover_every_mandatory_family_without_inventing_quality(tmp_path):
    module=api();profiles=module.verification_profiles()
    assert set(profiles)=={'gated','ordinary_neural','realmlp','tabm','native_trees','node','simple'}
    families=[f for v in profiles.values() for f in v['families']]
    assert sorted(families)==[f'P{i:02}' for i in range(14)]
    for row in profiles.values():
        assert row['test_files'] and row['native_workload_qualified'] is False
        assert row['protected_access_authorized'] is False
    profiles['simple']['test_files'].clear()
    assert module.verification_profiles()['simple']['test_files']==['tests/test_p13_simple.py']


@pytest.mark.parametrize('body',['','<testcase name="skipped"><skipped/></testcase>',
    '<testcase name="failed"><failure/></testcase>','<testcase name="error"><error/></testcase>'])
def test_empty_skipped_or_failed_cases_cannot_be_qualified_by_a_zero_process_exit(tmp_path,body):
    module=api();path=tmp_path/'junit.xml';path.write_text('<testsuites><testsuite>'+body+'</testsuite></testsuites>')
    with pytest.raises(ValueError):module.checked_junit(path)


def test_actual_fresh_simple_profile_authenticates_tests_source_and_closed_process_only(tmp_path):
    module=api();out=tmp_path/'verification'
    result=module.verify_native_implementation(profile='simple',output=out,interpreter=sys.executable)
    assert result['status']=='completed_software_verification' and result['software_verified'] is True
    assert result['process']['child_reaped'] is True and result['process']['timeout_ns'] is None
    proof=module.load_native_implementation_verification(result['reference'],expected_profile='simple')
    assert proof['software_verified'] is True and proof['families']==['P13'] and proof['test_cases']>0
    assert proof['native_workload_qualified'] is proof['quality_eligible'] is proof['T_accessed'] is False
    for change in ('profile','bytes'):
        reference=copy.deepcopy(result['reference']);profile='simple'
        if change=='profile':profile='node'
        else:reference['inventory_sha256']='0'*64
        with pytest.raises(ValueError):module.load_native_implementation_verification(reference,expected_profile=profile)


def test_real_owned_memory_stop_cannot_supply_software_qualification(tmp_path):
    module=api();result=module.verify_native_implementation(profile='simple',output=tmp_path/'stopped',
        interpreter=sys.executable,rss_limit_bytes=1)
    assert result['status']=='resource_exhausted' and result['software_verified'] is False
    assert result['process']['child_reaped'] is True and result['reference'] is None


def test_fresh_tree_profile_explicitly_binds_its_overlay_and_owns_all_parity_outputs(tmp_path, monkeypatch):
    module=api();root=Path(__file__).resolve().parents[1]
    overlay=root/'environment/research-v2-tree-site'
    assert overlay.is_dir(), 'Pinned tree environment must be copied and byte-verified before this integration check'
    # Neither an ambient overlay nor an ambient output directory may control
    # which code the fixed profile runs or where its tests write evidence.
    monkeypatch.setenv('ATN_TREE_OVERLAY',str(tmp_path/'forbidden-overlay'))
    monkeypatch.setenv('ATN_TREE_ADAPTER_TEST_OUTPUT',str(tmp_path/'forbidden-output'))
    output=tmp_path/'tree-verification'
    result=module.verify_native_implementation(profile='native_trees',output=output,
        interpreter=sys.executable,extra_pythonpath=(overlay,))
    assert result['status']=='completed_software_verification' and result['software_verified'] is True
    process=json.loads((output/'process/process.json').read_text())
    assert process['environment_overrides']['ATN_TREE_OVERLAY']==str(overlay)
    assert process['environment_overrides']['ATN_TREE_ADAPTER_TEST_OUTPUT']==str(output/'worker/native-tree-parity')
    assert not (tmp_path/'forbidden-overlay').exists() and not (tmp_path/'forbidden-output').exists()
    proof=module.load_native_implementation_verification(result['reference'],expected_profile='native_trees')
    assert proof['software_verified'] is True and proof['test_cases']==8
    assert proof['native_workload_qualified'] is proof['quality_eligible'] is False


def test_non_tree_profile_explicitly_clears_inherited_tree_test_controls(tmp_path, monkeypatch):
    module=api()
    monkeypatch.setenv('ATN_TREE_OVERLAY',str(tmp_path/'forbidden-overlay'))
    monkeypatch.setenv('ATN_TREE_ADAPTER_TEST_OUTPUT',str(tmp_path/'forbidden-output'))
    result=module.verify_native_implementation(profile='simple',output=tmp_path/'simple',interpreter=sys.executable)
    assert result['software_verified'] is True
    process=json.loads((tmp_path/'simple/process/process.json').read_text())
    assert process['environment_overrides']['ATN_TREE_OVERLAY'] is None
    assert process['environment_overrides']['ATN_TREE_ADAPTER_TEST_OUTPUT'] is None
    module.load_native_implementation_verification(result['reference'],expected_profile='simple')
    # Recommit a controlled, internally consistent metadata forgery so byte
    # hashes alone cannot be the reason it rejects. The profile/environment
    # semantic join must also reject an unexplained inherited test control.
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    process['environment_overrides']['ATN_TREE_OVERLAY']=str(tmp_path/'unbound-overlay')
    saved=json.loads((tmp_path/'simple/result.json').read_text());saved['process']=process
    (tmp_path/'simple/process/process.json').write_text(json.dumps(process))
    (tmp_path/'simple/result.json').write_text(json.dumps(saved))
    reference=copy.deepcopy(result['reference']);reference['inventory_sha256']=canonical_hash(_inventory(tmp_path/'simple'))
    with pytest.raises(ValueError,match='overlay/output controls'):
        module.load_native_implementation_verification(reference,expected_profile='simple')


def test_tree_profile_requires_an_explicit_overlay_before_creating_a_worker(tmp_path):
    with pytest.raises(ValueError, match='one explicit'):
        api().verify_native_implementation(profile='native_trees',output=tmp_path/'missing',interpreter=sys.executable)
    assert not (tmp_path/'missing').exists()
