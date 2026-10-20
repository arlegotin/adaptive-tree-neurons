"""Software parity and full real native workloads are separate qualifications."""
import copy
import importlib.util
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
