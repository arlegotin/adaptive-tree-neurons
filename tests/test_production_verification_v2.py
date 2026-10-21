"""Fixed production fixtures cannot be replaced by a caller's green summary."""
import importlib.util
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.production_verification_v2'),'Missing production fixture qualification'
    from treeneuron.research import production_verification_v2
    return production_verification_v2


def junit(path,cases):
    path.write_text('<testsuites><testsuite>'+''.join(
        f'<testcase classname="{module}" name="{name}">{child}</testcase>' for module,name,child in cases)+
        '</testsuite></testsuites>')


def test_junit_requires_every_fixed_input_file_and_rejects_unrelated_passing_cases(tmp_path):
    module=api();path=tmp_path/'junit.xml';files=['tests/test_alpha.py','tests/test_beta.py']
    junit(path,[('tests.test_alpha','first',''),('tests.test_beta.AClass','second','')])
    assert module._checked_profile_junit(path,files)==2
    for cases in [[('tests.test_alpha','only','')],
                  [('tests.test_alpha','first',''),('tests.test_beta','second',''),('tests.test_other','extra','')],
                  [('tests.test_alpha','first',''),('tests.test_beta','second','<skipped/>')],
                  [('tests.test_alpha','first',''),('tests.test_beta','second','<failure/>')],[]]:
        junit(path,cases)
        with pytest.raises(ValueError):module._checked_profile_junit(path,files)


@pytest.mark.parametrize('profile',['tests/test_production_verification_v2.py','user-selected','',True])
def test_public_verifier_rejects_caller_selected_tests_before_any_output(tmp_path,profile):
    import sys
    with pytest.raises(ValueError):
        api().verify_production_implementation(profile=profile,output=tmp_path/'never',interpreter=Path(sys.executable))
    assert not (tmp_path/'never').exists()


def test_green_junit_without_actual_supervised_profile_receipts_is_not_qualification(tmp_path):
    root=tmp_path/'invented';root.mkdir();junit(root/'junit.xml',[('tests.test_alpha','pass','')])
    with pytest.raises((ValueError,FileNotFoundError)):
        api().load_production_verification(dict(root=str(root),inventory_sha256='a'*64),expected_profile='protection')


def test_changed_source_or_profile_request_fails_before_worker_output(tmp_path):
    import hashlib,json
    from treeneuron.research.native_verification_v2 import verification_identity
    module=api();identity=verification_identity();identity['scientific_source_sha256']='0'*64
    path=tmp_path/'request.json';path.write_text(json.dumps(dict(schema='research/v2-production-verification-request',
        profile='protection',identity=identity)))
    with pytest.raises(ValueError):
        module._worker(path,hashlib.sha256(path.read_bytes()).hexdigest(),tmp_path/'never-worker')
    assert not (tmp_path/'never-worker').exists()
