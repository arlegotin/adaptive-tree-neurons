"""Readiness cannot be asserted from review or software metadata alone."""
import copy
import importlib.util

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.production_readiness_v2'),'Missing required actual readiness gate'
    from treeneuron.research import production_readiness_v2
    return production_readiness_v2


def review():
    reference=dict(path='/known/untrusted-review.json',sha256='f'*64)
    return dict(schema='research/v2-production-branch-review',reviewed_commit='a'*40,final_commit='b'*40,
        final_execution_sha256='c'*64,source_sha256='d'*64,
        review_package_reference=reference,review_response_reference=reference,
        reviewer=dict(agent_id='known-inert-fixture',model='known-inert-model',reasoning_effort='known-inert'),findings=[])


def snapshot():
    return dict(pinned_commit='b'*40,checkout_commit='b'*40,files={'src/known.py':'e'*64},execution_sha256='c'*64)


@pytest.mark.parametrize('entry',['seal','load'])
def test_readiness_opens_actual_native_gate_before_foreign_software_review_or_source(tmp_path,entry):
    module=api();reference=dict(path=str(tmp_path/'forbidden.json'),sha256='a'*64)
    qualification=dict(root=str(tmp_path/'false-registry'),inventory_sha256='b'*64)
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        if entry=='seal':module.seal_production_readiness(qualification_reference=qualification,
            verification_references={},review_reference=reference,pinned_commit='bad',output=tmp_path/'never.json')
        else:module.load_production_readiness(reference,qualification_reference=qualification)
    assert 'forbidden.json' not in str(error.value) and not (tmp_path/'never.json').exists()


def test_review_metadata_can_check_empty_disposition_but_cannot_qualify_native_or_source():
    value=api()._inspect_review_metadata(review(),snapshot(),source_sha256='d'*64,changed_execution_files=[])
    assert value['unresolved_material_findings']==0 and value['changed_execution_files']==[]
    assert value['production_source_ready'] is value['protected_access_authorized'] is False


@pytest.mark.parametrize('fault',['unresolved','uncovered_change','source','final_commit','package','unknown','reviewer','fixed_without_red','unrelated_fix'])
def test_incomplete_review_or_unreviewed_scientific_delta_cannot_become_ready(fault):
    value=review();changed=[]
    if fault=='unresolved':value['findings']=[dict(finding_id='one',severity='Important',resolution='deferred',changed_files=[],red_reference=None,green_reference=None)]
    elif fault=='uncovered_change':changed=['src/known.py']
    elif fault=='source':value['source_sha256']='0'*64
    elif fault=='final_commit':value['final_commit']='0'*40
    elif fault=='package':value['final_execution_sha256']='0'*64
    elif fault=='unknown':value['production_source_ready']=True
    elif fault=='reviewer':value['reviewer']={}
    else:
        changed=['src/known.py'];ref=dict(path='/known/untrusted.log',sha256='f'*64)
        value['findings']=[dict(finding_id='one',severity='Important',resolution='fixed',changed_files=['src/known.py' if fault!='unrelated_fix' else 'src/other.py'],
            red_reference=None if fault=='fixed_without_red' else ref,green_reference=ref)]
    with pytest.raises(ValueError):api()._inspect_review_metadata(value,snapshot(),source_sha256='d'*64,changed_execution_files=changed)


def test_fixed_finding_names_exact_reviewed_to_final_source_delta_with_evidence_refs():
    value=review();ref=dict(path='/known/untrusted.log',sha256='f'*64)
    value['findings']=[dict(finding_id='one',severity='Important',resolution='fixed',changed_files=['src/known.py'],red_reference=ref,green_reference=ref)]
    checked=api()._inspect_review_metadata(value,snapshot(),source_sha256='d'*64,changed_execution_files=['src/known.py'])
    assert checked['changed_execution_files']==['src/known.py'] and checked['production_source_ready'] is False


def test_docs_only_checkout_commit_change_keeps_pinned_scientific_identity_but_source_drift_fails():
    saved=snapshot();current=copy.deepcopy(saved);current['checkout_commit']='a'*40
    assert api()._same_execution_snapshot(saved,current)
    for key,replacement in [('files',{'src/known.py':'0'*64}),('execution_sha256','0'*64),('pinned_commit','0'*40)]:
        bad=copy.deepcopy(current);bad[key]=replacement
        assert not api()._same_execution_snapshot(saved,bad)
