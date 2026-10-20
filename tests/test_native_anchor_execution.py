"""Anchor execution needs the right software proof and native inspected model."""
import copy
import importlib.util
import json
import sys

import pytest

from test_native_generation_chain import actual


def api():
    assert importlib.util.find_spec('treeneuron.research.native_anchor_v2') is not None,'Missing native anchor execution qualification'
    from treeneuron.research import native_anchor_v2
    return native_anchor_v2


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    saved=[]
    def get():
        if not saved:
            from treeneuron.research.native_verification_v2 import verify_native_implementation
            root=tmp_path_factory.mktemp('native-anchor-evidence')
            with pytest.MonkeyPatch.context() as patch:args,old=actual(root,patch)
            proof=verify_native_implementation(profile='simple',output=root/'software',interpreter=sys.executable)
            assert proof['software_verified'] is True
            saved.append((args,old,proof['reference']))
        return saved[0]
    return get


def qualify(module,args,old,software,output,**kwargs):
    return module.qualify_native_anchor(implementation_reference=software,generation_arguments=args,
        immutable_generation_root=old,output=output,interpreter=sys.executable,**kwargs)


def accesses(args):return (args['prepared_root']/'access.jsonl').read_bytes()


def test_actual_anchor_joins_current_software_native_state_replay_and_primary_V_without_promotion(tmp_path,evidence):
    module=api();args,old,software=evidence()
    from treeneuron.research.workqueue import _inventory
    before=_inventory(old);result=qualify(module,args,old,software,tmp_path/'anchor')
    assert result['status']=='authenticated_native_anchor_execution' and result['native_anchor_execution_authenticated'] is True
    assert result['family_id']=='P13' and result['anchor_id']=='mean' and result['dataset']=='energy'
    assert result['software_verified'] is True and result['prediction_reload_exact'] is True
    assert result['V_score_used_for_qualification'] is False
    assert result['raw_acquisition_authenticated'] is result['native_workload_qualified'] is result['quality_eligible'] is result['T_accessed'] is False
    assert result['elapsed_ns']>=result['inspection_span_ns']>0 and _inventory(old)==before
    assert json.loads((tmp_path/'anchor/result.json').read_text())==result


@pytest.mark.parametrize('change',['software','seed','non_anchor'])
def test_wrong_profile_seed_or_nonanchor_configuration_rejects_before_new_V_access(tmp_path,evidence,change):
    module=api();original,old,software=evidence();args=copy.deepcopy(original);software=copy.deepcopy(software)
    if change=='software':software['inventory_sha256']='0'*64
    elif change=='seed':args['expected_fit_seed']=18
    else:
        from treeneuron.research.search_supports_v2 import family_search_spec
        args['expected_configuration']=family_search_spec('P13',args['expected_configuration']['feature_schema'])['finite_configurations'][1]
    before=accesses(original);output=tmp_path/'invalid'
    with pytest.raises(ValueError):qualify(module,args,old,software,output)
    assert accesses(original)==before and not output.exists()


def test_owned_replay_resource_stop_retains_failure_without_qualifying_or_reading_V(tmp_path,evidence):
    module=api();args,old,software=evidence();before=accesses(args)
    result=qualify(module,args,old,software,tmp_path/'stopped',rss_limit_bytes=1)
    assert result['status']=='resource_exhausted' and result['native_anchor_execution_authenticated'] is False
    assert result['quality_eligible'] is result['native_workload_qualified'] is False
    validation=lambda raw:[row for row in map(json.loads,raw.splitlines()) if 'V_labels' in row['roles']]
    assert validation(accesses(args))==validation(before)


def test_qualification_output_cannot_change_the_original_generation_container(tmp_path,evidence):
    module=api();args,old,software=evidence();output=old/'forbidden'
    with pytest.raises(ValueError):qualify(module,args,old,software,output)
    assert not output.exists()
