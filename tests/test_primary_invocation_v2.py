"""Production preflight must precede every output; metadata never substitutes qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.primary_invocation_v2'), 'Missing qualified primary invocation preflight'
    from treeneuron.research import primary_invocation_v2
    return primary_invocation_v2


def test_unqualified_preflight_cannot_open_plan_or_create_any_output(tmp_path):
    module=api()
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.prepare_primary_invocation(practical_plan_reference=dict(path=str(tmp_path/'forbidden-plan'),sha256='a'*64),
            expected_practical_plan_hash='b'*64,job_id='airfoil/outer-0/repeat-0/P13',
            support_reference=dict(path=str(tmp_path/'forbidden-support'),sha256='c'*64),
            qualification_reference=dict(root=str(tmp_path/'fake-registry'),inventory_sha256='d'*64),
            request_path=tmp_path/'request.json',worker_output=tmp_path/'worker',
            control_output=tmp_path/'control',search_directory=tmp_path/'search')
    assert 'forbidden-plan' not in str(error.value) and 'forbidden-support' not in str(error.value)
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('output_kind',['request','worker','control','search'])
@pytest.mark.parametrize('protected_kind',['data','proof','interpreter','executable','plan','support','prior_worker'])
def test_each_invocation_output_preserves_all_inputs(tmp_path,output_kind,protected_kind):
    module=api();project=Path(__file__).resolve().parents[1]
    protected=dict(data=tmp_path/'working',proof=tmp_path/'proof',interpreter=Path(sys.prefix).resolve(),
        executable=project/'src',plan=tmp_path/'plan.json',support=tmp_path/'support.json',prior_worker=tmp_path/'prior-worker')[protected_kind]
    paths=dict(request=tmp_path/'request.json',worker=tmp_path/'worker',control=tmp_path/'control',search=tmp_path/'search')
    paths[output_kind]=protected/'new-output'
    with pytest.raises(ValueError):
        module._check_invocation_locations(paths=paths,working_roots=[str(tmp_path/'working')],
            proof_roots=[str(tmp_path/'proof')],input_files=[tmp_path/'plan.json',tmp_path/'support.json',tmp_path/'prior-worker'],resume=False)
    assert not paths[output_kind].exists()


@pytest.mark.parametrize('kind',['equal','nested','request_exists','search_exists','worker_exists','relative'])
def test_disjoint_fresh_output_contract_before_publication(tmp_path,kind):
    module=api()
    paths=dict(request=tmp_path/'request.json',worker=tmp_path/'worker',control=tmp_path/'control',search=tmp_path/'search')
    if kind=='equal':paths['control']=paths['worker']
    if kind=='nested':paths['request']=paths['control']/'request.json'
    if kind=='request_exists':paths['request'].write_text('retained')
    if kind in ('search_exists','worker_exists'):paths[kind.split('_')[0]].mkdir()
    if kind=='relative':paths['control']=Path('relative-control')
    with pytest.raises(ValueError):module._check_invocation_locations(paths=paths,working_roots=[],proof_roots=[],input_files=[],resume=False)


def test_clean_resume_preserves_existing_search_but_requires_fresh_episode_outputs(tmp_path):
    module=api();search=tmp_path/'search';search.mkdir();(search/'retained').write_text('unchanged')
    paths=dict(request=tmp_path/'request.json',worker=tmp_path/'worker',control=tmp_path/'control',search=search)
    assert module._check_invocation_locations(paths=paths,working_roots=[],proof_roots=[],input_files=[],resume=True) is None
    assert sorted(p.name for p in tmp_path.iterdir())==['search']
    assert (search/'retained').read_text()=='unchanged'


def test_prior_command_paths_are_protected_inputs_not_new_outputs(tmp_path):
    module=api();root=tmp_path/'prior-control';root.mkdir()
    request=tmp_path/'prior-request.json';worker=tmp_path/'prior-worker'
    argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2',
        '--request',str(request),'--expected-request-sha256','a'*64,'--output',str(worker)]
    (root/'started.json').write_text(json.dumps(dict(argv=argv)))
    prior=[dict(process_root=str(root),process_receipt_sha256='b'*64,process_started_sha256='c'*64,body_outcome_sha256='d'*64)]
    assert module._prior_input_locations(prior)==[root,request,worker]
    assert not request.exists() and not worker.exists()


def test_parent_preflight_clock_is_paid_by_actual_cold_controller(tmp_path):
    """Real process/controller fixture; no native or full-registry success is faked."""
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process,inspect_cooperative_process
    script=tmp_path/'worker.py'
    script.write_text('''
import os, hashlib
from pathlib import Path
from treeneuron.research.cooperative_process_v2 import cooperative_worker
with cooperative_worker(identity=dict(fixture='parent-preflight-cost')):
    from treeneuron.research.search_controller_v2 import make_search_plan
    from treeneuron.research.primary_search_worker_v2 import _run_prepared_search
    root=Path(os.environ['ATN_SEARCH_CONTROL_ROOT']).parent
    plan=make_search_plan(identity=dict(job_id='tiny/P13',family_id='P13',search_repeat=0,source_sha256='a'*64,runtime_sha256='b'*64,support_sha256='c'*64,split_sha256='d'*64),fit_seed=17,hpo_seed32=19,anchors=[dict(constant=0)],mode='finite',support=None,candidates=[],max_attempts=1,deadline_ns=None,prefixes_ns=[200000000])
    def callback(request):
        path=Path(request['attempt_directory'])/'constant.txt';path.write_text('software-only')
        return dict(status='completed_eligible',V_score=1.,files=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())],checkpoint_files=[],reason=None)
    _run_prepared_search(plan=plan,callback=callback,search_directory=root/'search',output=root/'worker',previous_outer_tail_ns=0,resume=False,expected_journal_hash=None)
''')
    origin=time.perf_counter_ns();time.sleep(.25)
    result=supervise_cooperative_process(argv=[sys.executable,'-P','-B',str(script)],cwd=str(tmp_path),
        env_overrides=dict(PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'),PYTHONDONTWRITEBYTECODE='1'),
        output=tmp_path/'control',identity=dict(fixture='parent-preflight-cost'),admission_identity=dict(policy='software-only'),
        admission_check=lambda:dict(admitted=True),rss_pause_bytes=256*1024**2,invocation_origin_ns=origin)
    assert result['returncode']==0
    start=json.loads((tmp_path/'control/started.json').read_text())
    assert start['supervisor_started_monotonic_ns']==origin
    assert start['caller_preparation_ns']>=250_000_000
    body=json.loads((tmp_path/'worker/search-result.json').read_text())
    assert result['elapsed_ns']>=body['total_charged_ns']>=start['caller_preparation_ns']
    assert body['winner'] is not None
    assert body['prefixes']['200000000']['winner'] is None
    sha=lambda name:hashlib.sha256((tmp_path/'control'/name).read_bytes()).hexdigest()
    proof=inspect_cooperative_process(root=tmp_path/'control',expected_receipt_sha256=sha('process.json'),expected_started_sha256=sha('started.json'))
    assert proof['caller_preparation_clock_observed'] is True
    assert proof['whole_training_procedure_cost_authenticated'] is False


@pytest.mark.parametrize('origin',[True,-1,10**30])
def test_invalid_parent_origin_rejects_without_creating_control(tmp_path,origin):
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process
    with pytest.raises(ValueError):supervise_cooperative_process(argv=[sys.executable,'-c','pass'],cwd=str(tmp_path),
        env_overrides={},output=tmp_path/'control',identity=dict(fixture=True),admission_identity=dict(policy='test'),
        admission_check=lambda:dict(admitted=True),rss_pause_bytes=256*1024**2,invocation_origin_ns=origin)
    assert not (tmp_path/'control').exists()
