"""Real metadata CLI operations and actual native denials; no model/T work."""
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests'))
from test_practical_workload import fixtures
from treeneuron.research.cli import main
from treeneuron.research.deployment_worker_v2 import _source
from treeneuron.research.qualify import qualification_locks_sha256
from treeneuron.config import canonical_hash

root=ROOT/'artifacts/research-v2/WP10/practical-cli-software'
root.mkdir(parents=True,exist_ok=False);source=_source();started=time.perf_counter_ns();calls=[]
def write(name,value):
    path=root/name;path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');return path
def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def control(name,operation,arguments):
    return write(name,dict(schema='research/v2-practical-command',source_sha256=source,operation=operation,arguments=arguments))
def invoke(argv,expected):
    out,err=io.StringIO(),io.StringIO();begin=time.perf_counter_ns()
    with redirect_stdout(out),redirect_stderr(err):status=main(argv)
    calls.append(dict(argv=argv,cwd=str(ROOT),interpreter=sys.executable,
        invocation_mode='in_process_research_cli_main',exit_status=status,
        stdout=out.getvalue(),stderr=err.getvalue(),elapsed_ns=time.perf_counter_ns()-begin))
    assert status==expected,(argv,out.getvalue(),err.getvalue())

support_control=control('support-control.json','support_lock',{})
invoke(['plan','--config',str(support_control),'--output',str(root/'command-plan.json')],0)
support=root/'supports.json'
invoke(['run','--config',str(support_control),'--output',str(support)],0)
before=support.read_bytes();invoke(['run','--config',str(support_control),'--output',str(support)],2)
assert before==support.read_bytes()
manifests={name:ref(write(name+'-known-manifest.json',value)) for name,value in fixtures().items()}
plan_control=control('plan-control.json','practical_plan',dict(data_manifest_references=manifests,
    support_reference=ref(support),lock_sha256=qualification_locks_sha256()))
plan=root/'practical-plan.json';invoke(['run','--config',str(plan_control),'--output',str(plan)],0)
value=json.loads(plan.read_bytes());assert len(value['jobs'])==840
procedure_control=control('procedure-control.json','procedure',dict(practical_plan_reference=ref(plan),
    expected_practical_plan_hash=value['plan_hash'],support_reference=ref(support)))
invoke(['freeze','--config',str(procedure_control),'--output',str(root/'procedure.json')],0)
fake=dict(path=str(root/'forbidden-models.json'),sha256='a'*64)
inputs=dict(models_reference=fake,global_barrier_reference=fake,preT_costs_reference=fake,
    readiness_reference=fake,qualification_reference=dict(root=str(root/'unqualified'),inventory_sha256='b'*64))
payload=dict(source_sha256=source,**inputs)
protocol=write('known-unqualified-protocol.json',dict(schema='research/v2-final-practical-prediction-protocol',
    payload=payload,payload_hash=canonical_hash(payload)))
invoke(['confirm','--protocol',str(protocol),'--output',str(root/'never-study')],2)
assert 'forbidden-models' not in calls[-1]['stderr'] and not (root/'never-study').exists()
assert _source()==source
write('commands.json',calls)
observation=dict(schema='research/v2-practical-cli-software-capture',source_sha256=source,
    metadata_commands=len(calls),metadata_jobs=840,primary_metadata_jobs=168,secondary_metadata_jobs=672,
    actual_immutable_support_and_procedure_published=True,overwrite_rejected_without_mutation=True,
    actual_unqualified_confirmation_rejected_before_output=True,
    manifest_inputs_are_artificial_metadata=True,role_arrays_exist=False,
    actual_complete_qualified_public_positive=False,actual_benchmark_models_fit_or_predicted=0,
    benchmark_labels_read=0,T_accessed=False,protected_access_authorized=False,
    elapsed_ns=time.perf_counter_ns()-started,
    scope='CLI metadata and genuine native denial only; no protected evaluation or predictive result')
write('observation.json',observation);print(json.dumps(ref(root/'observation.json'),sort_keys=True))
