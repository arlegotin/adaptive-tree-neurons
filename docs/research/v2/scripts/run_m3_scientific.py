"""Supervise the admitted immutable M3 study, with no wall-time cutoff."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--expected-source-sha256',required=True)
parser.add_argument('--interpreter',type=Path,required=True)
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--resume',action='store_true')
args=parser.parse_args();source=args.source.resolve();output=args.output.resolve()
config=args.config.resolve() if args.config.is_absolute() else (source/args.config).resolve()
sys.path.insert(0,str(source/'src'))
from treeneuron.train import source_hash
from treeneuron.research.ledger import write_new_json
from treeneuron.research.process_supervision_v2 import supervise_process

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert source_hash()==args.expected_source_sha256,'Held M3 source digest differs'
raw=json.loads(config.read_text())
backend=Path(raw['base_experiment']['execution']['backend_report'])
admission=Path(raw['resource_admission'])
if not backend.is_absolute():backend=source/backend
if not admission.is_absolute():admission=source/admission
assert backend.is_file() and admission.is_file() and args.interpreter.is_file()
record=json.loads(admission.read_text())
assert record['source_sha256']==args.expected_source_sha256
identity=dict(schema='research/v2-M3-scientific-launch',source_root=str(source),
    source_sha256=source_hash(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
    config_path=str(config),config_sha256=sha(config),backend_receipt_sha256=sha(backend),
    admission_path=str(admission),admission_sha256=sha(admission),target_plan_hash=record['target_plan_hash'],
    operator_sha256=sha(__file__),interpreter=str(args.interpreter.absolute()),
    native_horizons_preserved=True,wall_deadline=None,T_authorized=False,
    evidence_scope='Fresh development teachers; fixed bounded M3 attempt list and one prospective extension only')
if args.resume:
    assert output.is_dir() and json.loads((output/'launch.json').read_text())==identity,'Immutable M3 launch changed'
else:
    output.mkdir(parents=True,exist_ok=False);write_new_json(output/'launch.json',identity)
index=0
while (output/f'attempt-{index:03}').exists():index+=1
attempt=output/f'attempt-{index:03}';attempt.mkdir()
argv=[str(args.interpreter.absolute()),'-B','-m','treeneuron.research.cli','run','--config',str(config),
    '--output',str(output/'queue')]
if args.resume:argv.append('--resume')
environment={key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}
environment.update(PYTHONPATH=str(source/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0',KMP_DUPLICATE_LIB_OK=None)
result=supervise_process(argv=argv,cwd=source,env_overrides=environment,output=attempt/'process',
    timeout_ns=None,rss_limit_bytes=3*1024**3,poll_interval_ns=10**9,identity=identity)
summary={key:result[key] for key in ('execution_status','returncode','elapsed_ns','observed_peak_root_rss_bytes')}
write_new_json(attempt/'terminal.json',dict(summary,T_accessed=False,
    interpretation='Consult complete queue terminal inventory; process completion alone is no efficacy claim'))
print(json.dumps(summary),flush=True)
raise SystemExit(0 if result['execution_status']=='exited' and result['returncode']==0 else 3)
