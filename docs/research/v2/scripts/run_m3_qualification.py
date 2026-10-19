"""Supervise the full-horizon M3 engineering spine under its held source."""
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
assert source_hash()==args.expected_source_sha256,'Held M3 source digest differs'
raw=json.loads(config.read_text());backend=Path(raw['base_experiment']['execution']['backend_report'])
if not backend.is_absolute():backend=source/backend
assert backend.is_file() and config.is_file() and args.interpreter.is_file()
identity=dict(schema='research/v2-M3-qualification-launch',source_root=str(source),
    source_sha256=source_hash(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
    config_path=str(config),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    backend_receipt_sha256=hashlib.sha256(backend.read_bytes()).hexdigest(),
    operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interpreter=str(args.interpreter.absolute()),
    native_horizons_preserved=True,wall_deadline=None,scientific_M3_admission=False,T_authorized=False)
if args.resume:
    assert output.is_dir() and json.loads((output/'launch.json').read_text())==identity,'Immutable qualification launch changed'
else:
    output.mkdir(parents=True,exist_ok=False);write_new_json(output/'launch.json',identity)
index=0
while (output/f'attempt-{index:03}').exists():index+=1
attempt=output/f'attempt-{index:03}';attempt.mkdir()
argv=[str(args.interpreter.absolute()),'-B','-m','treeneuron.research.cli','qualify','--config',str(config),'--output',str(output/'queue')]
if args.resume:argv.append('--resume')
environment={key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}
environment.update(PYTHONPATH=str(source/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0',KMP_DUPLICATE_LIB_OK=None)
result=supervise_process(argv=argv,cwd=source,env_overrides=environment,output=attempt/'process',
    timeout_ns=None,rss_limit_bytes=3*1024**3,poll_interval_ns=10**9,identity=identity)
summary={key:result[key] for key in ('execution_status','returncode','elapsed_ns','observed_peak_root_rss_bytes')}
write_new_json(attempt/'terminal.json',dict(summary,scientific_M3_admission=False,T_accessed=False))
print(json.dumps(summary),flush=True)
raise SystemExit(0 if result['execution_status']=='exited' and result['returncode']==0 else 3)
