"""One fresh measured numerical process, explicit label and retained failures."""
import os,sys,time,json,hashlib,subprocess
from pathlib import Path
label=sys.argv[1]
script={'probes':Path('environment/h1-v2/probe-calibration/measure.py'),
    'rows':Path('environment/h1-v2/precision/loss-timing.py'),
    'mc':Path('environment/h1-v2/precision/monte-carlo.py')}[label]
base=script.parent;env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
command=[sys.executable,str(script)];tick=time.perf_counter_ns()
with (base/(label+'.stdout')).open('xb') as out,(base/(label+'.stderr')).open('xb') as err:
    run=subprocess.run(command,env=env,stdout=out,stderr=err)
record=dict(label=label,command=command,exit_code=run.returncode,outer_ns=time.perf_counter_ns()-tick,
    script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    environment={key:env[key] for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK','PYTHONPATH')})
with (base/(label+'-invocation.json')).open('x') as out:json.dump(record,out,indent=2)
print(json.dumps(record),flush=True);raise SystemExit(run.returncode)
