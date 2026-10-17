"""Stdlib-only outer boundary; exactly one fresh serial numerical child."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
base=Path('environment/h1-v2/calibration');env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
command=[sys.executable,str(base/'parent-first.py'),str(base/'parent-b4096-lr0p005.json')]
started=time.perf_counter_ns()
with (base/'parent-first.stdout').open('xb') as out,(base/'parent-first.stderr').open('xb') as err:
    child=subprocess.run(command,env=env,stdout=out,stderr=err)
record=dict(command=command,outer_ns=time.perf_counter_ns()-started,exit_code=child.returncode,
    boundary='fresh child interpreter,imports,strictsourcevalidation,data,parent,G/checkpoints,measurement publication,exit',
    env={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK','PYTHONPATH')},
    files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (base/'parent-first.py',base/'launch-parent-first.py',base/'parent-b4096-lr0p005.json')])
with (base/'invocations.json').open('x') as out:json.dump([record],out,indent=2)
print(json.dumps(record),flush=True)
raise SystemExit(child.returncode)
