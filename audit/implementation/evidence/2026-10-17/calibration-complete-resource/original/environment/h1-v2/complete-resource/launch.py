"""One fresh authorized engineering process, retain exact outer cost/failures."""
import os,sys,json,time,hashlib,subprocess
from pathlib import Path
base=Path('environment/h1-v2/complete-resource');script=base/'measure.py'
env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0')
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
command=[sys.executable,str(script)];tick=time.perf_counter_ns()
with (base/'run.stdout').open('xb') as out,(base/'run.stderr').open('xb') as err:
    result=subprocess.run(command,env=env,stdout=out,stderr=err)
record=dict(command=command,exit_code=result.returncode,outer_ns=time.perf_counter_ns()-tick,
    wrapper_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    config_sha256=hashlib.sha256((base/'candidate.json').read_bytes()).hexdigest(),policy_sha256=hashlib.sha256((base/'policy.json').read_bytes()).hexdigest(),
    environment={key:env[key] for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK','PYTHONPATH')})
with (base/'invocation.json').open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record),flush=True);raise SystemExit(result.returncode)
