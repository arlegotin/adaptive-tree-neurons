"""Fresh outer measured genuine-pilot invocation, retain failed/incomplete attempt."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
base=Path('environment/h1-v2/genuine-pilot');script=base/'run.py'
env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
command=[sys.executable,str(script)];tick=time.perf_counter_ns()
with (base/'run.stdout').open('xb') as out,(base/'run.stderr').open('xb') as err:
    run=subprocess.run(command,env=env,stdout=out,stderr=err)
record=dict(command=command,existing_cli_arguments=['calibrate','--config','environment/h1-v2/scientific-ready/scientific-config.yaml'],
    exit_code=run.returncode,external_seconds=(time.perf_counter_ns()-tick)/1e9,
    wrapper_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    environment={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK','PYTHONPATH')})
with (base/'invocation.json').open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record),flush=True);raise SystemExit(run.returncode)
