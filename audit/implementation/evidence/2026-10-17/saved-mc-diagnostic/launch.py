"""Fresh serial numerical process; fixed saved-data diagnostic only."""
import time
started=time.perf_counter_ns()
import hashlib,json,os,subprocess,sys,uuid
from pathlib import Path
base=Path('environment/h1-v2/pilot-saved-mc-supplement');script=base/'run.py';root=Path('artifacts/h1-v2/pilot-saved-mc-supplement')/uuid.uuid4().hex
env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',H1_SAVED_MC_ROOT=str(root))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
with (base/'run.stdout').open('xb') as out,(base/'run.stderr').open('xb') as err,(base/'caffeinate.stdout').open('xb') as co,(base/'caffeinate.stderr').open('xb') as ce:
    process=subprocess.Popen([sys.executable,str(script)],env=env,stdout=out,stderr=err)
    assertion=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(process.pid)],stdout=co,stderr=ce)
    with (base/'launch-state.json').open('x') as stream:json.dump(dict(worker_pid=process.pid,launcher_pid=os.getpid(),caffeinate_pid=assertion.pid,artifact_root=str(root),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),stream,indent=2)
    code=process.wait();assertion.wait(timeout=10)
record=dict(command=[sys.executable,str(script)],exit_code=code,external_seconds=(time.perf_counter_ns()-started)/1e9,
    script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    environment={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK')})
with (base/'invocation.json').open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record),flush=True);raise SystemExit(code)
