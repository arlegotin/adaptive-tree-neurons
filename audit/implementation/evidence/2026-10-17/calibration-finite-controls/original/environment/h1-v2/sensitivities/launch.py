"""Finite serial fresh-interpreter sensitivity launcher; no recipe tuning."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
base=Path('environment/h1-v2/sensitivities');env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
records=[]
for point in json.loads((base/'coordinates.json').read_text())['coordinates']:
    name=point['name'];path=Path(point['config']);command=[sys.executable,str(base/'sensitivity.py'),str(path)];tick=time.perf_counter_ns()
    with (base/(name+'.stdout')).open('xb') as out,(base/(name+'.stderr')).open('xb') as err:
        run=subprocess.run(command,env=env,stdout=out,stderr=err)
    item=dict(coordinate=name,command=command,exit_code=run.returncode,outer_ns=time.perf_counter_ns()-tick,
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256((base/'sensitivity.py').read_bytes()).hexdigest())
    records.append(item);(base/'invocations.json').write_text(json.dumps(records,indent=2));print(json.dumps(item),flush=True)
    if run.returncode:raise SystemExit(run.returncode)
