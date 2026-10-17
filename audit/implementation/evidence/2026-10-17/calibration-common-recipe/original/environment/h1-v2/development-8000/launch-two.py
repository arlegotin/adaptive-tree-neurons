"""Sequential fresh numerical children; stop before B if A has an improving ceiling."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
base=Path('environment/h1-v2/development-8000');env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
records=[]
for letter in ('a','b'):
    config_path=base/f'primary-{letter}.json';config=json.loads(config_path.read_text())
    command=[sys.executable,str(base/'paired-development.py'),str(config_path)];tick=time.perf_counter_ns()
    with (base/f'primary-{letter}.stdout').open('xb') as out,(base/f'primary-{letter}.stderr').open('xb') as err:
        run=subprocess.run(command,env=env,stdout=out,stderr=err)
    item=dict(problem=letter,command=command,exit_code=run.returncode,outer_ns=time.perf_counter_ns()-tick,
        config_file_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256((base/'paired-development.py').read_bytes()).hexdigest(),
        environment={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK','PYTHONPATH')})
    records.append(item);(base/'invocations.json').write_text(json.dumps(records,indent=2));print(json.dumps(item),flush=True)
    if run.returncode:raise SystemExit(run.returncode)
    measurement=Path(config['execution']['artifacts_root'])/config['study_id']/'measurements.json'
    result=json.loads(measurement.read_text())
    if result['improving_ceilings']:
        print('STOP_BEFORE_NEXT_TEACHER_IMPROVING_CEILING',json.dumps(result['improving_ceilings']),flush=True);break
