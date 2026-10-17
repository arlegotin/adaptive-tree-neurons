"""Numerics-free serial fresh-process launcher and outer wall boundaries."""
import os,json,sys,time,subprocess,hashlib
from pathlib import Path
base=Path('environment/h1-v1/calibration');root=Path('artifacts/h1-v1/parent-safety')
original=json.loads((base/'primary-stopping-4000-a.json').read_text())
env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
records=[]
for batch,lr in ((128,.003),(128,.005),(256,.005),(256,.003),(512,.003),(512,.005)):
    config=json.loads(json.dumps(original));name=f'parent-b{batch}-lr{str(lr).replace(".","p")}-v2'
    config['study_id']=name
    config['training'].update(batch_size=batch,learning_rate=lr,base_updates=4000,continuation_updates=4000,
        validation_interval=25,checkpoint_interval=25,patience=20)
    config['execution']['artifacts_root']=str(root)
    path=base/(name+'.json');path.write_text(json.dumps(config,indent=2))
    command=[sys.executable,str(base/'parent-safety.py'),str(path)]
    start=time.perf_counter_ns()
    with (base/(name+'.stdout')).open('wb') as out,(base/(name+'.stderr')).open('wb') as err:
        run=subprocess.run(command,env=env,stdout=out,stderr=err)
    item=dict(name=name,command=command,status=run.returncode,outer_ns=time.perf_counter_ns()-start,
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256((base/'parent-safety.py').read_bytes()).hexdigest())
    records.append(item);(base/'parent-safety-launches-v2.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(item),flush=True)
    if run.returncode:raise RuntimeError(f'Parent candidate failed, retained: {name}')
