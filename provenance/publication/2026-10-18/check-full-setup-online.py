import subprocess,os,json,re,time
from pathlib import Path
R=Path.cwd();B=R/'environment/publication-20261018';old=json.loads((B/'clean-clone-v2.json').read_text());C=Path(old['clone']);assert old['checks'][-1]['returncode']==1
failed=[line.split(' ',2)[1] for line in (B/'clone-full-suite.log').read_text().splitlines() if line.startswith('FAILED ')];assert len(failed)==74
(B/'failed-test-ids.json').write_text(json.dumps(failed));env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTORCH_ENABLE_MPS_FALLBACK='0',UV_CACHE_DIR=str(R/'.uv-cache'),UV_PYTHON_INSTALL_DIR=str(R/'.uv-python'),PYTHONPATH=str(C/'src'))
checks=[]
def run(name,argv):
 t=time.perf_counter();p=subprocess.run(argv,cwd=C,env=env,capture_output=True,text=True);(B/(name+'.log')).write_text(p.stdout+p.stderr);checks.append({'name':name,'argv':argv,'returncode':p.returncode,'seconds':time.perf_counter()-t});print(name,p.returncode,flush=True)
 if p.returncode:print((p.stdout+p.stderr)[-6000:],flush=True)
 assert p.returncode==0
for name,req,target in [('node','locks/node/reference-requirements.txt','environment/node-reference-site'),('search','locks/search/requirements.txt','environment/search-site')]:
 run('clone-'+name+'-overlay-online',['uv','pip','install','--python',str(R/'.venv/bin/python'),'--target',str(C/target),'--require-hashes','-r',req])
code='import torch,pytest,json; from pathlib import Path; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main(["-q", *json.loads(Path("environment/publication-20261018/failed-test-ids.json").read_text()),"tests/test_algebra.py","tests/test_growth.py","tests/test_score.py","tests/test_probe_diagnostic.py","tests/test_train.py","tests/test_checkpoint.py"]))'
run('clone-configured-regressions',[str(R/'.venv/bin/python'),'-c',code])
(B/'configured-checks.json').write_text(json.dumps({'clone':str(C),'tested_source_commit':old['tested_commit'],'initial_full_core_attempt':{'passed':928,'skipped':48,'failed':74,'seconds':1920.73},'causes':{'single_thread_initialization':49,'optional_dependencies':25},'core_lock_changed':False,'scientific_source_changed':False,'checks':checks},indent=2)+'\n')
