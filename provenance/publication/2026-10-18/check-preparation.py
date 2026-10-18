import subprocess,os,json,time,re,sys,hashlib
from pathlib import Path
B=Path('environment/publication-20261018'); R=Path.cwd(); env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTORCH_ENABLE_MPS_FALLBACK='0',UV_CACHE_DIR=str(R/'.uv-cache'),UV_PYTHON_INSTALL_DIR=str(R/'.uv-python'))
checks=[]
def run(name,argv):
 tick=time.perf_counter();p=subprocess.run(argv,env=env,capture_output=True,text=True);(B/(name+'.log')).write_text(p.stdout+p.stderr)
 checks.append(dict(name=name,argv=argv,returncode=p.returncode,seconds=time.perf_counter()-tick));print(name,p.returncode,flush=True)
 if p.returncode:print((p.stdout+p.stderr)[-4000:],flush=True)
 return p.returncode
run('sync',['uv','sync','--frozen','--offline','--python','3.12'])
code=re.search(r'```python\n(.*?)\n```',Path('README.md').read_text(),re.S).group(1);p=B/'readme-example.py';p.write_text(code+'\n')
run('readme-example',[str(R/'.venv/bin/python'),str(p)])
run('appendix-a',[str(R/'.venv/bin/python'),'tests/reference_math_checks.py'])
run('appendix-b',[str(R/'.venv/bin/python'),'tests/reference_score_checks.py'])
run('cli-help',[str(R/'.venv/bin/python'),'-m','treeneuron.cli','--help'])
run('readme-checks',[str(R/'.venv/bin/python'),'-m','pytest','-q','tests/test_algebra.py','tests/test_growth.py','tests/test_score.py','tests/test_probe_diagnostic.py'])
run('build',['uv','build','--offline','--out-dir',str(B/'dist-verified')])
(B/'preparation-checks.json').write_text(json.dumps({'checks':checks,'thread_environment':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK']}},indent=2)+'\n')
raise SystemExit(any(v['returncode'] for v in checks))
