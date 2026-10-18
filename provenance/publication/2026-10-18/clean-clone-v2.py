"""Check source from an archive-free clone using the existing locked interpreter."""
import subprocess,os,json,tempfile,time,re,hashlib
from pathlib import Path
R=Path.cwd();B=R/'environment/publication-20261018';clone=Path('/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/treeneuron-public-ld_3_uvh/repo')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=clone,text=True).strip()==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert not any((clone/n).exists() for n in ['artifacts','runs','.venv','.uv-cache','environment/long-probes-20261018'])
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONPATH=str(clone/'src'))
python=R/'.venv/bin/python';checks=[]
def run(name,args):
 start=time.perf_counter();p=subprocess.run([str(python),*args],cwd=clone,env=env,capture_output=True,text=True);(B/(name+'.log')).write_text(p.stdout+p.stderr)
 checks.append(dict(name=name,argv=[str(python),*args],returncode=p.returncode,seconds=time.perf_counter()-start));print(name,p.returncode,flush=True)
 if p.returncode:print((p.stdout+p.stderr)[-5000:],flush=True)
 return p.returncode
run('clone-import',['-c','import treeneuron,pathlib; p=pathlib.Path(treeneuron.__file__).resolve(); print(p); assert p.is_relative_to(pathlib.Path.cwd())'])
code=re.search(r'```python\n(.*?)\n```',(clone/'README.md').read_text(),re.S).group(1)
run('clone-example',['-c',code]);run('clone-math',['tests/reference_math_checks.py']);run('clone-score',['tests/reference_score_checks.py'])
run('clone-readme-tests',['-m','pytest','-q','tests/test_algebra.py','tests/test_growth.py','tests/test_score.py','tests/test_probe_diagnostic.py'])
run('clone-full-suite',['-m','pytest','-q'])
result={'clone':str(clone),'tested_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=clone,text=True).strip(),'source_import_from_clone':True,'external_research_archive_absent':True,'dependencies_reused_from_locked_environment':str(R/'.venv'),'environment':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK','PYTHONPATH']},'checks':checks}
(B/'clean-clone-v2.json').write_text(json.dumps(result,indent=2)+'\n');print('CLEAN_CLONE_REPORT',str(B/'clean-clone-v2.json'),flush=True)
raise SystemExit(any(r['returncode'] for r in checks))
