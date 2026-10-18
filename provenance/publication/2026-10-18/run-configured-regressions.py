import subprocess,os,json,time
from pathlib import Path
R=Path.cwd();B=R/'environment/publication-20261018';old=json.loads((B/'clean-clone-v2.json').read_text());C=Path(old['clone']);ids=json.loads((B/'failed-test-ids.json').read_text());assert len(ids)==74
(B/'configured-ids-path-checker-failed.log').write_bytes((B/'clone-configured-regressions.log').read_bytes())
selected=ids+['tests/test_algebra.py','tests/test_growth.py','tests/test_score.py','tests/test_probe_diagnostic.py','tests/test_train.py','tests/test_checkpoint.py']
code='import torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main('+repr(['-q',*selected])+'))'
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONPATH=str(C/'src'))
start=time.perf_counter();p=subprocess.run([str(R/'.venv/bin/python'),'-c',code],cwd=C,env=env,capture_output=True,text=True)
(B/'configured-regressions.log').write_text(p.stdout+p.stderr)
record={'tested_source_commit':old['tested_commit'],'clone':str(C),'initial_full_core_attempt':{'passed':928,'skipped':48,'failed':74,'seconds':1920.73},'causes':{'single_thread_initialization':49,'optional_dependencies':25},'core_lock_changed':False,'scientific_source_changed':False,'selected_tests':selected,'returncode':p.returncode,'seconds':time.perf_counter()-start,'command_configuration':'torch.set_num_threads(1); torch.set_num_interop_threads(1); pytest -q all74 previously failed nodes + algebra/growth/score/diagnostic/train/checkpoint','optional_overlays':['locks/node/reference-requirements.txt','locks/search/requirements.txt'],'claim_scope':'Affected setup failures rerun plus core regressions; this is not a second monolithic full-suite pass.'}
(B/'configured-checks.json').write_text(json.dumps(record,indent=2)+'\n');print((p.stdout+p.stderr)[-6500:],flush=True);raise SystemExit(p.returncode)
