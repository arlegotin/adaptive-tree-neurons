"""One complete repository software attempt after the final review fixes."""
import json, os, sys
from pathlib import Path
from treeneuron.research.process_supervision_v2 import supervise_process
from treeneuron.research.production_verification_v2 import load_production_verification
from treeneuron.research.native_verification_v2 import verification_identity
from treeneuron.research.method_worker import scientific_source_sha256
from treeneuron.research.ledger import write_new_json
ROOT=Path.cwd(); source=scientific_source_sha256()
assert source=='6ca3cd4ec41ecc8222fc9e8add58ceb0ae07b5fd9c86d1fa1cd87fae0933bbac'
for profile in ('search','reporting','pipeline','protection'):
    loaded=json.loads((ROOT/'artifacts/research-v2/WP10'/('production-'+profile+'-'+source[:7]+'-loaded.json')).read_bytes())
    load_production_verification(loaded['reference'],expected_profile=profile)
root=ROOT/'artifacts/research-v2/WP11'/('whole-suite-'+source[:7])
root.mkdir(exist_ok=False)
identity=verification_identity(); write_new_json(root/'source-before.json',identity)
env={key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD')}
env.update(PYTHONPATH=str(ROOT/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0',KMP_DUPLICATE_LIB_OK=None,PYTEST_ADDOPTS=None,PYTEST_PLUGINS=None,
    ATN_TREE_OVERLAY=str(ROOT/'environment/research-v2-tree-site'),ATN_TREE_ADAPTER_TEST_OUTPUT=str(root/'tree-fixtures'))
code="import torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main("+repr(['-q','--tb=short','--basetemp='+str(root/'temporary'),'--junitxml='+str(root/'junit.xml')])+"))"
argv=[str(ROOT/'.venv-research-v2/bin/python'),'-P','-B','-c',code]
print(json.dumps(dict(status='starting_complete_suite',source_sha256=source,root=str(root))),flush=True)
process=supervise_process(argv,cwd=ROOT,environment_overrides=env,output=root/'process',timeout_ns=None,rss_limit_bytes=4*1024**3,
    identity=dict(purpose='complete_repository_software_verification',source_sha256=source))
assert scientific_source_sha256()==source and verification_identity()==identity
write_new_json(root/'result.json',dict(schema='research/v2-final-review-whole-suite-attempt',source_sha256=source,identity=identity,
    process=process,suite_passed=process['execution_status']=='exited' and process['returncode']==0,
    native_workload_qualified=False,quality_eligible=False,protected_access_authorized=False,T_accessed=False))
print(json.dumps(dict(status='complete_suite_attempt_finished',execution_status=process['execution_status'],returncode=process['returncode'],root=str(root))),flush=True)
