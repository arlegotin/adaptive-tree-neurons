"""Fixed production-fixture proofs, separate from native model qualification."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from .deployment_worker_v2 import _ROOT,_THREADS,_source,_json,_bytes
from .native_verification_v2 import verification_identity,checked_junit
from .ledger import write_new_json
from .workqueue import _inventory
from treeneuron.config import canonical_hash


_PROFILES={
 'search':('practical_workload','practical_support_lock_v2','practical_procedure_v2',
    'primary_search_declaration_v2','primary_search_worker_v2','primary_caller_boundary_v2',
    'primary_resource_binding_v2','primary_invocation_v2','primary_program_v2',
    'qualified_primary_models_v2','practical_global_freeze_v2','secondary_resource_v2',
    'secondary_program_v2','secondary_native_search_v2','qualified_practical_models_v2',
    'native_resource_forecast_v2','native_qualification_registry_v2',
    'validation_composition_v2','snapshot_boundary_v2'),
 'protection':('protected_journal_v2','protected_query_v2','protected_query_worker_v2',
    'protected_query_artifact_v2','protected_prediction_queue_v2','protected_prediction_gate_v2',
    'protected_label_evaluator_v2','independent_raw_audit_v2','raw_audit_readback_v2'),
 'pipeline':('program_process_v2','cold_confirmation_process_v2','confirm_programme_v2',
    'deployment_query_v2','native_deployment_worker_v2','native_deployment_plan_v2',
    'native_deployment_execution_v2','practical_pipeline_costs_v2','practical_preT_costs_v2'),
 'reporting':('report_v2','frozen_practical_analysis_v2','practical_comparisons_v2',
    'practical_family_report_v2','protected_practical_report_v2','practical_cli_v2',
    'practical_T_protocol_v2','production_verification_v2','production_readiness_v2'),
}
_FALSE=dict(native_workload_qualified=False,quality_eligible=False,protected_access_authorized=False,T_accessed=False)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def production_verification_profiles():
    return {name:['tests/test_'+stem+'.py' for stem in files] for name,files in _PROFILES.items()}


def _checked_profile_junit(path,files):
    """Validate supplied XML arithmetic only; no process authority inferred."""
    count=checked_junit(path);modules={Path(f).stem for f in files};covered=set()
    for case in ET.parse(path).getroot().iter('testcase'):
        parts=case.get('classname','').split('.')
        matches=modules.intersection(parts)
        if len(matches)!=1:raise ValueError('JUnit case is outside the fixed production profile')
        covered.update(matches)
    if covered!=modules:raise ValueError('Every fixed production test file must contribute passing cases')
    return count


def _request(path,expected):
    path=Path(path)
    if path.is_symlink():raise ValueError('Ordinary fixed production verification request required')
    value=_json(_bytes(path,expected));profiles=production_verification_profiles()
    if (type(value) is not dict or set(value)!={'schema','profile','identity'}
            or value['schema']!='research/v2-production-verification-request'
            or type(value['profile']) is not str or value['profile'] not in profiles
            or value['identity']!=verification_identity()
            or any(f not in value['identity']['test_and_lock_files'] for f in profiles[value['profile']])):
        raise ValueError('Fixed production profile/source/test/lock identity required')
    return value


def _worker(path,expected,output):
    request=_request(path,expected);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='research/v2-production-verification-worker',status='software_verification_failed',
        request_sha256=expected,profile=request['profile'],identity=request['identity'],software_verified=False,
        test_cases=None,**_FALSE)
    try:
        if not sys.dont_write_bytecode or any(os.environ.get(k)!='1' for k in _THREADS):
            raise ValueError('Declared production verification threads required')
        import torch,pytest
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        files=production_verification_profiles()[request['profile']]
        result['pytest_returncode']=int(pytest.main(['-q','--junitxml='+str(output/'junit.xml'),*files]))
        if result['pytest_returncode']!=0:raise ValueError('Production fixture checks failed; no software proof')
        count=_checked_profile_junit(output/'junit.xml',files)
        if _request(path,expected)!=request:raise ValueError('Production verification inputs changed during checks')
        result.update(status='completed_software_verification',software_verified=True,test_cases=count,junit_sha256=_sha(output/'junit.xml'))
    except BaseException as error:result.update(error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
    finally:write_new_json(output/'terminal.json',result)
    return result['software_verified']


def _environment(root,overlays):
    env={k:'1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD')}
    env.update(PYTHONPATH=os.pathsep.join([str(_ROOT/'src'),*overlays]),KMP_DUPLICATE_LIB_OK=None,
        PYTORCH_ENABLE_MPS_FALLBACK='0',PYTEST_ADDOPTS=None,PYTEST_PLUGINS=None,PYTEST_CURRENT_TEST=None,
        MPLCONFIGDIR=str(root/'cache/mpl'),XDG_CACHE_HOME=str(root/'cache/xdg'))
    return env


def _argv(interpreter,root,digest):
    return [str(interpreter),'-P','-B','-m','treeneuron.research.production_verification_v2',
        '--request',str(root/'request.json'),'--expected-request-sha256',digest,'--output',str(root/'worker')]


def verify_production_implementation(*,profile,output,interpreter,extra_pythonpath=(),rss_limit_bytes=4*1024**3):
    started=time.perf_counter_ns();original=Path(output);root=original.resolve();interpreter=Path(interpreter)
    if (type(profile) is not str or profile not in _PROFILES or original.is_symlink() or root.exists()
            or not interpreter.is_absolute() or not interpreter.is_file() or type(rss_limit_bytes) is not int
            or not 0<rss_limit_bytes<=16*1024**3 or not isinstance(extra_pythonpath,(tuple,list))):
        raise ValueError('Fresh fixed production verification profile required')
    from .primary_program_v2 import _check_program_location
    _check_program_location(root,working_roots=[],proof_roots=[],input_files=[])
    overlays=[]
    for item in extra_pythonpath:
        p=Path(item).resolve()
        if not p.is_dir() or not p.is_relative_to(_ROOT/'environment'):raise ValueError('Project verification overlay required')
        overlays.append(str(p))
    root.mkdir(parents=True,exist_ok=False);request=dict(schema='research/v2-production-verification-request',profile=profile,identity=verification_identity())
    path=root/'request.json';write_new_json(path,request);digest=_sha(path);_request(path,digest)
    from .process_supervision_v2 import supervise_process
    process=supervise_process(argv=_argv(interpreter,root,digest),cwd=_ROOT,env_overrides=_environment(root,overlays),
        output=root/'process',timeout_ns=None,rss_limit_bytes=rss_limit_bytes,
        identity=dict(purpose='production_fixture_verification',profile=profile,request_sha256=digest))
    status=process['execution_status'];verified=False
    if status=='exited':
        worker=_json((root/'worker/terminal.json').read_bytes());status=worker['status'];verified=worker['software_verified']
        if (process['returncode']==0)!=(verified is True):raise ValueError('Production verification worker/process disagree')
    if _request(path,digest)!=request:raise ValueError('Verification source changed before final publication')
    result=dict(schema='research/v2-production-verification-result',status=status,profile=profile,identity=request['identity'],
        request_sha256=digest,process=process,software_verified=verified,elapsed_before_publication_ns=time.perf_counter_ns()-started,**_FALSE)
    write_new_json(root/'result.json',result)
    return dict(result,reference=dict(root=str(root),inventory_sha256=canonical_hash(_inventory(root))) if verified else None)


def load_production_verification(reference,*,expected_profile):
    if (type(reference) is not dict or set(reference)!={'root','inventory_sha256'}
            or type(expected_profile) is not str or expected_profile not in _PROFILES):
        raise ValueError('Externally bound fixed production verification reference required')
    root=Path(reference['root'])
    if (not root.is_absolute() or root.resolve()!=root or root.is_symlink() or not root.is_dir()
            or canonical_hash(_inventory(root))!=reference['inventory_sha256']):raise ValueError('Production verification bytes differ')
    saved=_json((root/'result.json').read_bytes());request=_request(root/'request.json',saved['request_sha256'])
    worker=_json((root/'worker/terminal.json').read_bytes());process=_json((root/'process/process.json').read_bytes())
    if (saved.get('schema')!='research/v2-production-verification-result' or worker.get('schema')!='research/v2-production-verification-worker'
            or saved.get('profile')!=expected_profile or request['profile']!=expected_profile or worker.get('profile')!=expected_profile
            or saved.get('identity')!=request['identity'] or worker.get('identity')!=request['identity']
            or saved.get('process')!=process or worker.get('request_sha256')!=saved['request_sha256'] or worker.get('pytest_returncode')!=0
            or any(v.get('status')!='completed_software_verification' or v.get('software_verified') is not True for v in (saved,worker))
            or any(v.get(k) is not x for v in (saved,worker) for k,x in _FALSE.items())
            or process.get('execution_status')!='exited' or process.get('returncode')!=0 or process.get('child_reaped') is not True
            or process.get('timeout_ns') is not None or process.get('cwd')!=str(_ROOT)
            or process.get('identity')!=dict(purpose='production_fixture_verification',profile=expected_profile,request_sha256=saved['request_sha256'])):
        raise ValueError('Actual closed supervised production verification evidence required')
    argv=process['argv'];env=process['environment_overrides'];paths=env.get('PYTHONPATH','').split(os.pathsep)
    if (argv!=_argv(Path(sys.executable).absolute(),root,saved['request_sha256']) or not paths or paths[0]!=str(_ROOT/'src')
            or any(not Path(p).resolve().is_relative_to(_ROOT/'environment') for p in paths[1:])
            or env!=_environment(root,paths[1:])):raise ValueError('Fixed production verification invocation/environment changed')
    count=_checked_profile_junit(root/'worker/junit.xml',production_verification_profiles()[expected_profile])
    if (worker.get('test_cases')!=count or worker.get('junit_sha256')!=_sha(root/'worker/junit.xml')
            or request['identity']!=verification_identity() or canonical_hash(_inventory(root))!=reference['inventory_sha256']):
        raise ValueError('Production test result/source/evidence changed')
    return dict(profile=expected_profile,test_cases=count,identity=request['identity'],reference=reference,
        software_verified=True,**_FALSE,scope='Fixed final-source software fixtures only; native model qualification and readiness remain separate')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True)
    parser.add_argument('--expected-request-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    return 0 if _worker(args.request,args.expected_request_sha256,args.output) else 1


if __name__=='__main__':raise SystemExit(main())
