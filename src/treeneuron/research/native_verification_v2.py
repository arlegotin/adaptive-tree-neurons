"""Fixed native implementation checks, distinct from real workload admission."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from treeneuron.config import canonical_hash
from .deployment_worker_v2 import _ROOT,_THREADS,_source,_bytes,_json
from .ledger import write_new_json
from .process_supervision_v2 import supervise_process
from .workqueue import _inventory


_PROFILES={
    'gated':(['P00','P01','P02','P05'],['test_gated_native_core.py','test_gated_native_fit.py','test_gated_native_snapshot.py']),
    'ordinary_neural':(['P03','P04'],['test_ordinary_neural_native.py']),
    'realmlp':(['P06'],['test_realmlp_native.py','test_realmlp_snapshot_contract.py','test_realmlp_search.py']),
    'tabm':(['P07'],['test_tabm_native.py']),
    'native_trees':(['P08','P09','P10','P11'],['test_native_tree_parity.py']),
    'node':(['P12'],['test_node_v2.py','test_node_training_recovery.py','test_native_node_trace_reference.py']),
    'simple':(['P13'],['test_p13_simple.py'])}


def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verification_profiles():
    return {k:dict(families=copy.deepcopy(families),test_files=['tests/'+p for p in files],
        native_workload_qualified=False,protected_access_authorized=False) for k,(families,files) in _PROFILES.items()}


def verification_identity():
    paths=[*_ROOT.joinpath('tests').rglob('*.py'),*(_ROOT/'locks').rglob('*'),_ROOT/'uv.lock',_ROOT/'pyproject.toml']
    files={}
    for path in sorted(paths):
        if path.is_symlink():raise ValueError('Verification inputs cannot use symlinks')
        if path.is_file():files[path.relative_to(_ROOT).as_posix()]=_sha(path)
    return dict(scientific_source_sha256=_source(),test_and_lock_files=files)


def checked_junit(path):
    root=ET.fromstring(Path(path).read_bytes())
    if root.tag not in ('testsuites','testsuite'):raise ValueError('Complete native verification JUnit required')
    cases=list(root.iter('testcase'));seen=set()
    if not cases:raise ValueError('Zero collected cases cannot qualify an implementation')
    for case in cases:
        key=(case.get('classname'),case.get('name'))
        if not all(key) or key in seen or any(child.tag in ('failure','error','skipped') for child in case):
            raise ValueError('Failed/skipped/duplicate native checks cannot supply software qualification')
        seen.add(key)
    if any(node.tag in ('failure','error','skipped') for node in root.iter()):raise ValueError('Incomplete native verification report')
    return len(cases)


def _request(path,expected):
    path=Path(path)
    if path.is_symlink():raise ValueError('Ordinary native verification request required')
    value=_json(_bytes(path,expected))
    if (type(value) is not dict or set(value)!={'schema','profile','identity'}
            or value['schema']!='research/v2-native-implementation-request'
            or type(value['profile']) is not str or value['profile'] not in _PROFILES
            or value['identity']!=verification_identity()):raise ValueError('Fixed native profile/source/test/lock identity differs')
    for relative in verification_profiles()[value['profile']]['test_files']:
        if relative not in value['identity']['test_and_lock_files']:raise ValueError('Missing fixed native verification fixture')
    return value


def _worker(request_path,expected,output):
    started=time.perf_counter_ns();request=_request(request_path,expected);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='research/v2-native-implementation-worker',status='software_verification_failed',
        request_sha256=expected,profile=request['profile'],identity=request['identity'],test_cases=None,
        software_verified=False,native_workload_qualified=False,quality_eligible=False,T_accessed=False)
    try:
        if not sys.dont_write_bytecode or any(os.environ.get(k)!='1' for k in _THREADS):raise ValueError('Declared fresh verification threads required')
        if request['profile'] not in ('simple','native_trees'):
            import torch
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
        import pytest
        result['pytest_returncode']=int(pytest.main(['-q','--junitxml='+str(output/'junit.xml'),
            *verification_profiles()[request['profile']]['test_files']]))
        if result['pytest_returncode']!=0:raise ValueError('Native implementation checks failed; retained without qualification')
        result['test_cases']=checked_junit(output/'junit.xml')
        if _request(request_path,expected)!=request:raise ValueError('Verification inputs changed during tests')
        result.update(status='completed_software_verification',software_verified=True,junit_sha256=_sha(output/'junit.xml'))
    except BaseException as error:result.update(error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
    finally:
        result['elapsed_before_publication_ns']=time.perf_counter_ns()-started
        write_new_json(output/'terminal.json',result)
    return result


def _tree_test_environment(profile,overlays,output):
    if profile=='native_trees':
        if len(overlays)!=1:raise ValueError('Native tree profile requires one explicit pinned project overlay')
        return dict(ATN_TREE_OVERLAY=overlays[0],
            ATN_TREE_ADAPTER_TEST_OUTPUT=str(Path(output).resolve()/'worker/native-tree-parity'))
    return dict(ATN_TREE_OVERLAY=None,ATN_TREE_ADAPTER_TEST_OUTPUT=None)


def verify_native_implementation(*,profile,output,interpreter,extra_pythonpath=(),rss_limit_bytes=4*1024**3):
    started=time.perf_counter_ns();output=Path(output);interpreter=Path(interpreter)
    if (type(profile) is not str or profile not in _PROFILES or output.exists() or output.is_symlink()
            or not interpreter.is_absolute() or not interpreter.is_file() or type(rss_limit_bytes) is not int
            or rss_limit_bytes<=0 or not isinstance(extra_pythonpath,(list,tuple))):raise ValueError('Fresh fixed native implementation profile required')
    overlays=[]
    for value in extra_pythonpath:
        path=Path(value).resolve()
        if not path.is_dir() or not path.is_relative_to((_ROOT/'environment').resolve()):raise ValueError('Pinned project verification overlay required')
        overlays.append(str(path))
    tree_environment=_tree_test_environment(profile,overlays,output)
    request=dict(schema='research/v2-native-implementation-request',profile=profile,identity=verification_identity())
    output.mkdir(parents=True,exist_ok=False);path=output/'request.json';write_new_json(path,request);expected=_sha(path)
    _request(path,expected)
    env={k:'1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}
    env.update(PYTHONPATH=os.pathsep.join([str(_ROOT/'src'),*overlays]),KMP_DUPLICATE_LIB_OK=None,PYTORCH_ENABLE_MPS_FALLBACK='0',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS=None,PYTEST_PLUGINS=None,PYTEST_CURRENT_TEST=None,
        MPLCONFIGDIR=str((output/'cache/mpl').resolve()),XDG_CACHE_HOME=str((output/'cache/xdg').resolve()))
    env.update(tree_environment)
    process=supervise_process(argv=[str(interpreter),'-B','-m','treeneuron.research.native_verification_v2','--request',str(path.resolve()),
        '--expected-request-sha256',expected,'--output',str((output/'worker').resolve())],cwd=_ROOT,env_overrides=env,
        output=output/'process',timeout_ns=None,rss_limit_bytes=rss_limit_bytes,
        identity=dict(purpose='native_implementation_verification',profile=profile,request_sha256=expected))
    status=process['execution_status'];worker=None
    if status=='exited':
        worker=_json((output/'worker/terminal.json').read_bytes());status=worker['status']
        if (process['returncode']==0)!=(status=='completed_software_verification'):raise ValueError('Verification worker/process disagree')
    if _request(path,expected)!=request:raise ValueError('Verification source/test/lock inputs changed')
    result=dict(schema='research/v2-native-implementation-result',status=status,profile=profile,request_sha256=expected,
        identity=request['identity'],process=process,software_verified=status=='completed_software_verification',
        native_workload_qualified=False,quality_eligible=False,T_accessed=False,
        elapsed_before_publication_ns=time.perf_counter_ns()-started,
        clock_scope='Helper entry through result preparation; process time is nested; caller pays final publication/inventory/return')
    write_new_json(output/'result.json',result)
    reference=dict(root=str(output.resolve()),inventory_sha256=canonical_hash(_inventory(output))) if result['software_verified'] else None
    return dict(result,reference=reference)


def load_native_implementation_verification(reference,*,expected_profile):
    if (type(reference) is not dict or set(reference)!={'root','inventory_sha256'} or type(expected_profile) is not str
            or expected_profile not in _PROFILES):raise ValueError('Externally committed fixed native profile required')
    root=Path(reference['root'])
    if (not root.is_absolute() or root.resolve()!=root or root.is_symlink() or not root.is_dir()
            or canonical_hash(_inventory(root))!=reference['inventory_sha256']):raise ValueError('Native verification bytes differ')
    saved=_json((root/'result.json').read_bytes());request=_request(root/'request.json',saved['request_sha256'])
    worker=_json((root/'worker/terminal.json').read_bytes());process=_json((root/'process/process.json').read_bytes())
    if (saved['schema']!='research/v2-native-implementation-result' or worker['schema']!='research/v2-native-implementation-worker'
            or saved['profile']!=expected_profile or request['profile']!=expected_profile or worker['profile']!=expected_profile
            or saved['process']!=process or saved['identity']!=request['identity'] or worker['identity']!=request['identity']
            or worker['request_sha256']!=saved['request_sha256'] or worker['pytest_returncode']!=0
            or any(v['status']!='completed_software_verification' or v['software_verified'] is not True for v in (saved,worker))
            or any(v[k] is not False for v in (saved,worker) for k in ('native_workload_qualified','quality_eligible','T_accessed'))
            or process['execution_status']!='exited' or process['returncode']!=0 or process['child_reaped'] is not True
            or process['timeout_ns'] is not None or process['cwd']!=str(_ROOT)
            or process['identity']!=dict(purpose='native_implementation_verification',profile=expected_profile,request_sha256=saved['request_sha256'])):
        raise ValueError('Closed native software verification provenance differs')
    argv=process['argv'];expected=['-B','-m','treeneuron.research.native_verification_v2','--request',str(root/'request.json'),
        '--expected-request-sha256',saved['request_sha256'],'--output',str(root/'worker')]
    env=process['environment_overrides'];paths=env.get('PYTHONPATH','').split(os.pathsep)
    if (Path(argv[0]).resolve()!=Path(sys.executable).resolve() or argv[1:]!=expected
            or any(env.get(k)!='1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD'))
            or any(env.get(k) is not None for k in ('PYTEST_ADDOPTS','PYTEST_PLUGINS','KMP_DUPLICATE_LIB_OK'))
            or not paths or paths[0]!=str(_ROOT/'src') or any(not Path(p).resolve().is_relative_to((_ROOT/'environment').resolve()) for p in paths[1:])):
        raise ValueError('Native verification managed invocation differs')
    if any(env.get(key)!=value or key not in env for key,value in
            _tree_test_environment(expected_profile,paths[1:],root).items()):
        raise ValueError('Native tree fixture overlay/output controls differ from their managed profile')
    count=checked_junit(root/'worker/junit.xml')
    if worker['test_cases']!=count or worker['junit_sha256']!=_sha(root/'worker/junit.xml'):raise ValueError('Native verification test result changed')
    if request['identity']!=verification_identity() or canonical_hash(_inventory(root))!=reference['inventory_sha256']:
        raise ValueError('Verification source or evidence changed during loading')
    return dict(schema='research/v2-authenticated-native-implementation-verification',profile=expected_profile,
        families=verification_profiles()[expected_profile]['families'],test_cases=count,software_verified=True,
        identity=request['identity'],reference=reference,native_workload_qualified=False,quality_eligible=False,T_accessed=False,
        scope='Fixed source/test/lock-bound software and direct-author parity fixtures, not real native anchor/workload/support admission or predictive efficacy')


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--expected-request-sha256',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    result=_worker(args.request,args.expected_request_sha256,args.output)
    return 0 if result['software_verified'] else 1


if __name__=='__main__':raise SystemExit(main())
