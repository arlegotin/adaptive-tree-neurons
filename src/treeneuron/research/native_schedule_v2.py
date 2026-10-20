"""Owned label-free RealMLP schedule preparation; no fit or quality authority."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from treeneuron.config import canonical_hash,canonical_json
from .deployment_worker_v2 import _ROOT,_THREADS,_source,_bytes,_json
from .ledger import write_new_json
from .process_supervision_v2 import supervise_process
from .seeds import seed64
from .workqueue import _inventory


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_request(path,expected):
    path=Path(path)
    if path.is_symlink():raise ValueError('Ordinary native schedule request required')
    value=_json(_bytes(path,expected));fields={'schema','scientific_source_sha256','feature_schema','hpo_seed'}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-native-schedule-request'
            or value['scientific_source_sha256']!=_source() or type(value['hpo_seed']) is not int
            or not 0<=value['hpo_seed']<2**63):raise ValueError('Native schedule source/seed/request differs')
    schema=value['feature_schema']
    if (type(schema) is not dict or set(schema)!={'columns','categorical'}
            or type(schema['columns']) is not list or not 1<=len(schema['columns'])<=128
            or any(type(v) is not str or not v for v in schema['columns'])
            or len(set(schema['columns']))!=len(schema['columns'])
            or type(schema['categorical']) is not list or any(type(v) is not str for v in schema['categorical'])
            or len(set(schema['categorical']))!=len(schema['categorical'])
            or not set(schema['categorical'])<=set(schema['columns'])):
        raise ValueError('Native schedule receives semantic feature names only, never labels/files')
    return value


def _worker(request_path,expected,output):
    started=time.perf_counter_ns();request=checked_request(request_path,expected);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    record=dict(schema='research/v2-native-schedule-worker',status='input_or_dependency_failure',
        request_sha256=expected,scientific_source_sha256=request['scientific_source_sha256'],schedule=None,
        native_fits=0,labels_read=[],quality_eligible=False,protected_access_authorized=False,T_accessed=False)
    try:
        if not sys.dont_write_bytecode or any(os.environ.get(k)!='1' for k in _THREADS):
            raise ValueError('Native schedule requires the declared fresh process/thread treatment')
        from .method_worker import _initialize_native_threads
        _initialize_native_threads('P06')
        from .realmlp_search import realmlp_schedule
        record['schedule']=realmlp_schedule(request['feature_schema'],hpo_seed=request['hpo_seed'])
        if checked_request(request_path,expected)!=request:raise ValueError('Native schedule request changed')
        record['status']='completed_native_schedule'
    except BaseException as error:
        record.update(error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
    finally:
        record['elapsed_before_publication_ns']=time.perf_counter_ns()-started
        write_new_json(output/'terminal.json',record)
    return record


def prepare_realmlp_native_schedule(*,feature_schema,hpo_seed,output,interpreter,
        extra_pythonpath=(),rss_limit_bytes=4*1024**3):
    started=time.perf_counter_ns();output=Path(output);interpreter=Path(interpreter)
    if (output.exists() or output.is_symlink() or not interpreter.is_absolute() or not interpreter.is_file()
            or type(rss_limit_bytes) is not int or rss_limit_bytes<=0 or not isinstance(extra_pythonpath,(tuple,list))):
        raise ValueError('Fresh schedule output, qualified interpreter and owned RSS guard required')
    overlays=[]
    for value in extra_pythonpath:
        path=Path(value).resolve()
        if not path.is_dir() or not path.is_relative_to((_ROOT/'environment').resolve()):raise ValueError('Pinned project schedule overlay required')
        overlays.append(str(path))
    request=dict(schema='research/v2-native-schedule-request',scientific_source_sha256=_source(),
        feature_schema=feature_schema,hpo_seed=hpo_seed)
    # Validate before creating any process/output. This temporary serialization
    # is not a label-bearing input and cannot add arbitrary worker fields.
    if type(hpo_seed) is not int or not 0<=hpo_seed<2**63:raise ValueError('Fixed native63-bit HPO seed required')
    from .search_supports_v2 import family_search_spec
    family_search_spec('P06',feature_schema)
    output.mkdir(parents=True,exist_ok=False);path=output/'request.json';write_new_json(path,request);expected=_sha(path)
    checked_request(path,expected)
    env={k:'1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}
    env.update(PYTHONPATH=os.pathsep.join([str(_ROOT/'src'),*overlays]),PYTORCH_ENABLE_MPS_FALLBACK='0',
        KMP_DUPLICATE_LIB_OK=None,MPLCONFIGDIR=str((output/'cache/mpl').resolve()),XDG_CACHE_HOME=str((output/'cache/xdg').resolve()))
    process=supervise_process(argv=[str(interpreter),'-B','-m','treeneuron.research.native_schedule_v2',
        '--request',str(path.resolve()),'--expected-request-sha256',expected,'--output',str((output/'worker').resolve())],
        cwd=_ROOT,env_overrides=env,output=output/'process',timeout_ns=None,rss_limit_bytes=rss_limit_bytes,
        identity=dict(purpose='native_schedule',source_sha256=request['scientific_source_sha256'],request_sha256=expected))
    status=process['execution_status']
    if status=='exited':
        terminal=_json((output/'worker/terminal.json').read_bytes())
        status=terminal['status']
        if (process['returncode']==0)!=(status=='completed_native_schedule'):raise ValueError('Native schedule process/worker outcomes contradict')
    record=dict(schema='research/v2-native-schedule-result',status=status,process=process,
        request_sha256=expected,source_sha256=request['scientific_source_sha256'],native_fits=0,labels_read=[],
        quality_eligible=False,protected_access_authorized=False,T_accessed=False,
        elapsed_before_publication_ns=time.perf_counter_ns()-started,
        clock_scope='Helper entry through result preparation. Caller additionally pays final publication/inventory/return; process time is nested.')
    if checked_request(path,expected)!=request:raise ValueError('Native schedule inputs/source changed')
    write_new_json(output/'result.json',record)
    reference=dict(root=str(output.resolve()),inventory_sha256=canonical_hash(_inventory(output))) if status=='completed_native_schedule' else None
    return dict(record,reference=reference)


def load_realmlp_native_schedule(reference,*,expected_hpo_seed,expected_feature_schema):
    if type(reference) is not dict or set(reference)!={'root','inventory_sha256'}:raise ValueError('External native schedule inventory required')
    root=Path(reference['root']);source=_source()
    if (not root.is_absolute() or root.resolve()!=root or root.is_symlink() or not root.is_dir()
            or canonical_hash(_inventory(root))!=reference['inventory_sha256']):raise ValueError('Native schedule bytes changed')
    saved=_json((root/'result.json').read_bytes());request=checked_request(root/'request.json',saved['request_sha256'])
    terminal=_json((root/'worker/terminal.json').read_bytes());process=_json((root/'process/process.json').read_bytes())
    if (saved['schema']!='research/v2-native-schedule-result' or saved['status']!='completed_native_schedule'
            or saved['source_sha256']!=source or saved['process']!=process or request['hpo_seed']!=expected_hpo_seed
            or type(expected_hpo_seed) is not int or request['feature_schema']!=expected_feature_schema
            or terminal['schema']!='research/v2-native-schedule-worker' or terminal['status']!='completed_native_schedule'
            or terminal['scientific_source_sha256']!=source or terminal['request_sha256']!=saved['request_sha256']
            or any(v.get(k) is not False for v in (saved,terminal) for k in ('quality_eligible','protected_access_authorized','T_accessed'))
            or any(v.get('native_fits')!=0 or v.get('labels_read')!=[] for v in (saved,terminal))
            or process['execution_status']!='exited' or process['returncode']!=0 or process['child_reaped'] is not True
            or process['timeout_ns'] is not None or process['cwd']!=str(_ROOT)
            or process['identity']!=dict(purpose='native_schedule',source_sha256=source,request_sha256=saved['request_sha256'])):
        raise ValueError('Closed native schedule process/source/seed/schema differs')
    argv=process['argv'];expected=['-B','-m','treeneuron.research.native_schedule_v2','--request',str(root/'request.json'),
        '--expected-request-sha256',saved['request_sha256'],'--output',str(root/'worker')]
    env=process['environment_overrides'];paths=env.get('PYTHONPATH','').split(os.pathsep)
    if (Path(argv[0]).resolve()!=Path(sys.executable).resolve() or argv[1:]!=expected
            or any(env.get(k)!='1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE'))
            or env.get('KMP_DUPLICATE_LIB_OK') is not None or not paths or paths[0]!=str(_ROOT/'src')
            or any(not Path(p).resolve().is_relative_to((_ROOT/'environment').resolve()) for p in paths[1:])):
        raise ValueError('Native schedule managed invocation differs')
    schedule=terminal['schedule']
    from .realmlp_native import RealMLPConfig,_source as native_source
    if (schedule['schema']!='research/v2-realmlp-search-schedule' or schedule['hpo_seed']!=expected_hpo_seed
            or schedule['source']!=native_source() or schedule['native_fits']!=0 or schedule['labels_read']!=[]
            or schedule['eligible_predictive_result'] is not False or schedule['protected_access_authorized'] is not False
            or len(schedule['suggestions'])!=32):raise ValueError('Pinned official schedule identity changed')
    configs=[]
    for i,row in enumerate(schedule['suggestions']):
        cfg=RealMLPConfig.parse(row['config']).to_state()
        if (row['family_id']!='P06' or row['attempt_index']!=i or row['hpo_seed']!=expected_hpo_seed
                or row['native_sampler_seed']!=(None if i==0 else seed64(expected_hpo_seed,'P06','official-default-sampler',str(i)))
                or cfg['feature_schema']!=expected_feature_schema or cfg['mode']!=('native_anchor' if i==0 else 'search')
                or row['native_distribution']!=dict(is_classification=False,hpo_space_name='default')
                or row['native_fits']!=0 or row['labels_read']!=[] or row['eligible_predictive_result'] is not False
                or row['protected_access_authorized'] is not False
                or i==0 and cfg['parameters']!={}
                or i>0 and (set(cfg['parameters'])!={'num_emb_type','add_front_scale','lr','p_drop','wd','plr_sigma','act','hidden_sizes'}
                    or any(row['official_sampled_recipe'][k]!=v for k,v in cfg['parameters'].items()))):
            raise ValueError('Official native schedule row/configuration/seed differs')
        configs.append(cfg)
    if source!=_source() or canonical_hash(_inventory(root))!=reference['inventory_sha256']:raise ValueError('Native schedule changed during reading')
    return dict(schema='research/v2-authenticated-native-schedule',configurations=configs,
        paid_preparation_ns=saved['elapsed_before_publication_ns'],clock_scope=saved['clock_scope'],
        native_fits=0,labels_read=[],reference=reference,independently_resampled=False,
        native_workload_qualified=False,protected_access_authorized=False,T_accessed=False)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    parser.add_argument('--expected-request-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);result=_worker(args.request,args.expected_request_sha256,args.output)
    return 0 if result['status']=='completed_native_schedule' else 1


if __name__=='__main__':raise SystemExit(main())
