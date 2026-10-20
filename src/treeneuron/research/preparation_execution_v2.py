"""Owned current-source raw-table preparation; no modeling or protected evaluation.

The dedicated worker parses raw targets to seal roles. The proof reader only
hashes role archives. Managed generation is trusted execution, not an independent
numerical reconstruction of those archives by the reader.
"""
import argparse
import base64
import copy
import csv
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time

from treeneuron.config import canonical_hash
from treeneuron.data.tabular import DATASETS
from .deployment_worker_v2 import _json,_path
from .immutable_inputs import stage_immutable_file
from .ledger import write_new_json
from .method_worker import _ROOT,_THREADS,_sha,_digest,scientific_source_sha256
from .native_workload_v2 import _bound_reader,_process_observation
from .prepared_lineage_v2 import audit_current_prepared_transfer_suite
from .process_supervision_v2 import supervise_process
from .workqueue import _inventory


_LOCKS=('locks/research-v2/core.json','locks/research-v2/data-requirements.txt','locks/research-v2/data-environment.json')


def _read(path):return _json(Path(path).read_bytes())
def _normal(name):return name.lower().replace('_','-')
def _locks():return {name:_sha(_ROOT/name) for name in _LOCKS}


def _inputs(config_path,expected):
    path=Path(config_path)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file() or not _digest(expected) or _sha(path)!=expected):
        raise ValueError('Preparation configuration commitment differs')
    cfg=_read(path)
    if (type(cfg) is not dict or set(cfg)!={'schema','datasets','acquisition_manifest','acquisition_manifest_sha256'}
            or cfg['schema']!='research/v2-data-preparation' or cfg['datasets']!=list(DATASETS)
            or type(cfg['acquisition_manifest']) is not str or not _digest(cfg['acquisition_manifest_sha256'])):
        raise ValueError('Preparation configuration schema differs')
    acq=Path(cfg['acquisition_manifest']);acq=acq if acq.is_absolute() else _ROOT/acq
    if acq.is_symlink() or not acq.is_file() or _sha(acq)!=cfg['acquisition_manifest_sha256']:
        raise ValueError('Preparation acquisition bytes differ')
    entries=_read(acq).get('datasets')
    if (type(entries) is not list or len(entries)!=6 or any(type(e) is not dict or type(e.get('dataset_id')) is not int for e in entries)
            or {e['dataset_id'] for e in entries}!={s.uci_id for s in DATASETS.values()}):
        raise ValueError('Preparation requires six exact official acquisitions')
    refs=[dict(path=str(path),sha256=expected),dict(path=str(acq.resolve()),sha256=cfg['acquisition_manifest_sha256'])]
    for schema in DATASETS.values():
        row=next(e for e in entries if e['dataset_id']==schema.uci_id);members=row.get('matched_members')
        if row.get('status')!='raw_hash_matched' or type(members) is not list or len(members)!=1:
            raise ValueError('Missing official raw preparation asset')
        member=members[0];raw=_path(acq.parent,member['path'])
        if (member.get('sha256')!=schema.raw_sha256 or raw.is_symlink() or not raw.is_file()
                or type(member.get('bytes')) is not int or raw.stat().st_size!=member['bytes'] or _sha(raw)!=schema.raw_sha256):
            raise ValueError('Official raw preparation bytes differ')
        refs.append(dict(path=str(raw),sha256=schema.raw_sha256))
    return dict(config=cfg,config_path=str(path),config_sha256=expected,acquisition_manifest=str(acq.resolve()),
        acquisition_manifest_sha256=cfg['acquisition_manifest_sha256'],files=refs)


def _overlay(directory):
    root=Path(directory);expected=_read(_ROOT/_LOCKS[2])
    if (not root.is_absolute() or root.resolve()!=root or root.is_symlink()
            or not root.is_dir() or not root.is_relative_to((_ROOT/'environment').resolve())):
        raise ValueError('Preparation needs one pinned project data overlay')
    wanted={r['name']:r for r in expected['distributions']};actual={};files={}
    for dist in importlib.metadata.distributions(path=[str(root)]):
        name=_normal(dist.metadata['Name'])
        if name in actual or name not in wanted:raise ValueError('Duplicate or unexpected private data distribution')
        record=Path(dist._path)/'RECORD';row=wanted[name]
        if dist.version!=row['version'] or _sha(record)!=row['record_sha256']:
            raise ValueError('Private data version or locked RECORD differs')
        actual[name]=dict(version=dist.version,distribution_path=str(Path(dist._path).resolve()),record_sha256=_sha(record))
        for entry in csv.reader(record.read_text().splitlines()):
            p=root/entry[0]
            if '__pycache__' in p.parts or p.suffix=='.pyc':continue
            if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
                raise ValueError('Private data file escaped its locked overlay')
            digest=_sha(p)
            if entry[1]:
                mode,encoded=entry[1].split('=',1)
                if (mode!='sha256' or base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')!=encoded
                        or entry[2] and p.stat().st_size!=int(entry[2])):raise ValueError('Private data file differs from its RECORD')
            files[str(p.relative_to(root))]=digest
    if set(actual)!=set(wanted) or {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}!=set(files):
        raise ValueError('Data overlay must contain exactly the three locked distributions without generated bytecode')
    return dict(root=str(root),distributions=actual,files=files)


def _runtime(overlay):
    expected=_read(_ROOT/_LOCKS[0]);wanted={_normal(r['name']):r for r in expected['distributions']};actual={}
    if sys.executable!=expected['executable'] or sys.version!=expected['python'] or platform.platform()!=expected['platform']:
        raise ValueError('Preparation interpreter/platform differs from its qualified core')
    for dist in importlib.metadata.distributions():
        name=_normal(dist.metadata['Name'])
        if name in actual:raise ValueError('Duplicate preparation runtime distribution')
        actual[name]=dist
    if set(actual)!=set(wanted)|set(overlay['distributions']):raise ValueError('Preparation runtime requires exactly27core+3data distributions')
    for name,row in wanted.items():
        dist=actual[name]
        if (dist.version!=row['version'] or str(Path(dist._path).resolve())!=row['path'] or _sha(Path(dist._path)/'RECORD')!=row['record_sha256']):
            raise ValueError('Preparation core runtime identity differs')
    for name,row in overlay['distributions'].items():
        dist=actual[name]
        if dist.version!=row['version'] or str(Path(dist._path).resolve())!=row['distribution_path']:
            raise ValueError('Preparation private import environment differs')
    from .transfer import _data_environment
    dependencies=_data_environment();imports={}
    for name in ('numpy','pandas','openpyxl','xlrd','et-xmlfile'):
        module=importlib.import_module(name.replace('-','_'));p=Path(module.__file__).resolve();dist=actual[name]
        relative=p.relative_to(Path(dist.locate_file('')).resolve()).as_posix()
        record=next(r for r in csv.reader(dist.read_text('RECORD').splitlines()) if r[0]==relative)
        if not record[1] or record[1]!='sha256='+base64.urlsafe_b64encode(bytes.fromhex(_sha(p))).decode().rstrip('='):
            raise ValueError('Actual preparation import bytes differ from their locked RECORD')
        imports[name]=dict(path=str(p),sha256=_sha(p))
    from .method_worker import loaded_method_modules
    if loaded_method_modules():raise ValueError('Model runtime imported during dedicated data preparation')
    result=dict(schema='research/v2-data-runtime',interpreter=sys.executable,interpreter_sha256=_sha(Path(sys.executable)),python=sys.version,
        platform=platform.platform(),
        core_manifest_sha256=_sha(_ROOT/_LOCKS[0]),distribution_count=len(actual),data_dependencies=dependencies,
        imports=imports,model_modules_imported=[])
    _validate_runtime_record(result,overlay)
    return result


def _validate_runtime_record(runtime,overlay):
    """Check captured import identities without importing private packages here."""
    core=_read(_ROOT/_LOCKS[0]);by_name={_normal(r['name']):r for r in core['distributions']}
    fields={'schema','interpreter','interpreter_sha256','python','platform','core_manifest_sha256',
        'distribution_count','data_dependencies','imports','model_modules_imported'}
    if (type(runtime) is not dict or set(runtime)!=fields or runtime['schema']!='research/v2-data-runtime'
            or runtime['interpreter']!=core['executable'] or runtime['python']!=core['python'] or runtime['platform']!=core['platform']
            or runtime['interpreter_sha256']!=_sha(Path(core['executable']))
            or runtime['core_manifest_sha256']!=_sha(_ROOT/_LOCKS[0])
            or type(runtime['distribution_count']) is not int or runtime['distribution_count']!=30
            or runtime['model_modules_imported']!=[]):raise ValueError('Preparation runtime record differs')
    dependencies={};imports={}
    for name in ('numpy','pandas','openpyxl','xlrd','et-xmlfile'):
        if name in by_name:
            row=by_name[name];dist_path=Path(row['path'])
        else:
            row=overlay['distributions'][name];dist_path=Path(row['distribution_path'])
        base=dist_path.parent;module_path=(base/name.replace('-','_')/'__init__.py').resolve()
        record=dist_path/'RECORD'
        if record.is_symlink() or _sha(record)!=row['record_sha256'] or module_path.is_symlink():
            raise ValueError('Preparation import RECORD or location differs')
        entries=list(csv.reader(record.read_text().splitlines()));relative=module_path.relative_to(base).as_posix()
        entry=next((r for r in entries if r[0]==relative),None);digest=_sha(module_path)
        if not entry or entry[1]!='sha256='+base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('='):
            raise ValueError('Preparation import bytes differ from locked RECORD')
        imports[name]=dict(path=str(module_path),sha256=digest)
        dependencies[name]=dict(version=row['version'],import_path=str(module_path),distribution_path=str(base),record_sha256=row['record_sha256'])
    if runtime['imports']!=imports or runtime['data_dependencies']!=dependencies:
        raise ValueError('Preparation recorded import identity differs from actual pinned files')


def _request(path,expected):
    if not _digest(expected) or Path(path).is_symlink() or _sha(Path(path))!=expected:raise ValueError('Preparation request bytes differ')
    value=_read(path)
    if (type(value) is not dict or set(value)!={'schema','source_sha256','locks','inputs','staged_config','overlay'}
            or value['schema']!='research/v2-managed-preparation-request'
            or value['source_sha256']!=scientific_source_sha256() or value['locks']!=_locks()
            or value['overlay']!=_overlay(value['overlay']['root'])
            or value['inputs']!=_inputs(value['inputs']['config_path'],value['inputs']['config_sha256'])
            or _sha(Path(value['staged_config']))!=value['inputs']['config_sha256']):
        raise ValueError('Managed preparation source/lock/input/runtime request differs')
    return value


def _suite(data,request):
    path=data/'suite.json';suite=_read(path)
    execution=data.parent.parent
    argv=[str(_ROOT/'src/treeneuron/research/preparation_execution_v2.py'),'--request',str(execution/'request.json'),
        '--expected-request-sha256',_sha(execution/'request.json'),'--output',str(data.parent)]
    env={k:'1' for k in _THREADS};env['PYTORCH_ENABLE_MPS_FALLBACK']='0'
    if (suite.get('source_sha256')!=request['source_sha256'] or suite.get('config')!=request['inputs']['config']
            or suite.get('models_fitted') is not False or suite.get('protected_labels_read_for_modeling') is not False):
        raise ValueError('Prepared suite source/config/authority differs')
    if (suite.get('argv')!=argv or suite.get('cwd')!=str(_ROOT) or suite.get('environment_overrides')!=env
            or _read(data/'preparation-status.json')!=suite):
        raise ValueError('Prepared suite invocation or published closure differs')
    return audit_current_prepared_transfer_suite(prepared_root=data,expected_suite_sha256=_sha(path),
        acquisition_manifest=request['inputs']['acquisition_manifest'],
        expected_acquisition_sha256=request['inputs']['acquisition_manifest_sha256'])


def _worker(path,expected,output):
    request=_request(path,expected);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    runtime=_runtime(request['overlay'])
    from .transfer import prepare_data
    prepare_data(request['staged_config'],root/'data')
    checked=_suite(root/'data',request)
    if _request(path,expected)!=request or _runtime(request['overlay'])!=runtime:
        raise ValueError('Preparation identity changed during raw parsing/sealing')
    suite=_read(root/'data/suite.json')
    if suite['dependencies']!=runtime['data_dependencies']:raise ValueError('Actual preparation imports differ from runtime proof')
    write_new_json(root/'terminal.json',dict(schema='research/v2-managed-preparation-worker',status='completed_preparation',
        source_sha256=request['source_sha256'],request_sha256=expected,locks=request['locks'],runtime=runtime,
        suite_sha256=_sha(root/'data/suite.json'),datasets=checked['datasets'],role_archives_hashed=checked['role_archives_hashed'],
        protected_label_archives_hashed=checked['protected_label_archives_hashed'],models_fit=0,
        raw_targets_parsed_for_preparation=True,protected_labels_prepared=True,protected_labels_read_for_modeling=False,
        quality_eligible=False,protected_access_authorized=False))


def run_managed_preparation(config_path,*,expected_config_sha256,output,interpreter,data_overlay,rss_limit_bytes=4*1024**3):
    started=time.perf_counter_ns();path=Path(config_path).absolute();inputs=_inputs(path,expected_config_sha256)
    overlay=_overlay(Path(data_overlay));source=scientific_source_sha256();locks=_locks();original_output=Path(output)
    if original_output.is_symlink():raise ValueError('Ordinary fresh preparation output required')
    root=original_output.resolve()
    interpreter=Path(interpreter)
    if (root.exists() or root.is_symlink() or not interpreter.is_absolute() or not interpreter.is_file()
            or type(rss_limit_bytes) is not int or rss_limit_bytes<=0
            or any(root.resolve().is_relative_to((_ROOT/name).resolve()) for name in ('src','tests','locks','configs','protocol'))):
        raise ValueError('Fresh preparation output, qualified interpreter and owned RSS guard required')
    root.mkdir(parents=True);(root/'input').mkdir();staged=root/'input/config.json'
    stage_immutable_file(path,staged,expected_sha256=expected_config_sha256)
    request=dict(schema='research/v2-managed-preparation-request',source_sha256=source,locks=locks,
        inputs=inputs,staged_config=str(staged),overlay=overlay)
    write_new_json(root/'request.json',request);digest=_sha(root/'request.json')
    argv=[str(interpreter),'-B','-m','treeneuron.research.preparation_execution_v2','--request',str(root/'request.json'),
        '--expected-request-sha256',digest,'--output',str(root/'worker')]
    env={k:'1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}
    env.update(PYTHONPATH=os.pathsep.join([str(_ROOT/'src'),overlay['root']]),KMP_DUPLICATE_LIB_OK=None,PYTORCH_ENABLE_MPS_FALLBACK='0')
    process=supervise_process(argv=argv,cwd=_ROOT,env_overrides=env,output=root/'process',timeout_ns=None,
        rss_limit_bytes=rss_limit_bytes,identity=dict(purpose='official_data_preparation',request_sha256=digest))
    worker=_read(root/'worker/terminal.json') if (root/'worker/terminal.json').is_file() else None
    completed=(process['execution_status']=='exited' and process['returncode']==0 and process['child_reaped'] is True
        and worker is not None and worker.get('status')=='completed_preparation')
    if _request(root/'request.json',digest)!=request:raise ValueError('Preparation request changed during managed execution')
    result=dict(schema='research/v2-managed-preparation-result',status='completed_preparation' if completed else 'preparation_failed',
        request_sha256=digest,process_sha256=_sha(root/'process/process.json'),
        worker_terminal_sha256=_sha(root/'worker/terminal.json') if worker is not None else None,
        elapsed_before_publication_ns=time.perf_counter_ns()-started,
        clock_scope='Containing entry through result preparation; caller pays publication/inventory/return',
        native_workload_qualified=False,quality_eligible=False,protected_access_authorized=False)
    write_new_json(root/'result.json',result)
    return dict(result,reference=dict(root=str(root),inventory_sha256=canonical_hash(_inventory(root))))


def load_preparation_execution(reference):
    reference=copy.deepcopy(reference)
    if type(reference) is not dict or set(reference)!={'root','inventory_sha256'} or not _digest(reference['inventory_sha256']):
        raise ValueError('Exact external preparation inventory commitment required')
    root=Path(reference['root'])
    if not root.is_absolute() or root.is_symlink() or root.resolve()!=root:raise ValueError('Ordinary closed preparation root required')
    inventory=_inventory(root)
    if canonical_hash(inventory)!=reference['inventory_sha256']:raise ValueError('Preparation inventory bytes differ')
    read,refs=_bound_reader(root,inventory);result=read('result.json');request=_request(root/'request.json',result['request_sha256'])
    worker=read('worker/terminal.json');process=read('process/process.json')
    result_fields={'schema','status','request_sha256','process_sha256','worker_terminal_sha256','elapsed_before_publication_ns',
        'clock_scope','native_workload_qualified','quality_eligible','protected_access_authorized'}
    if type(result) is not dict or set(result)!=result_fields or result['schema']!='research/v2-managed-preparation-result':
        raise ValueError('Managed preparation result schema differs')
    if (result.get('status')!='completed_preparation'
            or type(result['elapsed_before_publication_ns']) is not int or result['elapsed_before_publication_ns']<=0
            or any(result.get(k) is not False for k in ('native_workload_qualified','quality_eligible','protected_access_authorized'))
            or worker.get('schema')!='research/v2-managed-preparation-worker'
            or worker.get('status')!='completed_preparation' or worker.get('request_sha256')!=result['request_sha256']
            or worker.get('source_sha256')!=request['source_sha256'] or worker.get('locks')!=request['locks']
            or result['process_sha256']!=refs['process/process.json'] or result['worker_terminal_sha256']!=refs['worker/terminal.json']
            or process.get('execution_status')!='exited' or process.get('returncode')!=0 or process.get('child_reaped') is not True
            or process.get('timeout_ns') is not None or process.get('cwd')!=str(_ROOT)
            or process.get('identity')!=dict(purpose='official_data_preparation',request_sha256=result['request_sha256'])
            or type(worker.get('models_fit')) is not int or worker['models_fit']!=0
            or worker.get('raw_targets_parsed_for_preparation') is not True or worker.get('protected_labels_prepared') is not True
            or any(worker.get(k) is not False for k in ('protected_labels_read_for_modeling','quality_eligible','protected_access_authorized'))):
        raise ValueError('Managed preparation process/runtime/authority record differs')
    argv=process['argv'];env=process['environment_overrides'];runtime=worker['runtime']
    _validate_runtime_record(runtime,request['overlay'])
    expected=['-B','-m','treeneuron.research.preparation_execution_v2','--request',str(root/'request.json'),
        '--expected-request-sha256',result['request_sha256'],'--output',str(root/'worker')]
    if (argv[1:]!=expected or argv[0]!=runtime.get('interpreter') or _sha(Path(argv[0]))!=runtime['interpreter_sha256']
            or any(env.get(k)!='1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE'))
            or env.get('PYTHONPATH')!=os.pathsep.join([str(_ROOT/'src'),request['overlay']['root']])
            or env.get('KMP_DUPLICATE_LIB_OK') is not None or env.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0'
            or runtime.get('distribution_count')!=30 or runtime.get('model_modules_imported')!=[]
            or runtime.get('core_manifest_sha256')!=request['locks'][_LOCKS[0]]):
        raise ValueError('Managed preparation invocation/runtime identity differs')
    checked=_suite(root/'worker/data',request);suite=read('worker/data/suite.json')
    if (worker['suite_sha256']!=refs['worker/data/suite.json'] or worker['datasets']!=checked['datasets']
            or worker['role_archives_hashed']!=checked['role_archives_hashed']
            or worker['protected_label_archives_hashed']!=checked['protected_label_archives_hashed']
            or suite['dependencies']!=runtime['data_dependencies']):raise ValueError('Managed preparation outputs/imports differ')
    observation=_process_observation(root,read,refs)
    if _request(root/'request.json',result['request_sha256'])!=request or _inventory(root)!=inventory:
        raise ValueError('Preparation proof inputs/output changed while reading')
    return dict(schema='research/v2-authenticated-preparation-execution',reference=reference,source_sha256=request['source_sha256'],
        preparation_execution_authenticated=True,preparation_runtime_authenticated=True,raw_to_sealed_generation_chain_authenticated=True,
        datasets=checked['datasets'],role_archives_hashed=checked['role_archives_hashed'],
        protected_label_archives_hashed=checked['protected_label_archives_hashed'],process_observation=observation,
        arrays_or_labels_decoded=False,data_content_independently_rederived=False,models_fit=0,
        raw_targets_parsed_by_preparation_worker=True,protected_labels_read_for_modeling=False,
        native_workload_qualified=False,quality_eligible=False,protected_access_authorized=False,
        scope='Trusted dedicated managed raw-table parsing/sealing, current source/locks/runtime and exact raw/sealed bytes; reader hashes opaque archives without independent numerical reconstruction or modeling/protected authority')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--expected-request-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
    _worker(args.request,args.expected_request_sha256,args.output)


if __name__=='__main__':main()
