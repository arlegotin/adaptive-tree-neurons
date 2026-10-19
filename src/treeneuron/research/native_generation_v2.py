"""Read-only primary native-fit lineage; no qualification or search authority.

This checks source-bound receipts and byte/role alignment. It does not retrain a
model, reproduce its optimization, qualify a native implementation or score V.
"""
import os
from pathlib import Path
import sys

import numpy as np

from treeneuron.config import canonical_hash
from .method_worker import _ROOT,_THREADS,_digest,_read_json,_sha,_validate_request,scientific_source_sha256
from .transfer import TransferStore
from .trial_bridge_v2 import EngineeringTrialBridge
from .workload import _checked_manifest
from .workqueue import _inventory


def audit_native_generation(*,request_path,expected_request_sha256,native_reference,
        prepared_root,expected_manifest_sha256,outer_fold,expected_fit_seed,expected_configuration):
    request_path=Path(request_path).resolve();prepared_root=Path(prepared_root).resolve()
    manifest_path=prepared_root/'manifest.json';before=scientific_source_sha256()
    if (not _digest(expected_manifest_sha256) or manifest_path.is_symlink()
            or not manifest_path.is_file() or _sha(manifest_path)!=expected_manifest_sha256):
        raise ValueError('Externally committed primary prepared manifest required')
    store=TransferStore(prepared_root);manifest=store.manifest
    _checked_manifest(manifest['dataset'],manifest)
    if type(outer_fold) is not int or outer_fold!=(4 if manifest['dataset']=='bike' else 0):
        raise ValueError('This generation verifier admits primary roles only')
    split=store._split(outer_fold)
    request,_=_validate_request(request_path,expected_request_sha256,
        recoverable_training=_read_json(request_path).get('family_id')=='P12')
    family=request['family_id']
    if (request['budget']!={'native_stopping':True} or type(expected_fit_seed) is not int
            or request['fit_seed']!=expected_fit_seed or request['config']!=expected_configuration
            or family!='P13' and request['config'].get('mode') not in ('native_anchor','search')
            or request['config']['feature_schema']!=dict(columns=manifest['feature_names'],categorical=manifest['categorical_features'])
            or request['observation_identity']!=dict(dataset=manifest['dataset'],split_kind='temporal' if manifest['dataset']=='bike' else 'grouped',split_sha256=split['split_hash'])):
        raise ValueError('Native generation seed/configuration/budget/primary split differs')
    for role in ('F','G','V_features'):
        relative=f'outer-{outer_fold}/'+('V-features.npz' if role=='V_features' else role+'.npz')
        if (request['files'][role]['sha256']!=manifest['files'][relative]['sha256']
                or _sha(prepared_root/relative)!=request['files'][role]['sha256']):
            raise ValueError('Native fitting roles differ from their committed primary dataset')
    if (type(native_reference) is not dict or set(native_reference)!={'root','inventory_sha256','request_sha256'}
            or type(native_reference['root']) is not str or not _digest(native_reference['inventory_sha256'])
            or native_reference['request_sha256']!=expected_request_sha256):
        raise ValueError('External closed native-attempt inventory commitment required')
    root=Path(native_reference['root'])
    if (not root.is_absolute() or root.resolve()!=root or root.is_symlink() or not root.is_dir()
            or canonical_hash(_inventory(root))!=native_reference['inventory_sha256']):
        raise ValueError('Native attempt bytes differ from their external inventory')
    native=_read_json(root/'result.json');process=_read_json(root/'process/process.json')
    if (native.get('schema')!='research/v2-native-fit-result' or native.get('status')!='completed_ineligible'
            or native.get('request_sha256')!=expected_request_sha256 or native.get('source_sha256')!=before
            or native.get('family_id')!=family or native.get('fit_seed')!=expected_fit_seed
            or native.get('child_reaped') is not True or native.get('child_started') is not True
            or any(native.get(k) is not False for k in ('quality_eligible','V_labels_read','T_accessed'))
            or native.get('process_receipt_sha256')!=_sha(root/'process/process.json')
            or process.get('schema')!='research/v2-process-receipt' or process.get('execution_status')!='exited'
            or process.get('returncode')!=0 or process.get('child_reaped') is not True
            or process.get('timeout_ns') is not None or process.get('cwd')!=str(_ROOT)):
        raise ValueError('Successful closed managed native fit required; no failed/capped promotion')
    argv=process.get('argv',[])
    expected_argv=[sys.executable,'-B','-m','treeneuron.research.method_worker','--request',str(request_path),
        '--expected-request-sha256',expected_request_sha256,'--output',str(root/'worker')]
    if family=='P12':
        expected_argv+=['--training-checkpoints',str(root/'training-checkpoints')]
        if native.get('recovery',{}).get('resume') is True:expected_argv.append('--resume')
    if (not argv or Path(argv[0]).resolve()!=Path(sys.executable).resolve() or argv[1:]!=expected_argv[1:]
            or process.get('identity')!=dict(request_sha256=expected_request_sha256,scientific_source_sha256=before,
                family_id=family,fit_seed=expected_fit_seed,native_stopping=True,recovery=native.get('recovery'))):
        raise ValueError('Native process is not the declared managed source/request invocation')
    env=process.get('environment_overrides',{})
    paths=env.get('PYTHONPATH','').split(os.pathsep)
    if (any(env.get(k)!='1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE'))
            or env.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or env.get('KMP_DUPLICATE_LIB_OK') is not None
            or not paths or paths[0]!=str(_ROOT/'src')
            or any(not Path(p).resolve().is_relative_to((_ROOT/'environment').resolve()) for p in paths[1:])):
        raise ValueError('Native process source/thread/overlay environment differs')
    refs=process.get('files',[])
    if {r.get('path') for r in refs}!={'started.json','spawned.json','stdout.log','stderr.log','resource-samples.jsonl'}:
        raise ValueError('Closed managed-process evidence inventory differs')
    for ref in refs:
        path=root/'process'/ref['path']
        if path.stat().st_size!=ref['bytes'] or _sha(path)!=ref['sha256']:raise ValueError('Process evidence bytes changed')
    worker=EngineeringTrialBridge._worker_terminal(root/'worker',request,expected_request_sha256)
    if (worker.get('status')!='completed_ineligible' or worker.get('snapshot_reload_exact') is not True
            or native.get('worker_terminal_sha256')!=_sha(root/'worker/terminal.json')
            or worker.get('budget_resolution')!=dict(requested={'native_stopping':True},
                adapter={'native_stopping':True} if family=='P12' else {},policy='native_stopping')):
        raise ValueError('Native closed worker/reload/budget provenance differs')
    snapshot_name='state.json' if family in ('P08','P09','P10','P11') else 'manifest.json'
    snapshot=root/'worker/snapshot'/snapshot_name
    if worker.get('snapshot_commitment')!=dict(path='snapshot/'+snapshot_name,sha256=_sha(snapshot)):
        raise ValueError('Closed native snapshot pointer differs')
    store._access(outer_fold,['F_labels','G_labels','V_features'])
    _,arrays=_validate_request(request_path,expected_request_sha256,recoverable_training=family=='P12')
    query=arrays['V_features'];prediction_path=root/'worker/predictions.npz'
    ids=['row_id','group_id']+(['timestamps'] if 'timestamps' in query else [])
    with np.load(prediction_path,allow_pickle=False) as saved:
        if (set(saved.files)!={'prediction',*ids} or len(saved.files)!=1+len(ids)
                or any(saved[k].dtype!=query[k].dtype or not np.array_equal(saved[k],query[k]) for k in ids)
                or saved['prediction'].dtype not in (np.dtype('float32'),np.dtype('float64'))
                or saved['prediction'].shape!=query['row_id'].shape or not np.isfinite(saved['prediction']).all()):
            raise ValueError('Native prediction rows/types differ from primary V features')
    if (scientific_source_sha256()!=before or _sha(manifest_path)!=expected_manifest_sha256
            or canonical_hash(_inventory(root))!=native_reference['inventory_sha256']):
        raise ValueError('Native generation evidence changed during verification')
    return dict(schema='research/v2-native-generation-chain-audit',status='authenticated_source_bound_chain',
        model_generation_chain_authenticated=True,family_id=family,fit_seed=expected_fit_seed,
        source_sha256=before,request_sha256=expected_request_sha256,native_inventory=native_reference,
        dataset=manifest['dataset'],outer_fold=outer_fold,split_hash=split['split_hash'],manifest_sha256=expected_manifest_sha256,
        configuration_sha256=canonical_hash(expected_configuration),query_rows=len(query['row_id']),
        process_receipt_sha256=_sha(root/'process/process.json'),worker_terminal_sha256=_sha(root/'worker/terminal.json'),
        prediction=dict(path=str(prediction_path),sha256=_sha(prediction_path)),snapshot=dict(path=str(snapshot),sha256=_sha(snapshot)),
        model_training_independently_reexecuted=False,model_predictions_independently_reexecuted=False,
        native_workload_qualified=False,HPO_journal_authenticated=False,quality_eligible=False,V_labels_read=False,T_accessed=False,
        scope='Trusted current-source managed receipts and immutable byte/primary-role chain; no independent optimization or prediction replay, native implementation/workload qualification, support/HPO/V arithmetic or protected authority')
