"""Join fixed native anchor software and execution proofs, without promotion."""
import copy
import hashlib
import os
from pathlib import Path
import sys
import time

from treeneuron.config import canonical_hash
from .deployment_worker_v2 import _ROOT,_source,_json
from .ledger import write_new_json
from .native_candidate_v2 import inspect_primary_native_candidate
from .native_verification_v2 import verification_profiles,load_native_implementation_verification
from .search_supports_v2 import family_search_spec
from .workqueue import _inventory


def _closed_reference(reference):
    from .method_worker import _digest
    if (type(reference) is not dict or set(reference)!={'root','inventory_sha256'}
            or type(reference['root']) is not str or not _digest(reference['inventory_sha256'])):
        raise ValueError('Externally committed closed native proof required')
    root=Path(reference['root'])
    if not root.is_absolute() or root.resolve()!=root or root.is_symlink() or not root.is_dir():
        raise ValueError('Ordinary immutable native proof directory required')
    inventory=_inventory(root)
    if canonical_hash(inventory)!=reference['inventory_sha256']:raise ValueError('Closed native proof bytes changed')
    return root,inventory


def _bound_json(root,inventory,relative):
    path=root/relative
    refs={row['path']:row['sha256'] for row in inventory['files']}
    if relative not in refs or path.is_symlink():raise ValueError('Missing committed proof JSON')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=refs[relative]:raise ValueError('Proof JSON changed during decoding')
    return _json(raw)


def load_native_anchor_execution(reference,*,prepared_root,expected_manifest_sha256,
        expected_family,expected_anchor_id,expected_dataset):
    """Recheck a closed anchor and its primary V arithmetic as coordinator.

    No fitting or new prediction is performed. Full workload/support/acquisition
    qualification still requires the separate registry. Legitimate arithmetic
    rechecking reads primary V labels through its logged coordinator boundary.
    """
    from .method_worker import _THREADS,_sha,_read_json,_digest
    from .native_generation_v2 import audit_native_generation
    from .prediction_replay_v2 import _request as replay_request
    from .validation_scoring_v2 import score_primary_validation_prediction
    from .workload import _checked_manifest
    root,inventory=_closed_reference(reference);source=_source()
    record=_bound_json(root,inventory,'result.json');prepared=Path(prepared_root)
    if (not prepared.is_absolute() or prepared.resolve()!=prepared or prepared.is_symlink()
            or not _digest(expected_manifest_sha256) or (prepared/'manifest.json').is_symlink()):
        raise ValueError('Exact externally committed primary prepared data required')
    raw=(prepared/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_manifest_sha256:raise ValueError('Prepared manifest bytes changed')
    manifest=_json(raw);_checked_manifest(expected_dataset,manifest)
    if manifest['dataset']!=expected_dataset:raise ValueError('Expected anchor dataset differs')
    spec=family_search_spec(expected_family,dict(columns=manifest['feature_names'],categorical=manifest['categorical_features']))
    choices=[row['config'] for row in spec['anchors'] if row['anchor_id']==expected_anchor_id]
    if (len(choices)!=1 or record.get('schema')!='research/v2-native-anchor-execution'
            or record.get('status')!='authenticated_native_anchor_execution'
            or record.get('source_sha256')!=source or record.get('family_id')!=expected_family
            or record.get('anchor_id')!=expected_anchor_id or record.get('dataset')!=expected_dataset
            or record.get('fit_seed')!=17 or record.get('support_sha256')!=spec['support_hash']
            or record.get('configuration_sha256')!=canonical_hash(choices[0])
            or any(record.get(k) is not True for k in ('native_anchor_execution_authenticated','software_verified','prediction_reload_exact'))
            or any(record.get(k) is not False for k in ('V_score_used_for_qualification','raw_acquisition_authenticated',
                'native_workload_qualified','quality_eligible','HPO_journal_authenticated','protected_access_authorized','T_accessed'))):
        raise ValueError('Native anchor source/coordinate or unearned authority differs')
    profile=next(name for name,row in verification_profiles().items() if expected_family in row['families'])
    software=load_native_implementation_verification(record['implementation_reference'],expected_profile=profile)
    inspection_root,inspection_inventory=_closed_reference(record['inspection_reference'])
    if inspection_root!=root/'inspection':raise ValueError('Native anchor inspection path differs')
    inspection=_bound_json(inspection_root,inspection_inventory,'result.json')
    if (inspection.get('schema')!='research/v2-native-candidate-inspection'
            or inspection.get('status')!='completed_candidate_inspection' or inspection.get('source_sha256')!=source
            or inspection.get('V_coordinator_invoked') is not True or inspection.get('V_labels_read_by_coordinator') is not True
            or any(inspection.get(k) is not False for k in ('native_workload_qualified','HPO_journal_authenticated',
                'quality_eligible','T_accessed','model_training_independently_reexecuted'))):
        raise ValueError('Closed native anchor inspection provenance differs')
    replay=_bound_json(inspection_root,inspection_inventory,'replay/terminal.json')
    process=_bound_json(inspection_root,inspection_inventory,'process/process.json')
    request_path=inspection_root/'input/request.json'
    replay_sha=hashlib.sha256(request_path.read_bytes()).hexdigest()
    replay_value,replayed_snapshot,_,_,_,_,_,_=replay_request(request_path,replay_sha)
    expected_argv=['-B','-m','treeneuron.research.prediction_replay_v2','--request',str(request_path),
        '--expected-request-sha256',replay_sha,'--output',str(inspection_root/'replay')]
    env=process.get('environment_overrides',{});paths=env.get('PYTHONPATH','').split(os.pathsep)
    argv=process.get('argv',[])
    if (replay!=inspection.get('replay') or process!=inspection.get('process')
            or replay.get('schema')!='research/v2-prediction-replay-terminal' or replay.get('status')!='completed_ineligible'
            or replay.get('prediction_reload_exact') is not True or replay.get('labels_read') is not False
            or replay.get('T_accessed') is not False or replay.get('request_sha256')!=replay_sha
            or replay.get('family_id')!=expected_family or replay.get('scientific_source_sha256')!=source
            or replay_value['family_id']!=expected_family or replay_value['feature_schema']!=spec['feature_schema']
            or process.get('schema')!='research/v2-process-receipt' or process.get('execution_status')!='exited'
            or process.get('returncode')!=0 or process.get('child_reaped') is not True or process.get('timeout_ns') is not None
            or process.get('cwd')!=str(_ROOT) or not argv or Path(argv[0]).resolve()!=Path(sys.executable).resolve()
            or argv[1:]!=expected_argv
            or process.get('identity')!=dict(purpose='candidate_inspection',request_sha256=replay_sha,source_sha256=source)
            or any(env.get(k)!='1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE'))
            or env.get('KMP_DUPLICATE_LIB_OK') is not None or env.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0'
            or not paths or paths[0]!=str(_ROOT/'src')
            or any(not Path(path).resolve().is_relative_to((_ROOT/'environment').resolve()) for path in paths[1:])
            or canonical_hash(_inventory(inspection_root/'replay'))!=inspection.get('replay_artifact_inventory_sha256')
            or canonical_hash(_inventory(inspection_root/'process'))!=inspection.get('replay_process_inventory_sha256')):
        raise ValueError('Managed independent replay proof differs')
    generation=inspection['generation'];native_ref=generation['native_inventory']
    native_process=_read_json(Path(native_ref['root'])/'process/process.json')
    native_argv=native_process.get('argv',[])
    if native_argv.count('--request')!=1:raise ValueError('Native managed generation request is missing/ambiguous')
    native_path=Path(native_argv[native_argv.index('--request')+1])
    native_request=_read_json(native_path)
    args=dict(request_path=native_path,expected_request_sha256=generation['request_sha256'],native_reference=native_ref,
        prepared_root=prepared,expected_manifest_sha256=expected_manifest_sha256,
        outer_fold=4 if expected_dataset=='bike' else 0,expected_fit_seed=17,expected_configuration=choices[0])
    container=Path(inspection['immutable_generation_root'])
    original_snapshot=Path(generation['snapshot']['path']).parent
    if (not native_path.resolve().is_relative_to(container)
            or not Path(native_ref['root']).resolve().is_relative_to(container)
            or replay_value['snapshot']['metadata_sha256']!=generation['snapshot']['sha256']
            or replay_value['query']['sha256']!=native_request['files']['V_features']['sha256']
            or replay_value['prediction']['sha256']!=generation['prediction']['sha256']
            or replay.get('query_sha256')!=replay_value['query']['sha256']
            or replay.get('prediction_sha256')!=replay_value['prediction']['sha256']
            or replay.get('snapshot_metadata_sha256')!=generation['snapshot']['sha256']
            or replay.get('query_rows')!=generation['query_rows']
            or _inventory(replayed_snapshot)!=_inventory(original_snapshot)
            or replay.get('snapshot_files')!=_inventory(original_snapshot)['files']):
        raise ValueError('Independent replay is not bound to the original generation')
    if (not container.is_absolute() or container.resolve()!=container or container.is_symlink()
            or canonical_hash(_inventory(container))!=inspection['immutable_generation_inventory_sha256']
            or audit_native_generation(**args)!=generation):raise ValueError('Original native generation proof differs')
    prediction=Path(generation['prediction']['path'])
    validation=score_primary_validation_prediction(root=prepared,outer_fold=args['outer_fold'],
        expected_manifest_sha256=expected_manifest_sha256,prediction_path=prediction,
        expected_prediction_sha256=generation['prediction']['sha256'])
    if validation!=inspection['validation']:raise ValueError('Independent primary validation arithmetic differs')
    if (audit_native_generation(**args)!=generation or _source()!=source
            or _sha(prepared/'manifest.json')!=expected_manifest_sha256
            or canonical_hash(_inventory(container))!=inspection['immutable_generation_inventory_sha256']
            or _inventory(root)!=inventory
            or load_native_implementation_verification(record['implementation_reference'],expected_profile=profile)!=software):
        raise ValueError('Native anchor proof changed during loading')
    for ref in validation['sources']:
        path=Path(ref['path'])
        if path.is_symlink() or _sha(path)!=ref['sha256']:raise ValueError('Final primary arithmetic input changed')
    return dict(schema='research/v2-authenticated-native-anchor-execution',reference=copy.deepcopy(reference),
        source_sha256=source,family_id=expected_family,anchor_id=expected_anchor_id,dataset=expected_dataset,
        native_anchor_execution_authenticated=True,software_verified=True,prediction_reload_exact=True,
        validation_recomputed=True,V_labels_read_by_coordinator=True,V_score_used_for_qualification=False,
        native_workload_qualified=False,quality_eligible=False,T_accessed=False,
        software=software,generation=generation,validation=validation,
        scope='Reopened fixed native anchor software, managed generation, independent inference evidence and primary V arithmetic. No raw acquisition, complete workload/support registry or production/protected authority.')


def qualify_native_anchor(*,implementation_reference,generation_arguments,immutable_generation_root,
        output,interpreter,extra_pythonpath=(),rss_limit_bytes=4*1024**3):
    """Authenticate a native anchor's source/software/state/replay/primary V.

    This does not authenticate raw acquisition or whole support/workload bounds.
    A complete qualified registry and production protocol remain prerequisites.
    """
    started=time.perf_counter_ns();args=copy.deepcopy(generation_arguments);output=Path(output)
    fields={'request_path','expected_request_sha256','native_reference','prepared_root','expected_manifest_sha256',
        'outer_fold','expected_fit_seed','expected_configuration'}
    if type(args) is not dict or set(args)!=fields:raise ValueError('Exact native generation arguments required')
    container=Path(immutable_generation_root).resolve();prepared=Path(args['prepared_root']).resolve()
    if (not container.is_dir() or output.exists() or output.is_symlink() or output.resolve().is_relative_to(container)
            or output.resolve().is_relative_to(prepared)
            or any(output.resolve().is_relative_to(_ROOT/p) for p in ('src','tests','locks','configs','protocol'))):
        raise ValueError('Fresh external qualification output must preserve immutable/executable inputs')
    cfg=args['expected_configuration']
    if type(cfg) is not dict or 'family' not in cfg or 'feature_schema' not in cfg:
        raise ValueError('Exact mandatory native anchor configuration required')
    spec=family_search_spec(cfg['family'],cfg['feature_schema'])
    matches=[a['anchor_id'] for a in spec['anchors'] if a['config']==cfg]
    if len(matches)!=1 or type(args['expected_fit_seed']) is not int or args['expected_fit_seed']!=17:
        raise ValueError('Qualification requires one declared native anchor and fixed seed17')
    profile=next(name for name,row in verification_profiles().items() if cfg['family'] in row['families'])
    software=load_native_implementation_verification(implementation_reference,expected_profile=profile);source=_source()
    interpreter=Path(interpreter)
    if (not interpreter.is_absolute() or not interpreter.is_file() or type(rss_limit_bytes) is not int or rss_limit_bytes<=0):
        raise ValueError('Qualified interpreter and owned replay RSS guard required')
    output.mkdir(parents=True,exist_ok=False);output=output.resolve()
    record=dict(schema='research/v2-native-anchor-execution',status='input_or_dependency_failure',
        native_anchor_execution_authenticated=False,source_sha256=source,family_id=cfg['family'],anchor_id=matches[0],
        dataset=None,software_verified=True,implementation_reference=copy.deepcopy(implementation_reference),
        configuration_sha256=canonical_hash(cfg),support_sha256=spec['support_hash'],fit_seed=17,
        prediction_reload_exact=False,inspection_span_ns=0,inspection_reference=None,
        V_score_used_for_qualification=False,raw_acquisition_authenticated=False,native_workload_qualified=False,
        quality_eligible=False,HPO_journal_authenticated=False,protected_access_authorized=False,T_accessed=False,
        scope='Fixed native anchor software plus managed current-source generation, closed prediction replay and primary V. No raw acquisition, complete support/workload registry or production/protected authority.')
    try:
        inspection_started=time.perf_counter_ns()
        inspection=inspect_primary_native_candidate(generation_arguments=args,immutable_generation_root=container,
            output=output/'inspection',interpreter=interpreter,extra_pythonpath=extra_pythonpath,rss_limit_bytes=rss_limit_bytes)
        record['inspection_span_ns']=time.perf_counter_ns()-inspection_started
        saved=_json((output/'inspection/result.json').read_bytes())
        if saved!=inspection:raise ValueError('Returned and published native inspection differ')
        if (load_native_implementation_verification(implementation_reference,expected_profile=profile)!=software
                or _source()!=source):raise ValueError('Native anchor software/source changed during inspection')
        record['inspection_reference']=dict(root=str(output/'inspection'),inventory_sha256=canonical_hash(_inventory(output/'inspection')))
        record['dataset']=inspection['generation']['dataset']
        record['status']=inspection['status']
        if inspection['status']=='completed_candidate_inspection':
            record.update(status='authenticated_native_anchor_execution',native_anchor_execution_authenticated=True,
                prediction_reload_exact=inspection['replay']['prediction_reload_exact'])
        return record
    except BaseException as error:
        record.update(error_type=type(error).__name__,reason=str(error))
        raise
    finally:
        record['elapsed_ns']=time.perf_counter_ns()-started
        record['clock_scope']='Function entry through result preparation; inspection is nested; caller pays final publication/return'
        write_new_json(output/'result.json',record)
