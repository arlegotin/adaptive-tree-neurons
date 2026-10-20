"""Reopen a primary native search inspection without fitting or inference."""
import copy
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

from treeneuron.config import canonical_hash
from .deployment_worker_v2 import _ROOT, _source
from .method_worker import _THREADS, _sha, _read_json
from .native_anchor_v2 import _closed_reference
from .native_generation_v2 import audit_native_generation
from .native_workload_v2 import _bound_reader, _process_observation
from .prediction_replay_v2 import _request as replay_request
from .validation_scoring_v2 import score_primary_validation_prediction
from .workqueue import _inventory


_ARGUMENTS={'request_path','expected_request_sha256','native_reference','prepared_root',
    'expected_manifest_sha256','outer_fold','expected_fit_seed','expected_configuration'}


def _header(path,name):
    """Read bounded NPY declarations without allocating query/model arrays."""
    with zipfile.ZipFile(path) as archive:
        member=archive.getinfo(name+'.npy')
        with archive.open(member) as stream:
            version=np.lib.format.read_magic(stream)
            if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(stream)
            elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(stream)
            else:raise ValueError('Unsupported replay numeric header')
            if (dtype not in (np.dtype('float32'),np.dtype('float64'))
                    or any(type(v) is not int or v<0 for v in shape)
                    or member.file_size!=stream.tell()+math.prod(shape)*dtype.itemsize):
                raise ValueError('Replay numeric dtype/shape/bytes differ')
    return shape,str(dtype)


def load_primary_native_inspection(reference,*,generation_arguments,immutable_generation_root):
    """Authenticate generation, independent saved replay and coordinator V.

    This is a reusable search-config reader, not native qualification, a paid
    journal, a selected model or a protected evaluator. V arithmetic is read
    through its declared primary coordinator; T remains unopened.
    """
    reference=copy.deepcopy(reference);args=copy.deepcopy(generation_arguments)
    if type(args) is not dict or set(args)!=_ARGUMENTS:raise ValueError('Exact native generation arguments required')
    root,inventory=_closed_reference(reference);read,refs=_bound_reader(root,inventory)
    source=_source();record=read('result.json');container=Path(immutable_generation_root)
    if (not container.is_absolute() or container.is_symlink() or container.resolve()!=container
            or not container.is_dir() or root.is_relative_to(container)
            or root.is_relative_to(Path(args['prepared_root']).resolve())
            or not Path(args['request_path']).resolve().is_relative_to(container)
            or not Path(args['native_reference']['root']).resolve().is_relative_to(container)):
        raise ValueError('Exact containing immutable generation and external inspection required')
    if (record.get('schema')!='research/v2-native-candidate-inspection'
            or record.get('status')!='completed_candidate_inspection' or record.get('source_sha256')!=source
            or record.get('immutable_generation_root')!=str(container)
            or record.get('immutable_generation_inventory_sha256')!=canonical_hash(_inventory(container))
            or any(record.get(k) is not True for k in ('V_coordinator_invoked','V_labels_read_by_coordinator'))
            or any(record.get(k) is not False for k in ('native_workload_qualified','HPO_journal_authenticated',
                'quality_eligible','T_accessed','model_training_independently_reexecuted'))):
        raise ValueError('Closed native inspection source/schema/authority/generation differs')
    generation=audit_native_generation(**args)
    if record.get('generation')!=generation:raise ValueError('Inspection belongs to another original generation')
    replay=read('replay/terminal.json');process=read('process/process.json')
    path=root/'input/request.json';request_sha=_sha(path)
    replay_value,replayed_snapshot,_,_,_,_,_,_=replay_request(path,request_sha)
    family=generation['family_id'];original_request=_read_json(args['request_path'])
    expected_argv=['-B','-m','treeneuron.research.prediction_replay_v2','--request',str(path),
        '--expected-request-sha256',request_sha,'--output',str(root/'replay')]
    env=process.get('environment_overrides',{});paths=env.get('PYTHONPATH','').split(os.pathsep);argv=process.get('argv',[])
    if (replay!=record.get('replay') or process!=record.get('process')
            or replay.get('schema')!='research/v2-prediction-replay-terminal' or replay.get('status')!='completed_ineligible'
            or replay.get('prediction_reload_exact') is not True or replay.get('request_sha256')!=request_sha
            or replay.get('family_id')!=family or replay.get('scientific_source_sha256')!=source
            or any(replay.get(k) is not False for k in ('labels_read','quality_eligible',
                'model_generation_authenticated','protected_access_authorized','T_accessed'))
            or process.get('schema')!='research/v2-process-receipt' or process.get('execution_status')!='exited'
            or process.get('returncode')!=0 or process.get('child_reaped') is not True or process.get('child_started') is not True
            or process.get('timeout_ns') is not None or process.get('cwd')!=str(_ROOT)
            or not argv or Path(argv[0]).resolve()!=Path(sys.executable).resolve() or argv[1:]!=expected_argv
            or process.get('identity')!=dict(purpose='candidate_inspection',request_sha256=request_sha,source_sha256=source)
            or any(env.get(k)!='1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE'))
            or env.get('KMP_DUPLICATE_LIB_OK') is not None or env.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0'
            or env.get('MPLCONFIGDIR')!=str(root/'cache/mpl') or env.get('XDG_CACHE_HOME')!=str(root/'cache/xdg')
            or not paths or paths[0]!=str(_ROOT/'src')
            or any(not Path(p).resolve().is_relative_to((_ROOT/'environment').resolve()) for p in paths[1:])
            or canonical_hash(_inventory(root/'replay'))!=record.get('replay_artifact_inventory_sha256')
            or canonical_hash(_inventory(root/'process'))!=record.get('replay_process_inventory_sha256')):
        raise ValueError('Managed independent replay request/process/authority differs')
    expected_files={'started.json','spawned.json','stdout.log','stderr.log','resource-samples.jsonl'}
    files=process.get('files',[])
    if type(files) is not list or len(files)!=5 or {r.get('path') for r in files}!=expected_files:
        raise ValueError('Complete managed replay process file commitments required')
    for row in files:
        relative='process/'+row['path'];actual=root/relative
        if (row.get('bytes')!=actual.stat().st_size or row.get('sha256')!=refs.get(relative)
                or _sha(actual)!=row['sha256']):raise ValueError('Replay raw process file bytes differ')
    observation=_process_observation(root,read,refs)
    snapshot=Path(generation['snapshot']['path']).parent
    query_shape,query_dtype=_header(root/'input'/replay_value['query']['path'],'X')
    prediction_shape,prediction_dtype=_header(root/'input'/replay_value['prediction']['path'],'prediction')
    if (query_shape!=(generation['query_rows'],len(args['expected_configuration']['feature_schema']['columns']))
            or prediction_shape!=(generation['query_rows'],) or query_dtype!='float64'
            or replay.get('query_dtype')!=query_dtype or replay.get('prediction_dtype')!=prediction_dtype):
        raise ValueError('Replay numeric dtype/geometry claims differ from actual headers')
    if (replay_value['family_id']!=family or replay_value['feature_schema']!=args['expected_configuration']['feature_schema']
            or replay_value['snapshot']['metadata_sha256']!=generation['snapshot']['sha256']
            or replay_value['query']['sha256']!=original_request['files']['V_features']['sha256']
            or replay_value['prediction']['sha256']!=generation['prediction']['sha256']
            or replay.get('query_sha256')!=replay_value['query']['sha256']
            or replay.get('prediction_sha256')!=replay_value['prediction']['sha256']
            or replay.get('snapshot_metadata_sha256')!=generation['snapshot']['sha256']
            or replay.get('query_rows')!=generation['query_rows']
            or _inventory(replayed_snapshot)!=_inventory(snapshot)
            or replay.get('snapshot_files')!=_inventory(snapshot)['files']):
        raise ValueError('Replay query/prediction/snapshot is not the original generation')
    validation=score_primary_validation_prediction(root=args['prepared_root'],outer_fold=args['outer_fold'],
        expected_manifest_sha256=args['expected_manifest_sha256'],prediction_path=Path(generation['prediction']['path']),
        expected_prediction_sha256=generation['prediction']['sha256'])
    if validation!=record.get('validation'):raise ValueError('Independent primary V arithmetic differs')
    if (audit_native_generation(**args)!=generation or _source()!=source or _inventory(root)!=inventory
            or canonical_hash(_inventory(container))!=record['immutable_generation_inventory_sha256']):
        raise ValueError('Original source/generation/inspection changed while reading')
    for row in validation['sources']:
        original=Path(row['path'])
        if original.is_symlink() or _sha(original)!=row['sha256']:raise ValueError('Primary validation bytes changed after arithmetic')
    return dict(schema='research/v2-authenticated-primary-native-inspection',reference=reference,source_sha256=source,
        inspection_authenticated=True,generation=generation,validation=validation,
        prediction_reload_exact=True,V_labels_read_by_coordinator=True,process_observation=observation,
        new_fit_or_inference=False,native_workload_qualified=False,HPO_journal_authenticated=False,
        quality_eligible=False,protected_access_authorized=False,T_accessed=False,
        scope='Reopened current-source primary native search generation, saved managed independent replay and exact coordinator V arithmetic. No new fit/inference, software/workload/support qualification, paid journal, selected model or protected authority.')
