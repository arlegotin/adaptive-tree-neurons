"""Compose closed generation, fresh prediction replay and primary V arithmetic.

This evidence packet supplies no workload/support/HPO/freeze qualification and
cannot itself promote an engineering fit to a production search candidate.
"""
import copy
import os
from pathlib import Path
import shutil
import time

from treeneuron.config import canonical_hash
from .ledger import write_new_json
from .immutable_inputs import stage_immutable_file
from .method_worker import _ROOT,_THREADS,_read_json,_sha,scientific_source_sha256
from .native_generation_v2 import audit_native_generation
from .process_supervision_v2 import supervise_process
from .validation_scoring_v2 import score_primary_validation_prediction
from .workqueue import _inventory


def inspect_primary_native_candidate(*,generation_arguments,immutable_generation_root,
        output,interpreter,extra_pythonpath=(),rss_limit_bytes=4*1024**3):
    started=time.perf_counter_ns();args=copy.deepcopy(generation_arguments)
    container=Path(immutable_generation_root).resolve();output=Path(output)
    if (type(args) is not dict or not container.is_dir() or output.exists() or output.is_symlink()
            or output.resolve().is_relative_to(container)
            or output.resolve().is_relative_to(Path(args['prepared_root']).resolve())
            or not Path(args['native_reference']['root']).resolve().is_relative_to(container)
            or not Path(args['request_path']).resolve().is_relative_to(container)):
        raise ValueError('Fresh external inspection and a containing immutable generation directory required')
    interpreter=Path(interpreter);overlays=[]
    if (not interpreter.is_absolute() or not interpreter.is_file() or type(rss_limit_bytes) is not int
            or rss_limit_bytes<=0 or not isinstance(extra_pythonpath,(tuple,list))):
        raise ValueError('Qualified interpreter and explicit owned replay RSS guard required')
    for value in extra_pythonpath:
        path=Path(value).resolve()
        if not path.is_dir() or not path.is_relative_to((_ROOT/'environment').resolve()):
            raise ValueError('Replay overlays must remain in the pinned environment')
        overlays.append(str(path))
    generation=audit_native_generation(**args)
    before=_inventory(container);source=scientific_source_sha256()
    request=_read_json(args['request_path']);snapshot=Path(generation['snapshot']['path']).parent
    query=Path(args['request_path']).parent/request['files']['V_features']['path']
    prediction=Path(generation['prediction']['path'])
    staged_snapshot=_inventory(snapshot)
    required_bytes=sum(v['bytes'] for v in staged_snapshot['files'])+query.stat().st_size+prediction.stat().st_size
    parent=output.absolute().parent
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<required_bytes+512*1024**2:
        raise OSError('Insufficient disk for the complete replay transaction; no V access')
    output.mkdir(parents=True,exist_ok=False);output=output.resolve()
    record=dict(schema='research/v2-native-candidate-inspection',status='input_or_dependency_failure',
        generation=generation,replay=None,validation=None,process=None,replay_supervisor_span_ns=None,
        source_sha256=source,immutable_generation_root=str(container),
        immutable_generation_inventory_sha256=canonical_hash(before),
        native_workload_qualified=False,HPO_journal_authenticated=False,quality_eligible=False,
        V_coordinator_invoked=False,V_labels_read_by_coordinator=False,T_accessed=False,
        model_training_independently_reexecuted=False,
        input_materialization=[],
        scope='Current-source generation chain, independent closed-model inference and primary V arithmetic only. Native implementation/workload/support qualification, paid search journal and protected freeze remain separate.')
    try:
        inputs=output/'input';inputs.mkdir()
        committed={row['path']:row['sha256'] for row in staged_snapshot['files']}
        def stage_snapshot(original,target):
            row=stage_immutable_file(original,target,expected_sha256=committed[Path(original).relative_to(snapshot).as_posix()])
            record['input_materialization'].append(row)
            return str(target)
        shutil.copytree(snapshot,inputs/'snapshot',copy_function=stage_snapshot)
        record['input_materialization'].append(stage_immutable_file(query,inputs/'query.npz',
            expected_sha256=request['files']['V_features']['sha256']))
        record['input_materialization'].append(stage_immutable_file(prediction,inputs/'prediction.npz',
            expected_sha256=generation['prediction']['sha256']))
        write_new_json(inputs/'materialization.json',dict(files=record['input_materialization'],
            policy='Byte-authenticated immutable inference inputs, hard-linked on the same filesystem or copied across devices; containing clocks include staging'))
        if (_inventory(inputs/'snapshot')!=staged_snapshot
                or _sha(inputs/'query.npz')!=request['files']['V_features']['sha256']
                or _sha(inputs/'prediction.npz')!=generation['prediction']['sha256']):
            raise ValueError('Replay staging differs from the original closed artifacts')
        replay_request=dict(schema='research/v2-prediction-replay-request',purpose='engineering_qualification',
            family_id=request['family_id'],scientific_source_sha256=source,
            feature_schema=request['config']['feature_schema'],
            snapshot=dict(path='snapshot',metadata_sha256=generation['snapshot']['sha256']),
            query=dict(path='query.npz',sha256=_sha(inputs/'query.npz')),
            prediction=dict(path='prediction.npz',sha256=_sha(inputs/'prediction.npz')))
        path=inputs/'request.json';write_new_json(path,replay_request)
        env={key:'1' for key in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}
        env.update(PYTHONPATH=os.pathsep.join([str(_ROOT/'src'),*overlays]),PYTORCH_ENABLE_MPS_FALLBACK='0',
            KMP_DUPLICATE_LIB_OK=None,MPLCONFIGDIR=str(output/'cache/mpl'),XDG_CACHE_HOME=str(output/'cache/xdg'))
        replay_started=time.perf_counter_ns()
        process=supervise_process(argv=[str(interpreter),'-B','-m','treeneuron.research.prediction_replay_v2',
            '--request',str(path),'--expected-request-sha256',_sha(path),'--output',str(output/'replay')],
            cwd=_ROOT,env_overrides=env,output=output/'process',timeout_ns=None,rss_limit_bytes=rss_limit_bytes,
            identity=dict(purpose='candidate_inspection',request_sha256=_sha(path),source_sha256=source))
        record['replay_supervisor_span_ns']=time.perf_counter_ns()-replay_started;record['process']=process
        if process['execution_status']!='exited':
            record['status']=process['execution_status']
            return record
        replay=_read_json(output/'replay/terminal.json');record['replay']=replay
        if (process['returncode']!=0 or replay.get('status')!='completed_ineligible'
                or replay.get('request_sha256')!=_sha(path) or replay.get('prediction_reload_exact') is not True
                or replay.get('labels_read') is not False or replay.get('T_accessed') is not False):
            raise ValueError('Independent closed-model replay did not pass; no V scoring')
        replay_evidence=_inventory(output/'replay');process_evidence=_inventory(output/'process')
        record['replay_artifact_inventory_sha256']=canonical_hash(replay_evidence)
        record['replay_process_inventory_sha256']=canonical_hash(process_evidence)
        # No primary labels are opened until both generation and independent
        # inference pass. Repeat generation validation before and after V.
        if audit_native_generation(**args)!=generation or _inventory(container)!=before:
            raise ValueError('Original generation changed during independent replay')
        record['V_coordinator_invoked']=True;record['V_labels_read_by_coordinator']=None
        record['validation']=score_primary_validation_prediction(root=args['prepared_root'],outer_fold=args['outer_fold'],
            expected_manifest_sha256=args['expected_manifest_sha256'],prediction_path=prediction,
            expected_prediction_sha256=generation['prediction']['sha256'])
        record['V_labels_read_by_coordinator']=True
        if (audit_native_generation(**args)!=generation or _inventory(container)!=before
                or scientific_source_sha256()!=source or _inventory(inputs/'snapshot')!=staged_snapshot
                or _sha(inputs/'query.npz')!=replay_request['query']['sha256']
                or _sha(inputs/'prediction.npz')!=replay_request['prediction']['sha256']
                or _sha(path)!=replay['request_sha256']
                or _inventory(output/'replay')!=replay_evidence or _inventory(output/'process')!=process_evidence):
            raise ValueError('Original or staged candidate evidence changed during inspection')
        # Generation deliberately does not read V labels. Its final audit must
        # not leave the separate scorer's label commitments unchecked afterward.
        for ref in record['validation']['sources']:
            original=Path(ref['path'])
            if original.is_symlink() or not original.is_file() or _sha(original)!=ref['sha256']:
                raise ValueError('Validation evidence changed after candidate inspection arithmetic')
        record['status']='completed_candidate_inspection'
        return record
    except BaseException as error:
        record.update(error_type=type(error).__name__,reason=str(error))
        raise
    finally:
        record['elapsed_ns']=time.perf_counter_ns()-started
        record['clock_scope']='Function entry through final result preparation; supervisor span is nested. Caller pays publication, hashing, return and all earlier fitting/search separately.'
        write_new_json(output/'result.json',record)
