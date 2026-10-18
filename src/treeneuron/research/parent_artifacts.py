"""F/G-only parent prerequisites with committed actual and selected snapshots."""
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

from treeneuron import train
from treeneuron.artifacts import create_run,finish_run
from treeneuron.checkpoint import _descriptor,load_checkpoint,save_checkpoint
from treeneuron.config import canonical_hash
from .boundaries import current_boundary
from .ledger import write_new_json
from .roles import FittingData
from .snapshots import SnapshotSink


@dataclass(frozen=True)
class ParentSpec:
    snapshot_updates: tuple[int,...]=()
    fixed_budget: bool=False

    def __post_init__(self):
        if (not isinstance(self.snapshot_updates,tuple) or type(self.fixed_budget) is not bool
                or any(type(u) is not int or u<0 for u in self.snapshot_updates)
                or list(self.snapshot_updates)!=sorted(set(self.snapshot_updates))):
            raise ValueError('Parent snapshots require ordered unique actual updates')


def _reference(path,root):
    return dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fit_parent_artifact(data,config,output,spec,*,control=None,guard=lambda:None):
    """One finite attempt; the outer queue owns any declared same-seed retry."""
    if (not isinstance(data,FittingData) or not isinstance(spec,ParentSpec)
            or config.backend.device!='cpu' or config.model.coordinates!='selective'
            or any(u>config.training.base_updates for u in spec.snapshot_updates)
            or (control is not None and (spec.snapshot_updates or spec.fixed_budget))):
        raise ValueError('Parent artifact requires the declared ordinary F/G-only CPU treatment')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    report=dict(schema='research/v2-parent-artifact',status='running',source_sha256=train.source_hash(),
        request=asdict(spec),config_hash=canonical_hash(config.to_dict()),
        F_G_hashes=train._fg_hashes(data),manifest_sha256=canonical_hash(data.manifest.to_dict()),
        V_accessed=False,E_accessed=False,T_accessed=False,
        parent_kind='trained' if control is None else 'predeclared_unfitted_control')
    write_new_json(root/'request.json',report)
    before=train.capture_rng();started=time.perf_counter_ns()
    original_backend=train.BACKEND_REPORT_PATH
    try:
        train.BACKEND_REPORT_PATH=Path(config.execution.backend_report)
        guard()
        paths=create_run(root/'runs',config.to_dict())
        report['run_path']=str(paths.directory.resolve())
        sink=SnapshotSink(root/'snapshots',spec.snapshot_updates)
        def capture(label,snapshot):
            guard();sink(label,snapshot)
        if control is None:
            state=train.fit_parent(data,config,paths,snapshot_updates=spec.snapshot_updates,
                on_snapshot=capture,stop_by_patience=not spec.fixed_budget)
        else:
            from .controls import initialize_control_state
            state=initialize_control_state(control,config,paths,data=data)
            sink('selected',train.pack_state(state))
        guard()
        sink.finalize()
        pointer=root/'selected.json';save_checkpoint(state,pointer)
        finish_run(paths,{'selected_g_mse':state.stage['best_mse']},'completed')
        report.update(status='completed_eligible',identity_hashes=dict(state.identity_hashes),
            selected_update=state.update,executed_updates=state.stage['updates_completed'],
            selected_g_mse=state.stage['best_mse'],selected=_reference(pointer,root),
            capture=_reference(root/'snapshots'/'capture.json',root),
            unavailable_snapshots=sorted(set(spec.snapshot_updates)-sink.captured))
        return report
    except BaseException as error:
        report.update(status=('interrupted_recoverable' if isinstance(error,KeyboardInterrupt)
            else 'resource_exhausted' if isinstance(error,(MemoryError,OSError))
            else 'numerical_failure' if isinstance(error,FloatingPointError) else 'input_or_dependency_failure'),
            error_type=type(error).__name__,reason=str(error))
        raise
    finally:
        active=sys.exc_info()[1]
        report['enclosing_ns']=time.perf_counter_ns()-started
        try:
            try:write_new_json(root/'result.json',report)
            except BaseException as error:
                if active is None:raise
                active.add_note(f'Parent result publication also failed: {error}')
        finally:
            train.BACKEND_REPORT_PATH=original_backend
            train.restore_rng(before)


def _committed(root,reference):
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
        raise ValueError('Invalid parent artifact commitment')
    path=(root/reference['path']).resolve()
    if (not path.is_relative_to(root.resolve()) or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']):
        raise ValueError('Parent artifact commitment mismatch')
    return path


def load_parent_input(root,snapshot,data):
    """Return None only for a recorded unreached update; corrupt assets fail."""
    if not isinstance(data,FittingData):raise ValueError('Parent loading requires F/G-only data')
    root=Path(root);report=json.loads((root/'result.json').read_text())
    if (report['schema']!='research/v2-parent-artifact' or report['status']!='completed_eligible'
            or report['source_sha256']!=train.source_hash() or report['F_G_hashes']!=train._fg_hashes(data)
            or report['manifest_sha256']!=canonical_hash(data.manifest.to_dict())
            or report['identity_hashes']['raw']!=data.manifest.dataset_id
            or report['identity_hashes']['split']!=data.manifest.assignment_hash):
        raise ValueError('Parent source/data/status differs from its prerequisite')
    capture=json.loads(_committed(root,report['capture']).read_text())
    before=train.capture_rng()
    try:
        if snapshot=='late_G_best':
            path=_committed(root,report['selected'])
            state=load_checkpoint(path,expected_hashes=report['identity_hashes'],device='cpu')
        else:
            if (not isinstance(snapshot,str) or not snapshot.startswith('actual_')
                    or not snapshot[7:].isdigit()):
                raise ValueError('Unknown planned parent snapshot')
            update=int(snapshot[7:])
            if update not in capture['requested']:
                raise ValueError('Requested actual update was not planned')
            if update in capture['unavailable']:
                return None
            records=[r for r in capture['records'] if r['label']=='current' and r['update']==update]
            if len(records)!=1:raise ValueError('Missing or duplicate actual snapshot record')
            record=records[0];path=Path(record['pointer']).resolve()
            if not path.is_relative_to(root.resolve()):raise ValueError('Actual snapshot escaped parent root')
            pointer,_,_=_descriptor(path)
            if pointer['sha256']!=record['metadata_sha256']:
                raise ValueError('Actual snapshot commitment mismatch')
            captured=load_checkpoint(path,expected_hashes=report['identity_hashes'],device='cpu')
            state=current_boundary(train.pack_state(captured),data,requested_update=update)
        if (state.identity_hashes!=train._identities(data,state.config,state.preprocessing)
                or state.data_hashes!=train._fg_hashes(data)):
            raise ValueError('Loaded parent differs from supplied F/G-only context')
        return state
    finally:train.restore_rng(before)
