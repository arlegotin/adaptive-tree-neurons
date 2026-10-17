"""Fresh same-seed replay against standalone prechange numerical dictionaries."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.artifacts import create_run, finish_run
from treeneuron.checkpoint import load_checkpoint
from treeneuron.config import resolve_experiment, canonical_hash, canonical_json
from treeneuron.data.synthetic import make_from_config, sample_teacher
from treeneuron.data.types import array_hash
from treeneuron.experiments.paired import write_json, file_hash
from treeneuron.protocol import execution_snapshot

from types import SimpleNamespace
base = Path(__file__).parent
out = base/'replay-4-completed-2'
out.mkdir(exist_ok=False)
ref = Path('environment/h1-v1/calibration/old-source-reference')
old = json.loads((ref/'provenance.json').read_text())
commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
snapshot = execution_snapshot(Path.cwd(),commit)
assert snapshot == json.loads((base/'strict-snapshot.json').read_text())
record = json.loads((base/'replay-4/replay-measurement.json').read_text())
record['fit_source_measurement_path'] = str(base/'replay-4/replay-measurement.json')
record['preserved_postprocessing_failure'] = record['failures']
record['failures'] = []
record['status'] = 'comparing_retained_same_source_fit'
record['completion_wrapper_sha256'] = file_hash(Path(__file__))
config = resolve_experiment(json.loads((base/'replay-4/replay-config.json').read_text()))
teacher,data = make_from_config(config)
E = sample_teacher(teacher,old['E']['rows'],old['E']['seed'],config.data.noise_sigma)
assert canonical_json(teacher.to_dict()) == canonical_json(json.loads((ref/'teacher.json').read_text()))
assert canonical_json(data.manifest.to_dict()) == canonical_json(json.loads((ref/'splits.json').read_text()))
arrays = np.load(ref/'observations.npz')
for name,rows in [('F',data.F),('G',data.G),('V',data.V),('E',E)]:
    for field in ('X','y','row_id','group_id','noise_free_target'):
        value=getattr(rows,field)
        if value is not None:
            key=f'{name}_{field}'
            assert value.dtype == arrays[key].dtype and value.tobytes() == arrays[key].tobytes()
            assert array_hash(value) == old['arrays'][key]['sha256']
paths = SimpleNamespace(directory=Path(record['parent_path']))
meta = json.loads((base/'replay-4/selected-metadata.json').read_text())
state = load_checkpoint(paths.directory/'checkpoints/best.json',expected_hashes=meta['identity_hashes'],device='cpu')
started = time.perf_counter_ns()
try:
    # Compare every tensor, scalar and container in the selected plain snapshot.
    # This does not construct or load an old-source TrainingState.
    expected = torch.load(ref/'selected-snapshot.pt', map_location='cpu', weights_only=True)
    actual = train.pack_state(state)
    torch.save(actual, out/'selected-snapshot.pt')
    write_json(out/'selected-metadata.json', actual['metadata'])
    tensor_hashes = {}
    def exact(a, b, path):
        assert type(a) is type(b), (path, type(a), type(b))
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and a.shape == b.shape and a.device == b.device, path
            x, y = a.detach().contiguous().numpy().tobytes(), b.detach().contiguous().numpy().tobytes()
            assert x == y, path
            tensor_hashes[path] = dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(x).hexdigest())
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a: exact(a[k], b[k], f'{path}.{k}')
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), path
            for i, (x,y) in enumerate(zip(a,b)): exact(x,y,f'{path}[{i}]')
        else: assert a == b, (path,a,b)
    exact(expected['tensors'], actual['tensors'], 'tensors')
    selected_tensor_count = len(tensor_hashes)
    write_json(out/'exact-tensor-hashes.json', tensor_hashes)
    allowed = {'metadata.config.execution.artifacts_root','metadata.config.execution.backend_report',
               'metadata.identity_hashes.code','metadata.identity_hashes.config','metadata.backend.evidence_sha256'}
    differences = {}
    def metadata(a,b,path):
        if path in allowed:
            if a != b: differences[path] = dict(old=a,new=b)
            return
        if isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a: metadata(a[k],b[k],f'{path}.{k}')
        elif isinstance(a, (list,tuple)):
            assert len(a) == len(b), path
            for i,(x,y) in enumerate(zip(a,b)):metadata(x,y,f'{path}[{i}]')
        else: assert a == b, (path,a,b)
    metadata(expected['metadata'], actual['metadata'], 'metadata')
    assert set(differences) == allowed
    write_json(out/'metadata-differences.json', differences)
    expected_predictions = np.load(ref/'predictions.npz')
    predictions = {name:state.predict(rows.X) for name,rows in [('F',data.F),('G',data.G),('E',E)]}
    for name,value in predictions.items():
        assert value.dtype == expected_predictions[name].dtype
        assert value.tobytes() == expected_predictions[name].tobytes(), name
        assert array_hash(value) == old['predictions'][name]['sha256']
    np.savez_compressed(out/'predictions.npz', **predictions)
    record['exact_prediction_hashes'] = {k:array_hash(v) for k,v in predictions.items()}
    def events(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]
    raw_events = events(paths.directory/'events.jsonl')
    old_events = events(old['raw_events_path'])
    assert len(raw_events) == len(old_events)
    checkpoint_differences = 0
    for a,b in zip(old_events, raw_events):
        if a['type'] == 'checkpoint':
            assert a['sha256'] != b['sha256']
            a, b = dict(a), dict(b)
            a.pop('sha256'); b.pop('sha256'); checkpoint_differences += 1
        assert a == b, a.get('type')
    updates = []
    reference_updates = json.loads((ref/'ordered-update-commitments.json').read_text())
    for event in raw_events:
        if event['type'] != 'update': continue
        ids = np.asarray(event['row_ids'], dtype=np.int64)
        updates.append(dict(update=event['update'],stage_update=event['stage_update'],rows=event['rows'],
                            array_hash=array_hash(ids),ordered_int64_bytes_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                            loss_standardized_mse=event['loss_standardized_mse'],learning_rate=event['learning_rate']))
    assert updates == reference_updates['updates']
    history_events = [e for e in raw_events if e['type'] in {'update','evaluation','checkpoint','selection'}]
    assert len(history_events) == len(state.history)
    for event,saved in zip(history_events,state.history):
        expected_event = dict(event)
        if event['type'] == 'update':
            ids = np.asarray(expected_event.pop('row_ids'), dtype=np.int64)
            expected_event['row_ids_commitment'] = dict(schema_version=1,encoding='ordered_int64_array_hash',count=len(ids),sha256=array_hash(ids))
        assert saved == expected_event
    write_json(out/'ordered-update-commitments.json', dict(count=len(updates),examples=sum(e['rows'] for e in updates),updates=updates))
    old_history = [e for e in old_events if e['type'] in {'update','evaluation','checkpoint','selection'}]
    scaling = []
    for n in (25,100,1000,2000,4000):
        selected = lambda hist: [e for e in hist if e.get('update',e.get('last_update',0)) <= n]
        oh, nh = selected(old_history), selected(state.history)
        scaling.append(dict(updates=n,old_history_bytes=len(canonical_json(dict(history=oh)).encode()),
                            compact_history_bytes=len(canonical_json(dict(history=nh)).encode()),
                            examples=sum(e['rows'] for e in nh if e['type']=='update')))
    write_json(out/'history-scaling.json', dict(measurements=scaling))
    # New-source reload validates receipt and preserved sampler look-ahead.
    loaded = load_checkpoint(paths.directory/'checkpoints/best.json',expected_hashes=state.identity_hashes,device='cpu')
    exact(actual['tensors'],train.pack_state(loaded)['tensors'],'reloaded_tensors')
    assert loaded.history == state.history
    assert np.array_equal(loaded.next_row_ids(data.F),state.next_row_ids(data.F))
    write_json(out/'next-row-ids.json', dict(count=len(state.next_row_ids(data.F)),sha256=array_hash(state.next_row_ids(data.F))))
    meta_files = sorted((paths.directory/'checkpoints').glob('checkpoint-*.meta.json'))
    old_measurement = json.loads(Path(old['measurement_path']).read_text())
    old_parent = Path(old_measurement['parent_path'])
    record.update(selected_update=state.update,executed_updates=len(updates),actual_examples=sum(e['rows'] for e in updates),
                  exact_selected_tensor_count=selected_tensor_count,metadata_differences=differences,
                  exact_raw_events=len(raw_events),checkpoint_hash_differences=checkpoint_differences,
                  checkpoint_metadata=[dict(path=str(p),bytes=p.stat().st_size,sha256=file_hash(p)) for p in meta_files],
                  two_metadata_bytes=sum(p.stat().st_size for p in meta_files),
                  old_fit_enclosing_ns=old_measurement['parent_enclosing_ns'],old_peak_memory_bytes=old_measurement['peak_memory_bytes'],
                  old_parent_directory_bytes=sum(p.stat().st_size for p in old_parent.rglob('*') if p.is_file()),
                  old_two_metadata_bytes=old_measurement['two_metadata_probe_floor_bytes'],
                  raw_events_sha256=file_hash(paths.directory/'events.jsonl'),
                  raw_update_events_sha256=hashlib.sha256(canonical_json(dict(events=[e for e in raw_events if e['type']=='update'])).encode()).hexdigest(),
                  old_raw_update_events_sha256=hashlib.sha256(canonical_json(dict(events=[e for e in old_events if e['type']=='update'])).encode()).hexdigest(),
                  exact_count_and_ordered_row_commitments=True,exact_model_optimizer_schedule_sampler_rng=True,
                  exact_F_G_E_predictions=True,exact_data_arrays=True,new_source_reload=True,
                  final_source_snapshot_unchanged=execution_snapshot(Path.cwd(),commit)==snapshot,status='passed')
    assert record['final_source_snapshot_unchanged']
except BaseException as error:
    record['status'] = 'failed'
    record['failures'].append(f'{type(error).__name__}: {error}')
    raise
finally:
    record.update(comparison_enclosing_ns=time.perf_counter_ns()-started,
                  process_peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(out/'replay-measurement.json',record)
print(json.dumps({k:record[k] for k in ('status','selected_update','executed_updates','actual_examples','fit_enclosing_ns','parent_directory_bytes','two_metadata_bytes','fit_peak_memory_bytes')}),flush=True)
