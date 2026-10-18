#!/usr/bin/env python3
"""Finite raw-artifact audit; never imports a model loader or fits/restores a state."""
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import resource
import struct
import sys
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[3]
ROOTS = [MAIN / 'artifacts/research-v2/WP02' / name for name in
         ('M0-native-qualification', 'M0-journal-qualification')]
START = time.perf_counter_ns()
INVENTORY = {}
COUNTS = collections.Counter()
STAGES = []
ARRAYS = []
TENSORS = []
FAILURES = []
MISMATCHES = []
ALLOWANCES = []
VERIFIED_GROWTH_HASHES = set()
MISSING = object()


def emit(value):
    line = json.dumps(value, sort_keys=True, allow_nan=False)
    print(line, flush=True)
    with (HERE / 'audit.log').open('a') as stream:
        stream.write(line + '\n')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def raw(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root) for root in ROOTS):
        raise ValueError(f'Input outside declared audit roots: {path}')
    value = path.read_bytes()
    item = dict(path=str(path.relative_to(MAIN)), bytes=len(value), sha256=sha(value))
    prior = INVENTORY.setdefault(str(path), item)
    if prior != item:
        raise ValueError(f'Input changed during audit: {path}')
    return value


def load(path):
    return json.loads(raw(path))


def record_mismatch(path, old, new):
    MISMATCHES.append(dict(path=path, old=old, new=new))


def brief(value):
    if value is MISSING:
        return {'missing': True}
    if isinstance(value, torch.Tensor):
        a = value.detach().cpu().contiguous().numpy()
        return dict(kind='tensor', dtype=str(value.dtype), shape=list(value.shape),
                    sha256=sha(a.tobytes()))
    return value


def floats_equal(a, b):
    return struct.pack('>d', a) == struct.pack('>d', b)


def compare(a, b, path, allowance=None):
    if a is MISSING and b is MISSING:
        return
    if allowance is not None and allowance(path, a, b):
        if a is MISSING or b is MISSING or type(a) is not type(b) or a != b:
            ALLOWANCES.append(dict(path=path, old=brief(a), new=brief(b)))
        return
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        COUNTS['tensor_leaves'] += 1
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            record_mismatch(path, brief(a), brief(b))
            return
        aa, bb = a.detach().cpu().contiguous().numpy(), b.detach().cpu().contiguous().numpy()
        same = a.dtype == b.dtype and a.shape == b.shape and aa.tobytes() == bb.tobytes()
        TENSORS.append(dict(path=path, old=brief(a), new=brief(b), byte_exact=same))
        if not same:
            record_mismatch(path, brief(a), brief(b))
            if a.dtype == b.dtype and a.shape == b.shape:
                width = aa.dtype.itemsize
                left = np.frombuffer(aa.tobytes(), dtype=np.uint8).reshape(-1, width)
                right = np.frombuffer(bb.tobytes(), dtype=np.uint8).reshape(-1, width)
                for i in np.flatnonzero(np.any(left != right, axis=1)):
                    record_mismatch(f'{path}/element/{int(i)}', left[i].tobytes().hex(), right[i].tobytes().hex())
        return
    if a is MISSING or b is MISSING:
        record_mismatch(path, brief(a), brief(b))
    elif type(a) is not type(b):
        record_mismatch(path, brief(a), brief(b))
    elif isinstance(a, dict):
        for key in sorted(a.keys() | b.keys(), key=str):
            compare(a.get(key, MISSING), b.get(key, MISSING), f'{path}/{key}', allowance)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            record_mismatch(path + '/length', len(a), len(b))
        for i, (x, y) in enumerate(itertools.zip_longest(a, b, fillvalue=MISSING)):
            compare(x, y, f'{path}/{i}', allowance)
    elif isinstance(a, float):
        if not floats_equal(a, b):
            record_mismatch(path, {'value': a, 'float64_hex': struct.pack('>d', a).hex()},
                            {'value': b, 'float64_hex': struct.pack('>d', b).hex()})
    elif a != b:
        record_mismatch(path, a, b)


def safe(label, function, *args):
    try:
        return function(*args)
    except Exception as error:
        failure = dict(label=label, exception=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
        FAILURES.append(failure)
        emit({'failure': failure})


def compare_npz(left, right, label):
    old_bytes, new_bytes = raw(left), raw(right)
    import io
    with np.load(io.BytesIO(old_bytes), allow_pickle=False) as a, np.load(io.BytesIO(new_bytes), allow_pickle=False) as b:
        compare(sorted(a.files), sorted(b.files), label + '/keys')
        for key in sorted(set(a.files) & set(b.files)):
            x, y = a[key], b[key]
            same = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
            ARRAYS.append(dict(path=label + '/' + key, dtype=str(x.dtype), shape=list(x.shape),
                               old_sha256=sha(x.tobytes()), new_sha256=sha(y.tobytes()), byte_exact=same))
            COUNTS[label + '_arrays'] += 1
            if not same:
                record_mismatch(label + '/' + key, dict(dtype=str(x.dtype), shape=list(x.shape), sha256=sha(x.tobytes())),
                                dict(dtype=str(y.dtype), shape=list(y.shape), sha256=sha(y.tobytes())))
                if x.dtype == y.dtype and x.shape == y.shape:
                    width = x.dtype.itemsize
                    aa = np.frombuffer(x.tobytes(), dtype=np.uint8).reshape(-1, width)
                    bb = np.frombuffer(y.tobytes(), dtype=np.uint8).reshape(-1, width)
                    for i in np.flatnonzero(np.any(aa != bb, axis=1)):
                        record_mismatch(f'{label}/{key}/element/{int(i)}', aa[i].tobytes().hex(), bb[i].tobytes().hex())


def events(path):
    return [json.loads(line) for line in raw(path).splitlines()]


def normalize_history(rows):
    output = []
    for event in rows:
        item = dict(event)
        if item.get('type') == 'checkpoint':
            item.pop('sha256', None)
        output.append(item)
    return output


def compare_stream(left, right, label):
    a, b = events(left / 'events.jsonl'), events(right / 'events.jsonl')
    growth_hashes = []
    for directory, rows in ((left, a), (right, b)):
        initialized = [e for e in rows if e['type'] == 'probe_initialized']
        if initialized:
            descriptor, _, _ = authenticate(directory / 'checkpoints/best.json')
            growth = json.loads(json.dumps(descriptor['state']['growth_metadata']))
            for key in ('executed_updates', 'selected_stage_updates', 'stopping_reason', 'hit_ceiling'):
                growth['greedy_probe'].pop(key, None)
            expected = sha(json.dumps(growth, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())
            assert len(initialized) == 1 and initialized[0]['growth_hash'] == expected, (label, 'probe initial growth hash')
            growth_hashes.append(expected)
    if growth_hashes:
        assert len(growth_hashes) == 2
        VERIFIED_GROWTH_HASHES.add(tuple(growth_hashes))
    # Preserve list order and multiplicity; never index events by update.
    compare(normalize_history(a), normalize_history(b), label + '/raw_ordered_events', metadata_allowance)
    for side, rows in (('old', a), ('new', b)):
        starts = [e for e in rows if e['type'] == 'stage_started']
        updates = [e for e in rows if e['type'] == 'update']
        selections = [e for e in rows if e['type'] == 'selection']
        assert len(starts) == len(selections) == 1, (label, side, 'stage boundaries')
        start, selection = starts[0]['update'], selections[0]
        assert [e['update'] for e in updates] == list(range(start + 1, selection['last_update'] + 1)), (label, side, 'update sequence')
        assert selection['executed_updates'] == len(updates), (label, side, 'executed count')
        selection_index = next(i for i, e in enumerate(rows) if e['type'] == 'selection')
        assert all(e['type'] == 'probe_stopping_evidence' for e in rows[selection_index + 1:]), (label, side, 'selection terminal to training')
        for e in updates:
            assert e['stage_update'] == e['update'] - start and e['rows'] == len(e['row_ids']), (label, side, 'raw update completeness')
        g = [e for e in rows if e['type'] == 'evaluation' and e['split'] == 'G']
        assert all(math.isfinite(e['metrics']['mse']) for e in g)
        assert min(g, key=lambda e: e['metrics']['mse'])['update'] == selection['selected_update'], (label, side, 'earliest G-best')
    summary = dict(label=label, old_path=str(left.relative_to(MAIN)), new_path=str(right.relative_to(MAIN)),
                   updates=sum(e['type'] == 'update' for e in a),
                   G_evaluations=sum(e['type'] == 'evaluation' and e['split'] == 'G' for e in a),
                   checkpoint_events=sum(e['type'] == 'checkpoint' for e in a),
                   selection=[e for e in a if e['type'] == 'selection'][0])
    STAGES.append(summary)
    COUNTS['native_stages'] += 1
    COUNTS['raw_update_pairs'] += summary['updates']
    COUNTS['raw_G_evaluation_pairs'] += summary['G_evaluations']
    emit({'compared_stage': summary})


def metadata_allowance(path, a, b):
    # Each allowed difference below is recorded verbatim, never silently discarded.
    if path.endswith('/execution/artifacts_root'):
        return a == str(ROOTS[0].relative_to(MAIN) / 'pair') and b == str(ROOTS[1].relative_to(MAIN) / 'pair')
    if path.endswith('/execution/backend_report'):
        return a == 'environment/research-v2-WP02-cpu.json' and b == 'environment/research-v2-journal-qualified-cpu.json'
    if path.endswith('/identity_hashes/code') or path.endswith('/parent_identity_hashes/code') or path.endswith('/backend/causal_cpu_evidence/source_hash'):
        return a == RESULTS[0]['source_sha256'] and b == RESULTS[1]['source_sha256']
    if path.endswith('/identity_hashes/config') or path.endswith('/parent_identity_hashes/config'):
        return isinstance(a, str) and isinstance(b, str) and len(a) == len(b) == 64
    if path.endswith('/backend/evidence_sha256'):
        return a == RESULTS[0]['execution_identity']['backend_receipt_sha256'] and b == RESULTS[1]['execution_identity']['backend_receipt_sha256']
    if path.endswith('/greedy_probe/parent_checkpoint_sha256'):
        return a == GREEDY[0]['parent_checkpoint_sha256'] and b == GREEDY[1]['parent_checkpoint_sha256']
    if path.endswith('/growth_hash'):
        return (a, b) in VERIFIED_GROWTH_HASHES
    if path.endswith('/greedy_probe/coordinator_run_id'):
        return a == GREEDY_PATHS[0].parent.name and b == GREEDY_PATHS[1].parent.name
    if '/policy/' in path and a is MISSING:
        field = path.rsplit('/', 1)[1]
        return ((field == 'capture_schema' and b == 2)
                or (field == 'selection_prefixes' and b == [])
                or (field == 'snapshot_updates' and b in ([], [250, 1000]))
                or (field == 'stop_by_patience' and b is True))
    return False


def authenticate(pointer, expected_pointer_sha=None):
    pointer_bytes = raw(pointer)
    if expected_pointer_sha is not None:
        assert sha(pointer_bytes) == expected_pointer_sha, ('pointer checksum', str(pointer))
    p = json.loads(pointer_bytes)
    descriptor_path = pointer.parent / p['metadata']
    descriptor_bytes = raw(descriptor_path)
    assert sha(descriptor_bytes) == p['sha256'], ('descriptor checksum', str(pointer))
    descriptor = json.loads(descriptor_bytes)
    payload_path = descriptor_path.parent / descriptor['payload']
    payload_bytes = raw(payload_path)
    assert sha(payload_bytes) == descriptor['sha256'], ('tensor payload checksum', str(pointer))
    import io
    payload = torch.load(io.BytesIO(payload_bytes), map_location='cpu', weights_only=True)
    history = descriptor['history']
    if isinstance(history, dict):
        content = raw(descriptor_path.parent / history['file'])[:history['bytes']]
        assert len(content) == history['bytes'] and sha(content) == history['sha256'], ('history prefix checksum', str(pointer))
        history_rows = [json.loads(line) for line in content.splitlines()]
        assert len(history_rows) == history['count'], ('history record count', str(pointer))
    else:
        assert isinstance(history, list)
        history_rows = history
    for state in (descriptor['state'], descriptor['best_state']):
        if state is not None:
            config_digest = sha(json.dumps(state['config'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())
            assert state['identity_hashes']['config'] == config_digest, ('resolved config digest', str(pointer))
    return descriptor, payload, history_rows


def compare_checkpoint(left, right, label, expected_old=None, expected_new=None):
    a, x, ah = authenticate(left, expected_old)
    b, y, bh = authenticate(right, expected_new)
    assert a['schema_version'] == 1 and b['schema_version'] == 2, (label, 'declared metadata migration')
    ALLOWANCES.append(dict(path=label + '/descriptor_schema_version', old=a['schema_version'], new=b['schema_version']))
    ALLOWANCES.append(dict(path=label + '/history_storage', old='inline JSON list', new='authenticated append-only journal prefix'))
    compare(a['state'], b['state'], label + '/metadata/state', metadata_allowance)
    compare(a['best_state'], b['best_state'], label + '/metadata/best_state', metadata_allowance)
    compare(normalize_history(ah), normalize_history(bh), label + '/history_semantics')
    compare(x, y, label + '/payload')
    COUNTS['authenticated_checkpoint_pairs'] += 1
    COUNTS['byte_identical_tensor_payload_files'] += a['sha256'] == b['sha256']


def compare_stage(left, right, label):
    safe(label + '/stream', compare_stream, left, right, label)
    for pointer in ('best.json', 'last.json'):
        safe(label + '/' + pointer, compare_checkpoint,
             left / 'checkpoints' / pointer, right / 'checkpoints' / pointer, label + '/' + pointer)


def same_keys(a, b, keys, path):
    for key in keys:
        compare(a.get(key, MISSING), b.get(key, MISSING), path + '/' + key, metadata_allowance)


def main():
    torch.set_num_threads(1)
    emit(dict(action='finite same-case storage equivalence audit', roots=[str(r) for r in ROOTS],
              source_state_restoration=False, fitting=False, protected_T_access=False,
              independent_teacher=False, exactness='dtype/shape/byte equality for arrays/tensors; exact ordered raw events and float64 bits'))
    compare_npz(ROOTS[0] / 'observations.npz', ROOTS[1] / 'observations.npz', 'observations')
    compare_npz(ROOTS[0] / 'pair/predictions.npz', ROOTS[1] / 'pair/predictions.npz', 'predictions')
    for file in ('teacher.json', 'split.json'):
        compare(load(ROOTS[0] / file), load(ROOTS[1] / file), file)
    for field in ('status', 'teacher_index', 'repetition', 'final_roles', 'nominal_probe_trials', 'protected_labels_accessed', 'script_sha256'):
        compare(RESULTS[0][field], RESULTS[1][field], 'qualification_result/' + field)
    assert all(r['status'] == 'completed' and r['final_roles'] == 8 and r['nominal_probe_trials'] == 65 for r in RESULTS)
    same_keys(*PAIRS, ('teacher_id', 'optimizer_seed', 'data_hash', 'capacity', 'controls', 'failures', 'status', 'T_accessed', 'V_accessed',
                       'greedy_keep_decision', 'pairing_audit', 'parent_pairing'), 'pair')
    compare(sorted(PAIRS[0]['methods']), sorted(PAIRS[1]['methods']), 'pair/methods/keys')
    assert len(PAIRS[0]['methods']) == 8
    parent_dirs = [r / 'pair/runs' / p['parent_run_id'] for r, p in zip(ROOTS, PAIRS)]
    for side in range(2):
        assert sha(raw(parent_dirs[side] / 'checkpoints/best.json')) == PAIRS[side]['parent_checkpoint_sha256']
    compare_stage(*parent_dirs, 'parent')
    for role in sorted(PAIRS[0]['methods']):
        a, b = [p['methods'][role] for p in PAIRS]
        same_keys(a, b, ('allocation', 'executed_row_ids', 'initial_pairing', 'method_id', 'recipe',
                         'selected_g_mse', 'status', 'total_parameters'), 'methods/' + role)
        compare(len(a['stages']), len(b['stages']), 'methods/' + role + '/stage_count')
        for index, (old, new) in enumerate(zip(a['stages'], b['stages'])):
            same_keys(old, new, tuple(k for k in old if k not in ('run_id', 'path')), f'methods/{role}/stages/{index}')
            compare_stage(ROOTS[0] / 'pair' / old['path'], ROOTS[1] / 'pair' / new['path'], f'final/{role}/stage{index}')
        model_refs = [p['models'][role] for p in PAIRS]
        safe('export/' + role, compare_checkpoint, *(Path(ref['path']) for ref in model_refs),
             'export/' + role, *(ref['sha256'] for ref in model_refs))
    a, b = GREEDY
    for side in range(2):
        # Greedy's private parent pointer is distinct from the main parent's
        # best pointer; authenticate both rather than assuming their bytes match.
        private_parent = GREEDY_PATHS[side].parent / 'parent.json'
        descriptor_bytes = raw(private_parent)
        pointer = json.loads(descriptor_bytes)
        assert pointer['sha256'] == GREEDY[side]['parent_checkpoint_sha256']
        authenticate(private_parent)
    same_keys(a, b, ('T_accessed', 'V_accessed', 'allocation', 'complete', 'cost_boundary', 'effective_count', 'eligible_ids',
                     'improvements', 'keep_decision', 'keep_g_mse', 'method', 'not_attempted', 'probe_updates', 'recipe',
                     'requested_count', 'schema_version', 'score_hash', 'selection_split', 'stopping_mode'), 'greedy')
    compare(a['policy'], b['policy'], 'greedy/policy', metadata_allowance)
    assert len(a['trials']) == len(b['trials']) == 65
    compare([t['candidate'] for t in a['trials']], [t['candidate'] for t in b['trials']], 'greedy/ordered_candidates')
    for side in range(2):
        score_path = GREEDY_PATHS[side].parent / GREEDY[side]['score_artifact']['path']
        assert sha(raw(score_path)) == GREEDY[side]['score_artifact']['sha256']
    compare_npz(GREEDY_PATHS[0].parent / a['score_artifact']['path'], GREEDY_PATHS[1].parent / b['score_artifact']['path'], 'score_table')
    for index, (old, new) in enumerate(zip(a['trials'], b['trials'])):
        label = f'probe/{index:02d}/{old["candidate"]}'
        same_keys(old, new, tuple(k for k in old if k not in ('nanoseconds', 'parent_identity_hashes', 'path', 'run_id')), label + '/trial_record')
        compare_stage(GREEDY_PATHS[0].parent / old['path'], GREEDY_PATHS[1].parent / new['path'], label)
        COUNTS['probe_trials'] += 1


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


RESULTS = [load(root / 'result.json') for root in ROOTS]
PAIRS = [load(root / 'pair/pair.json') for root in ROOTS]
GREEDY_PATHS = [Path(pair['greedy_report']).resolve() for pair in PAIRS]
GREEDY = [load(path) for path in GREEDY_PATHS]
safe('audit', main)
if COUNTS['native_stages'] != 77 or COUNTS['probe_trials'] != 65 or COUNTS['authenticated_checkpoint_pairs'] != 162:
    record_mismatch('required_finite_coverage', dict(native_stages=77, probe_trials=65, authenticated_checkpoint_pairs=162),
                    {k: COUNTS[k] for k in ('native_stages', 'probe_trials', 'authenticated_checkpoint_pairs')})
for absolute, entry in list(INVENTORY.items()):
    safe('input_stability/' + entry['path'], raw, Path(absolute))
write_json(HERE / 'input-hashes.json', sorted(INVENTORY.values(), key=lambda e: e['path']))
write_json(HERE / 'allowed-metadata-differences.json', ALLOWANCES)
write_json(HERE / 'array-comparisons.json', ARRAYS)
write_json(HERE / 'tensor-comparisons.json', TENSORS)
write_json(HERE / 'stage-comparisons.json', STAGES)
write_json(HERE / 'mismatches.json', MISMATCHES)
write_json(HERE / 'failures.json', FAILURES)
result = dict(schema='research/v2-native-storage-equivalence-audit', status='passed' if not FAILURES and not MISMATCHES else 'failed',
              scope='same first predeclared exploratory teacher/repetition; source/storage engineering equivalence only',
              gate_A='pending; no efficacy or independent-teacher evidence', fitting_performed=False, state_restored=False,
              protected_T_accessed=False, transfer_labels_accessed=False, comparison_counts=dict(COUNTS),
              failures=len(FAILURES), mismatches=len(MISMATCHES), allowed_metadata_differences=len(ALLOWANCES),
              inputs=[dict(root=str(root.relative_to(MAIN)), source_commit=r['source_commit'], source_sha256=r['source_sha256'],
                           containing_wall_seconds=r['elapsed_ns'] / 1e9, peak_rss_bytes=r['peak_rss_bytes'],
                           max_checkpoint_transaction_bytes=r['max_checkpoint_transaction_bytes'], retained_bytes=r['retained_bytes'],
                           result_sha256=sha(raw(root / 'result.json')), pair_sha256=sha(raw(root / 'pair/pair.json')))
                      for root, r in zip(ROOTS, RESULTS)],
              resource_interpretation='Two contended single-case observations; variance uncalibrated, clocks nested; no formal efficiency ratio or full resource admission.',
              allowed_difference_policy=['Source/config/backend receipt identities and execution output paths are retained and checked separately.',
                                        'Metadata schema1 inline histories versus schema2 authenticated append-only journal prefixes; history event semantics compared exactly except checkpoint descriptor hashes.',
                                        'Checkpoint UUIDs/descriptor hashes/run IDs/artifact paths and containing timings/resources are bookkeeping, not numeric equivalence targets.',
                                        'New capture_schema2, empty selection prefixes, declared snapshot capture points and stop_by_patience=True record formerly implicit/callback behavior; raw trajectories still compared exactly.'],
              audit_runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                                 elapsed_ns=time.perf_counter_ns() - START, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
              script_sha256=sha(Path(__file__).read_bytes()))
write_json(HERE / 'result.json', result)
emit({'audit_result': result})
sys.exit(0 if result['status'] == 'passed' else 1)
