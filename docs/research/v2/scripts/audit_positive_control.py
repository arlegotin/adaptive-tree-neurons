#!/usr/bin/env python3
"""Independent NumPy/JSON M1 audit. No repository imports, Torch, restore or fit."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[3]
ROOT = MAIN / 'artifacts/research-v2/WP02/M1-native-qualification'
ROLES = ('keep', 'top', 'random', 'oracle_current', 'oracle_true')
COMMIT = 'b11bbff5c4c106f83a342f8b8e8f667bb07dbd2a'
START = time.perf_counter_ns()
INPUTS = {}
FAILURES = []
CHECKS = []
ARRAY_CHECKS = []
METRICS = {}
NOISE = {}
STAGES = {}
ALLOCATIONS = {}
SOURCES = {}
MAX_METRIC_DIFFERENCE = 0.


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())


def seed64(root, *labels):
    data = json.dumps([root, *labels], separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big') & ((1 << 63)-1)


def read(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Artifact escaped declared run: {path}')
    content = path.read_bytes()
    item = dict(path=str(path.relative_to(MAIN)), bytes=len(content), sha256=sha(content))
    if INPUTS.setdefault(str(path), item) != item:
        raise ValueError(f'Input changed during audit: {path}')
    return content


def load(path):
    return json.loads(read(path))


def npz(path):
    with np.load(io.BytesIO(read(path)), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def check(label, condition, detail=None):
    CHECKS.append(dict(label=label, passed=bool(condition), detail=detail))
    if not condition:
        FAILURES.append(dict(label=label, detail=detail))


def exact(label, actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    item = dict(label=label, dtype=str(a.dtype), shape=list(a.shape), actual_sha256=sha(a.tobytes()),
                expected_sha256=sha(b.tobytes()), exact=same)
    ARRAY_CHECKS.append(item)
    check(label, same, item)


def array_hash(a):
    a = np.ascontiguousarray(a)
    return sha(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def snapshot_source(path):
    content = subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=MAIN)
    SOURCES[path] = dict(commit=COMMIT, sha256=sha(content))
    return content


def authenticate(pointer, expected_pointer_hash=None):
    content = read(pointer)
    if expected_pointer_hash is not None:
        check(str(pointer) + '/pointer_authentication', sha(content) == expected_pointer_hash)
    index = json.loads(content)
    descriptor_path = pointer.parent / index['metadata']
    content = read(descriptor_path)
    check(str(pointer) + '/descriptor_authentication', sha(content) == index['sha256'])
    descriptor = json.loads(content)
    payload_path = descriptor_path.parent / descriptor['payload']
    check(str(pointer) + '/payload_checksum_without_deserialization', sha(read(payload_path)) == descriptor['sha256'])
    if isinstance(descriptor['history'], dict):
        h = descriptor['history']
        prefix = read(descriptor_path.parent / h['file'])[:h['bytes']]
        check(str(pointer) + '/history_prefix_authentication', len(prefix) == h['bytes'] and sha(prefix) == h['sha256'])
    return descriptor


def verify_stage(role, metadata, directories, record):
    matches = [path for path, desc in directories if desc['sha256'] == metadata['sha256']]
    check(role + '/unique_native_stream_by_payload_hash', len(matches) == 1, [str(p) for p in matches])
    if len(matches) != 1:
        return
    directory = matches[0]
    rows = [json.loads(line) for line in read(directory / 'events.jsonl').splitlines()]
    starts = [e for e in rows if e['type'] == 'stage_started']
    selections = [e for e in rows if e['type'] == 'selection']
    updates = [e for e in rows if e['type'] == 'update']
    evaluations = [e for e in rows if e['type'] == 'evaluation' and e['split'] == 'G']
    check(role + '/one_native_stage_and_selection', len(starts) == len(selections) == 1)
    first, chosen = starts[0], selections[0]
    state = metadata['state']
    check(role + '/starts_from_unfitted_parent', first['update'] == state['stage']['start_update'] == 0)
    check(role + '/contiguous_ordered_updates', [e['update'] for e in updates] == list(range(1, chosen['last_update']+1)))
    check(role + '/selected_executed_counters', chosen['selected_update'] == state['update'] == record['selected_absolute_update']
          == record['selected_stage_update'] and chosen['executed_updates'] == len(updates)
          == record['executed_stage_updates'] == state['stage']['updates_completed'])
    check(role + '/within_native_budget', 0 <= chosen['selected_update'] <= chosen['last_update'] <= 8000)
    check(role + '/strict_earliest_G_best', min(evaluations, key=lambda e: e['metrics']['mse'])['update'] == chosen['selected_update'])
    check(role + '/patience_tail', chosen['last_update']-chosen['selected_update'] == 25*20)
    check(role + '/full_G_schedule', [e['update'] for e in evaluations] == list(range(0, chosen['last_update']+1, 25)))
    check(role + '/F_only_ordered_minibatch_IDs', all(e['rows'] == len(e['row_ids']) == 128
          and all(0 <= v < 512 for v in e['row_ids']) and e['stage_update'] == e['update']
          and e['learning_rate'] == .005 and np.isfinite(e['loss_standardized_mse']) for e in updates))
    check(role + '/native_terminal_status', load(directory / 'manifest.json')['identity']['status'] == 'completed')
    STAGES[role] = dict(native_directory=str(directory.relative_to(MAIN)), selection=chosen,
        raw_updates=len(updates), G_evaluations=len(evaluations), total_parameters=record['total_parameters'])


def main():
    global MAX_METRIC_DIFFERENCE
    result, manifest = load(ROOT / 'result.json'), load(ROOT / 'manifest.json')
    teacher, plan = load(ROOT / 'teacher.json'), load(ROOT / 'parent-plan.json')
    pair, coordinator = load(ROOT / 'pair/control-pair.json'), load(ROOT / 'pair/coordinator-metrics.json')
    observations, predictions, score = npz(ROOT / 'observations.npz'), npz(ROOT / 'pair/predictions.npz'), npz(ROOT / 'pair/score-table.npz')
    split = load(ROOT / 'split.json')
    check('terminal_run_and_pair', result['status'] == pair['status'] == 'completed')
    check('source_commit', result['source_commit'] == manifest['source_commit'] == COMMIT)
    source_paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', COMMIT, 'src/treeneuron'], cwd=MAIN, text=True).splitlines()
    package_digest = hashlib.sha256()
    for path in sorted(p for p in source_paths if p.endswith('.py')):
        package_digest.update(path.removeprefix('src/treeneuron/').encode())
        package_digest.update(snapshot_source(path))
    check('source_package_hash_from_bound_git_commit', package_digest.hexdigest() == result['source_sha256'] == manifest['source_sha256'] == plan['source_sha256'], package_digest.hexdigest())
    recipe = snapshot_source('docs/research/v2/scripts/measure_positive_control.py').decode()
    check('qualification_recipe_hash', sha(recipe.encode()) == result['script_sha256'] == manifest['script_sha256'])
    config_source = json.loads(snapshot_source('configs/research/v2/M1-native-qualification.json'))
    check('fixed_configuration', config_source['study_id'] == 'M1-controls' and config_source['kind'] == 'm1_controls'
          and config_source['seed_root'] == 2026101801 and config_source['teachers'] == 4 and config_source['repetitions'] == 2)
    retained_plan_hash = plan['plan_hash']
    check('canonical_parent_plan_hash', canonical({k:v for k,v in plan.items() if k != 'plan_hash'}) == retained_plan_hash
          == result['parent_plan_hash'] == manifest['parent_plan_hash'])
    attempts = [a for a in plan['attempts'] if a['population'] == 'positive' and a['gamma'] == 2. and a['sigma'] == .1
                and a['teacher_index'] == 0 and a['repetition'] == 0]
    check('exact_fixed_cell_and_role_order', attempts == result['attempts'] == manifest['attempts'] and [a['role'] for a in attempts] == list(ROLES))
    a = attempts[0]
    labels = ('M1-controls', 'positive', 'teacher', '0')
    expected_seeds = {name + '_seed': seed64(2026101801, *labels, name) for name in ('teacher','data','reference','evaluation')}
    paired = (*labels, 'repeat', '0')
    expected_seeds.update(optimizer_seed=seed64(2026101801, *paired, 'optimizer', 'library32', '0') & ((1 << 32)-1),
        optimizer_collision_counter=0, allocation_seed=seed64(2026101801, *paired, 'allocation'), contrast_seed=seed64(2026101801, *paired, 'contrast'))
    check('paired_seed_lineage', all(all(attempt[key] == value for key,value in expected_seeds.items()) for attempt in attempts), expected_seeds)
    check('teacher_matches_selected_seeds', teacher['instance_seed'] == a['teacher_seed'] and teacher['data_seed'] == a['data_seed'])
    check('teacher_before_observations_recipe_order_only', teacher['committed_before_observation_generation'] is True
          and recipe.index("write_new_json(root/'teacher.json'") < recipe.index('control=make_positive_control('))
    base = np.asarray(((1,0,2),(1,0,2),(-1,0,0),(0,1,2),(0,-1,2),(1,1,3),(-1,1,3),(0,0,-2)), dtype=np.float64)
    factors = np.random.default_rng(a['teacher_seed']).uniform(.95,1.05,8)
    factors[1] = factors[0]
    head = np.asarray((.5,.5,1.,.2,-.2,.1,-.1,.1)) * factors
    contrast = np.asarray(((0.,.5,0.),(.5,0.,0.)))
    gate = np.asarray(((2.,0.,0.),(0.,2.,0.)))
    exact('raw_base_coefficients', np.asarray(teacher['base']), base)
    exact('raw_head_coefficients_and_shared_duplicate_factors', np.asarray(teacher['head']), head)
    exact('raw_planted_contrasts', np.asarray(teacher['contrasts']), contrast)
    exact('raw_planted_gates', np.asarray(teacher['gates']), gate)
    check('declared_teacher_cell', teacher['generating_ids'] == [0,2] and teacher['gamma'] == 2. and teacher['sigma'] == .1)
    teacher_id = canonical(dict(version='research/v2-planted-stump', base=base.tolist(), head=head.tolist(), gamma=2., sigma=.1, instance_seed=a['teacher_seed']))
    check('canonical_teacher_identity', teacher_id == split['teacher_id'])
    offset = 0
    for role, count in (('F',512),('G',256),('V',256),('E',16384)):
        x, target, y = [observations[f'{role}_{name}'] for name in ('X','evaluator_noisefree','y')]
        derived_seed = seed64(a['data_seed'], 'positive', role)
        rng = np.random.default_rng(derived_seed)
        expected_x = rng.uniform(-1,1,(count,2))
        expected_noise = teacher['sigma'] * rng.standard_normal(count)
        exact(role + '/input_seed_reproduction', x, expected_x)
        ids = np.arange(offset, offset+count, dtype=np.int64)
        exact(role + '/row_IDs', observations[f'{role}_row_id'], ids)
        exact(role + '/group_IDs', observations[f'{role}_group_id'], ids)
        offset += count
        augmented = np.column_stack((x, np.ones(count)))
        activation = augmented @ base.T
        modulation = np.tanh((augmented @ gate.T)/2) * (augmented @ contrast.T)
        activation[:,[0,2]] += modulation
        independent_target = np.maximum(activation,0) @ head
        error = float(np.max(np.abs(independent_target-target)))
        check(role + '/independent_planted_equation', error <= 1e-13, dict(max_abs_error=error, tolerance=1e-13))
        exact(role + '/observed_y_from_retained_target_and_independent_noise', y, target+expected_noise)
        observed_noise = y-target
        check(role + '/sampled_noise_with_subtraction_rounding', float(np.max(np.abs(observed_noise-expected_noise))) <= 2e-15)
        NOISE[role] = dict(rows=count, actual_observation_seed=derived_seed, mse=float(np.mean(observed_noise**2)),
                           max_noise_reconstruction_error=float(np.max(np.abs(observed_noise-expected_noise))))
    variance = float(np.var(observations['F_y'], ddof=0))
    for role in NOISE:
        NOISE[role]['nmse_F_variance'] = NOISE[role]['mse']/variance
    check('sampled_E_noise_NMSE', NOISE['E']['nmse_F_variance'] == result['sampled_noise_NMSE'])
    check('exact_planned_role_sets', set(pair['models']) == set(pair['methods']) == set(coordinator['metrics']) == set(result['metrics']) == set(ROLES))
    for role in ('V','E'):
        exact(role + '/prediction_archive_target_ID_alignment', predictions[role+'_row_id'], observations[role+'_row_id'])
        exact(role + '/prediction_archive_label_alignment', predictions[role+'_y'], observations[role+'_y'])
    for role in ROLES:
        METRICS[role] = {}
        check(role + '/terminal_completion', pair['methods'][role]['status'] == 'completed')
        for split_role in ('V','E'):
            y, prediction = observations[split_role+'_y'], predictions[role+'_'+split_role]
            check(f'{role}/{split_role}/finite_complete_predictions', prediction.shape == y.shape and np.isfinite(prediction).all())
            residual = prediction-y
            mse = float(np.mean(residual**2)); nmse = mse/variance
            for owner, metrics in (('result',result['metrics']),('coordinator',coordinator['metrics'])):
                for name, value in (('mse',mse),('normalized_mse',nmse)):
                    difference = abs(value-metrics[role][split_role][name])
                    MAX_METRIC_DIFFERENCE = max(MAX_METRIC_DIFFERENCE,difference)
                    check(f'{role}/{split_role}/{owner}/{name}', difference <= 1e-14, difference)
            METRICS[role][split_role] = dict(mse=mse, normalized_mse=nmse, n=len(y),
                evaluator_noisefree_mse=float(np.mean((prediction-observations[split_role+'_evaluator_noisefree'])**2)))
    check('F_G_V_manifest_ID_assignment', split['row_id'] == list(range(1024)) and split['group_id'] == list(range(1024))
          and split['assignments'] == ['F']*512+['G']*256+['V']*256)
    for name, digest in split['hashes']:
        split_role, field = name.split('.')
        check('split_raw_hash/' + name, digest == array_hash(observations[f'{split_role}_{field}']))
    check('protected_labels_and_fit_V_flags', not result['protected_labels_accessed'] and not plan['protected_labels_accessed']
          and pair['fit_V_accessed'] is False and pair['T_accessed'] is False and coordinator['T_accessed'] is False)
    parent = authenticate(ROOT / 'pair/parent.json')
    check('supplied_unfitted_parent_metadata', parent['state']['update'] == parent['state']['stage']['start_update'] == 0
          and parent['state']['stage']['updates_completed'] == 0 and not parent['state']['model']['layer']['growth_applied'])
    directories = []
    parent_runs = []
    for directory in sorted((ROOT / 'pair/runs').iterdir()):
        if (directory / 'checkpoints/best.json').exists():
            directories.append((directory, authenticate(directory / 'checkpoints/best.json')))
        else:
            parent_runs.append(directory)
            rows = [json.loads(line) for line in read(directory / 'events.jsonl').splitlines()]
            check('unfitted_parent_native_stream_has_no_updates', all(e['type'] != 'update' for e in rows))
    check('native_run_coverage', len(directories) == 5 and len(parent_runs) == 1)
    check('generating_IDs_current_eligible', pair['generating_eligibility'] == {'0':True,'2':True} and score['eligible'][0] and score['eligible'][2])
    for role, ref in pair['models'].items():
        metadata = authenticate(Path(ref['path']), ref['sha256'])
        state = metadata['state']
        check(role + '/state_source_F_variance_IDs', state['identity_hashes']['code'] == result['source_sha256']
              and state['identity_hashes']['split'] == split['assignment_hash'] and state['fit_target_variance'] == variance)
        check(role + '/config_seed_lineage', all(state['config']['seeds'][name] == a[name+'_seed']
              for name in ('teacher','data','reference','optimizer','allocation','contrast')))
        for name, digest in state['data_hashes'].items():
            split_role, field = name.split('.')
            check(role + '/state_raw_hash/' + name, digest == array_hash(observations[f'{split_role}_{field}']))
        check(role + '/model_and_training_cell', state['config']['model']['d'] == 2 and state['config']['model']['h'] == 8
              and state['config']['growth']['m_requested'] == 2 and state['stage']['policy']['max_updates'] == 8000)
        columns = state['preprocessing']['columns']
        check(role + '/F_only_preprocessing_statistics',
              np.allclose([c['mean'] for c in columns], observations['F_X'].mean(axis=0), rtol=0, atol=1e-15)
              and np.allclose([c['scale'] for c in columns], observations['F_X'].std(axis=0,ddof=0), rtol=0, atol=1e-15)
              and abs(state['preprocessing']['target_mean']-float(observations['F_y'].mean())) <= 1e-15
              and abs(state['preprocessing']['target_scale']-float(observations['F_y'].std(ddof=0))) <= 1e-15)
        verify_stage(role, metadata, directories, pair['methods'][role])
        if role != 'keep':
            allocation = state['growth_metadata']['allocation']
            ALLOCATIONS[role] = allocation
            ids = allocation['ids']
            check(role + '/common_growth_count_seed_and_birth', allocation['requested_count'] == allocation['effective_count'] == len(ids) == 2
                  and allocation['seed'] == a['allocation_seed'] and state['growth_metadata']['insertion_update'] == 0)
            if role in ('oracle_current','oracle_true'):
                check(role + '/generating_IDs', ids == [0,2])
            if role == 'top':
                ranked = sorted(np.flatnonzero(score['eligible']).tolist(),key=lambda i:(-score['adam_scores'][i],int(score['ids'][i])))
                check('top/current_score_selected_IDs', ids == sorted(ranked[:2]))
            if role == 'oracle_true':
                columns = state['preprocessing']['columns']
                mu = np.asarray([v['mean'] for v in columns]);scale = np.asarray([v['scale'] for v in columns])
                transported = contrast.copy();transported[:,:2] *= scale;transported[:,2] += contrast[:,:2] @ mu
                exact(role + '/true_contrast_F_coordinate_transport', np.asarray(allocation['contrasts']), transported)
            else:
                exact(role + '/unchanged_current_per_ID_contrasts', np.asarray(allocation['contrasts']), score['contrasts'][ids])
            exact(role + '/declared_inserted_float32_contrasts', np.asarray(state['growth_metadata']['inserted_contrasts']),
                  np.asarray(allocation['contrasts'],dtype=np.float32).astype(np.float64))
    measurement = load(Path(result['measurement']['path']))
    check('measurement_receipt_authentication', sha(read(Path(result['measurement']['path']))) == result['measurement']['sha256'])
    return dict(F_target_variance_ddof0=variance, teacher_id=teacher_id, metrics=METRICS, sampled_noise=NOISE,
        max_reported_metric_absolute_difference=MAX_METRIC_DIFFERENCE, stages=STAGES, allocations=ALLOCATIONS,
        fixed_cell=dict(teacher_index=0,repetition=0,gamma=2.,sigma=.1,roles=list(ROLES),seeds=expected_seeds),
        actual_E_seed_policy='seed64(data_seed,positive,E); the generic plan evaluation/reference/contrast seed fields are not used to generate planted E/teacher/current contrasts',
        source_commit=COMMIT, source_sha256=result['source_sha256'], containing_elapsed_seconds=result['elapsed_ns']/1e9,
        resources=measurement, teacher_precommit_evidence='Committed source recipe writes teacher.json before calling the observation constructor; retained flag agrees. No independent filesystem timing or prior-existence proof is claimed.')


summary = {}
try:
    summary = main()
except Exception as error:
    FAILURES.append(dict(label='audit_exception',exception=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
for absolute, item in list(INPUTS.items()):
    if sha(Path(absolute).read_bytes()) != item['sha256']:
        FAILURES.append(dict(label='input_changed',path=absolute))
result = dict(schema='research/v2-M1-control-independent-audit',status='passed' if not FAILURES else 'failed',
    failures=len(FAILURES), checks=len(CHECKS), exact_array_checks=len(ARRAY_CHECKS), input_files=len(INPUTS),
    fitting_performed=False, state_restored=False, protected_T_accessed=False, repository_modules_imported=False,
    teacher_units=1,repetitions=1, Gate_A='pending',
    interpretation=['One predeclared constructed positive-control instance/repetition at gamma2/sigma.1.',
                    'Oracle-current is advantaged by generating IDs; oracle-true additionally receives generating contrasts. These are diagnostics.',
                    'Growth beating keep in this instance does not establish placement superiority, cohort oracle sensitivity or primary-population efficacy.',
                    'V/E are development observations; no independent teacher or confidence interval is created by this arithmetic audit.',
                    'Predictions are authenticated retained arrays; no model was loaded to independently regenerate them. Resource observations are contended, not calibrated performance claims.'],
    script_sha256=sha(Path(__file__).read_bytes()),numpy_version=np.__version__,audit_elapsed_ns=time.perf_counter_ns()-START,**summary)
outputs={'result.json':result,'checks.json':CHECKS,'array-checks.json':ARRAY_CHECKS,'failures.json':FAILURES,
         'input-hashes.json':sorted(INPUTS.values(),key=lambda v:v['path']),'recipe-sources.json':SOURCES}
for name,value in outputs.items():
    with (HERE/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
with (HERE/'audit.log').open('x') as stream:
    json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
print(json.dumps(result,sort_keys=True))
sys.exit(0 if not FAILURES else 1)
