"""Read-only execution verification and inventory; no prediction analysis or model load."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()
ENV = ROOT / 'environment/long-probes-20261018'
OUT = ROOT / 'artifacts/long-probes-20261018/cohort'
REPORT = ROOT / '.superpowers/sdd/2026-10-18-long-probe-allocation-diagnostic/task-2-cohort-report.md'
SOURCE = '089b03d13e1950b8e0c1ff54a147a121cf78db66acaaa2dcedd9ebf29afc3543'
ROLES = {'keep', 'top', 'random', 'greedy', 'greedy_long', 'widen_upper',
    'ordinary_lower', 'ordinary_upper', 'static_partial'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def sealed(path):
    assert read(path.with_name(path.stem + '-seal.json')) == {'sha256': sha(path)}, str(path)
    return read(path)


def within(path):
    path = path.resolve()
    assert path.is_relative_to(OUT.resolve()), str(path)
    return path


def publish_inventory(failures):
    paths = sorted({p for p in OUT.rglob('*') if p.is_file()} |
        {p for p in ENV.glob('cohort-*') if p.is_file()} | {REPORT})
    records = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]
    inventory = dict(schema_version=1, purpose='All new fixed-cohort execution evidence, before independent audit',
        artifacts=records, artifact_count=len(records), total_logical_bytes=sum(r['bytes'] for r in records),
        excludes=['environment/long-probes-20261018/cohort-inventory.json',
            'environment/long-probes-20261018/cohort-inventory-seal.json'],
        failures=failures, automatic_retries=0, scientific_analysis_run=False)
    path = ENV / 'cohort-inventory.json'
    write(path, inventory)
    write(ENV / 'cohort-inventory-seal.json', dict(sha256=sha(path)))
    return path, records


outer = read(ENV / 'cohort-outer-measurement.json')
stop = read(ENV / 'cohort-stop.json')
preflight = read(ENV / 'cohort-preflight.json')
errors = []
summary = dict(status='not_verified', phase='cohort execution only; STOP before independent audit',
    source_sha256=SOURCE, costs=outer, members=[], searches={}, failures=[],
    automatic_retries=0, effect_inspection=False, prediction_analysis=False,
    model_load=False, replay=False, bootstrap=False, T_accessed=False, V_accessed=False)
try:
    assert outer['returncode'] == stop['returncode'] == 0
    assert stop['tracked_unchanged'] and stop['prior_evidence_unchanged']
    for rel, digest in preflight['tracked_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in preflight['prior_evidence_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    assert not outer['monitor_errors']
    declaration = sealed(ROOT / 'protocol/diagnostics/2026-10-18-long-probes-lock.json')
    decision = sealed(ROOT / 'pilot/long-probes-20261018/resource-decision.json')
    ledger = sealed(OUT / 'members.json')
    lock = sealed(OUT / 'study-lock.json')
    assert ledger['status'] == 'completed' and ledger['phase'] == 'study' and ledger['failures'] == []
    expected = [dict(teacher_index=i, repetition=j) for i in range(8) for j in range(2)]
    assert ledger['expected_members'] == declaration['expected_members'] == expected
    assert [{k: member[k] for k in ('teacher_index', 'repetition')} for member in ledger['members']] == expected
    assert ledger['spec'] == lock['spec'] == declaration['spec']
    assert lock['provenance'] == ledger['provenance']
    for key, value in declaration['execution_provenance'].items():
        if key != 'commit':
            assert ledger['provenance'][key] == value, key
    assert ledger['provenance']['commit'] == preflight['commit']
    assert ledger['provenance']['source'] == SOURCE
    assert lock['resource_decision'] == ledger['resource_decision'] == decision
    assert lock['resource_decision_sha256'] == declaration['resource_decision_sha256']
    assert lock['wrapper_arguments'] == dict(command='run', output=str(OUT),
        resource_decision=str(ROOT / declaration['wrapper_arguments']['resource_decision']),
        reference_probe_updates=8000)
    assert lock['policies'] == ledger['policies'] == declaration['policies']
    totals = {name: dict(trial_count=0, eligible_count=0, executed_updates=0,
        ceiling_hits=0, stopping_reasons=Counter(), outcomes=Counter(), nanoseconds=0)
        for name in ('short', 'long')}
    artifact_count = 0
    role_counts = Counter()
    parent_ids = set()
    teacher_ids = {}
    for entry in ledger['members']:
        assert entry['status'] == 'completed'
        pairroot = within(OUT / entry['pair_path'])
        pair = read(pairroot / 'pair.json')
        assert pair['status'] == 'completed' and pair['failures'] == []
        assert set(pair['methods']) == ROLES
        assert pair['source_sha256'] == SOURCE and not pair['T_accessed'] and not pair['V_accessed']
        for name, key in [('pair.json', 'pair_sha256'), ('E-predictions.npz', 'predictions_sha256'),
                ('artifacts.json', 'artifacts_sha256')]:
            assert sha(pairroot / name) == entry[key]
        observations = within(OUT / entry['observations_path'])
        for key, path in [('observations_sha256', observations),
                ('teacher_sha256', observations.parent / 'teacher.json'),
                ('splits_sha256', observations.parent / 'splits.json')]:
            assert sha(path) == entry[key]
        assert entry['teacher_id'] == teacher_ids.setdefault(entry['teacher_index'], entry['teacher_id'])
        parent_ids.add(pair['parent_run_id'])
        artifacts = read(pairroot / 'artifacts.json')
        actual = {str(p.relative_to(pairroot)) for p in pairroot.rglob('*')
            if p.is_file() and p.name != 'artifacts.json'}
        assert set(artifacts) == actual
        for rel, digest in artifacts.items():
            assert sha(within(pairroot / rel)) == digest, rel
        artifact_count += len(artifacts)
        warm = [m for m in pair['methods'].values() if m.get('parent_run_id')]
        assert len(warm) == pair['pairing_audit']['warm_forks'] == 6
        assert pair['pairing_audit']['passed'] and pair['long_probe_audit']['passed']
        keep = pair['methods']['keep']['executed_row_ids']
        for method in warm:
            assert method['parent_run_id'] == pair['parent_run_id']
            assert method['parent_checkpoint_sha256'] == pair['parent_checkpoint_sha256']
            assert method['initial_pairing'] == pair['parent_pairing']
            assert method['executed_row_ids'] == method['stages'][-1]['row_hashes']
            assert len(method['executed_row_ids']) == method['stages'][-1]['executed_updates']
            n = min(len(keep), len(method['executed_row_ids']))
            assert keep[:n] == method['executed_row_ids'][:n]
        assert pair['pairing_audit']['common_prefix_updates'] == min(len(m['executed_row_ids']) for m in warm)
        score = read(pairroot / 'score-table.json')
        contrasts = dict(zip(score['ids'], score['contrasts']))
        reference = pair['methods']['top']['allocation']
        for name in ('top', 'random', 'greedy', 'greedy_long'):
            allocation = pair['methods'][name]['allocation']
            for key in ('requested_count', 'effective_count', 'table_hash', 'seed'):
                assert allocation[key] == reference[key]
            assert allocation['table_hash'] == score['table_hash']
            assert all(q == contrasts[unit] for unit, q in zip(allocation['ids'], allocation['contrasts_float64']))
        searches = {}
        member_searches = {}
        for name, key, mode, allowance in [('short', 'greedy_report', 'fixed_updates', 320),
                ('long', 'greedy_long_report', 'patience', 8000)]:
            path = within(Path(pair[key]))
            search = read(path)
            assert search['complete'] and search['not_attempted'] == [] and search['stopping_mode'] == mode
            eligible = search['eligible_ids']
            trials = search['trials']
            assert Counter(t['unit_id'] for t in trials) == Counter([None, *eligible])
            assert search['effective_count'] == min(16, len(eligible)) == reference['effective_count']
            assert search['score_hash'] == score['table_hash']
            method = pair['methods']['greedy' if name == 'short' else 'greedy_long']
            assert method['allocation']['ids'] == search['allocation']['ids']
            for trial in trials:
                assert trial['outcome'] == 'completed' and trial['allowance'] == allowance
                assert trial['stopping_mode'] == mode
                assert trial['parent_identity_hashes'] == search['parent_identity_hashes']
                executed = trial['executed_updates']
                assert 0 <= trial['selected_stage_updates'] <= executed <= allowance
                assert trial['stopping_reason'] in ('patience', 'ceiling')
                assert trial['hit_ceiling'] == (executed == allowance)
                assert (trial['stopping_reason'] == 'ceiling') == trial['hit_ceiling']
                if name == 'short':
                    assert executed == 320 and trial['hit_ceiling']
                manifest = read(within(path.parent / trial['path'] / 'manifest.json'))
                assert manifest['identity']['status'] == 'completed'
                assert manifest['identity']['code_revision'] == SOURCE
            reasons = Counter(t['stopping_reason'] for t in trials)
            outcomes = Counter(t['outcome'] for t in trials)
            member_searches[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                eligible_count=len(eligible), trial_count=len(trials), effective_count=search['effective_count'],
                allowance=allowance, mode=mode, outcomes=dict(outcomes), stopping_reasons=dict(reasons),
                ceiling_hits=sum(t['hit_ceiling'] for t in trials),
                executed_updates_total=sum(t['executed_updates'] for t in trials),
                executed_updates_min=min(t['executed_updates'] for t in trials),
                executed_updates_max=max(t['executed_updates'] for t in trials), nanoseconds=search['nanoseconds'])
            total = totals[name]
            total['trial_count'] += len(trials)
            total['eligible_count'] += len(eligible)
            total['executed_updates'] += member_searches[name]['executed_updates_total']
            total['ceiling_hits'] += member_searches[name]['ceiling_hits']
            total['stopping_reasons'].update(reasons)
            total['outcomes'].update(outcomes)
            total['nanoseconds'] += search['nanoseconds']
            searches[name] = search
        for key in ('parent_identity_hashes', 'parent_state_sha256', 'score_hash', 'eligible_ids'):
            assert searches['short'][key] == searches['long'][key]
        original = pair['methods']['greedy_long']['original_parent_audit']
        assert original['passed']
        assert searches['long']['parent_state_sha256'] == original['complete_state_sha256']
        assert pair['long_probe_audit']['parent_state_sha256'] == original['complete_state_sha256']
        assert pair['long_probe_audit']['effective_count'] == reference['effective_count']
        assert pair['long_probe_audit']['allocation_seed'] == reference['seed']
        roles = {}
        for name, method in pair['methods'].items():
            assert method['status'] == 'completed'
            role_counts[name] += 1
            for stage in method['stages']:
                assert stage['executed_updates'] <= stage['max_updates'] == 8000
                assert stage['selected_update'] >= 0
            roles[name] = dict(status=method['status'], total_parameters=method['total_parameters'],
                enclosing_ns=method['enclosing_ns'], stage_count=len(method['stages']),
                executed_updates=sum(s['executed_updates'] for s in method['stages']))
        capacity = pair['capacity']
        assert capacity['effective_count'] == reference['effective_count']
        assert capacity['target_parameters'] == 18 * 64 + 1 + 34 * reference['effective_count']
        assert capacity['ordinary_parameters'] == [18 * h + 1 for h in capacity['ordinary_widths']]
        assert capacity['ordinary_parameters'][0] <= capacity['target_parameters'] <= capacity['ordinary_parameters'][1]
        assert pair['methods']['keep']['total_parameters'] == 18 * 64 + 1
        for name in ('top', 'random', 'greedy', 'greedy_long', 'static_partial'):
            assert pair['methods'][name]['total_parameters'] == capacity['target_parameters']
        for side, count in zip(('lower', 'upper'), capacity['ordinary_parameters']):
            assert pair['methods']['ordinary_' + side]['total_parameters'] == count
            assert len(pair['methods']['ordinary_' + side]['stages']) == 2
        assert pair['methods']['widen_upper']['total_parameters'] == capacity['ordinary_parameters'][1]
        summary['members'].append(dict(teacher_index=entry['teacher_index'], repetition=entry['repetition'],
            pair_path=entry['pair_path'], pair_sha256=entry['pair_sha256'],
            parent_run_id=pair['parent_run_id'], parent_checkpoint_sha256=pair['parent_checkpoint_sha256'],
            enclosing_ns=entry['enclosing_ns'], pair_enclosing_ns=pair['enclosing_ns'],
            costs=pair['costs'], searches=member_searches, roles=roles, capacity=capacity,
            pairing_audit=pair['pairing_audit'], long_probe_audit=pair['long_probe_audit'],
            original_parent_audit=original, pair_artifact_hashes_verified=len(artifacts)))
    assert len(parent_ids) == 16 and len(teacher_ids) == len(set(teacher_ids.values())) == 8
    assert role_counts == Counter({name: 16 for name in ROLES})
    files = sorted(p for p in OUT.rglob('*') if p.is_file())
    assert len(files) == outer['final_post_publication_file_count']
    assert sum(p.stat().st_size for p in files) == outer['final_post_publication_retained_bytes']
    assert sum(p.stat().st_blocks * 512 for p in files) == outer['final_post_publication_allocated_bytes']
    observations = ledger['resource_observations']
    assert all(o['counter_reason'] is None for o in observations)
    assert max(o['peak_rss_bytes'] for o in observations) <= 6 * 1024**3
    assert all(o['free_disk_bytes'] >= 10 * 1024**3 for o in observations)
    summary.update(status='passed', provenance=ledger['provenance'],
        expected_members=expected, completed_members=16, independent_teachers=8,
        role_counts=dict(role_counts), searches=totals, pair_artifact_hashes_verified=artifact_count,
        ledger_sha256=sha(OUT / 'members.json'), study_lock_sha256=sha(OUT / 'study-lock.json'),
        declaration_sha256=sha(ROOT / 'protocol/diagnostics/2026-10-18-long-probes-lock.json'),
        resource_decision_sha256=sha(ROOT / 'pilot/long-probes-20261018/resource-decision.json'),
        internal_enclosing_ns=ledger['enclosing_ns'], internal_peak_rss_bytes=ledger['peak_rss_bytes'],
        internal_retained_bytes=ledger['retained_bytes'],
        resources=dict(observation_count=len(observations),
            available_percent_min=min(o['available_percent'] for o in observations),
            swapouts_min=min(o['swapouts'] for o in observations),
            swapouts_max=max(o['swapouts'] for o in observations),
            free_disk_bytes_min=min(o['free_disk_bytes'] for o in observations),
            peak_rss_bytes=max(o['peak_rss_bytes'] for o in observations), counter_failures=[]),
        failures=[], seals_verified=4)
except Exception as error:
    errors.append(f'{type(error).__name__}: {error}')
    summary.update(status='failed', failures=errors)
finally:
    write(ENV / 'cohort-verification.json', summary)

if errors:
    ledger = read(OUT / 'members.json') if (OUT / 'members.json').exists() else {}
    completed = sum(m.get('status') == 'completed' for m in ledger.get('members', []))
    text = '\n'.join([
        '# Task 2 Phase B: retained cohort failure — numerical STOP', '',
        f"The single authorized invocation returned {outer['returncode']}; {completed}/16 members completed. Execution verification did not pass: `{errors}`. The raw cohort ledger retains status `{ledger.get('status')}` and failures `{ledger.get('failures')}`.", '',
        f"Outer launch-to-exit cost: {outer['outer_elapsed_ns'] / 1e9:.9f}s; retained cohort bytes: {outer['final_post_publication_retained_bytes']}; files: {outer['final_post_publication_file_count']}. All complete, failed and incomplete evidence remains in place. Zero automatic retries; no changed recipe, added teacher, prediction-effect inspection, scientific analysis, replay, bootstrap, calibration, freeze or T access.", '',
        'Invocation, native counters, outer clocks, stop record and failed verification are in `environment/long-probes-20261018/cohort-*`. All new cohort/environment/report evidence is hash-inventoried by `cohort-inventory.json` and its seal. The inventory excludes itself and its seal. No complete-result claim is made. Root must independently audit the retained evidence before any next numerical step.', '',
    ])
    with REPORT.open('x') as stream:
        stream.write(text)
    inventory_path, records = publish_inventory(errors)
    print(json.dumps(dict(status='failed', errors=errors, completed_members=completed,
        report=str(REPORT.relative_to(ROOT)), inventory=str(inventory_path.relative_to(ROOT)),
        inventory_sha256=sha(inventory_path), inventoried_files=len(records)), sort_keys=True, indent=2))
    sys.exit(1)

seconds = outer['outer_elapsed_ns'] / 1e9
short, long = summary['searches']['short'], summary['searches']['long']
lines = [
    '# Task 2 Phase B: fixed cohort execution — numerical STOP',
    '',
    'The one authorized cohort invocation completed all eight fresh teachers with two paired repetitions each, in teacher0..7/R0..1 order. All sixteen members, both exhaustive searches and all nine final roles completed with zero failures and zero automatic retries. This is execution evidence only; independent audit and all scientific analysis remain pending.',
    '',
    '## Immutable execution and prospective declarations',
    '',
    f"- Source: `{SOURCE}`; implementation commit `b2a993721b85aa559fd5b3d89fa80a33a93892a0`; execution HEAD `{summary['provenance']['commit']}`.",
    f"- Reused CPU128 receipt: `{summary['provenance']['receipt']}`. No receipt/backend probe was rerun.",
    f"- Prospective declaration SHA256: `{summary['declaration_sha256']}`; committed resource decision SHA256: `{summary['resource_decision_sha256']}`.",
    f"- Runtime source/config/lock/runtime/receipt match the accepted measurement identities. All tracked bytes and all {stop['prior_evidence_files']} prior evidence files remained unchanged.",
    '- Fresh `.venv` interpreter sets CPU1/inter-op1 before module entry; all five BLAS thread settings1 and MPS fallback0 precede imports. Process-scoped caffeinate; serial numerical execution; no elapsed-time kill, pause, retry, added member or resource-policy change.',
    '',
    '## Actual measured costs',
    '',
    f"- Outer launch-to-exit: {seconds:.9f}s ({seconds / 3600:.6f}h), measured by Python `perf_counter_ns`; complete publication included. Internal enclosing clock: {summary['internal_enclosing_ns'] / 1e9:.9f}s. Nested clocks are retained separately and are never added to enclosing elapsed time.",
    f"- Final retained cohort: {outer['final_post_publication_retained_bytes']:,} logical bytes; {outer['final_post_publication_allocated_bytes']:,} allocated bytes; {outer['final_post_publication_file_count']:,} regular files. Environment/report/inventory evidence is additional and separately inventoried.",
    f"- Outer RUSAGE_CHILDREN peak: {outer['child_peak_rss_bytes']:,} bytes; internal numerical-process peak: {summary['internal_peak_rss_bytes']:,} bytes. Outer child user/system: {outer['child_user_seconds']:.6f}/{outer['child_system_seconds']:.6f}s. Outer counters include lightweight native monitor children; the internal maximum independently identifies numerical RSS.",
    f"- Resource observations: {summary['resources']['observation_count']}; minimum coarse memory availability {summary['resources']['available_percent_min']}%; swapouts {summary['resources']['swapouts_min']}..{summary['resources']['swapouts_max']}; minimum observed free disk {summary['resources']['free_disk_bytes_min']:,} bytes. No native-counter failures, RSS-cap violation, severe-pressure stop or disk-reserve stop occurred.",
    '',
    '## Completion, stopping and pairing',
    '',
    f"- Short: {short['trial_count']} completed trials over {short['eligible_count']} eligible units plus sixteen keeps; {short['executed_updates']:,} executed updates; {short['ceiling_hits']} ceiling hits; stopping reasons `{dict(short['stopping_reasons'])}`.",
    f"- Long: {long['trial_count']} completed trials over {long['eligible_count']} eligible units plus sixteen keeps; {long['executed_updates']:,} executed updates; {long['ceiling_hits']} ceiling hits; stopping reasons `{dict(long['stopping_reasons'])}`.",
    '- Every short trial executed exactly320 updates. Every long trial completed under its8000 allowance with explicit patience/ceiling stopping; selected/executed counts and hit-ceiling flags agree. Every search retained complete eligible-plus-keep coverage and no unattempted/failed trial.',
    '- All nine roles have16 completed instances (144 total). Six warm forks per pair retain original parent checkpoint, optimizer/sampler/RNG/schedule and shared executed-row prefix. Both coordinators share original parent complete-state/identity hashes, score table, eligible IDs, allocation seed/counts and per-ID contrasts. Long joint continuation binds the original parent complete state. Capacity counts and ordinary brackets were recalculated from actual eligibility.',
    f"- Read-only execution verification rehashed {summary['pair_artifact_hashes_verified']:,} pair-artifact bindings and every observation/teacher/split/ledger/lock link. The final inventory hashes every new cohort/environment/report artifact once. No bulk pair/row arrays are copied into a cumulative ledger or verification JSON.",
    '',
    '## Evidence and handoff',
    '',
    '- `environment/long-probes-20261018/cohort-invocation.json`, `cohort-preflight.json`, `cohort-progress.jsonl`, `cohort-outer-measurement.json`, `cohort-stop.json`, `cohort-verification.json` retain invocation, native counters, clocks, costs, immutable-byte checks and compact member audits.',
    '- `artifacts/long-probes-20261018/cohort/members.json` and seal bind all16 complete pair files; the study lock and seal bind resolved config, wrapper arguments and resource decision.',
    '- `environment/long-probes-20261018/cohort-inventory.json` and seal enumerate all new cohort artifacts, cohort environment evidence and this report. The inventory excludes itself and its seal to avoid a self-hash cycle.',
    '- No E risk/effect inspection, saved-prediction analysis, ranking replay, checkpoint restore, bootstrap, scientific report generation, calibration, freeze, T access/generation, dependency installation, old/full tests or unrelated experiment was performed. V remained unused. No commits were made. Numerical work stops here for independent audit.',
    '',
]
with REPORT.open('x') as stream:
    stream.write('\n'.join(lines))
inventory_path, records = publish_inventory([])
print(json.dumps(dict(status=summary['status'], completed_members=16,
    role_counts=summary['role_counts'], searches=summary['searches'],
    outer_seconds=seconds, retained_bytes=outer['final_post_publication_retained_bytes'],
    peak_rss_bytes=summary['internal_peak_rss_bytes'], resources=summary['resources'],
    report=str(REPORT.relative_to(ROOT)), inventory=str(inventory_path.relative_to(ROOT)),
    inventory_sha256=sha(inventory_path), inventoried_files=len(records)), sort_keys=True, indent=2))
