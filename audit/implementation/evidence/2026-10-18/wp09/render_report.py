"""Readable WP09 report from unchanged, hash-bound saved inputs; stdlib only.

Published usage: python render_report.py OUTPUT.md
Place this script beside the unchanged report-inputs/ directory.
"""
import hashlib
import json
from pathlib import Path
import sys

INPUT = Path(__file__).resolve().parent / 'report-inputs'
def read(name):
    return json.loads((INPUT / name).read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = read('manifest.json')
for item in manifest['inputs']:
    assert sha(INPUT / item['local_path']) == item['sha256'], item['local_path']
raw = read('raw-audit.json')
clean = read('clean-readout.json')
restore = read('clean-restoration.json')
joined = read('joined-summary.json')
summary = joined['summary']
design = read('joined-design.json')['design']
mc = read('supplement-512000.json')
states = read('saved-state-comparison.json')
appendices = read('appendix-invocations.json')
assert all(r['status'] == 'completed' for r in (raw, clean, states))
assert all(r['exit_code'] == 0 for r in appendices)
def number(value):
    return 'null' if value is None else format(value, '.17g')
def interval(value):
    return '[' + ', '.join(number(v) for v in value) + ']'
lines = []
def paragraph(text):
    lines.extend([text, ''])
def table(headers, rows):
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join('---' for _ in headers) + '|')
    lines.extend('|' + '|'.join(str(v) for v in row) + '|' for row in rows)
    lines.append('')

paragraph('# Selective tree neurons: exploratory evidence and reproduction')
paragraph(f"The experiment does not establish that score-selected growth improves population risk. Across **16 independent teachers**, the mean H1 effect was **{number(raw['h1_mean'])}**, with a separately authorized 512,000-draw diagnostic interval of **{interval(mc['h1_interval'])}**. The interval includes zero. The result is **exploratory and inconclusive**; the recorded design branch remains **{design['branch']}**, without a freeze or protected-test confirmation.")
paragraph('H1 is normalized E risk(random) minus risk(top): a positive value favors score-selected growth. Static is risk(ordinary_upper) minus risk(static_partial): positive favors static partial. Risks use original-unit squared errors divided by the F target population variance (ddof=0). These saved diagnostics favor ordinary_upper over static_partial. Software reproduction succeeded; it adds no independent efficacy teacher and establishes no practical or architectural novelty.')
paragraph('## Evidence and decision')
table(['Item', 'Retained status'], [
    ['Independent units', '16 teachers; two optimizer repetitions per teacher; 262,144 shared E rows per teacher'],
    ['Completed evidence', '32 pairs, 256 role results, all 2,080 finite 320-update B6 probes; zero failed scientific pairs'],
    ['Original producer', 'Failed after the user-requested pause; original raw ledger and seal unchanged'],
    ['Continuation', 'Only teacher15/R0 and R1, the two never-started pairs; no completed pair retried'],
    ['Canonical runtime evidence', 'Only the first eight-teacher batch accepted; second canonical runtime batch remains missing'],
    ['Current design', 'calibrate; no final n_T, K_calc, planning effect, powered forecast or valid frozen protocol'],
    ['Protected testing', 'No T generation/access, freeze or independent confirmation'],
])
paragraph('Repetitions and E rows remain paired within teachers. Repetitions are averaged within each teacher before the 16-teacher mean; the independent sample size is 16. All zero, tied and unfavorable outcomes remain included. The linked 16-teacher saved analysis is explicitly noncanonical.')
table(['Diagnostic', 'Original 8,000 draws', 'Fixed 512,000 draws'], [
    ['H1 mean', number(summary['h1_mean']), number(mc['h1_mean'])],
    ['H1 95% interval', interval(summary['h1_interval']), interval(mc['h1_interval'])],
    ['Static mean', number(summary['static_mean']), number(mc['static_mean'])],
    ['Static 95% interval', interval(summary['static_interval']), interval(mc['static_interval'])],
    ['Teacher SD interval', interval(summary['sd_interval']), interval(mc['sd_interval'])],
    ['Maximum endpoint MC SE', number(summary['endpoint_mc_error']), number(mc['endpoint_mc_error'])],
    ['Computational MC gate', 'Failed', 'Passed'],
    ['Intrinsic-variance lower bound', number(summary['between_teacher_variance_lower']), number(mc['between_teacher_variance_lower'])],
    ['Paired-row variance upper bound', number(summary['paired_row_variance_upper']), number(mc['paired_row_variance_upper'])],
])
paragraph(f"The unchanged endpoint tolerance is {number(mc['endpoint_gate'])}. The supplement's SD interval ratio remains {number(mc['sd_interval_ratio'])}, above 2; the intrinsic-variance lower bound remains zero. The original 8k branch and failed computational gate are preserved. More draws reduced computational noise without increasing independent-teacher information or filling the missing runtime batch. Both calculations used seed {mc['seed']}; the supplement was prospectively authorized and ran exactly once. This report performs no resampling.")
paragraph(f"Student-t sensitivity interval: {interval(mc['student_t_interval'])}. Empirical whole-teacher bootstrap and t sensitivity do not establish nominal population coverage or unseen-tail behavior. The paired nuisance decomposition is conservative in expectation and remains unresolved. The complete original/supplement endpoints, component intervals, eight endpoint SEs, ten chunk vectors and index/RNG commitments remain in `joined-summary.json` and `supplement-512000.json`; original 8k endpoint details are also retained in the accepted MC bundle's `parity-8000.json`.")
table(['512k endpoint', 'Monte Carlo SE'], [(name, number(mc['endpoint_SEs'][name])) for name in mc['endpoint_order']])
paragraph('## All eight method risks')
paragraph('Mean normalized E risk, averaging R2 within each teacher and then all 16 teachers:')
table(['Role', 'Mean risk'], [(role, number(value)) for role, value in raw['method_means'].items()])
paragraph('The complete 32-by-eight risk table, F variances, ordered input identities, per-pair effects and all outcomes are retained in `raw-audit.json`.')
paragraph('## Complete teacher distribution')
table(['Teacher', 'H1 mean', 'Static mean', 'Within-R H1 variance', 'Paired-row H1 variance'], [
    [t['index'], number(t['h1_mean']), number(t['static_mean']), number(t['within_teacher_H1_variance']), number(t['paired_row_H1_variance'])]
    for t in raw['teachers']
])
paragraph('Paired-row variance is computed after averaging repetitions on each E row, retaining their correlation. Within-R variance uses repetition means. No row or repetition is treated as an independent teacher.')
paragraph('## Independent saved-prediction audit')
paragraph('The audit read actual E predictions and the original F_y, E_y and ordered E row IDs for every pair. Original targets, row arrays, dtypes, source membership and consumed-file hashes matched. It independently recomputed finite original-unit losses and F population variance, all eight risks, random-minus-top and ordinary_upper-minus-static_partial row differences, teacher means and both hierarchical variance summaries. Saved h1_rows was a reconciliation target, not the sole calculation input. Reads were bounded per pair; the giant cumulative raw ledger was not decoded.')
table(['Reconciliation target', 'Maximum absolute difference'], [(name, number(value)) for name, value in raw['maximum_differences'].items()])
paragraph('Saved score versus finite G-probe improvement association is descriptive within each fit. Every eligible unit is retained; constant cases are null and tied Spearman ranks are averaged. A 320-update probe does not establish final convergence or held-out generalization. Full eligibility, ties, growth and nonpositive-probe records remain in `raw-audit.json`.')
table(['Teacher', 'R', 'Eligible', 'Pearson', 'Spearman', 'Nonpositive improvements', 'Growth', 'Near ties'], [
    [p['teacher_index'], p['repetition'], p['score_association']['n'], number(p['score_association']['pearson']), number(p['score_association']['spearman']), p['score_association']['nonpositive_improvements'], p['effective_growth'], p['near_ties']]
    for p in raw['pairs']
])
paragraph('## Clean locked reproduction and durable recovery')
paragraph(f"A disposable checkout at commit `{restore['head']}` used a newly created, separate .venv and Python 3.12.13. The exact cached installation command was `{' '.join(restore['sync_command'])}` (exit {restore['sync_exit_code']}). Generated editable egg-info was preserved and hashed outside execution source. The strict source guard was not waived. Imports resolved to `{clean['environment']['import_path']}`; actual dependencies, source map, command/environment records and install metadata are retained in `clean-readout.json` and `clean-restoration.json`.")
paragraph(f"Execution315 SHA `07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c`, package SHA `d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c`, and original receipt SHA `4cfb4983fbe8d152decf7e621dc79b2e77b7f4e3825cb379534ce6a4e05e9e42` remained unchanged. The engineering full config key was `{clean['config_key']}`. All 12 regenerated F/G/V/E arrays matched before fitting. This recipe's teacher0/R0 is a same-seed engineering check, with no added efficacy unit.")
paragraph('Separate fresh processes ran Appendices A and B, so their float64 default did not contaminate training. Training used CPU float32, scoring CPU float64, CPU1/inter-op1, all five BLAS settings1 and fallback0. Numerical jobs were serial and long workers had process-scoped caffeinate assertions. Native/sandbox MPS visibility matched the original CPU reference; no RNG field was dropped.')
table(['Check', 'Scope', 'Result'], [
    ['Appendix A', 'Fresh standalone reference process', 'Passed'],
    ['Appendix B', 'Separate fresh standalone reference process', 'Passed'],
    ['Full paired replay', 'All eight roles and all 65 keep/unit probes', 'Completed'],
    ['Scores and allocations', 'Full score table, selected IDs/contrasts and all probe outcomes', 'Exact'],
    ['Eight selected role states', 'Topology, model tensors, all Adam moments/ages/groups, scheduler, full RNG/sampler/cursor/next rows; F/G/E predictions', 'Exact; maximum difference 0'],
    ['Parent and all 65 probe states', 'Complete numerical states, next rows and G predictions', 'Exact; maximum difference 0'],
    ['All 11 role stages', 'Executed row hashes, losses, selected updates and G evaluations', 'Exact'],
    ['Actual growth recovery', 'Durable interruption, reload unfinished stage, linked retry', 'Exact; maximum difference 0'],
])
table(['Role', 'Selected global update', 'Executed continuation updates', 'Selected G MSE', 'State/predictions'], [
    [role, clean['comparisons'][role]['update'], clean['comparisons'][role]['stage']['updates_completed'], number(clean['comparisons'][role]['stage']['best_mse']), 'Exact']
    for role in raw['method_means']
])
recovery = clean['controlled_interruption']
updates = states['recovery_updates']
paragraph(f"The real checkpoint event was appended before a controlled KeyboardInterrupt at global update {recovery['update']}, after growth insertion at {recovery['insertion_update']}. The interrupted attempt remains retained. `checkpoints/last.json` was reloaded with unchanged expected hashes while its stage was unfinished. A linked retry continued the original allowance without extra updates: {updates['interrupted']} interrupted-attempt updates plus {updates['retry']} retry updates exactly matched the uninterrupted {updates['uninterrupted']} update events, including full row IDs, loss and learning rate. Selected full state and F/G/E predictions matched the uninterrupted clean top continuation.")
paragraph(f"Durable last-pointer SHA: `{recovery['last_pointer_sha256']}`. The interrupted/retry paths, every state comparison, exact next-row hashes, topology and full sampler/RNG fields remain in `clean-readout.json` and `saved-state-comparison.json`. Incidental IDs, timestamps, clocks and serialized checkpoint hashes were excluded from numerical equality targets.")
paragraph('## Actual costs and retained failures')
table(['WP09 work', 'Perf seconds', 'Peak RSS bytes'], [
    ['Independent raw prediction audit', number(raw['seconds']), raw['peak_rss_bytes']],
    ['Clean replay/recovery process, enclosing', number(clean['seconds']), clean['peak_rss_bytes']],
    ['Full paired replay, nested', number(clean['timings']['full_replay_seconds']), 'Included in process peak'],
    ['Actual growth recovery, nested', number(clean['timings']['growth_recovery_seconds']), 'Included in process peak'],
    ['Saved parent/probe/stage comparison', number(states['seconds']), states['peak_rss_bytes']],
    ['Appendix A, separate process', number(appendices[0]['seconds']), 'Not separately recorded'],
    ['Appendix B, separate process', number(appendices[1]['seconds']), 'Not separately recorded'],
])
paragraph(f"The clean worker's UTC span was {clean['utc_start']} to {clean['utc_end']}. Nested times must not be added to enclosing costs. Restored reference subset: {len(restore['files'])} files / {restore['logical_restored_bytes']} logical bytes; initial clean physical du was `{restore['physical_du_kib']}` KiB. Initial disk free was {restore['disk_free_bytes']} bytes and a 10 GiB reserve was verified. The accepted technical WP09 new-only inventory contains 1,030 files / 481,577,468 bytes. This presentation revision has its own new-only inventory and does not change those original bytes.")
paragraph('Historical costs remain separate and unchanged. The original producer enclosing 47,057.135737208 perf seconds includes a documented 40,020.992005584-second controller suspension; derived outside-suspension time7,036.143731624 seconds is not CPU utilization. Original sleep-affected pair clocks remain retained. The two continuation pairs cost251.615493833 and176.384032458 wrapper seconds. Two operational setup failures preceded numerical work: tuple-versus-JSON-list metadata comparison (21.683983625 seconds) and str-versus-Path API argument (1.295538625 seconds). They caused no completed-pair retry or completed bootstrap. Original-plus-continuation indexed retention was12,970,486,988 bytes; actual producer footprint12,953,209,885 bytes. The saved MC supplement worker used .635482459 seconds and347,013,120 bytes peak RSS. The one canonical runtime batch measured406.949557520875 seconds/teacher; no valid powered full-study forecast or cost curve exists.')
paragraph('No unexpected scientific or numerical WP09 failure occurred. The controlled KeyboardInterrupt is an intentional retained recovery event. One report-renderer AST preflight found an unterminated string before rendering; its failed script/error were retained, and only the closing quote was corrected. No numerical calculation was repeated for that correction. This v2 revision changes presentation only and preserves the accepted original technical report, its two outputs and its inventory.')
paragraph('## Scope, limitations and saved details')
paragraph('Completed: independent raw risk/variance/source audit; all prescribed roles and probes with unfavorable/zero/tied cases retained; finite-probe association where defined; clean frozen environment/source/import checks; isolated A/B references; actual full engineering replay; durable growth interruption/reload/linked recovery; exact complete-state and row-stream comparisons; saved-input report regeneration and hash inventories. Absent: independent powered confirmation, valid freeze, protected T, transfer/depth/hard-routing evidence, practical superiority and architectural novelty. These absences are limitations, not hypothesis rejection. Existing closest-prior-art limits still apply.')
paragraph('The original full scientific config key9152a59bb3180da35f780b44a78d9eee59f7efa3e1af9677e4acf307a1bbb073 remains distinct from scientific recipe4579ba87ffc6be1678b935c5297e849e0a6a53adb0e055ecedea84596ab29c6a. The operational continuation, joined E diagnostics, fixed saved-MC supplement and engineering reproduction retain their separate lineages and scopes. No changed statistical gate, extra science, inference eligibility or experimental authority follows from report formatting.')
paragraph('All detailed records below are supplied unchanged in `report-inputs/`. Their exact bytes are verified before rendering:')
table(['Saved record', 'Detailed content', 'SHA256'], [
    [item['local_path'], {
        'raw-audit.json': '32 pair risks, 16 teacher summaries, input hashes, growth/ties, probe associations and audit differences',
        'clean-readout.json': 'Environment/dependencies, source315, exact role states, full recovery identities and timings',
        'clean-restoration.json': 'Restored file hashes/bytes, offline lock command, metadata relocation and disk accounting',
        'appendix-invocations.json': 'Exact fresh Appendix A/B commands, statuses and durations',
        'saved-state-comparison.json': 'Parent/all65 probe state comparisons, 11 stage streams and recovery update events',
        'joined-summary.json': 'Unchanged 8k summary, uncertainty/decomposition and canonical-status limits',
        'joined-design.json': 'Original calibrate branch, missing runtime batch, unresolved design/forecast',
        'supplement-512000.json': 'Fixed supplement endpoints, SEs/chunks, component intervals and RNG/index commitments',
    }[item['local_path']], '`' + item['sha256'] + '`'] for item in manifest['inputs']
])
paragraph('To regenerate this reader report, place `render_report.py` beside `report-inputs/` and run `python render_report.py OUTPUT.md`. The renderer uses only the Python standard library and the saved hash-bound inputs; it neither imports numerical packages nor runs experiments. Two independent renders are byte compared, with no generated timestamps. The accepted original technical report remains `.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/scientific-wp09-report.md`. **Numerical STOP.**')
Path(sys.argv[1]).write_text('\n'.join(lines))
