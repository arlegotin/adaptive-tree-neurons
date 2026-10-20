"""Rebuild the Parkinsons development appendix from retained scalar evidence."""
import csv
import io
import json
from pathlib import Path
import sys

root, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
values = json.loads((root / 'inputs.json').read_text())
scores, audit, anchors = (values[k] for k in ('scores', 'completion_audit', 'original_anchors'))
new = next(r for r in scores['rows'] if r['dataset'] == 'parkinsons')
assert len(anchors) == 20 and len(scores['rows']) == 5
assert new['independent_max_absolute_discrepancy'] == new['F_variance_discrepancy'] == 0
grown = float(next(r['V_group_day_NMSE'] for r in anchors if r['family'] == 'P00'))
cat = float(next(r['V_group_day_NMSE'] for r in anchors if r['family'] == 'P08'))
table = io.StringIO()
writer = csv.DictWriter(table, fieldnames=['family', 'anchor_id', 'original_V_NMSE', 'new_followup_V_NMSE'])
writer.writeheader()
for row in anchors:
    writer.writerow(dict(family=row['family'], anchor_id=row['anchor_id'],
        original_V_NMSE=row['V_group_day_NMSE'],
        new_followup_V_NMSE=repr(new['V_score']) if row['family'] == 'P12' else ''))
(output / 'parkinsons.csv').write_text(table.getvalue())
(output / 'parkinsons-summary.json').write_text(json.dumps(values, indent=2, sort_keys=True) + '\n')
lines = ['# Native anchor follow-up: Parkinsons NODE', '',
    f"NODE’s primary Parkinsons validation NMSE is **{new['V_score']:.9f}**, versus grown **{grown:.9f}** and CatBoost **{cat:.9f}**. Lower is better. NODE has {100*(new['V_score']/grown-1):.2f}% higher point error than grown; CatBoost remains substantially lower than both. This is one fixed setting, split and seed, with no tuned or protected-test comparison.", '',
    'The predeclared five-cell follow-up discloses every completed cell. This tranche adds Parkinsons and rechecks Airfoil and Concrete exactly. Earlier tranches and the original 115-score survey remain unchanged. There are now 118 of 120 native engineering completions; CCPP is running and Bike follows serially.', '',
    '| Fixed follow-up cell | Status | V NMSE |', '|---|---|---:|']
for row in scores['rows']:
    score = '—' if row['V_score'] is None else format(row['V_score'], '.9f')
    lines.append(f"| {row['dataset']} | {row['status'].replace('_', ' ')} | {score} |")
lines += ['', 'All twenty original Parkinsons anchors remain visible. P13 is the mean anchor, rather than the full reference search.', '',
    '| Family / anchor | Original V NMSE | New follow-up V NMSE |', '|---|---:|---:|']
for row in anchors:
    old = row['V_group_day_NMSE']
    old = 'unavailable' if old in ('', 'None') else format(float(old), '.9f')
    fresh = format(new['V_score'], '.9f') if row['family'] == 'P12' else '—'
    lines.append(f"| {row['family']} / {row['anchor_id']} | {old} | {fresh} |")
lines += ['',
    f"Full native NODE (2,048 trees, depth six, three output channels, effective batch 512) executed {audit['executed_updates']:,} updates and selected update {audit['selected_step']:,}, with {audit['native_G_evaluations']} G evaluations. Its trace, rolling checkpoint selection and first native patience endpoint were independently checked. Selected predictions replay exactly. Training, RNG transitions and weight updates were not independently reexecuted.", '',
    f"The original attempt hit a shared disk-full error. Its published update-4,000 checkpoint passed the complete original-source loader before a same-state, same-seed continuation. The successful continuation took {audit['containing_elapsed_ns']/3.6e12:.3f} hours with {audit['observed_peak_root_rss_bytes']/1024**3:.3f} GiB sampled root RSS; the original attempt has at least {audit['prior_interruption_observations']['last_resource_sample_elapsed_ns']/3.6e12:.3f} hours in retained resource samples. Its missing terminal cost tail remains unavailable. These are separate attempt costs; no complete total-cost or efficiency claim follows. Audit time ({audit['audit_elapsed_ns']/1e9:.3f} seconds) is separate. No wall cap, data subsampling or model-size reduction was used.", '',
    f"F has {audit['training_rows']:,} rows and {audit['encoded_width']} encoded features. The selected model has {audit['trainable_parameters']:,} trainable / {audit['total_parameters']:,} total parameters. V has {new['V_rows']} rows from only {new['V_groups']} subjects. Group-weighted NMSE divides by F target variance (ddof=0). An independent fsum calculation agrees exactly on seven risk metrics and F variance. Group-weighted RMSE is {new['metrics']['primary_RMSE']:.6f}; row-weighted RMSE is {new['metrics']['row_RMSE']:.6f}. Six subjects do not support broad population confidence from this one development split.", '',
    'Producer source remains `c33a84e` / package `b680a26b...a65cc`. Scoring used coordinator `5356edf` / `65553d01...b8` without fitting or model inference. Engineering eligibility remains false. T and secondary-fold development stay sealed; M3’s negative Gate B stays closed.', '',
    '[All twenty points](../audit/research-v2/native-anchor-follow-up/parkinsons.csv), [complete scalar snapshot](../audit/research-v2/native-anchor-follow-up/parkinsons-summary.json), [completion/scoring receipt](../docs/research/v2/WP07-NODE-Parkinsons-completion-and-V-receipt.json), [storage recovery](../docs/research/v2/native-storage-continuation.md). Full common-source qualification, HPO and protected evaluation remain unfinished.']
(output / 'research-v2-native-anchors-parkinsons.md').write_text('\n'.join(lines) + '\n')
