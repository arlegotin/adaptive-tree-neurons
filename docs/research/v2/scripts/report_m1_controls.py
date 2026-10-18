"""Rebuild the descriptive M1 report from authenticated retained audit outputs."""
import argparse
import hashlib
import json
from pathlib import Path

from treeneuron.config import canonical_hash
from treeneuron.research.control_summary import summarize_control_audit


def checked(directory,name):
    invocation=json.loads((directory/'invocation.json').read_text());path=directory/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=invocation['outputs'][name]:
        raise ValueError('Retained audit/execution output differs from invocation commitment')
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',required=True,type=Path);parser.add_argument('--execution-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    audit=checked(args.audit_dir,'audit.json');summary=checked(args.audit_dir,'summary.json')
    execution=checked(args.execution_dir,'execution-checks.json');inventory=checked(args.execution_dir,'producer-inventory.json')
    if (canonical_hash(summary)!=canonical_hash(summarize_control_audit(audit)) or audit['status']!='completed'
            or audit['plan_hash']!='cfe48a305c3c721eae7afe058ffe9af8f27c00dd7321be63359a3999abd6859d'):
        raise ValueError('Report requires the complete independent256-fit arithmetic')
    lines=['# Research v2: complete M1 development controls','',
        'Added gated capacity is learnable in the planted tasks. The present cheap chooser did not establish a placement advantage: random growth has lower mean exploratory E error in every positive coordinate. The predeclared information-advantaged sensitivity screen passes, so further opportunity diagnostics are justified. Gate A still requires M2; no new selector, confirmation or modern practical benchmark is reported.','',
        'This is a development report, not the final WP11 canonical study report.','',
        '## Fixed experiment and verification','',
        'All64 transactions and256 final fits completed:160positive and96negative. Positive tasks use h8/count2, a predeclared ungrown parent and generating units0/2; the same four base instances recur across four magnitude/noise coordinates, each with two optimizer repeats. Negative populations each have four separate teachers and R2. The cells are not16 independent positive problems. The positive E set has16,384 rows; negative E has65,536. No protected T was created or read.','',
        f'Producer commit604cffb967d6bb90b96dbae72d1a069b48d0ec09; package `{audit["source_sha256"]}`; plan `{audit["plan_hash"]}`. A separate auditor reconstructed all512 V/E prediction vectors and3,584 scalar comparisons from retained arrays with maximum absolute disagreement{audit["maximum_absolute_metric_difference"]:.3g}. Row/target alignment and F population-variance normalization were checked. Models/teachers were not reexecuted and checkpoint tensor chains were not authenticated.','',
        'Risks below average R2 within each instance first, then equally average four instances. Lower is better. No confidence interval or population superiority claim is made from these diagnostic counts.','',
        '## Planted positive results','']
    for split in ('V','E'):
        lines += [f'{split} normalized MSE:','',
            '| Magnitude / noise | Keep | Cheap | Random | Known IDs/current contrasts | Known IDs/true contrasts |',
            '|---|---:|---:|---:|---:|---:|']
        for cell in summary['cells'][:4]:
            risk=cell['mean_risk'][split]
            lines.append('| '+f'{cell["gamma"]:g} / {cell["sigma"]:g}'+' | '+' | '.join(f'{risk[k]:.9g}' for k in ('keep','top','random','oracle_current','oracle_true'))+' |')
        lines.append('')
    lines += ['At magnitude2/noise.1, mean E error falls about24% for random and20% for cheap growth versus ordinary continuation. This supports useful added capacity in this constructed recipe, without a matched ordinary extra-capacity or strong-model comparison. The oracle roles use privileged generating information; neither is an attained global optimum.','',
        'V sensitivity was fixed to true-oracle versus random at magnitude2/noise.1. Mean R2 instance benefits are1.54%,3.08%,9.95% and−0.94%; exactly2of4 exceed2%, meeting the screen. This is a sensitivity check, not evidence that the implementable selector succeeds. Cheap placement has mixed instance effects; instance3 has about33% higher V error than random in this coordinate.','',
        '## Negative and smooth controls','',
        '| Population | Keep V | Cheap V | Random V | Keep E | Cheap E | Random E |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for cell in summary['cells'][4:]:
        risk=cell['mean_risk'];lines.append('| '+cell['population']+' | '+' | '.join(
            f'{risk[s][r]:.9g}' for s in ('V','E') for r in ('keep','top','random'))+' |')
    lines += ['',
        'For both pure noise and shuffled labels, all8 paired runs per population select stage zero in every arm, and saved V/E prediction arrays equal keep exactly. The affine population has one cheap-growth change, with slightly worse mean V/E; random equals keep in all8. The additive smooth population benefits from extra capacity on average, with random lower than cheap on both mean V and E. That result is consistent with generic capacity/optimization benefit and does not identify conditional structure. These small controls do not establish a general false-positive rate.','',
        '## Resources, limits and next decision','',
        f'The complete producer retains{inventory["total_files"]:,} files/{inventory["total_bytes"]:,} bytes ({inventory["total_bytes"]/1024**3:.3f}GiB), fully inventoried by relative path and SHA256. One training worker used one numerical thread. Enclosing launcher wall time was8,609.93s (about2h23m30s); coordinator time8,608.26s is nested and must not be added. These are contended observations, not a dedicated hardware or efficiency benchmark. Full-cohort peak RSS was not retained as a separate measurement. No failure/retry or missing prediction occurred.','',
        'The next scientific work is the fixed M2 timing/stream, matched ordinary capacity/recipe and small pair-oracle diagnostics. M2 native resource qualification and production dispatch remain pending. A joint complementary selector is gated on opportunity across multiple primary teachers. Modern MLP/ResNet/tree/RealMLP/TabM/NODE comparison and prospective confirmation remain unrun. No broad practical or novelty claim follows from M1.','',
        '[Machine-readable receipt and artifact paths](../docs/research/v2/WP02-M1-full-receipt.json). The raw archive is retained locally; it has not been published. Producer files contain original absolute locators, while the inventory supplies portable relative names. Relocation requires explicit locator mapping; raw bytes are never rewritten silently.','']
    with args.output.open('x') as stream:stream.write('\n'.join(lines))
    print(json.dumps(dict(report=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),T_accessed=False)))


if __name__=='__main__':main()
