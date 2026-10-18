"""Publish M1 retained-prediction arithmetic with a separate auditor identity."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from treeneuron.config import canonical_json
from treeneuron.research import control_audit,control_summary


def package_digest():
    root=Path(control_audit.__file__).parents[1];digest=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--expected-plan-hash',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter_ns();before=package_digest()
    audit=control_audit.audit_control_study(args.study,expected_plan_hash=args.expected_plan_hash)
    summary=control_summary.summarize_control_audit(audit)
    if package_digest()!=before:raise ValueError('Auditor package changed during arithmetic')
    args.output.mkdir(parents=True,exist_ok=False)
    for name,value in [('audit.json',audit),('summary.json',summary)]:
        (args.output/name).write_text(canonical_json(value)+'\n')
    paths=[Path(__file__),Path(control_audit.__file__),Path(control_summary.__file__)]
    receipt=dict(schema='research/v2-M1-arithmetic-invocation',argv=sys.argv,cwd=str(Path.cwd()),
        producer_source_sha256=audit['source_sha256'],producer_plan_hash=audit['plan_hash'],
        auditor_source_sha256=before,auditor_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        helper_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        python=sys.version,executable=sys.executable,platform=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','pandas')},
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()},
        elapsed_ns=time.perf_counter_ns()-started,
        scope='Retained prediction arithmetic and row alignment; no model/teacher reexecution or checkpoint tensor-chain authentication',
        model_predictions_reexecuted=False,checkpoint_tensor_chain_authenticated=False,
        teacher_data_regenerated=False,Gate_A_decided=False,T_accessed=False)
    (args.output/'invocation.json').write_text(canonical_json(receipt)+'\n')
    print(json.dumps(dict(status=audit['status'],audited_predictions=audit['audited_predictions'],
        metric_comparisons=audit['metric_comparisons'],maximum_absolute_metric_difference=audit['maximum_absolute_metric_difference'],
        positive_sensitivity=audit['positive_sensitivity'],status_counts=summary['status_counts']),indent=2))


if __name__=='__main__':main()
