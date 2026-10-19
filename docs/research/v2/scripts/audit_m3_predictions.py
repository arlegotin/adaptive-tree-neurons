"""Audit frozen M3 development prediction arithmetic with a separately bound current source."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from treeneuron.research import opportunity_prediction_audit
from treeneuron.research.ledger import write_new_json
from treeneuron.research.joint_analysis import summarize_joint_predictions
from treeneuron.research.joint_costs import collect_joint_costs


def package_digest():
    root=Path(opportunity_prediction_audit.__file__).parents[1];digest=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--expected-plan-hash',required=True)
    parser.add_argument('--expected-ledger-hash',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter_ns();before=package_digest()
    if args.output.resolve().is_relative_to(args.study.resolve()):raise ValueError('Audit output must not mutate the frozen producer queue')
    args.output.mkdir(parents=True,exist_ok=False);result=None;failure=None
    try:
        result=opportunity_prediction_audit.audit_joint_predictions(args.study,
            expected_plan_hash=args.expected_plan_hash,expected_ledger_hash=args.expected_ledger_hash)
        if package_digest()!=before:raise ValueError('Auditor source changed during arithmetic')
        write_new_json(args.output/'audit.json',result)
        if result['status']=='completed_arithmetic':
            write_new_json(args.output/'development-summary.json',summarize_joint_predictions(result))
            costs=collect_joint_costs(args.study,expected_plan_hash=args.expected_plan_hash,
                expected_ledger_hash=args.expected_ledger_hash)
            if package_digest()!=before:raise ValueError('Auditor source changed during cost collection')
            write_new_json(args.output/'experimental-costs.json',costs)
    except BaseException as error:
        failure=dict(exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        write_new_json(args.output/'error.json',failure)
        raise
    finally:
        receipt=dict(schema='research/v2-M3-prediction-arithmetic-invocation',argv=sys.argv,cwd=str(Path.cwd()),
            auditor_source_sha256=before,auditor_source_after_sha256=package_digest(),
            auditor_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            helper_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(opportunity_prediction_audit.__file__))},
            producer_source_sha256=None if result is None else result['producer_source_sha256'],
            expected_queue_hash=args.expected_plan_hash,expected_ledger_hash=args.expected_ledger_hash,
            python=sys.version,executable=sys.executable,platform=platform.platform(),numpy_version=np.__version__,
            elapsed_ns=time.perf_counter_ns()-started,failure=failure,
            outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()},
            scope='Queue/file/array alignment and independent scalar reconstruction only; no teacher/model reexecution or tensor-chain semantic authentication; Gate B/cost/local-prediction review separate',
            Torch_imported='torch' in sys.modules,Gate_B_decided=False,T_accessed=False)
        write_new_json(args.output/'invocation.json',receipt)
    print(json.dumps({k:result[k] for k in ('status','planned_evaluation_jobs','audited_evaluation_jobs',
        'audited_prediction_vectors','scalar_metric_comparisons','maximum_absolute_metric_difference')},sort_keys=True))


if __name__=='__main__':main()
