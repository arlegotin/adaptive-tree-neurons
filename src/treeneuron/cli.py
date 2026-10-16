"""Command-line entry point for the gated research tools."""
import argparse
import json
from pathlib import Path

from treeneuron.backend import collect_backend_report
from treeneuron.doctor import collect_system_report, write_report
from treeneuron.verification import collect_math_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="treeneuron")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Record local prerequisite observations")
    doctor.add_argument("--output", type=Path, required=True)
    verify_math = commands.add_parser(
        "verify-math", help="Validate CPU references, package math and actual backend kernels/batching"
    )
    verify_math.add_argument("--output", type=Path)
    verify_math.add_argument("--backend-report", type=Path, help="Revalidate an existing doctor report against current code")
    causal = commands.add_parser('verify-baselines', help='Execute bounded causal control correctness recipes')
    causal.add_argument('--track', choices=['causal'], required=True)
    causal.add_argument('--output', type=Path)
    causal.add_argument('--backend-report', type=Path)
    causal.add_argument('--artifacts-dir', type=Path, help='Retain diagnostic runs/checkpoints, including failures')
    args = parser.parse_args(argv)
    if args.command == 'verify-baselines':
        from treeneuron.baselines.causal_checks import collect_causal_report
        report = collect_causal_report(backend_report=args.backend_report, artifacts_dir=args.artifacts_dir)
        if args.output is not None: write_report(report, args.output)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report['implemented_checks_passed'] else 1
    if args.command == "verify-math":
        backend_report = None
        if args.backend_report is not None:
            try:
                artifact = json.loads(args.backend_report.read_text())
                backend_report = artifact.get("backend_probes", artifact)
            except (OSError, ValueError, AttributeError) as error:
                backend_report = {"status": "failed", "reason": str(error)}
        report = collect_math_report(backend_report=backend_report)
        if args.output is not None:
            write_report(report, args.output)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report["implemented_checks_passed"] else 1
    report = collect_system_report()
    report["backend_probes"] = collect_backend_report() if report["cpu_ready"] else {
        "status": "failed", "reason": "Indispensable CPU imports failed", "probes": {}, "decision": None}
    report["torch"]["kernel_gate"] = report["backend_probes"]["status"]
    write_report(report, args.output)
    return 0 if report["cpu_ready"] and report["backend_probes"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
