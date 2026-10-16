"""Command-line entry point for the gated research tools."""
import argparse
import json
from pathlib import Path

from treeneuron.doctor import collect_system_report, write_report
from treeneuron.verification import collect_math_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="treeneuron")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Record local prerequisite observations")
    doctor.add_argument("--output", type=Path, required=True)
    verify_math = commands.add_parser(
        "verify-math", help="Run CPU reference checks; package/backend gates remain pending"
    )
    verify_math.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify-math":
        report = collect_math_report()
        if args.output is not None:
            write_report(report, args.output)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report["reference_checks_passed"] else 1
    report = collect_system_report()
    write_report(report, args.output)
    return 0 if report["cpu_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
