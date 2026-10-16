"""Command-line entry point for the gated research tools."""
import argparse
from pathlib import Path

from treeneuron.doctor import collect_system_report, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="treeneuron")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Record local prerequisite observations")
    doctor.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = collect_system_report()
    write_report(report, args.output)
    return 0 if report["cpu_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
