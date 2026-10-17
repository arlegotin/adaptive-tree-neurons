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
    causal.add_argument('--track', choices=['causal','practical'], required=True)
    causal.add_argument('--output', type=Path)
    causal.add_argument('--backend-report', type=Path)
    causal.add_argument('--artifacts-dir', type=Path, help='Retain diagnostic runs/checkpoints, including failures')
    run = commands.add_parser('run', help='Execute a complete exploratory paired study (G/V only)')
    run.add_argument('--config', type=Path, required=True)
    report_parser = commands.add_parser('report', help='Rebuild exploratory metrics from retained artifacts')
    report_parser.add_argument('--mode', choices=['exploratory','audited','archived'], required=True)
    report_parser.add_argument('--study-dir', type=Path)
    report_parser.add_argument('--output', type=Path)
    report_parser.add_argument('--protocol', type=Path)
    report_parser.add_argument('--archive', type=Path)
    audit_parser=commands.add_parser('audit',help='Audit complete frozen evidence or a sealed saved-evidence archive')
    audit_parser.add_argument('--protocol',type=Path)
    audit_parser.add_argument('--study-dir',type=Path)
    audit_parser.add_argument('--archive',type=Path)
    audit_parser.add_argument('--output',type=Path)
    sealing=commands.add_parser('seal-archive',help='Copy validated live evidence before amendment/source drift')
    sealing.add_argument('--protocol',type=Path,required=True)
    sealing.add_argument('--study-dir',type=Path,required=True)
    sealing.add_argument('--output',type=Path,required=True)
    calibration = commands.add_parser('calibrate', help='Run bounded exploratory pilots; scientific purpose requires verified readiness')
    calibration.add_argument('--config', type=Path, required=True)
    freeze_parser = commands.add_parser('freeze', help='Freeze a verified canonical scientific design')
    freeze_parser.add_argument('--design', type=Path, required=True)
    freeze_parser.add_argument('--output', type=Path, default=Path('protocol/frozen.yaml'))
    preparation = commands.add_parser('prepare-cohort', help='Prepare new F/G/V and T row commitments before freeze')
    preparation.add_argument('--design', type=Path, required=True)
    preparation.add_argument('--output', type=Path, required=True)
    preparation.add_argument('--cohort-id', required=True)
    preparation.add_argument('--root-seed', type=int, required=True)
    preparation.add_argument('--publish', type=Path, required=True)
    confirmation = commands.add_parser('confirm', help='Execute the complete validated frozen synthetic cohort')
    confirmation.add_argument('--protocol', type=Path, required=True)
    confirmation.add_argument('--study', choices=['synthetic'], required=True)
    primary = commands.add_parser('analyze-primary', help='Rebuild teacher-level analysis from sealed rows')
    primary.add_argument('--protocol', type=Path, required=True)
    primary.add_argument('--study-dir', type=Path, required=True)
    amendment = commands.add_parser('amend', help='Preserve old identity and require fresh confirmation')
    amendment.add_argument('--protocol', type=Path, required=True)
    amendment.add_argument('--reason', required=True)
    amendment.add_argument('--replacement-study-id', required=True)
    amendment.add_argument('--archive', type=Path)
    args = parser.parse_args(argv)
    if args.command in ('audit','seal-archive') or (args.command=='report' and args.mode!='exploratory'):
        from treeneuron.analysis.audit import audit, audit_archive, seal_archive
        from treeneuron.analysis.report import render_report
        from treeneuron.protocol import FrozenProtocol, _read
        archive=getattr(args,'archive',None)
        if archive is not None:
            if args.protocol is not None: parser.error('Choose protocol or archive, not both')
            result=audit_archive(archive)
        else:
            if args.protocol is None or args.study_dir is None: parser.error('Live audit requires --protocol and --study-dir')
            try:
                value=_read(args.protocol); snapshot=value['payload']['snapshot']
                protocol=FrozenProtocol(args.protocol.resolve(),value['protocol_hash'],snapshot['pinned_commit'],snapshot['checkout_commit'])
            except (OSError,ValueError,KeyError):
                # Missing protocol produces a blocked report, never a fabricated freeze.
                protocol=FrozenProtocol(args.protocol.resolve(),'unavailable','unavailable','unavailable')
            if args.command=='seal-archive':
                print(seal_archive(args.study_dir,protocol,args.output)); return 0
            result=audit(args.study_dir,protocol)
        if args.command=='report':
            if args.mode=='archived' and archive is None: parser.error('Archived report requires --archive')
            if args.mode=='audited' and archive is not None: parser.error('Audited live report requires --protocol')
            if args.output is None: parser.error('Audited report requires --output')
            render_report(result,args.output)
        elif args.output is not None:
            write_report(result,args.output)
        print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
        return 0 if result['valid'] else 1
    if args.command == 'prepare-cohort':
        from treeneuron.protocol import load_design, publish_design
        from treeneuron.evaluation.confirm import prepare_cohort
        design, sources = load_design(args.design)
        sources = prepare_cohort(design, sources, args.output, cohort_id=args.cohort_id, root_seed=args.root_seed)
        publish_design(design, sources, args.publish)
        print(args.publish)
        return 0
    if args.command in ('confirm', 'analyze-primary'):
        from treeneuron.protocol import _read, validate_frozen
        payload = _read(args.protocol)['payload']
        protocol = validate_frozen(args.protocol, {'root': str(args.protocol.resolve().parent.parent)})
        if args.command == 'confirm':
            from treeneuron.evaluation.confirm import run_confirmation
            print(run_confirmation(protocol, args.study))
        else:
            from treeneuron.analysis.primary import analyze_primary
            print(json.dumps(analyze_primary(args.study_dir, protocol), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == 'freeze':
        from treeneuron.protocol import load_design, freeze
        design, sources = load_design(args.design)
        print(freeze(design, sources, args.output).protocol_hash)
        return 0
    if args.command == 'amend':
        from treeneuron.protocol import FrozenProtocol, _read, amend
        value = _read(args.protocol)
        snapshot = value['payload']['snapshot']
        protocol = FrozenProtocol(args.protocol, value['protocol_hash'], snapshot['pinned_commit'], snapshot['checkout_commit'])
        print(amend(protocol, reason=args.reason, replacement_study_id=args.replacement_study_id, archive=args.archive))
        return 0
    if args.command == 'calibrate':
        from dataclasses import asdict
        from treeneuron.config import load_experiment
        from treeneuron.pilot.calibrate import calibrate
        print(json.dumps(asdict(calibrate(load_experiment(args.config))), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == 'run':
        from treeneuron.config import load_experiment
        from treeneuron.experiments.paired import run_paired
        print(run_paired(load_experiment(args.config)))
        return 0
    if args.command == 'report':
        if args.study_dir is None: parser.error('Exploratory report requires --study-dir')
        from treeneuron.analysis.exploratory import summarize_exploratory
        report = summarize_exploratory(args.study_dir)
        if args.output is not None: write_report(report, args.output)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report['status'] == 'completed' else 1
    if args.command == 'verify-baselines':
        from treeneuron.baselines.causal_checks import collect_causal_report
        if args.track == 'causal':
            # This CLI owns its serial numerical runtime, matching fresh CPU
            # receipts. In-process callers retain the explicit runtime contract.
            import torch
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            report = collect_causal_report(backend_report=args.backend_report, artifacts_dir=args.artifacts_dir)
        else:
            from treeneuron.baselines.registry import collect_practical_report
            report = collect_practical_report()
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
