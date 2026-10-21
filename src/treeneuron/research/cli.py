"""Research/v2 command boundary; future commands fail with owning prerequisites."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'run', 'prepare-data', 'qualify', 'freeze', 'confirm', 'report'):
        cmd = subs.add_parser(name)
        key = 'protocol' if name == 'confirm' else 'study' if name == 'report' else 'config'
        cmd.add_argument('--'+key, type=Path, required=True)
        cmd.add_argument('--output', type=Path, required=True)
        if name in ('run','qualify'):
            cmd.add_argument('--resume', action='store_true')
        if name == 'report':
            cmd.add_argument('--development-manifest-sha256')
            cmd.add_argument('--repository-root',type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ('plan','run','freeze'):
            from .deployment_worker_v2 import _json
            metadata=_json(args.config.read_bytes())
            if isinstance(metadata,dict) and metadata.get('schema')=='research/v2-practical-command':
                from .practical_cli_v2 import plan_practical_command,execute_practical_command,practical_command_exit_code
                if args.command=='plan':
                    result=plan_practical_command(args.config,output=args.output)
                else:
                    result=execute_practical_command(args.config,output=args.output,
                        command=args.command,resume=getattr(args,'resume',False))
                reference=getattr(result,'reference',None)
                if reference is None and hasattr(result,'file_sha256'):
                    reference=dict(path=str(result.path),sha256=result.file_sha256)
                print(json.dumps(reference if reference is not None else result,sort_keys=True))
                return practical_command_exit_code(result)
        if args.command in ('plan', 'run'):
            if args.command=='plan':
                from .method_worker import _read_json
                metadata=_read_json(args.config)
                if isinstance(metadata,dict) and metadata.get('schema')=='research/v2-native-qualification':
                    from .qualify import qualification_plan
                    from .ledger import write_new_json
                    write_new_json(args.output,qualification_plan(args.config))
                    return 0
            from .config import load_research_config, expand
            config = load_research_config(args.config)
            plan = expand(config)
            if args.command == 'plan':
                from .ledger import write_new_json
                write_new_json(args.output, plan.to_dict())
            else:
                from .runner import run
                result=run(plan, args.output, resume=args.resume)
                if config.to_dict()['kind'] in ('m2_opportunity','m3_joint'):
                    print(json.dumps(dict(status=result['status'],statuses=result['statuses'],
                        charged_job_ns=result['charged_job_ns']),sort_keys=True))
                    return 0 if result['status']=='completed' else 4 if result['status']=='completed_with_failures' else 3
        elif args.command == 'qualify':
            from .method_worker import _read_json
            metadata=_read_json(args.config)
            if isinstance(metadata,dict) and metadata.get('schema')=='research/v2' and metadata.get('kind')=='m3_joint':
                from .config import load_research_config,expand
                from .joint_execution import run_joint_qualification
                result=run_joint_qualification(expand(load_research_config(args.config)).to_dict(),args.output,resume=args.resume)
            else:
                from .qualify import qualify
                result=qualify(args.config,args.output,resume=args.resume)
            print(json.dumps(dict(status=result['status'],statuses=result['statuses'],quality_eligible=False),sort_keys=True))
            return 0 if result['status']=='completed' else 3 if result['status']=='paused_before_job' else 4
        elif args.command == 'prepare-data':
            from .transfer import prepare_data
            prepare_data(args.config, args.output)
        elif args.command == 'report' and args.development_manifest_sha256 is not None:
            from treeneuron.analysis.audit_v2 import audit_development_study
            from treeneuron.analysis.report_v2 import write_report
            from treeneuron.protocol import _execution
            root=(args.repository_root or Path.cwd()).resolve();output=args.output.resolve()
            if output.is_relative_to(root) and _execution(output.relative_to(root).as_posix()):
                raise ValueError('Report output must not add/change executable scientific inputs')
            audit=audit_development_study(args.study/'development-manifest.json',
                expected_manifest_sha256=args.development_manifest_sha256,repository_root=root)
            print(json.dumps(write_report(audit,args.output),sort_keys=True))
        elif args.command=='confirm':
            from .practical_cli_v2 import confirm_practical_protocol
            result=confirm_practical_protocol(args.protocol,output=args.output)
            print(json.dumps(result,sort_keys=True))
            return 0 if result['status']=='completed' else 3 if result['status']=='paused' else 4
        elif args.command=='report':
            from .practical_cli_v2 import report_practical_study
            print(json.dumps(report_practical_study(args.study,output=args.output),sort_keys=True))
        else:
            owners = {'prepare-data': 'WP07', 'qualify': 'WP07', 'freeze': 'WP09',
                      'confirm': 'WP10', 'report': 'WP11'}
            raise NotImplementedError(f'{args.command} requires {owners[args.command]} implementation and qualification')
    except (ValueError, OSError, NotImplementedError) as error:
        print(f'research/v2: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
