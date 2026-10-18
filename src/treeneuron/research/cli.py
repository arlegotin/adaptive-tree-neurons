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
        if name == 'run':
            cmd.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command in ('plan', 'run'):
            from .config import load_research_config, expand
            config = load_research_config(args.config)
            plan = expand(config)
            if args.command == 'plan':
                from .ledger import write_new_json
                write_new_json(args.output, plan.to_dict())
            else:
                from .runner import run
                result=run(plan, args.output, resume=args.resume)
                if config.to_dict()['kind']=='m2_opportunity':
                    print(json.dumps(dict(status=result['status'],statuses=result['statuses'],
                        charged_job_ns=result['charged_job_ns']),sort_keys=True))
                    return 0 if result['status']=='completed' else 4 if result['status']=='completed_with_failures' else 3
        elif args.command == 'prepare-data':
            from .transfer import prepare_data
            prepare_data(args.config, args.output)
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
