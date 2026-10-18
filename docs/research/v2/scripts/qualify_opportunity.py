#!/usr/bin/env python3
"""Run the fixed M2 engineering spine, retaining enclosing process receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
THREADS=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
    PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONPATH=str(ROOT/'src'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())


def child(args):
    import resource
    import shutil
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from treeneuron import train
    from treeneuron.config import canonical_hash
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.identity import execution_identity
    from treeneuron.research.opportunity_graph import compile_bounded_graph
    from treeneuron.research.opportunity_qualification import (
        build_qualification_plan,observed_qualification_workers,collect_qualification_profiles)
    from treeneuron.research.opportunity_execution import live_resources,make_opportunity_admission
    from treeneuron.research.workqueue import run_queue
    target=compile_bounded_graph(expand(load_research_config(args.config)).to_dict())
    plan=build_qualification_plan(target);output=args.output.resolve()

    def identity(value):
        if (value!=plan['identity'] or train.source_hash()!=value['source_sha256']
                or execution_identity(value['study_config'])!=value['execution_identity']):
            raise ValueError('Engineering qualification source/runtime changed')

    def before_start(job,events):
        live=live_resources(output);free=live['free_disk_bytes'];memory=live['available_memory_bytes']
        # These are explicit operator capacity limits for engineering, not an
        # empirical scientific forecast. Full scientific admission waits for
        # every measured class and the conservative complete-graph projection.
        needed_disk=4*1024**3;needed_memory=2*1024**3
        admitted=free>=needed_disk and memory is not None and memory>=needed_memory
        return dict(admitted=admitted,reason='ready_engineering_capacity' if admitted else 'engineering_capacity_unavailable',
            required_disk_bytes=needed_disk,required_memory_bytes=needed_memory,live_observation=live,
            forecast=False,scientific_admission=False)

    def guard():
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if peak>2*1024**3:raise MemoryError('Engineering qualification exceeded explicit2GiB own RSS cap')
        if shutil.disk_usage(output).free<768*1024**2:
            raise OSError('Engineering qualification512MiB free floor +256MiB transaction allowance exhausted')

    worker=observed_qualification_workers(target,plan,guard=guard)
    result=run_queue(plan,output,worker,check_identity=identity,resume=args.resume,before_start=before_start)
    index=len(list(output.glob('qualification-collection-*.json')))
    collection=collect_qualification_profiles(output,target,expected_plan_hash=plan['plan_hash'])
    write(output/f'qualification-collection-{index}.json',collection)
    result.update(target_plan_hash=target['plan_hash'],qualification_plan_hash=plan['plan_hash'],
        collection_hash=canonical_hash(collection),resource_profile_status=collection['status'],
        scientific_admission=False,T_accessed=False)
    write(output/f'qualification-disposition-{index}.json',result)
    if collection['status']=='profiles_complete' and args.admission_output:
        record=make_opportunity_admission(target,output,available_worker_seconds=args.available_worker_seconds)
        write(args.admission_output,record)
    print(json.dumps(dict(status=result['status'],resource_profile_status=collection['status'],
        covered_classes=len({r['workload_hash'] for r in collection['profiles']}),
        missing_classes=len(collection['missing_workload_hashes']),T_accessed=False)))
    return 0 if collection['status']=='profiles_complete' else 3


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--admission-output',type=Path)
    parser.add_argument('--available-worker-seconds',type=float,default=604800,
        help='Prospective bounded dedicated-worker allowance; default7days, not predicted duration')
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.child:return child(args)
    output=args.output.resolve();launch=output.with_name(output.name+'-launch')
    if args.resume:
        if not launch.is_dir():raise ValueError('Resume requires original launcher evidence')
    else:launch.mkdir(parents=True,exist_ok=False)
    number=len(list(launch.glob('invocation-*')));invocation=launch/f'invocation-{number}';invocation.mkdir()
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--config',str(args.config.resolve()),
        '--output',str(output),'--available-worker-seconds',str(args.available_worker_seconds)]
    if args.resume:argv.append('--resume')
    if args.admission_output:argv.extend(['--admission-output',str(args.admission_output.resolve())])
    env=dict(os.environ,**THREADS);start=time.perf_counter_ns();started_utc=time.time_ns()
    write(invocation/'started.json',dict(argv=argv,cwd=str(ROOT),environment_overrides=THREADS,
        executable=sys.executable,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),started_utc_ns=started_utc,
        capacity_scope='Engineering4GiB live disk/2GiB live memory;2GiB own RSS cap; not a scientific forecast',
        T_accessed=False))
    process=None;code=None
    try:
        with (invocation/'child.log').open('xb') as log:
            process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            code=process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            process.send_signal(__import__('signal').SIGINT)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate();process.wait(timeout=10)
        raise
    finally:
        write(invocation/'terminal.json',dict(exit_code=code,pid=None if process is None else process.pid,
            complete_process_wall_ns=time.perf_counter_ns()-start,ended_utc_ns=time.time_ns(),
            clock_scope='Enclosing parent start before child spawn through child publication/termination; nested queue clocks not added',
            T_accessed=False))
    print(f'Engineering qualification child exit:{code}; evidence:{invocation}')
    return code


if __name__=='__main__':raise SystemExit(main())
