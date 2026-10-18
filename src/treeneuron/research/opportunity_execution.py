"""Measured M2 admission and live pauses preserve the complete scientific queue."""
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import time

from treeneuron import train
from treeneuron.config import canonical_hash
from .identity import execution_identity
from .ledger import write_new_json
from .opportunity_graph import compile_bounded_graph
from .opportunity_qualification import build_qualification_plan,collect_qualification_profiles
from .opportunity_resources import forecast_resources,remaining_storage
from .queue_observation import ObservedOpportunityWorkers
from .workqueue import run_queue


_ADMISSION_KEYS={'schema','source_sha256','target_plan_hash','qualification_directory',
    'qualification_plan_hash','collection_sha256','available_worker_seconds',
    'free_disk_margin_bytes','forecast'}


def live_resources(directory):
    path=Path(directory).resolve()
    while not path.exists():path=path.parent
    available=None;basis='unavailable';details={}
    if sys.platform=='darwin':
        try:result=subprocess.run(['/usr/bin/vm_stat'],capture_output=True,text=True,check=True)
        except (OSError,subprocess.CalledProcessError) as error:
            details=dict(probe_error=f'{type(error).__name__}: {error}');text=''
        else:text=result.stdout
        match=re.search(r'page size of (\d+) bytes',text)
        pages={name:int(value) for name,value in re.findall(r'^([^:\n]+):\s+(\d+)\.',text,re.M)}
        required=('Pages free','Pages inactive','Pages speculative')
        if match and all(name in pages for name in required):
            available=int(match.group(1))*sum(pages[name] for name in required)
            basis='vm_stat free+inactive+speculative pages; heuristic, no allocation guarantee'
            details=dict(page_size=int(match.group(1)),pages=pages)
    elif Path('/proc/meminfo').is_file():
        raw=Path('/proc/meminfo').read_text();match=re.search(r'^MemAvailable:\s+(\d+) kB$',raw,re.M)
        if match:available=int(match.group(1))*1024;basis='/proc/meminfo MemAvailable'
    return dict(free_disk_bytes=shutil.disk_usage(path).free,available_memory_bytes=available,
        memory_basis=basis,memory_details=details,platform=platform.platform(),observed_utc_ns=time.time_ns())


def resource_decision(target,forecast,job,events,live):
    jobs={j['job_id']:j for j in target['attempts']};name=job['job_id']
    if name not in jobs or canonical_hash(jobs[name])!=canonical_hash(job):raise ValueError('Resource current job differs')
    latest={e['attempt_id']:e for e in events};statuses={n:e['status'] for n,e in latest.items()}
    remaining=remaining_storage(target,forecast,statuses,attempted=set(latest))
    disk=live.get('free_disk_bytes');memory=live.get('available_memory_bytes')
    if type(disk) is not int or disk<0 or (memory is not None and (type(memory) is not int or memory<0)):
        raise ValueError('Invalid live resource quantities')
    required_disk=(forecast['job_retained_bytes'][name]+remaining['additional_retry_reserve_bytes']
        +forecast['max_checkpoint_transaction_bytes']+forecast['free_disk_margin_bytes'])
    required_memory=math.ceil(forecast['peak_rss_bytes']*1.25)+512*1024**2
    used=sum(e['elapsed_ns'] for e in latest.values())/1e9
    reason=('insufficient_live_disk' if disk<required_disk else 'unknown_live_memory' if memory is None
        else 'insufficient_live_memory' if memory<required_memory else 'worker_time_envelope_exhausted'
        if used>=forecast['available_worker_seconds'] else 'ready_for_next_immutable_job')
    return dict(admitted=reason=='ready_for_next_immutable_job',reason=reason,
        target_plan_hash=target['plan_hash'],forecast_hash=forecast['forecast_hash'],job_id=name,
        required_disk_bytes=required_disk,required_memory_bytes=required_memory,
        free_disk_bytes=disk,available_memory_bytes=memory,
        full_remaining_forecast_bytes=remaining['required_additional_bytes'],
        additional_retry_reserve_bytes=remaining['additional_retry_reserve_bytes'],
        charged_worker_seconds=used,live_observation=live,
        scope='Next full-ceiling job plus rolling retry, transaction and free-space margin; full queue retained')


def make_opportunity_admission(target,qualification_directory,*,available_worker_seconds,free_disk_margin_bytes=512*1024**2):
    qualified=build_qualification_plan(target)
    collection=collect_qualification_profiles(qualification_directory,target,expected_plan_hash=qualified['plan_hash'])
    if collection['status']!='profiles_complete':raise ValueError('M2 admission requires all53authenticated native/control workload profiles')
    if collection['worker_loss_upper_bounds_present']:raise ValueError('Worker-loss bounds cannot qualify measured admission clocks')
    forecast=forecast_resources(target,collection['profiles'],available_worker_seconds=available_worker_seconds,
        free_disk_margin_bytes=free_disk_margin_bytes)
    return dict(schema='research/v2-opportunity-admission',source_sha256=target['identity']['source_sha256'],
        target_plan_hash=target['plan_hash'],qualification_directory=str(Path(qualification_directory).resolve()),
        qualification_plan_hash=qualified['plan_hash'],collection_sha256=canonical_hash(collection),
        available_worker_seconds=available_worker_seconds,free_disk_margin_bytes=free_disk_margin_bytes,forecast=forecast)


def validate_opportunity_admission(target,locator):
    path=Path(locator)
    if path.is_symlink() or not path.is_file():raise ValueError('Measured M2 resource admission is required before run')
    raw=path.read_bytes();record=json.loads(raw)
    if set(record)!=_ADMISSION_KEYS or record.get('schema')!='research/v2-opportunity-admission':
        raise ValueError('Invalid M2 resource admission schema')
    actual=make_opportunity_admission(target,record['qualification_directory'],
        available_worker_seconds=record['available_worker_seconds'],free_disk_margin_bytes=record['free_disk_margin_bytes'])
    if record!=actual:raise ValueError('M2 resource admission source/observations/forecast changed')
    return record,dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest())


def run_opportunity(study,output,*,resume=False):
    target=compile_bounded_graph(study);config=study['config'];output=Path(output).resolve()
    locator=config.get('resource_admission')
    if not locator:raise ValueError('Measured M2 resource admission is required before run')
    from .runner import _threads
    _threads()
    admission,reference=validate_opportunity_admission(target,locator)
    forecast=admission['forecast'];binding=dict(schema='research/v2-opportunity-admission-binding',
        target_plan_hash=target['plan_hash'],admission=reference,record=admission)
    binding_path=output/'resource-admission-binding.json'

    def identity(value):
        if (value!=target['identity'] or train.source_hash()!=value['source_sha256']
                or execution_identity(config)!=value['execution_identity']):
            raise ValueError('M2 source/runtime/config changed')
        if hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest()!=reference['sha256']:
            raise ValueError('M2 original admission bytes changed')
        if binding_path.exists():
            if binding_path.is_symlink() or json.loads(binding_path.read_text())!=binding:
                raise ValueError('M2 saved admission binding changed')
        elif (output/'events.jsonl').is_file() and (output/'events.jsonl').stat().st_size:
            raise ValueError('Missing M2 admission binding after paid work')

    def before_start(job,events):
        if not binding_path.exists():write_new_json(binding_path,binding)
        return resource_decision(target,forecast,job,events,live_resources(output))

    def guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if rss>16*1024**3:raise MemoryError('M2 worker exceeded16GiB own peak RSS')
        if shutil.disk_usage(output).free<forecast['free_disk_margin_bytes']+forecast['max_checkpoint_transaction_bytes']:
            raise OSError('M2 checkpoint transaction margin exhausted; retain and resume identical queue')

    worker=ObservedOpportunityWorkers(target,guard=guard)
    return run_queue(target,output,worker,check_identity=identity,resume=resume,before_start=before_start)
