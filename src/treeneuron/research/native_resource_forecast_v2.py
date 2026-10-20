"""Measured practical envelopes and advisory live admission, without a wall cap.

An anchor-based projection is not a runtime quantile or an unseen-configuration
bound. Pure reducers grant no native, execution or protected-data authority.
"""
import copy
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from treeneuron.config import canonical_hash,canonical_json
from .deployment_worker_v2 import _source,_json
from .native_anchor_v2 import _closed_reference,_bound_json
from .native_qualification_registry_v2 import load_native_qualification_registry
from .practical_support_lock_v2 import load_practical_support_lock
from .primary_selection_v2 import _primary
from .qualify import qualification_locks_sha256


def _positive(value):return type(value) is int and value>0


def _closed_logical_footprint(references):
    """Measure unique closed files, including separately retained generation."""
    files={}
    for reference in references:
        root,inventory=_closed_reference(reference)
        for row in inventory['files']:
            name=str(root/row['path']);identity=(row['bytes'],row['sha256'])
            if name in files and files[name]!=identity:raise ValueError('Overlapping footprint bytes changed')
            files[name]=identity
    return dict(retained_logical_bytes=sum(identity[0] for identity in files.values()),unique_files=len(files),
        native_qualification_authenticated=False)


def _forecast_from_observations(plan,expected,observations,*,free_disk_margin_bytes):
    """Reduce untrusted measured metadata; no public qualification is implied."""
    value,primary=_primary(plan,expected)
    if value.get('budget_policy')!='native_quality' or not _positive(free_disk_margin_bytes) or free_disk_margin_bytes<1024**3:
        raise ValueError('Native-quality budget and at least1GiB free-space margin required')
    if type(observations) is not list:raise ValueError('Explicit resource observations required')
    coordinates={(j['dataset'],j['family_id']) for j in value['jobs']};groups={key:[] for key in coordinates}
    fields={'dataset','family_id','kind','process_elapsed_ns','observed_peak_root_rss_bytes','retained_logical_bytes'}
    for row in observations:
        if (type(row) is not dict or set(row)!=fields or row['kind'] not in ('anchor','support')
                or (row['dataset'],row['family_id']) not in coordinates
                or any(not _positive(row[k]) for k in ('process_elapsed_ns','observed_peak_root_rss_bytes','retained_logical_bytes'))):
            raise ValueError('Complete positive anchor/support measurements required')
        groups[row['dataset'],row['family_id']].append(row)
    envelopes={}
    for (dataset,family),rows in sorted(groups.items()):
        anchors=[r for r in rows if r['kind']=='anchor'];supports=[r for r in rows if r['kind']=='support']
        if not anchors or not supports:raise ValueError('Every dataset/family needs native anchors and short supports')
        peak=max(r['observed_peak_root_rss_bytes'] for r in rows);size=max(r['retained_logical_bytes'] for r in rows)
        envelopes[dataset+'/'+family]=dict(native_anchor_count=len(anchors),short_support_count=len(supports),
            native_anchor_min_ns=min(r['process_elapsed_ns'] for r in anchors),
            native_anchor_max_ns=max(r['process_elapsed_ns'] for r in anchors),
            largest_observed_root_rss_bytes=peak,largest_retained_logical_bytes=size,
            transaction_reserve_proxy_bytes=2*size+64*1024**2,
            recovery_copy_reserve_proxy_bytes=size,required_memory_proxy_bytes=(5*peak+3)//4+512*1024**2)
    jobs=[]
    for job in value['jobs']:
        envelope=envelopes[job['dataset']+'/'+job['family_id']];cap=job['budget']['max_attempts']
        configurations=29 if job['family_id']=='P13' else cap
        jobs.append(dict(job=copy.deepcopy(job),envelope_key=job['dataset']+'/'+job['family_id'],
            no_retry_configurations=configurations,
            native_anchor_proxy_ns=cap*envelope['native_anchor_max_ns'],
            retained_proxy_bytes=cap*envelope['largest_retained_logical_bytes']))
    tranche=lambda name:[r for r in jobs if r['job']['tranche']==name]
    main=tranche('primary');secondary=tranche('secondary')
    result=dict(schema='research/v2-native-practical-resource-forecast',source_sha256=value['source_sha256'],
        practical_plan_hash=expected,support_lock_sha256=value['support_lock_sha256'],
        free_disk_margin_bytes=free_disk_margin_bytes,
        counts=dict(jobs=len(jobs),primary=len(main),secondary=len(secondary),
            paid_attempt_cap=sum(r['job']['budget']['max_attempts'] for r in jobs),
            primary_paid_attempt_cap=sum(r['job']['budget']['max_attempts'] for r in main),
            secondary_paid_attempt_cap=sum(r['job']['budget']['max_attempts'] for r in secondary),
            no_retry_configurations=sum(r['no_retry_configurations'] for r in jobs),
            primary_no_retry_configurations=sum(r['no_retry_configurations'] for r in main),
            secondary_no_retry_configurations=sum(r['no_retry_configurations'] for r in secondary)),
        projection=dict(whole_native_anchor_proxy_ns=sum(r['native_anchor_proxy_ns'] for r in jobs),
            primary_native_anchor_proxy_ns=sum(r['native_anchor_proxy_ns'] for r in main),
            secondary_native_anchor_proxy_ns=sum(r['native_anchor_proxy_ns'] for r in secondary),
            whole_retained_proxy_bytes=sum(r['retained_proxy_bytes'] for r in jobs),
            primary_retained_proxy_bytes=sum(r['retained_proxy_bytes'] for r in main),
            secondary_retained_proxy_bytes=sum(r['retained_proxy_bytes'] for r in secondary)),
        envelopes=envelopes,jobs=jobs,worker_concurrency=1,elapsed_time_cutoff_ns=None,
        confidence_quantiles_available=False,unseen_configuration_upper_bound_available=False,
        native_qualification_authenticated=False,execution_authorized=False,secondary_release_authorized=False,
        protected_access_authorized=False,T_accessed=False,
        scope='Anchor maximum times paid-attempt cap is a heuristic projection, not a deadline, quantile, or runtime upper bound. Short supports inform sampled root RSS and logical footprint only. Storage2x proof+64MiB and memory1.25x+512MiB are disclosed reserve proxies, not aggregate-memory or unseen-configuration guarantees. Full retention projection is separate from next-attempt live admission. Secondary scaling is descriptive, not secondary workload qualification. Registry validation/planning and final outer costs are additional measured costs; native process durations exclude those containing costs.')
    result['forecast_hash']=canonical_hash(result)
    return result


def forecast_native_practical_resources(*,practical_plan,expected_practical_plan_hash,support_reference,
        qualification_reference,free_disk_margin_bytes=1024**3):
    """Read-only actual-qualified forecast; no output or model process is created."""
    started=time.perf_counter_ns();registry=load_native_qualification_registry(qualification_reference)
    value,jobs=_primary(practical_plan,expected_practical_plan_hash);law=load_practical_support_lock(support_reference)
    if (value.get('budget_policy')!='native_quality' or value['support_lock_sha256']!=support_reference['sha256']
            or value['source_sha256']!=law['source_sha256'] or value['runtime_sha256']!=law['core_manifest_sha256']
            or value['lock_sha256']!=qualification_locks_sha256()
            or registry['identity']['scientific_source_sha256']!=value['source_sha256']
            or registry['identity']['test_and_lock_files']['locks/research-v2/core.json']!=value['runtime_sha256']
            or any(registry['preparation']['datasets'][j['dataset']]['split_hashes'][str(j['outer_fold'])]!=j['split_hash'] for j in jobs)):
        raise ValueError('Forecast requires the exact qualified native-quality source/support/runtime/primary splits')
    observations=[];references=[]
    for kind,rows in (('anchor',registry['anchors']),('support',registry['supports'])):
        for row in rows:
            observed=[copy.deepcopy(row['reference'])]
            if kind=='anchor':
                root,inventory=_closed_reference(row['reference'])
                inspection=_bound_json(root,inventory,'inspection/result.json')
                observed.append(dict(root=inspection['immutable_generation_root'],
                    inventory_sha256=inspection['immutable_generation_inventory_sha256']))
            footprint=_closed_logical_footprint(observed);references.extend(observed)
            resources=row['resources']
            observations.append(dict(dataset=row['dataset'],family_id=row['family_id'],kind=kind,
                process_elapsed_ns=resources['process_elapsed_ns'],observed_peak_root_rss_bytes=resources['observed_peak_root_rss_bytes'],
                retained_logical_bytes=footprint['retained_logical_bytes']))
    result=_forecast_from_observations(practical_plan,expected_practical_plan_hash,observations,
        free_disk_margin_bytes=free_disk_margin_bytes)
    if _source()!=value['source_sha256'] or load_practical_support_lock(support_reference)!=law:
        raise ValueError('Resource forecast source or support changed')
    result.pop('forecast_hash')
    result.update(native_qualification_authenticated=True,qualification_reference=copy.deepcopy(qualification_reference),
        support_reference=copy.deepcopy(support_reference),observation_references=references,
        resource_observations=observations,forecast_preparation_ns=time.perf_counter_ns()-started)
    result['forecast_hash']=canonical_hash(result)
    return result


def _match_cached_forecast(saved,actual):
    """Metadata equality only; recomputed public qualification is separate."""
    for value in (saved,actual):
        if (type(value) is not dict or not _positive(value.get('forecast_preparation_ns'))
                or value.get('forecast_hash')!=canonical_hash({k:v for k,v in value.items() if k!='forecast_hash'})):
            raise ValueError('Exact cached resource forecast/hash/declared clock required')
    ignored={'forecast_hash','forecast_preparation_ns'}
    if canonical_json({k:v for k,v in saved.items() if k not in ignored})!=canonical_json({k:v for k,v in actual.items() if k not in ignored}):
        raise ValueError('Cached resource source, proof, policy or measured observations differ')


def load_native_practical_resource_forecast(reference,*,practical_plan,expected_practical_plan_hash,
        support_reference,qualification_reference):
    """Recompute genuine prerequisites before opening a cached forecast.

    The saved planning duration is an observation, not a cost authority; the
    current caller pays this entire revalidation using its outer clock.
    """
    actual=forecast_native_practical_resources(practical_plan=practical_plan,
        expected_practical_plan_hash=expected_practical_plan_hash,support_reference=support_reference,
        qualification_reference=qualification_reference)
    from .primary_search_worker_v2 import _reference,_sha
    path=_reference(reference)
    raw=path.read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('Resource forecast file commitment changed')
    saved=_json(raw)
    # The actual comparison uses the cache's declared margin only after basic
    # byte/structure checks, then revalidates the full policy and observations.
    if (type(saved) is not dict or type(saved.get('free_disk_margin_bytes')) is not int
            or saved['free_disk_margin_bytes']<1024**3):raise ValueError('Declared resource margin required')
    if saved['free_disk_margin_bytes']!=actual['free_disk_margin_bytes']:
        actual=forecast_native_practical_resources(practical_plan=practical_plan,
            expected_practical_plan_hash=expected_practical_plan_hash,support_reference=support_reference,
            qualification_reference=qualification_reference,free_disk_margin_bytes=saved['free_disk_margin_bytes'])
    _match_cached_forecast(saved,actual)
    if _sha(path)!=reference['sha256']:raise ValueError('Cached forecast changed during revalidation')
    return saved


def native_resource_decision(*,forecast,job,live):
    """Pure advisory decision; callers must separately authenticate every input."""
    return _native_resource_decision(forecast=forecast,job=job,live=live,expected_tranche='primary')


def _native_resource_decision(*,forecast,job,live,expected_tranche):
    if expected_tranche not in ('primary','secondary'):raise ValueError('Exact advisory resource tranche required')
    if (type(forecast) is not dict or forecast.get('schema')!='research/v2-native-practical-resource-forecast'
            or forecast.get('forecast_hash')!=canonical_hash({k:v for k,v in forecast.items() if k!='forecast_hash'})
            or forecast.get('source_sha256')!=_source() or forecast.get('elapsed_time_cutoff_ns') is not None
            or type(job) is not dict):raise ValueError('Exact unchanged current-source resource forecast required')
    matches=[r for r in forecast['jobs'] if r['job']==job]
    if len(matches)!=1:raise ValueError('Resource job differs from the complete forecast')
    if job['tranche']!=expected_tranche:raise PermissionError('Resource projection does not admit a different tranche')
    if (type(live) is not dict or not _positive(live.get('observed_utc_ns'))
            or type(live.get('free_disk_bytes')) is not int or live['free_disk_bytes']<0
            or live.get('available_memory_bytes') is not None and (type(live['available_memory_bytes']) is not int or live['available_memory_bytes']<0)
            or type(live.get('memory_basis')) is not str or not live['memory_basis']):
        raise ValueError('Explicit disk and advisory memory observations required')
    envelope=forecast['envelopes'][matches[0]['envelope_key']]
    disk=envelope['transaction_reserve_proxy_bytes']+envelope['recovery_copy_reserve_proxy_bytes']+forecast['free_disk_margin_bytes']
    memory=envelope['required_memory_proxy_bytes']
    reason=('insufficient_live_disk' if live['free_disk_bytes']<disk else 'unknown_live_memory' if live['available_memory_bytes'] is None
        else 'insufficient_live_memory' if live['available_memory_bytes']<memory else 'ready_for_next_immutable_job')
    return dict(schema='research/v2-native-resource-decision',admitted=reason=='ready_for_next_immutable_job',reason=reason,
        job_id=job['job_id'],forecast_hash=forecast['forecast_hash'],required_disk_proxy_bytes=disk,
        required_memory_proxy_bytes=memory,live_observation=copy.deepcopy(live),elapsed_time_cutoff_ns=None,
        native_qualification_authenticated=False,execution_authorized=False,protected_access_authorized=False,T_accessed=False,
        scope='Pure advisory next-attempt arithmetic, not proof of native qualification or a hard resource guarantee. Preserve data/configuration/seed and pause the immutable queue on shortage. Existing paid-recovery and exact NODE checkpoint transaction checks remain required.')


def observe_native_host_resources(directory):
    """Observe disk and a host-wide RAM proxy, without inspecting other processes."""
    path=Path(directory).resolve()
    while not path.exists():path=path.parent
    available=None;basis='unavailable';details={}
    if sys.platform=='darwin':
        try:probe=subprocess.run(['/usr/bin/vm_stat'],capture_output=True,text=True,check=True,timeout=5)
        except (OSError,subprocess.SubprocessError) as error:details=dict(probe_error=f'{type(error).__name__}: {error}')
        else:
            match=re.search(r'page size of (\d+) bytes',probe.stdout)
            pages={name:int(value) for name,value in re.findall(r'^([^:\n]+):\s+(\d+)\.',probe.stdout,re.M)}
            names=('Pages free','Pages inactive','Pages speculative')
            if match and all(name in pages for name in names):
                available=int(match[1])*sum(pages[name] for name in names)
                basis='vm_stat free+inactive+speculative pages; heuristic, no allocation guarantee'
                details=dict(page_size=int(match[1]),pages=pages)
    elif Path('/proc/meminfo').is_file():
        match=re.search(r'^MemAvailable:\s+(\d+) kB$',Path('/proc/meminfo').read_text(),re.M)
        if match:available=int(match[1])*1024;basis='/proc/meminfo MemAvailable; advisory'
    return dict(free_disk_bytes=shutil.disk_usage(path).free,available_memory_bytes=available,
        memory_basis=basis,memory_details=details,observed_utc_ns=time.time_ns())
