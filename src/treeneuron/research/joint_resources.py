"""Source-fixed M3 resource forecasts and live whole-job admission; no time cap.

Profile authentication belongs to the qualification collector. These arithmetic
helpers never grant scientific execution on the strength of caller-supplied rows.
"""
import math
import statistics

from treeneuron.config import canonical_hash
from .joint_qualification import checked_target,workload_key


def forecast_joint_resources(target,profiles,*,free_disk_margin_bytes=512*1024**2):
    checked_target(target);classes={};jobs={j['job_id']:j for j in target['attempts']}
    for job in jobs.values():
        key=workload_key(job)
        maximum=job['payload'].get('max_updates',0) if job['kind'] in ('fit','parent','probe') else 0
        classes.setdefault(key,dict(kind=job['kind'],maximum_updates=maximum,samples=[]))
        if classes[key]['maximum_updates']!=maximum:raise ValueError('M3 workload class changed its horizon')
    if type(free_disk_margin_bytes) is not int or free_disk_margin_bytes<=0:
        raise ValueError('Positive explicit free-space margin required')
    runtime=canonical_hash(target['identity']['execution_identity'])
    for profile in profiles:
        key=profile.get('workload_key');kind=classes.get(key,{}).get('kind')
        if (key not in classes or profile.get('source_sha256')!=target['identity']['source_sha256']
                or profile.get('execution_identity_hash')!=runtime or not isinstance(profile.get('provenance'),dict)
                or profile.get('status')!='completed_eligible' and not
                    (kind=='extension_decision' and profile.get('status')=='completed_ineligible')):
            raise ValueError('Resource profile source/runtime/class/native status is unqualified')
        for name in ('elapsed_ns','retained_bytes','peak_rss_bytes'):
            if type(profile.get(name)) is not int or profile[name]<=0:raise ValueError('Positive measured profile quantities required')
        updates=profile.get('executed_updates');maximum=classes[key]['maximum_updates']
        if type(updates) is not int or not 0<=updates<=maximum or maximum and updates==0:
            raise ValueError('Native training profile must include actual optimization')
        transaction=profile.get('max_checkpoint_transaction_bytes')
        if type(transaction) is not int or transaction<0:raise ValueError('Nonnegative explicit checkpoint transaction required')
        inherited=kind=='probe' or kind=='fit' and key!='fit:fresh_upper'
        prefix=profile.get('parent_selected_update',0)
        if inherited and (type(prefix) is not int or not 0<=prefix<=8000):
            raise ValueError('Observed inherited parent prefix is required')
        scale=max(1.,maximum/max(1,updates))
        history_scale=max(scale,(8000+maximum)/max(1,prefix+updates)) if inherited else scale
        classes[key]['samples'].append(dict(seconds=profile['elapsed_ns']*max(scale,history_scale)/1e9,
            retained_bytes=math.ceil(profile['retained_bytes']*history_scale*1.25)+transaction+64*1024,
            peak_rss_bytes=profile['peak_rss_bytes'],transaction_bytes=transaction))
    if any(not value['samples'] for value in classes.values()):raise ValueError('Missing measured M3 workload class')
    estimates={}
    for key,value in classes.items():
        samples=value['samples']
        estimates[key]=dict(profile_count=len(samples),typical_seconds=statistics.median(s['seconds'] for s in samples),
            slow_seconds=max(s['seconds'] for s in samples)*1.5,
            retained_bytes=max(s['retained_bytes'] for s in samples))
    allocated={name:estimates[workload_key(job)]['retained_bytes'] for name,job in jobs.items()}
    peak=max(s['peak_rss_bytes'] for c in classes.values() for s in c['samples'])
    transaction=max(s['transaction_bytes'] for c in classes.values() for s in c['samples'])
    if peak>3*1024**3:raise ValueError('Qualification exceeds the declared3GiB M3 own-worker RSS track')
    initial={}
    for name,job in jobs.items():
        initial[name]=(job['payload']['teacher_index']<8 if job['kind']=='data' else
            True if job['kind']=='extension_decision' else all(initial[d] for d in job['depends_on']))
    result=dict(schema='research/v2-M3-resource-forecast',target_plan_hash=target['plan_hash'],
        source_sha256=target['identity']['source_sha256'],execution_identity_hash=runtime,
        profiles_sha256=canonical_hash({'profiles':profiles}),class_estimates=estimates,job_retained_bytes=allocated,
        retry_reserve_bytes=max(allocated.values()),max_checkpoint_transaction_bytes=transaction,
        free_disk_margin_bytes=free_disk_margin_bytes,peak_rss_bytes=peak,
        own_worker_rss_limit_bytes=3*1024**3,job_initial_tranche=initial,
        forecast_initial_raw_bytes=sum(value for name,value in allocated.items() if initial[name]),
        forecast_complete_raw_bytes=sum(allocated.values()),
        maximum_planned_updates=sum(j['payload'].get('max_updates',0) for j in jobs.values() if j['kind'] in ('fit','parent','probe')),
        forecast_typical_worker_seconds=sum(estimates[workload_key(j)]['typical_seconds'] for j in jobs.values()),
        forecast_slow_worker_seconds=sum(estimates[workload_key(j)]['slow_seconds'] for j in jobs.values()),
        forecast_initial_typical_seconds=sum(estimates[workload_key(j)]['typical_seconds'] for name,j in jobs.items() if initial[name]),
        timing_interpretation='Median and1.5x maximum scaled profile heuristics, not calibrated population quantiles; no execution deadline',
        storage_policy='Whole next job plus rolling retry/transaction/free margin; disclose full raw forecast; inactive byte-identical log compression may be needed',
        production_admission=False,T_authorized=False)
    result['forecast_hash']=canonical_hash(result)
    return result


def joint_resource_decision(target,forecast,job,events,live):
    jobs={j['job_id']:j for j in target['attempts']};name=job['job_id']
    if (name not in jobs or canonical_hash(jobs[name])!=canonical_hash(job)
            or forecast.get('target_plan_hash')!=target['plan_hash']
            or forecast.get('forecast_hash')!=canonical_hash({k:v for k,v in forecast.items() if k!='forecast_hash'})):
        raise ValueError('M3 resource forecast or current immutable job differs')
    disk=live.get('free_disk_bytes');memory=live.get('available_memory_bytes')
    if (type(disk) is not int or disk<0 or memory is not None and (type(memory) is not int or memory<0)):
        raise ValueError('Explicit live resource quantities required')
    latest={event['attempt_id']:event for event in events}
    required_disk=(forecast['job_retained_bytes'][name]+forecast['retry_reserve_bytes']
        +forecast['max_checkpoint_transaction_bytes']+forecast['free_disk_margin_bytes'])
    required_memory=math.ceil(forecast['peak_rss_bytes']*1.25)+512*1024**2
    reason=('insufficient_live_disk' if disk<required_disk else 'unknown_live_memory' if memory is None
        else 'insufficient_live_memory' if memory<required_memory else 'ready_for_next_immutable_job')
    terminal={'completed_eligible','completed_ineligible','numerical_failure','input_or_dependency_failure','irrecoverable_missing','timeout','resource_exhausted'}
    remaining=sum(value for key,value in forecast['job_retained_bytes'].items()
        if latest.get(key,{}).get('status') not in terminal)
    return dict(admitted=reason=='ready_for_next_immutable_job',reason=reason,job_id=name,
        target_plan_hash=target['plan_hash'],forecast_hash=forecast['forecast_hash'],
        required_disk_bytes=required_disk,required_memory_bytes=required_memory,
        free_disk_bytes=disk,available_memory_bytes=memory,remaining_raw_forecast_bytes=remaining,
        charged_worker_seconds=sum(event['elapsed_ns'] for event in latest.values())/1e9,
        live_observation=live,scope='Full next immutable job/retry/transaction; no wall-time quality deadline')
