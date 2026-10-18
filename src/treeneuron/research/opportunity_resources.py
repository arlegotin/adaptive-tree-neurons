"""Complete M2 workload inventory and conservative measured-profile forecasts.

These helpers prepare forecasts. They do not authenticate native qualification
artifacts, admit production execution, or enable the public M2 dispatcher.
"""
from collections import Counter
import copy
import math
import statistics

from treeneuron.config import canonical_hash
from .ledger import TERMINAL


def _checked(plan):
    if (plan.get('schema')!='research/v2-workqueue'
            or plan.get('plan_hash')!=canonical_hash({k:v for k,v in plan.items() if k!='plan_hash'})
            or plan['identity'].get('admission')!='initial_and_bounded_conditions'
            or len(plan['attempts'])!=3853):
        raise ValueError('Requires the complete frozen bounded M2 graph')
    jobs={j['job_id']:j for j in plan['attempts']}
    if len(jobs)!=3853 or dict(Counter(j['kind'] for j in jobs.values()))!=plan['identity']['job_counts']:
        raise ValueError('M2 workload inventory is incomplete')
    return jobs


def workload_manifest(plan):
    jobs=_checked(plan);base=plan['identity']['study_config']['base_experiment']
    descriptors={};result={};classes={}

    def shape(job):
        rows=[jobs[n]['payload'] for n in job['depends_on'] if jobs[n]['kind']=='data']
        if job['kind']=='data':rows=[job['payload']]
        if len(rows)!=1:raise ValueError('Numerical workload requires exactly one data prerequisite')
        small=rows[0]['population']=='positive_small'
        return dict(population=rows[0]['population'],d=2 if small else base['model']['d'],
            parent_h=8 if small else base['model']['h'],
            n_fit=512 if small else base['data']['n_fit'],n_growth=256 if small else base['data']['n_growth'],
            n_validation=256 if small else base['data']['n_validation'],n_evaluation=rows[0]['evaluation_rows'],
            training_dtype='float32',scoring_dtype='float64',device='cpu',intra_threads=1,interop_threads=1)

    for job in plan['attempts']:
        kind,p=job['kind'],job['payload'];maximum=0
        descriptor=dict(kind=kind)
        if kind in ('data','parent','score','fit','head','evaluate','local_diagnostic'):
            descriptor['shape']=shape(job)
        if kind=='parent':
            maximum=0 if p['parent_recipe']=='raw_positive_control' else p['max_updates']
            descriptor.update(parent_recipe=p['parent_recipe'],maximum_updates=maximum,
                stop_by_patience=p.get('stop_by_patience',True),snapshot_updates=p.get('snapshot_updates',[]),
                training=p.get('training',base['training']),absolute_schedule_horizon=p.get('absolute_schedule_horizon'))
        elif kind=='fit':
            parent=[jobs[n]['payload'] for n in job['depends_on'] if jobs[n]['kind']=='parent']
            if len(parent)!=1:raise ValueError('Fitting workload requires one parent')
            training=copy.deepcopy(p.get('training',parent[0].get('training',base['training'])))
            h=descriptor['shape']['parent_h'];d=descriptor['shape']['d'];m=p['requested_count']
            role=p.get('role','horizon_slot')
            if p['matrix']=='M2c':m=0 if p['slot']=='keep' else 1
            grown=m if role not in ('keep','ordinary_keep','warm_lower','warm_upper') else 0
            if role in ('warm_lower','warm_upper'):
                numerator=h*(d+2)+2*m*(d+1)
                h=numerator//(d+2)+(role=='warm_upper' and numerator%(d+2)!=0)
            maximum=p['max_updates']
            descriptor.update(hidden_width=h,gate_count=grown,requested_count=m,
                role=role if role in ('top','random','warm_lower','warm_upper') else 'fixed_or_keep',
                maximum_updates=maximum,stop_by_patience=p['stop_by_patience'],
                selection_prefixes=p.get('selection_prefixes',[]),
                new_stream=p.get('treatment')=='new_stream_retained_adam',
                training=training,absolute_schedule_horizon=p.get('absolute_schedule_horizon'),
                epsilon=base['growth']['epsilon'],tau=base['model']['tau'])
        elif kind=='evaluate':
            producer=[jobs[n] for n in job['depends_on'] if jobs[n]['kind']!='data']
            if len(producer)!=1:raise ValueError('Evaluation workload requires one producer')
            source=producer[0]
            if source['kind']=='pair_oracle':source=jobs[source['depends_on'][0]]
            previous=descriptors[source['job_id']]
            descriptor.update(predictor_kind=source['kind'],hidden_width=previous.get('hidden_width',previous['shape']['parent_h']),
                gate_count=previous.get('gate_count',0),
                retained_models=1+len(previous.get('selection_prefixes',[])))
        elif kind=='head':descriptor.update(alphas=p['alphas'],unpenalized_intercept=True)
        elif kind=='score':descriptor.update(contrasts='current',eligibility='current',tau=base['model']['tau'],epsilon=base['growth']['epsilon'])
        elif kind=='local_diagnostic':descriptor.update(minibatches=64,step_size=1e-7)
        elif kind=='data':descriptor['raw_control_coordinate']=dict(gamma=2.,sigma=.1) if p['population']=='positive_small' else None
        elif kind=='horizon_binding':descriptor.update(top=4,remaining_sample=4,nominal_candidates=64)
        elif kind=='pair_oracle':descriptor.update(nominal_pair_count=len(job['depends_on']),select_on='G')
        elif kind=='fallback_gate':descriptor.update(snapshot_updates=p['snapshot_updates'])
        elif kind=='snapshot_gate':descriptor.update(snapshot=p['snapshot'])
        elif kind=='extension_decision':
            descriptor.update(input_count=len(job['depends_on']),contract=p['opportunity_decision_contract'])
        else:raise ValueError('Unknown numerical workload kind')
        key=canonical_hash(descriptor);descriptors[job['job_id']]=descriptor
        result[job['job_id']]=dict(workload_hash=key,maximum_updates=maximum)
        if key not in classes:classes[key]=dict(descriptor=descriptor,maximum_updates=maximum,
            representative_job_id=job['job_id'],job_count=0)
        classes[key]['job_count']+=1
    return dict(schema='research/v2-opportunity-workloads',queue_plan_hash=plan['plan_hash'],
        classes=classes,jobs=result,includes_all_conditional_slots=True,
        maximum_planned_updates=sum(r['maximum_updates'] for r in result.values()),
        production_admission=False,scope='shape/treatment inventory; not native qualification evidence')


def forecast_resources(plan, profiles, *, available_worker_seconds, free_disk_margin_bytes=512*1024**2):
    manifest=workload_manifest(plan);classes=manifest['classes'];grouped={key:[] for key in classes}
    for profile in profiles:
        key=profile.get('workload_hash')
        if (profile.get('schema')!='research/v2-queue-resource-profile'
                or profile.get('source_sha256')!=plan['identity']['source_sha256']
                or profile.get('execution_identity_hash')!=canonical_hash(plan['identity']['execution_identity'])
                or key not in grouped or profile.get('status')!='completed_eligible'
                or not isinstance(profile.get('provenance'),dict)):
            raise ValueError('Unqualified resource profile source/runtime/class/status')
        if any(type(profile.get(k)) is not int or profile[k]<=0 for k in
               ('elapsed_ns','retained_bytes','peak_rss_bytes')):
            raise ValueError('Resource profile requires positive measured quantities')
        maximum=classes[key]['maximum_updates'];updates=profile.get('executed_updates')
        if type(updates) is not int or not 0<=updates<=maximum:
            raise ValueError('Invalid measured execution counter')
        if maximum and updates==0:raise ValueError('Training profile has no measured optimization')
        if type(profile.get('max_checkpoint_transaction_bytes')) is not int or profile['max_checkpoint_transaction_bytes']<0:
            raise ValueError('Invalid checkpoint transaction measurement')
        grouped[key].append(profile)
    if any(not rows for rows in grouped.values()):raise ValueError('Missing measured workload class')
    if (type(available_worker_seconds) not in (int,float) or not math.isfinite(available_worker_seconds)
            or available_worker_seconds<=0 or type(free_disk_margin_bytes) is not int or free_disk_margin_bytes<=0):
        raise ValueError('Invalid available resource envelope')
    estimates={}
    for key,rows in grouped.items():
        maximum=classes[key]['maximum_updates']
        scales=[max(1.,maximum/max(1,r['executed_updates'])) for r in rows]
        seconds=[r['elapsed_ns']*scale/1e9 for r,scale in zip(rows,scales)]
        estimates[key]=dict(typical_seconds=statistics.median(seconds),slow_seconds=max(seconds)*1.5,
            retained_bytes=math.ceil(max(r['retained_bytes']*scale for r,scale in zip(rows,scales))*1.25)
                +max(r['max_checkpoint_transaction_bytes'] for r in rows)+64*1024)
    allocations={name:estimates[r['workload_hash']]['retained_bytes'] for name,r in manifest['jobs'].items()}
    reserve=max(allocations.values());typical=sum(estimates[r['workload_hash']]['typical_seconds'] for r in manifest['jobs'].values())
    slow=sum(estimates[r['workload_hash']]['slow_seconds'] for r in manifest['jobs'].values())
    peak=max(r['peak_rss_bytes'] for r in profiles);transaction=max(r['max_checkpoint_transaction_bytes'] for r in profiles)
    if peak>16*1024**3 or free_disk_margin_bytes<transaction or slow>available_worker_seconds:
        raise ValueError('Measured forecast exceeds declared time/RAM/transaction margin')
    record=dict(schema='research/v2-opportunity-resource-forecast',queue_plan_hash=plan['plan_hash'],
        source_sha256=plan['identity']['source_sha256'],execution_identity_hash=canonical_hash(plan['identity']['execution_identity']),
        profiles_sha256=canonical_hash({'profiles':profiles}),job_retained_bytes=allocations,
        retry_reserve_bytes=reserve,forecast_remaining_bytes=sum(allocations.values())+reserve,
        free_disk_margin_bytes=free_disk_margin_bytes,forecast_typical_seconds=typical,forecast_slow_seconds=slow,
        peak_rss_bytes=peak,max_checkpoint_transaction_bytes=transaction,available_worker_seconds=available_worker_seconds,
        workload_class_count=len(classes),class_estimates=estimates,
        maximum_planned_updates=manifest['maximum_planned_updates'],
        forecast_type='uncalibrated_full_ceiling_projection',production_admission=False,
        limitations=['Empirical observed speed with1.5x slow envelope is not a calibrated P90',
            'Whole retained bytes scale to update ceilings with25% storage headroom and64KiB publication allowance per job',
            'One rolling untouched-job retry reserve; previous failed bytes remain paid and present',
            'Native profile artifact authentication and live production admission remain separate prerequisites'])

    record['forecast_hash']=canonical_hash(record)
    return record


def remaining_storage(plan, forecast, statuses, *, attempted):
    _checked(plan);allocations=forecast['job_retained_bytes'];names={j['job_id'] for j in plan['attempts']}
    if (forecast.get('forecast_hash')!=canonical_hash({k:v for k,v in forecast.items() if k!='forecast_hash'})
            or forecast['source_sha256']!=plan['identity']['source_sha256']
            or forecast['execution_identity_hash']!=canonical_hash(plan['identity']['execution_identity'])
            or forecast['queue_plan_hash']!=plan['plan_hash'] or set(allocations)!=names
            or not set(statuses)<=names or set(statuses)!=set(attempted)
            or any(s not in TERMINAL|{'running'} for s in statuses.values())
            or any(type(v) is not int or v<=0 for v in allocations.values())):
        raise ValueError('Remaining storage inventory/status identity differs')
    terminal={name for name,status in statuses.items() if status in TERMINAL-{'interrupted_recoverable'}}
    reserve=max((v for name,v in allocations.items() if name not in attempted),default=0)
    return dict(required_additional_bytes=sum(v for name,v in allocations.items() if name not in terminal)+reserve,
        additional_retry_reserve_bytes=reserve,terminal_job_count=len(terminal),pending_job_count=len(names-terminal),
        basis='whole terminal jobs only; interrupted/partial work receives no allocation credit',production_admission=False)
