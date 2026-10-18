"""Prepare a measured, immutable full-M1 forecast; admission publication is separate."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def forecast(plan,samples,*,publication_allowance=1024**2):
    import numpy as np
    from treeneuron.config import canonical_hash
    from treeneuron.research.resources import case_id
    if plan['config']['kind']!='m1_controls' or set(samples)!={'positive','negative'}:
        raise ValueError('Requires the fixed M1 measured workload classes')
    groups={}
    for a in plan['attempts']:groups.setdefault(case_id(a),[]).append(a)
    allocations={};counts=dict(positive=0,negative=0);seconds=[0.,0.];details=[]
    t=plan['config']['base_experiment']['training']
    classes={}
    for key,sample in samples.items():
        stages=sample['stages']
        if not stages or any(type(s['executed_updates']) is not int or s['executed_updates']<=0
                or type(s['elapsed_ns']) is not int or s['elapsed_ns']<=0 for s in stages):
            raise ValueError('Every forecast stage requires measured positive execution and elapsed time')
        speeds=[s['elapsed_ns']/s['executed_updates'] for s in stages]
        classes[key]=dict(observed_updates=sum(s['executed_updates'] for s in stages),
            per_update_ns_p50=float(np.quantile(speeds,.5)),per_update_ns_p90=float(np.quantile(speeds,.9)))
    for key,attempts in groups.items():
        positive=attempts[0]['population']=='positive';kind='positive' if positive else 'negative'
        expected={'keep','top','random','oracle_current','oracle_true'} if positive else {'keep','top','random'}
        if {a['role'] for a in attempts}!=expected or len(attempts)!=len(expected):
            raise ValueError('Incomplete nominal M1 transaction cannot be forecast as a smaller procedure')
        maximum_updates=len(expected)*t['continuation_updates']+(0 if positive else t['base_updates'])
        sample=samples[kind];speed=classes[kind]
        allocations[key]=math.ceil(sample['retained_bytes']*maximum_updates/speed['observed_updates']*1.25
            )+sample['max_checkpoint_transaction_bytes']
        for i,q in enumerate(('p50','p90')):
            seconds[i]+=(maximum_updates*speed['per_update_ns_'+q]*(1.25 if i else 1.)+sample['overhead_ns'])/1e9
        counts[kind]+=1
        details.append(dict(case_id=key,cell=attempts[0]['cell'],teacher_index=attempts[0]['teacher_index'],
            repetition=attempts[0]['repetition'],workload_class=kind,maximum_updates=maximum_updates,
            retained_allocation_bytes=allocations[key]))
    reserve=max(allocations.values(),default=0)
    maximum_transaction=max(s['max_checkpoint_transaction_bytes'] for s in samples.values())
    record=dict(schema='research/v2-resource-admission',source_sha256=plan['source_sha256'],
        config_hash=canonical_hash(plan['config']),execution_identity_hash=canonical_hash(plan['execution_identity']),
        measurements=[],forecast_worker_seconds_p50=seconds[0],forecast_worker_seconds_p90=seconds[1],
        forecast_remaining_bytes=sum(allocations.values())+reserve+publication_allowance,
        free_disk_margin_bytes=max(512*1024**2,maximum_transaction),
        peak_rss_bytes=max(s['peak_rss_bytes'] for s in samples.values()),available_worker_seconds=24*3600,
        max_checkpoint_transaction_bytes=maximum_transaction,case_retained_bytes=allocations,retry_reserve_bytes=reserve)
    detail=dict(case_counts=counts,cases=details,measured_classes=classes,samples=samples,
        future_quantiles_validated=False,method='all declared update ceilings at empirical stage-speed quantiles; p90 speed and storage have25% contention/retention headroom',
        limitations=['P50/P90 describe measured stage-speed projections, not calibrated future cohort quantiles',
            'two measured cases do not cover every possible contention/failure pattern',
            'runtime forecast assumes every fit reaches its update ceiling; patience may finish earlier',
            'storage scales whole measured retained bytes by executed-to-ceiling update ratio; check actual disk during execution'],
        publication_allowance_bytes=publication_allowance,
        rolling_retry_policy='largest untouched pending full transaction; completed/partial records never select seeds')
    return record,detail


def load_sample(root,kind,plan):
    from treeneuron.config import canonical_hash
    root=Path(root);result=json.loads((root/'result.json').read_text())
    if (result['status']!='completed' or result['source_sha256']!=plan['source_sha256']
            or result['execution_identity']!=plan['execution_identity']):
        raise ValueError('Measured native case source/runtime/status differs')
    refs=result['measurements'] if kind=='negative' else [result['measurement']]
    profiles=[]
    for reference in refs:
        path=Path(reference['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']:
            raise ValueError('Resource measurement commitment differs')
        item=json.loads(path.read_text())
        if (item['source_sha256']!=plan['source_sha256'] or item['execution_identity_hash']!=canonical_hash(plan['execution_identity'])
                or item['training_contract_hash']!=canonical_hash({'training':plan['config']['base_experiment']['training']})):
            raise ValueError('Resource measurement recipe differs')
        profiles.append(item)
    pair=json.loads((root/'pair/control-pair.json').read_text());stages=[]
    if kind=='negative':
        pointer=json.loads((root/'pair/parent.json').read_text());path=root/'pair'/pointer['metadata']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pointer['sha256']:raise ValueError('Parent metadata commitment differs')
        meta=json.loads(path.read_text())
        stages.append(dict(role='parent',executed_updates=meta['state']['stage']['updates_completed'],elapsed_ns=pair['costs']['parent_ns']))
    for role,m in pair['methods'].items():
        if m['status']!='completed':raise ValueError('Qualification must retain complete native role measurements')
        stages.append(dict(role=role,executed_updates=m['executed_stage_updates'],elapsed_ns=m['enclosing_ns']))
    return dict(retained_bytes=max(p['retained_bytes'] for p in profiles),
        max_checkpoint_transaction_bytes=max(p['max_checkpoint_transaction_bytes'] for p in profiles),
        peak_rss_bytes=max(p['peak_rss_bytes'] for p in profiles),stages=stages,
        overhead_ns=max(1,result['elapsed_ns']-sum(s['elapsed_ns'] for s in stages))),refs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','positive','negative','output'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.ledger import write_new_json
    plan=expand(load_research_config(args.config)).to_dict();samples={};references=[]
    for kind,path in (('positive',args.positive),('negative',args.negative)):
        samples[kind],refs=load_sample(path,kind,plan);references.extend(refs)
    proposed,detail=forecast(plan,samples);proposed['measurements']=references
    args.output.mkdir(parents=True,exist_ok=False)
    write_new_json(args.output/'plan.json',plan);write_new_json(args.output/'proposed-admission.json',proposed)
    write_new_json(args.output/'forecast.json',dict(**detail,plan_hash=plan['plan_hash'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=plan['source_sha256'],admission_not_published=True))
    print(json.dumps(dict(cases=len(proposed['case_retained_bytes']),forecast_hours_p50=proposed['forecast_worker_seconds_p50']/3600,
        forecast_hours_p90=proposed['forecast_worker_seconds_p90']/3600,forecast_GiB=proposed['forecast_remaining_bytes']/1024**3,
        free_margin_GiB=proposed['free_disk_margin_bytes']/1024**3)),flush=True)


if __name__=='__main__':main()
