"""Timer attribution and paired cost arithmetic; no deployment claim authority."""
import hashlib
import math

import numpy as np

from treeneuron.config import canonical_hash
from treeneuron.research.seeds import seed64


def _ns(value,*,positive=False):
    if type(value) is not int or not (0<value<2**63 if positive else 0<=value<2**63):
        raise ValueError('Cost clocks require nonnegative signed64-bit integer nanoseconds')
    return value


def _overlap(a,b):
    return max(a['start_ns'],b['start_ns'])<min(a['end_ns'],b['end_ns'])


def attribute_costs(timers,procedures,*,cohort_timer):
    """Keep measured cohort wall time separate from standalone charged roots.

    Timers are inclusive. Dependency edges identify required shared work; parent
    edges identify containing clocks. One shared prerequisite is charged once to
    each standalone procedure. Parallel standalone-worker attribution needs a
    separate contract; overlapping charged roots are rejected here.
    """
    if not isinstance(timers,list) or not timers or not isinstance(procedures,dict) or not procedures:
        raise ValueError('Explicit timer tree and standalone procedures required')
    nodes={}
    for row in timers:
        if not isinstance(row,dict) or set(row)!={'timer_id','parent_id','start_ns','end_ns','requires'}:
            raise ValueError('Invalid inclusive timer fields')
        name=row['timer_id'];requires=row['requires']
        if (type(name) is not str or not name or name in nodes
                or row['parent_id'] is not None and (type(row['parent_id']) is not str or not row['parent_id'])
                or not isinstance(requires,list) or any(type(v) is not str or not v for v in requires)
                or len(requires)!=len(set(requires))):raise ValueError('Invalid or duplicate timer/dependency identity')
        start=_ns(row['start_ns']);end=_ns(row['end_ns'])
        if end<start:raise ValueError('Timer end precedes start')
        nodes[name]=dict(row,requires=list(requires))
    if cohort_timer not in nodes or nodes[cohort_timer]['parent_id'] is not None:
        raise ValueError('Declared cohort timer must be the unique physical root')

    def ancestors(name):
        seen={name};result=[];parent=nodes[name]['parent_id']
        while parent is not None:
            if parent not in nodes or parent in seen:raise ValueError('Missing or cyclic containing timer')
            seen.add(parent);result.append(parent);parent=nodes[parent]['parent_id']
        return result

    lineage={name:ancestors(name) for name in nodes}
    for name,node in nodes.items():
        if name!=cohort_timer and (not lineage[name] or lineage[name][-1]!=cohort_timer):
            raise ValueError('Every timer must belong to the measured cohort')
        if node['parent_id'] is not None:
            parent=nodes[node['parent_id']]
            if node['start_ns']<parent['start_ns'] or node['end_ns']>parent['end_ns']:
                raise ValueError('Child timer escapes its containing clock')
        for dependency in node['requires']:
            if dependency not in nodes or dependency==name or nodes[dependency]['end_ns']>node['start_ns']:
                raise ValueError('Missing, cyclic or unfinished prerequisite clock')

    children={name:[] for name in nodes}
    for name,node in nodes.items():
        if node['parent_id'] is not None:children[node['parent_id']].append(name)
    cached={}

    def closure(names,active=None):
        active=set() if active is None else active;result=set()
        for name in names:
            if name not in nodes or name in active:raise ValueError('Missing or cyclic cost prerequisite')
            if name not in cached:
                # A containing clock also inherits dependencies declared by its
                # nested work. Requiring callers to duplicate these edges on
                # every container would silently omit shared parent costs.
                cached[name]=frozenset({name}|closure(nodes[name]['requires']+children[name],active|{name}))
            result.update(cached[name])
        return result

    closure([cohort_timer])
    attributed={}
    for procedure,roots in procedures.items():
        if (type(procedure) is not str or not procedure or not isinstance(roots,list) or not roots
                or any(type(v) is not str for v in roots) or len(roots)!=len(set(roots))):
            raise ValueError('Invalid standalone procedure cost roots')
        if any(name not in nodes for name in roots):raise ValueError('Unknown standalone procedure timer')
        if any(set(lineage[name])&set(roots) for name in roots):
            raise ValueError('Containing and nested timers cannot both be explicitly charged')
        required=closure(roots)
        # A prerequisite already contained in a charged inclusive clock is paid
        # by that clock. Its own prerequisites remain in the closure.
        charged=sorted(name for name in required if not set(lineage[name])&required)
        if any(_overlap(nodes[a],nodes[b]) for i,a in enumerate(charged) for b in charged[i+1:]):
            raise ValueError('Standalone charged roots overlap; parallel attribution needs a separate contract')
        attributed[procedure]=dict(charged_ns=sum(nodes[name]['end_ns']-nodes[name]['start_ns'] for name in charged),
            charged_roots=charged,all_required_timers=sorted(required))
    cohort=nodes[cohort_timer];duration=cohort['end_ns']-cohort['start_ns']
    intervals=sorted((node['start_ns'],node['end_ns']) for node in nodes.values() if node['parent_id']==cohort_timer)
    covered=0;cursor=cohort['start_ns']
    for start,end in intervals:
        covered+=max(0,end-max(cursor,start));cursor=max(cursor,end)
    return dict(schema='research/v2-cost-attribution',physical_cohort_wall_ns=duration,
        standalone_procedures=attributed,standalone_attributed_sum_ns=sum(v['charged_ns'] for v in attributed.values()),
        unattributed_cohort_overhead_ns=duration-covered,
        timer_inputs_sha256=canonical_hash(dict(timers=timers,procedures=procedures,cohort_timer=cohort_timer)),
        full_pipeline_cost_qualified=False,cost_claim_authorized=False,
        scope='Inclusive sequential standalone-worker clocks; shared physical work charged once per procedure')


def paired_timing_savings(blocks,*,seed,quality_risk_ratio_upper,iterations=20000,chunk_size=256):
    """Paired percentile uncertainty for fixed randomized timing blocks.

    Quality upper bounds are supplied, not authenticated. These mathematical
    conditions do not certify native imports, preprocessing, CPU precision,
    workload, power/cache policy or fresh processes; the frozen evaluator must.
    """
    if (not isinstance(blocks,list) or len(blocks)<30 or type(seed) is not int or not 0<=seed<2**63
            or type(iterations) is not int or iterations<2 or type(chunk_size) is not int or chunk_size<1
            or type(quality_risk_ratio_upper) not in (int,float) or not math.isfinite(quality_risk_ratio_upper)
            or quality_risk_ratio_upper<=0):raise ValueError('Thirty paired timing blocks, seed and finite positive quality bound required')
    rng=np.random.default_rng(seed64(seed,'paired-deployment-timing','order'))
    expected=rng.integers(0,2,size=len(blocks));a=[];b=[]
    for index,row in enumerate(blocks):
        if (not isinstance(row,dict) or set(row)!={'block_id','order','selected_ns','baseline_ns'}
                or type(row['block_id']) is not int or row['block_id']!=index
                or row['order']!=('selected_first' if expected[index]==0 else 'baseline_first')):
            raise ValueError('Timing block identity or frozen paired randomized order changed')
        a.append(_ns(row['selected_ns'],positive=True));b.append(_ns(row['baseline_ns'],positive=True))
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    selected=math.fsum(a/len(a));baseline=math.fsum(b/len(b));point=1-selected/baseline
    rng=np.random.default_rng(seed64(seed,'paired-deployment-timing','bootstrap'))
    savings=np.empty(iterations);differences=np.empty(iterations);index_hash=hashlib.sha256()
    for start in range(0,iterations,chunk_size):
        count=min(chunk_size,iterations-start);indices=rng.integers(0,len(blocks),size=(count,len(blocks)),dtype=np.int64)
        index_hash.update(indices.astype('<i8',copy=False).tobytes())
        ma=np.mean(a[indices],axis=1);mb=np.mean(b[indices],axis=1)
        savings[start:start+count]=1-ma/mb;differences[start:start+count]=np.mean((b-a)[indices],axis=1)
    interval=np.quantile(savings,[.025,.975],method='linear').tolist()
    paired_interval=np.quantile(differences,[.025,.975],method='linear').tolist()
    passes=interval[0]>.20 and paired_interval[0]>0 and quality_risk_ratio_upper<1.02
    return dict(schema='research/v2-paired-timing-savings',block_count=len(blocks),
        mean_selected_ns=selected,mean_baseline_ns=baseline,savings_fraction=point,
        savings_fraction_interval=interval,paired_time_saving_interval_ns=paired_interval,
        quality_risk_ratio_upper=quality_risk_ratio_upper,statistical_savings_conditions_pass=bool(passes),
        bootstrap_savings_fractions=savings,bootstrap_paired_time_savings_ns=differences,
        bootstrap_index_sha256=index_hash.hexdigest(),iterations=iterations,seed=seed,
        input_blocks_sha256=canonical_hash(dict(blocks=blocks)),cost_claim_authorized=False,
        native_workload_authenticated=False,protected_access_authorized=False,
        scope='Conditional timing-block percentile uncertainty; quality bound provenance and native workload remain external prerequisites')
