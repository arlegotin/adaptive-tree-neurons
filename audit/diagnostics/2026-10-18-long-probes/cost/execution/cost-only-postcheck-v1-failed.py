"""Read-only hash/completeness checks; no model load, evaluation or fitting."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path.cwd()
ENV=ROOT/'environment/long-probes-20261018'
COST=ROOT/'artifacts/long-probes-20261018/cost'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sealed(path):
    assert read(path.with_name(path.stem+'-seal.json'))=={'sha256':sha(path)}
    return read(path)

ledger=sealed(COST/'members.json')
lock=sealed(COST/'study-lock.json')
decision=sealed(COST/'resource-decision.json')
assert ledger['status']=='completed' and ledger['phase']=='cost' and ledger['failures']==[]
assert ledger['expected_members']==[dict(teacher_index=0,repetition=0)]
assert len(ledger['members'])==1
entry=ledger['members'][0]
assert entry['status']=='completed'
pairroot=COST/entry['pair_path']
pair=read(pairroot/'pair.json')
roles={'keep','top','random','greedy','greedy_long','widen_upper','ordinary_lower','ordinary_upper','static_partial'}
assert pair['status']=='completed' and set(pair['methods'])==roles and pair['failures']==[]
assert pair['source_sha256']=='089b03d13e1950b8e0c1ff54a147a121cf78db66acaaa2dcedd9ebf29afc3543'
assert not pair['T_accessed'] and not pair['V_accessed']
assert pair['pairing_audit']['passed'] and pair['pairing_audit']['warm_forks']==6
assert pair['long_probe_audit']['passed'] and pair['methods']['greedy_long']['original_parent_audit']['passed']
for name,key in [('pair.json','pair_sha256'),('E-predictions.npz','predictions_sha256'),('artifacts.json','artifacts_sha256')]:
    assert sha(pairroot/name)==entry[key]
for key,path in [('observations_sha256',COST/entry['observations_path']),
        ('teacher_sha256',(COST/entry['observations_path']).parent/'teacher.json'),
        ('splits_sha256',(COST/entry['observations_path']).parent/'splits.json')]:
    assert sha(path)==entry[key]
artifacts=read(pairroot/'artifacts.json')
actual={str(p.relative_to(pairroot)) for p in pairroot.rglob('*') if p.is_file() and p.name!='artifacts.json'}
assert set(artifacts)==actual
for rel,digest in artifacts.items():
    assert sha(pairroot/rel)==digest,rel
assert decision['measurement_sha256']==sha(COST/'members.json')
assert decision['measurement_lock_sha256']==sha(COST/'study-lock.json')
assert decision['provenance']==ledger['provenance']==lock['provenance']
searches={}
raw_searches={}
for name,key,mode,allowance in [('short','greedy_report','fixed_updates',320),
        ('long','greedy_long_report','patience',8000)]:
    path=Path(pair[key]); search=read(path)
    assert path.is_relative_to(pairroot)
    assert search['complete'] and search['not_attempted']==[] and search['stopping_mode']==mode
    eligible=search['eligible_ids']; trials=search['trials']
    assert len(eligible)==64 and len(trials)==65
    assert Counter(t['unit_id'] for t in trials)==Counter([None,*eligible])
    assert all(t['outcome']=='completed' and t['allowance']==allowance for t in trials)
    assert all(t['stopping_mode']==mode for t in trials)
    assert search['effective_count']==16
    assert all(t['executed_updates']==320 and t['hit_ceiling'] and t['stopping_reason']=='ceiling'
        for t in trials) if name=='short' else all(t['executed_updates']<=8000 and
            t['stopping_reason'] in ('patience','ceiling') for t in trials)
    raw_searches[name]=search
    searches[name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),complete=True,
        mode=mode,allowance=allowance,eligible_count=len(eligible),trial_count=len(trials),
        outcomes=dict(Counter(t['outcome'] for t in trials)),
        stopping_reasons=dict(Counter(t['stopping_reason'] for t in trials)),
        ceiling_hits=sum(t['hit_ceiling'] for t in trials),
        executed_updates_total=sum(t['executed_updates'] for t in trials),
        executed_updates_min=min(t['executed_updates'] for t in trials),
        executed_updates_max=max(t['executed_updates'] for t in trials),
        nanoseconds=search['nanoseconds'],
        trials=[{k:t[k] for k in ('unit_id','candidate','allowance','executed_updates',
            'selected_stage_updates','stopping_reason','hit_ceiling','outcome','run_id','path')} for t in trials])
for key in ('parent_checkpoint_sha256','parent_identity_hashes','parent_state_sha256','score_hash','eligible_ids'):
    assert raw_searches['short'][key]==raw_searches['long'][key]
assert raw_searches['long']['parent_checkpoint_sha256']==pair['parent_checkpoint_sha256']
roles_summary={name:dict(status=m['status'],total_parameters=m['total_parameters'],
        enclosing_ns=m['enclosing_ns'],
        stages=[{k:s[k] for k in ('run_id','path','max_updates','executed_updates','selected_update','hit_ceiling')}
            for s in m['stages']]) for name,m in pair['methods'].items()}
assert all(m['status']=='completed' for m in pair['methods'].values())
outer=read(ENV/'cost-only-outer-measurement.json')
files=sorted(p for p in COST.rglob('*') if p.is_file())
assert len(files)==outer['final_post_publication_file_count']
assert sum(p.stat().st_size for p in files)==outer['final_post_publication_retained_bytes']
stop=read(ENV/'cost-only-stop.json')
assert stop['returncode']==0 and stop['tracked_unchanged'] and stop['prior_evidence_unchanged']
obs=ledger['resource_observations']
summary=dict(status='passed',member=dict(teacher_index=0,repetition=0,teacher_id=entry['teacher_id'],
        pair_path=entry['pair_path'],pair_sha256=entry['pair_sha256'],
        observations_sha256=entry['observations_sha256'],parent_run_id=pair['parent_run_id'],
        parent_checkpoint_sha256=pair['parent_checkpoint_sha256'],
        optimizer_seed=pair['optimizer_seed']),
    provenance=ledger['provenance'],costs=dict(outer_elapsed_ns=outer['outer_elapsed_ns'],
        internal_enclosing_ns=ledger['enclosing_ns'],member_enclosing_ns=entry['enclosing_ns'],
        pair_enclosing_ns=pair['enclosing_ns'],nested=pair['costs'],
        child_peak_rss_bytes=outer['child_peak_rss_bytes'],internal_peak_rss_bytes=ledger['peak_rss_bytes'],
        final_post_publication_retained_bytes=outer['final_post_publication_retained_bytes'],
        internal_retained_bytes=ledger['retained_bytes'],file_count=len(files)),
    searches=searches,roles=roles_summary,capacity=pair['capacity'],
    pairing_audit=pair['pairing_audit'],long_probe_audit=pair['long_probe_audit'],
    original_parent_audit=pair['methods']['greedy_long']['original_parent_audit'],
    resources=dict(observation_count=len(obs),
        available_percent_min=min(o['available_percent'] for o in obs if o['available_percent'] is not None),
        swapouts_min=min(o['swapouts'] for o in obs if o['swapouts'] is not None),
        swapouts_max=max(o['swapouts'] for o in obs if o['swapouts'] is not None),
        counter_failures=[o['counter_reason'] for o in obs if o['counter_reason']]),
    pair_artifact_hashes_verified=len(artifacts),seals_verified=3,
    failures=[],automatic_retries=0,cohort_run=False,receipt_producer_rerun=False,
    inspection_note='A read-only exploratory inspection first matched models/greedy.json (a descriptor) and raised KeyError trials. The corrected check uses exact pair.greedy_report/greedy_long_report paths. No model execution, mutation or retry resulted.')
with (ENV/'cost-only-verification.json').open('x') as stream:
    json.dump(summary,stream,sort_keys=True,indent=2); stream.write('\n')
print(json.dumps(dict(status=summary['status'],costs=summary['costs'],resources=summary['resources'],
    short={k:v for k,v in searches['short'].items() if k!='trials'},
    long={k:v for k,v in searches['long'].items() if k!='trials'},
    pair_artifact_hashes_verified=len(artifacts)),sort_keys=True,indent=2))
