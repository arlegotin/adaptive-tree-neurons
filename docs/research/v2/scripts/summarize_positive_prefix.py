"""Freeze the earliest closed M1 positive event prefix; raw audit stays pending."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

from treeneuron.config import canonical_hash, canonical_json
from treeneuron.research.control_analysis import positive_sensitivity
from treeneuron.research.ledger import AttemptLedger, TERMINAL


ROLES=('keep','top','random','oracle_current','oracle_true')


def summarize(plan, raw, expected_plan_hash):
    if (plan.get('plan_hash')!=expected_plan_hash or expected_plan_hash!=canonical_hash(
            {k:v for k,v in plan.items() if k!='plan_hash'})
            or plan.get('protected_labels_accessed') is not False):
        raise ValueError('Expected immutable nonprotected M1 plan')
    attempts={a['attempt_id']:a for a in plan['attempts']}
    if len(attempts)!=len(plan['attempts']):raise ValueError('Duplicate attempt IDs')
    positive={key:a for key,a in attempts.items() if a['population']=='positive'}
    expected={(g,s,t,r,role) for g in (.5,2.) for s in (0.,.1)
        for t in range(4) for r in range(2) for role in ROLES}
    coordinates=[(a['gamma'],a['sigma'],a['teacher_index'],a['repetition'],a['role']) for a in positive.values()]
    if len(coordinates)!=160 or set(coordinates)!=expected:raise ValueError('Expected all160 positive slots')
    validator=object.__new__(AttemptLedger);validator.ids=tuple(attempts);validator.events=[]
    previous='0'*64;last={};prefix=[]
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):break
        event=json.loads(line);payload={k:v for k,v in event.items() if k!='event_hash'}
        if event['previous_hash']!=previous or canonical_hash(payload)!=event['event_hash']:
            raise ValueError('Event commitment mismatch')
        validator._validate(payload);validator.events.append(event)
        previous=event['event_hash'];last[event['attempt_id']]=event;prefix.append(line)
        if all(key in last and last[key]['status'] in TERMINAL-{'interrupted_recoverable'} for key in positive):break
    else:pass
    if not all(key in last and last[key]['status'] in TERMINAL-{'interrupted_recoverable'} for key in positive):
        raise ValueError('Positive subset is not closed')
    rows=[]
    for key,a in positive.items():
        event=last[key];row=dict(a,status=event['status'])
        if event['status']=='completed_eligible':
            row['metrics']=event['metrics']
            for split in ('V','E'):
                value=row['metrics'][split]['normalized_mse']
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError('Invalid event risk')
            row['V_NMSE']=row['metrics']['V']['normalized_mse']
        rows.append(row)
    cells=[]
    for g in (.5,2.):
        for s in (0.,.1):
            index={(r['teacher_index'],r['repetition'],r['role']):r for r in rows if r['gamma']==g and r['sigma']==s}
            missing=[{k:r[k] for k in ('teacher_index','repetition','role','status')} for r in index.values()
                if r['status']!='completed_eligible']
            teachers=[];means=None
            if not missing:
                for t in range(4):
                    teachers.append(dict(teacher_index=t,risk={split:{role:math.fsum(
                        index[t,r,role]['metrics'][split]['normalized_mse'] for r in range(2))/2
                        for role in ROLES} for split in ('V','E')}))
                means={split:{role:math.fsum(t['risk'][split][role] for t in teachers)/4
                    for role in ROLES} for split in ('V','E')}
            cells.append(dict(gamma=g,sigma=s,base_instances=4,repeats=2,mean_risk=means,
                teachers=teachers,missing_coordinates=missing,information_advantaged_roles=['oracle_current','oracle_true']))
    prefix=b''.join(prefix)
    return dict(schema='research/v2-provisional-positive-event-arithmetic',plan_hash=expected_plan_hash,
        producer_source_sha256=plan['source_sha256'],prefix_event_count=len(validator.events),
        prefix_event_hash=previous,prefix_file_sha256=hashlib.sha256(prefix).hexdigest(),
        status_counts=dict(collections.Counter(r['status'] for r in rows)),rows=rows,cells=cells,
        positive_sensitivity=positive_sensitivity(rows),positive_base_instances_reused_across_cells=True,
        independent_raw_audit_pending=True,Gate_A_decided=False,efficacy_claim=False,T_accessed=False,
        scope='Earliest complete positive event subset only; no raw prediction/model authentication or confidence claim'),prefix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True,type=Path)
    parser.add_argument('--expected-plan-hash',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    plan_bytes=(args.study/'plan.json').read_bytes();plan=json.loads(plan_bytes)
    summary,prefix=summarize(plan,(args.study/'events.jsonl').read_bytes(),args.expected_plan_hash)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'plan.json').write_bytes(plan_bytes)
    (args.output/'event-prefix.jsonl').write_bytes(prefix)
    (args.output/'summary.json').write_text(canonical_json(summary)+'\n')
    import treeneuron.config as config_module
    import treeneuron.research.control_analysis as analysis_module
    import treeneuron.research.ledger as ledger_module
    paths=[Path(__file__),Path(config_module.__file__),Path(analysis_module.__file__),Path(ledger_module.__file__)]
    receipt=dict(schema='research/v2-provisional-event-invocation',argv=sys.argv,cwd=str(Path.cwd()),
        python=sys.version,executable=sys.executable,platform=platform.platform(),observed_utc_ns=time.time_ns(),
        analyst_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        helper_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()},T_accessed=False)
    (args.output/'invocation.json').write_text(canonical_json(receipt)+'\n')
    print(json.dumps(dict(prefix_event_count=summary['prefix_event_count'],status_counts=summary['status_counts'],
        provisional_sensitivity=summary['positive_sensitivity']),indent=2))


if __name__=='__main__':main()
