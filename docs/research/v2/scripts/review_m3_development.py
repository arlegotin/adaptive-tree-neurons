"""Export a complete M3 development review; never train, choose on T or admit a remedy."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from treeneuron.analysis.primary_v2 import analyze_teacher_risks
from treeneuron.config import canonical_hash
from treeneuron.research.joint_analysis import summarize_joint_predictions,summarize_joint_local_checks
from treeneuron.research.ledger import AttemptLedger,write_new_json
from treeneuron.research.seeds import seed64


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--expected-audit-sha256',required=True)
    parser.add_argument('--expected-plan-hash',required=True)
    parser.add_argument('--expected-ledger-hash',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter_ns()
    root=args.study.resolve();audit_path=args.audit.resolve()
    if args.output.resolve().is_relative_to(root):raise ValueError('Review cannot write into the producer queue')
    if audit_path.is_symlink() or sha(audit_path)!=args.expected_audit_sha256:
        raise ValueError('Independent prediction audit commitment differs')
    audit=json.loads(audit_path.read_text());summary=summarize_joint_predictions(audit)
    plan_path=root/'plan.json';plan=json.loads(plan_path.read_text())
    if (plan['plan_hash']!=args.expected_plan_hash
            or canonical_hash({k:v for k,v in plan.items() if k!='plan_hash'})!=args.expected_plan_hash):
        raise ValueError('M3 plan commitment differs')
    ledger=AttemptLedger.resume(root,plan);latest={v['attempt_id']:v for v in ledger.events}
    jobs={v['job_id']:v for v in plan['attempts']}
    if (set(latest)!=set(jobs) or any(e['status']!='completed_eligible' for e in latest.values())
            or ledger.events[-1]['event_hash']!=args.expected_ledger_hash
            or summary['ledger_event_hash']!=args.expected_ledger_hash
            or summary['queue_plan_hash']!=args.expected_plan_hash):
        raise ValueError('This review requires the complete eligible admitted M3 cohort')
    refs={};local=[];fits=[];traces=[];extension=None
    for name,job in jobs.items():
        if job['kind'] not in ('local_model','fit','placement','extension_decision'):continue
        ref=latest[name]['result'];path=Path(ref['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path)!=ref['sha256']:
            raise ValueError('Committed job marker differs')
        refs[str(path)]=ref['sha256'];outer=json.loads(path.read_text());value=outer['value'];p=job['payload']
        if outer['job_id']!=name or outer['job_hash']!=canonical_hash(job):raise ValueError('Wrong job identity')
        if job['kind']=='extension_decision':extension=value['decision'];continue
        labels={k:p[k] for k in ('teacher_index','repetition','role')}
        if job['kind']=='local_model':
            diagnostic={k:v for k,v in value['diagnostic'].items() if k not in ('solution','linear_solve')}
            diagnostic['linear_solve']={k:v for k,v in value['diagnostic']['linear_solve'].items() if k!='singular_values'}
            local.append(dict(labels,diagnostic=diagnostic))
        elif job['kind']=='fit':
            report=value['report'];fields=('selected_stage_update','executed_stage_updates','selected_absolute_update',
                'parent_update','selected_g_mse','selected_ids','total_parameters','effective_count')
            fits.append(dict(labels,**{k:report.get(k) for k in fields}))
        else:
            path=Path(value['selection_path'])
            if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path)!=value['selection_file_sha256']:
                raise ValueError('Committed selection bytes differ')
            refs[str(path)]=value['selection_file_sha256'];trace=json.loads(path.read_text())
            if canonical_hash(trace)!=value['trace_sha256']:raise ValueError('Selection trace identity differs')
            fields=('config','strategy','selected_ids','selection_order','predicted_gain','parent_rank',
                'parent_span_novelty_fraction','parent_solve_relative_residual','gram_asymmetry',
                'gate_gram_condition_number','effective_count','eligible_count','fallback_history')
            traces.append(dict(labels,**{k:trace[k] for k in fields}))
    expected={(t,r,role) for t in summary['teacher_ids'] for r in (0,1) for role in summary['role_means']}
    if {(v['teacher_index'],v['repetition'],v['role']) for v in fits}!=expected or len(fits)!=len(expected):
        raise ValueError('Missing/duplicate final fit metadata')
    review=summarize_joint_local_checks(local,teacher_ids=summary['teacher_ids'])
    comparisons={}
    for role in summary['role_means']:
        if role=='joint':continue
        comparisons[role]={}
        for target in ('V','E'):
            records=[]
            for row in audit['rows']:
                p=row['payload']
                if p['role'] not in ('joint',role):continue
                records.append(dict(teacher_id=str(p['teacher_index']),repetition=p['repetition'],
                    role='selected' if p['role']=='joint' else role,status=row['status'],
                    risk=row['metrics']['selected'][target]['normalized_mse']))
            result=analyze_teacher_risks([str(t) for t in summary['teacher_ids']],records,
                seed=seed64(2026101801,'M3-post-cohort-descriptive-review',role,target),baseline_role=role)
            comparisons[role][target]={k:v for k,v in result.items() if k not in ('bootstrap_mean_log_benefits','teachers')}
            comparisons[role][target]['scope']='Descriptive development intervals after the one declared precision extension; no confirmatory error-control claim'
    stage={}
    for role in summary['role_means']:
        rows=[v for v in fits if v['role']==role];stages=[v['selected_stage_update'] for v in rows]
        if any(type(s) is not int or s<0 or s>8000 for s in stages):raise ValueError('Invalid selected update')
        if any(v['executed_stage_updates']!=8000 for v in rows):raise ValueError('Changed fixed continuation budget')
        stage[role]=dict(fits=len(rows),stage_zero=sum(s==0 for s in stages),selected_min=min(stages),
            selected_max=max(stages),selected_mean=math.fsum(stages)/len(stages),executed_each=8000)
    trace_map={(r['teacher_index'],r['repetition'],r['role']):r for r in traces}
    overlaps=[]
    for t in summary['teacher_ids']:
        for repeat in (0,1):
            joint=trace_map[t,repeat,'joint']
            for role in ('local_singleton','unconditioned','pure_span'):
                other=trace_map[t,repeat,role]
                overlaps.append(dict(teacher_index=t,repetition=repeat,reference=role,
                    shared_ids=len(set(joint['selected_ids'])&set(other['selected_ids'])),
                    identical=set(joint['selected_ids'])==set(other['selected_ids'])))
    export=dict(schema='research/v2-M3-compact-review-input',summary=summary,local_checks=local,
        fits=fits,traces=traces,extension=extension,source_file_commitments=refs,
        independent_audit_sha256=args.expected_audit_sha256,queue_plan_hash=args.expected_plan_hash,
        ledger_event_hash=args.expected_ledger_hash,T_accessed=False)
    result=dict(schema='research/v2-M3-development-review',summary=summary,local_model=review,
        descriptive_comparisons=comparisons,fit_stage_summary=stage,set_overlaps=overlaps,
        terminal_statuses=dict(Counter(v['status'] for v in latest.values())),extension=extension,
        Gate_B_decided=False,remedy_admitted=False,T_accessed=False,
        evidence_scope='Prior independent retained-prediction arithmetic plus rehashed job/selection metadata. Saved producer local diagnostics are summarized, not independently replayed. No additional fitted model or protected evaluation.')
    args.output.mkdir(parents=True,exist_ok=False)
    write_new_json(args.output/'input.json',export);write_new_json(args.output/'review.json',result)
    write_new_json(args.output/'invocation.json',dict(argv=sys.argv,cwd=str(Path.cwd()),script_sha256=sha(__file__),
        analysis_module_sha256=sha(Path(sys.modules[summarize_joint_local_checks.__module__].__file__)),
        input_sha256=sha(args.output/'input.json'),review_sha256=sha(args.output/'review.json'),
        plan_file_sha256=sha(plan_path),ledger_file_sha256=sha(root/'events.jsonl'),
        audit_sha256=sha(audit_path),elapsed_ns=time.perf_counter_ns()-started,T_accessed=False))
    print(json.dumps(dict(teachers=summary['teachers'],fits=len(fits),local_checks=len(local),
        output=str(args.output),Gate_B_decided=False),sort_keys=True))


if __name__=='__main__':main()
