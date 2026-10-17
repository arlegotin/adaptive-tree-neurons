"""Retained-pair scalar/readout export only; no source imports or resampling."""
import hashlib,json,math,statistics
from pathlib import Path

base=Path('environment/h1-v2/genuine-pilot')
t=json.loads((base/'terminal-summary.json').read_text())
r=json.loads((base/'readout.json').read_text())
root=Path(t['attempt_root'])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
teachers=[]
for teacher in t['teachers']:
    records=[];pairs=[]
    for repetition in teacher['repetitions']:
        p=root/f"teacher-{teacher['index']}"/f"repetition-{repetition['repetition']}"/'pair.json'
        row=json.loads(p.read_text());records.append(row)
        pairs.append(dict(repetition=row['repetition'],pair_path=str(p),pair_sha256=sha(p),
            h1_mean=statistics.fmean(row['h1_rows']),static_mean=statistics.fmean(row['static_rows']),
            optimizer_seed=row['optimizer_seed'],evaluation_rows=len(row['h1_rows'])))
    assert len(records)==2
    n=len(records[0]['h1_rows']);assert all(len(x['h1_rows'])==n for x in records)
    row_mean=[(a+b)/2 for a,b in zip(records[0]['h1_rows'],records[1]['h1_rows'])]
    item=dict(teacher_index=teacher['index'],teacher_id=teacher['teacher_id'],repetitions=2,evaluation_rows=n,
        h1_mean=statistics.fmean(p['h1_mean'] for p in pairs),static_mean=statistics.fmean(p['static_mean'] for p in pairs),
        within_teacher_H1_variance=statistics.variance(p['h1_mean'] for p in pairs),
        within_teacher_static_mean_variance=statistics.variance(p['static_mean'] for p in pairs),
        paired_row_H1_variance=statistics.variance(row_mean),exact_pair_means=pairs)
    assert abs(item['h1_mean']-teacher['h1_mean'])<1e-12
    assert abs(item['static_mean']-teacher['static_mean'])<1e-12
    teachers.append(item)
summary8=json.loads((base/'summary-8-call1.json').read_text())
for i,item in enumerate(teachers[:8]):
    assert abs(item['h1_mean']-summary8['teacher_means'][i])<1e-12
    for key,source in [('within_teacher_H1_variance','within_teacher_variances'),('paired_row_H1_variance','paired_row_variances')]:
        assert abs(item[key]-summary8['diagnostics'][source][i])<1e-12
expected={(i,j) for i in range(16) for j in range(2)}
actual={(a['teacher_index'],a['repetition']) for a in t['attempts'] if a['status']=='completed'}
missing=sorted(expected-actual);assert missing==[(15,0),(15,1)]
pause=json.loads((base/'user-pause.json').read_text());resume=json.loads((base/'user-resume.json').read_text())
pub=max(r['raw_publications'],key=lambda x:x['nanoseconds'])
assert pub['attempts']==30 and pub['completed_attempts']==29
before=next(x for x in r['resource_observations'] if x['boundary'].endswith('teacher-14/repetition-0') and x['boundary'].startswith('after '))
after=next(x for x in r['resource_observations'] if x['boundary'].endswith('teacher-14/repetition-1') and x['boundary'].startswith('before '))
stages={s['path']:s for a in t['attempts'] if a['status']=='completed' for s in [a['parent']]+[s for m in a['methods'].values() for s in m['stages']]}
value=dict(scope='Descriptive retained original records only; 15 complete teachers are not the completed fixed16 pilot. No new bootstrap/design or clock-policy adjustment.',
    source_execution_sha256=r['source']['execution_sha256'],recipe_sha256=r['recipe_sha256'],
    original_raw_sha256=t['raw_sha256'],raw_seal_sha256=sha(root/'raw-seal.json'),
    original_status=t['status'],original_failure=t['failure'],failed_pairs=t['failed_pairs'],
    completed_pairs=len(actual),accepted_records=t['accepted_records'],missing_pairs=[dict(teacher_index=i,repetition=j) for i,j in missing],
    generated_unstarted_teacher=r['teachers'][-1],teachers=teachers,
    descriptive_means_only=dict(independent_teachers=len(teachers),h1=statistics.fmean(x['h1_mean'] for x in teachers),
        static=statistics.fmean(x['static_mean'] for x in teachers)),
    raw_publication_accounting=dict(count=len(r['raw_publications']),total_written_bytes=sum(x['bytes'] for x in r['raw_publications']),
        original_total_perf_seconds=sum(x['nanoseconds'] for x in r['raw_publications'])/1e9,max_publication=pub,
        limitation='Original max publication includes the stopped-process interval; it is not serialization throughput.'),
    clock_pause_evidence=dict(controller_pause_receipt_sha256=sha(base/'user-pause.json'),controller_resume_receipt_sha256=sha(base/'user-resume.json'),
        controller_UTC_pause_seconds=resume['user_paused_utc_seconds'],controller_perf_pause_seconds=resume['user_paused_perf_seconds'],
        producer_after_R0={k:before[k] for k in ('boundary','utc','elapsed_seconds')},producer_before_R1={k:after[k] for k in ('boundary','utc','elapsed_seconds')},
        producer_between_pair_perf_seconds=after['elapsed_seconds']-before['elapsed_seconds'],
        interpretation='The producer measured a 40033.386s raw publication spanning the controller pause and between these producer boundaries. Controller and producer values remain separate; no pause-subtracted cap or cost is asserted.'),
    distinct_full_stage_count=len(stages),hit_ceiling_stages=[s for s in stages.values() if s['hit_ceiling']],
    final_nonimprovement_counts=sorted(set(s['final_consecutive_nonimprovements'] for s in stages.values())),
    canonical_summary_published=(root/'summary.json').exists(),canonical_design_published=(root/'design_decision.json').exists(),numerical_STOP=True)
out=base/'retained-scalars-and-clocks.json';assert not out.exists()
out.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:value[k] for k in ('completed_pairs','missing_pairs','descriptive_means_only','distinct_full_stage_count','final_nonimprovement_counts')}))
