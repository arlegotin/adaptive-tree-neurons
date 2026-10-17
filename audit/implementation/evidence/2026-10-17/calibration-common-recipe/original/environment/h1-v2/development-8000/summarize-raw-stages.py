"""Stdlib-only final counts from retained stage events, not inherited history."""
import json,hashlib,time
from pathlib import Path
base=Path('environment/h1-v2/development-8000');root=Path('artifacts/h1-v2/development-8000')
result=dict(status='raw_stage_count_receipt',scope='authoritative per-stage examples/bytes from original stage events; no training/prediction',
    note='The executing wrapper stage.examples helper filters inherited checkpoint histories by global update and can include the parent executed-after-selected tail. Preserve original measurements; these direct stage-event counts supersede that diagnostic field only. Executed updates, selected states, curves, pairing, predictions and costs are unchanged.',
    problems=[])
for letter in ('a','b'):
    directory=root/f'h1-v2-primary-{letter}-8000';measurement=directory/'measurements.json'
    d=json.loads(measurement.read_text());assert d['status']=='completed'
    problem=dict(problem=letter,measurement_path=str(measurement),measurement_sha256=hashlib.sha256(measurement.read_bytes()).hexdigest(),
        source_sha256=d['package_sha256'],config_sha256=d['config_sha256'],roles=[],byte_buckets={})
    for role,r in d['methods'].items():
        for stage_index,s in enumerate(r['stages']):
            p=directory/s['path']/'events.jsonl';updates=[];evaluations=[];row_counts={};update_bytes=0
            with p.open('rb') as f:
                for line in f:
                    event=json.loads(line)
                    if event['type']=='update':
                        assert len(event['row_ids'])==event['rows']
                        updates.append(event);update_bytes+=len(line);row_counts[str(event['rows'])]=row_counts.get(str(event['rows']),0)+1
                    elif event['type']=='evaluation':evaluations.append(event)
            assert len(updates)==s['executed_updates']
            bad=0
            for e in reversed(evaluations):
                if e['improved']:break
                bad+=1
            assert bad==s['final_consecutive_nonimprovements']
            problem['roles'].append(dict(role=role,method_id=r['method_id'],stage_index=stage_index,
                path=str(p),events_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),events_bytes=p.stat().st_size,
                update_event_bytes=update_bytes,updates=len(updates),examples=sum(e['rows'] for e in updates),
                helper_examples=s['examples'],actual_row_counts=row_counts,selected_update=s['selected_update'],
                max_updates=s['max_updates'],hit_ceiling=s['hit_ceiling'],improving_at_ceiling=s['improving_at_ceiling'],
                final_nonimproving_evaluations=bad,early_stop_patience_reached=bad>=d['config']['training']['patience']))
    for p in directory.rglob('*'):
        if not p.is_file():continue
        if p.name=='events.jsonl':bucket='raw_events'
        elif p.name.endswith('.meta.json'):bucket='checkpoint_metadata'
        elif p.suffix=='.pt':bucket='tensor_payloads'
        elif p.suffix=='.npz':bucket='npz_data_predictions'
        else:bucket='other'
        problem['byte_buckets'][bucket]=problem['byte_buckets'].get(bucket,0)+p.stat().st_size
    problem.update(final_retained_bytes=sum(problem['byte_buckets'].values()),
        peak_rss_bytes=d['peak_rss_bytes'],enclosing_ns=d['enclosing_ns'],pairing_audit=d['pairing_audit'],sham_max_abs=d['sham_max_abs'])
    result['problems'].append(problem)
out=base/'raw-stage-summary.json';out.write_text(json.dumps(result,indent=2));print(out)
