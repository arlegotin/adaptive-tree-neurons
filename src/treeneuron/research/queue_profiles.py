"""Authenticate real queue observations before using them in resource forecasts."""
import hashlib
import json
from pathlib import Path

from treeneuron import train
from treeneuron.config import canonical_hash
from .opportunity_resources import workload_manifest
from .workqueue import authenticated_queue


def collect_resource_profiles(directory, *, expected_plan_hash):
    root=Path(directory).resolve()
    with authenticated_queue(root,expected_plan_hash=expected_plan_hash) as (plan,ledger,results):
        manifest=workload_manifest(plan);jobs={j['job_id']:j for j in plan['attempts']}
        source=plan['identity']['source_sha256'];runtime=canonical_hash(plan['identity']['execution_identity'])
        latest={e['attempt_id']:e for e in ledger.events};profiles=[];unqualified=[]
        for name,result in results.items():
            value=result['value'];event=latest[name]
            if value['status']!='completed_eligible':continue
            reference=value.get('resource_observation')
            if reference is None:
                unqualified.append(dict(job_id=name,reason='No committed resource observation'));continue
            job=jobs[name];path=Path(reference['path'])
            parent=Path(event['result']['path']).parent
            if (path.is_symlink() or path.resolve()!=parent/'resource-observation.json'
                    or hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']):
                raise ValueError('Resource observation reference commitment differs')
            observed=json.loads(path.read_text());workload=manifest['jobs'][name]
            original={k:result[k] for k in ('files','directories')}
            original['files']=[f for f in original['files'] if f['path']!='resource-observation.json']
            if (observed.get('schema')!='research/v2-queue-resource-observation'
                    or observed['job_id']!=name or observed['job_sha256']!=canonical_hash(job)
                    or observed['source_sha256']!=source or observed['execution_identity_hash']!=runtime
                    or observed['attempt']!=result['attempt'] or observed['procedure_status']!=value['status']
                    or observed.get('T_accessed') is not False or observed.get('production_admission') is not False
                    or observed['raw_inventory']!=original):
                raise ValueError('Observed job/source/runtime/raw inventory differs')
            report=value.get('report',{})
            counter=report.get('executed_updates',0) if job['kind']=='parent' else (
                report.get('executed_stage_updates',0) if job['kind']=='fit' else 0)
            if (type(counter) is not int or observed['executed_updates']!=counter
                    or not 0<=counter<=workload['maximum_updates']):
                raise ValueError('Observed native execution counter differs')
            for key in ('handler_and_observation_ns','peak_process_rss_bytes'):
                if type(observed[key]) is not int or observed[key]<=0:raise ValueError('Invalid resource measurement')
            if (type(observed['estimated_checkpoint_transaction_bytes']) is not int
                    or observed['estimated_checkpoint_transaction_bytes']<0
                    or observed['handler_and_observation_ns']>event['elapsed_ns']
                    or observed['retained_bytes_before_observation']!=sum(f['bytes'] for f in original['files'])):
                raise ValueError('Observed time/storage estimate contradicts its retained evidence')
            if workload['maximum_updates'] and counter==0:
                unqualified.append(dict(job_id=name,reason='No applied optimization updates measured'));continue
            profiles.append(dict(schema='research/v2-queue-resource-profile',source_sha256=source,
                execution_identity_hash=runtime,workload_hash=workload['workload_hash'],status='completed_eligible',
                executed_updates=counter,elapsed_ns=event['elapsed_ns'],
                retained_bytes=sum(f['bytes'] for f in result['files'])+Path(event['result']['path']).stat().st_size,
                peak_rss_bytes=observed['peak_process_rss_bytes'],
                max_checkpoint_transaction_bytes=observed['estimated_checkpoint_transaction_bytes'],
                provenance=dict(job_id=name,job_sha256=canonical_hash(job),attempt=result['attempt'],
                    terminal_event_hash=event['event_hash'],result=event['result'],observation=reference,
                    handler_and_observation_ns=observed['handler_and_observation_ns'],
                    elapsed_scope='cumulative charged queue job; preparation/publication and earlier paid failures included',
                    memory_scope=observed['memory_scope'],transaction_estimator=observed['transaction_estimator'])))
        covered={p['workload_hash'] for p in profiles};missing=sorted(set(manifest['classes'])-covered)
        invocations=[]
        for path in sorted(root.glob('invocation-*.json')):
            if path.is_symlink() or not path.is_file():raise ValueError('Unsafe invocation receipt')
            raw=path.read_bytes();row=json.loads(raw)
            if type(row.get('invocation_ns')) is not int or row['invocation_ns']<=0:
                raise ValueError('Invalid enclosing invocation clock')
            invocations.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),**row))
        if ledger.events and not invocations:raise ValueError('Missing enclosing invocation receipt')
        return dict(schema='research/v2-queue-resource-collection',queue_plan_hash=expected_plan_hash,
            producer_source_sha256=source,execution_identity_hash=runtime,collector_source_sha256=train.source_hash(),
            profiles=profiles,missing_workload_hashes=missing,unqualified_jobs=unqualified,
            status='incomplete_qualification' if missing else 'profiles_complete',
            ledger_event_hash=ledger.events[-1]['event_hash'] if ledger.events else '0'*64,
            charged_cumulative_job_ns=ledger.charged_ns,invocations=invocations,
            time_scope='Job profiles and enclosing run_queue invocations; process startup/imports and invocation publication remain separate',
            worker_loss_upper_bounds_present=any(e.get('cost_precision')=='wall_time_upper_bound_after_worker_loss' for e in ledger.events),
            memory_scope='process lifetime own peak RSS; not incremental or aggregate machine RAM',
            transaction_scope='conservative largest retained payload/metadata/history estimate; not a measured write transaction',
            production_admission=False,scientific_quality_qualification=False,T_accessed=False)
