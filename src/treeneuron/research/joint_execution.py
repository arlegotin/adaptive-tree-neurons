"""Authenticated M3 qualification/admission and exact source-bound queue execution."""
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys

import numpy as np

from treeneuron import train
from treeneuron.checkpoint import _descriptor,load_checkpoint
from treeneuron.config import canonical_hash,canonical_json
from .identity import execution_identity
from .joint_graph import compile_joint_graph
from .joint_observation import ObservedJointWorkers
from .joint_qualification import build_joint_qualification_plan,JointQualificationWorkers,workload_key
from .joint_resources import forecast_joint_resources,joint_resource_decision
from .ledger import write_new_json
from .opportunity_data import load_fitting_data
from .opportunity_execution import live_resources
from .runner import _threads
from .workqueue import authenticated_queue,run_queue


def _runtime(identity):
    path=Path(identity['study_config']['base_experiment']['execution']['backend_report'])
    if path.is_symlink() or not path.is_file():raise ValueError('M3 requires its real CPU backend receipt before execution')
    if (identity['source_sha256']!=train.source_hash()
            or execution_identity(identity['study_config'])!=identity['execution_identity']):
        raise ValueError('M3 source/runtime/backend receipt changed')


def _guard(output,transaction=128*1024**2):
    rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    if rss>3*1024**3:raise MemoryError('M3 own-worker peak exceeded the qualified3GiB RSS track')
    if shutil.disk_usage(output).free<512*1024**2+transaction:
        raise OSError('M3 checkpoint transaction margin exhausted; preserve and resume the immutable queue')


def run_joint_qualification(study,output,*,resume=False):
    target=compile_joint_graph(study);q=build_joint_qualification_plan(target);identity=q['identity']
    _runtime(identity);_threads();output=Path(output).resolve()
    def check(value):
        if value!=identity:raise ValueError('M3 qualification identity changed')
        _runtime(identity)
    def before(job,events):
        live=live_resources(output);memory=live['available_memory_bytes']
        reason=('insufficient_live_disk' if live['free_disk_bytes']<2*1024**3 else
            'unknown_live_memory' if memory is None else 'insufficient_live_memory' if memory<4*1024**3
            else 'ready_for_complete_qualification_job')
        return dict(admitted=reason=='ready_for_complete_qualification_job',reason=reason,
            required_disk_bytes=2*1024**3,required_memory_bytes=4*1024**3,live_observation=live,
            scope='Conservative engineering bootstrap admission; no wall-time deadline or scientific M3 admission')
    guard=lambda:_guard(output)
    native=JointQualificationWorkers(target,q,guard=guard)
    result=run_queue(q,output,ObservedJointWorkers(q,guard=guard,handler=native),
        check_identity=check,resume=resume,before_start=before)
    if result['status']=='completed':
        collected=collect_joint_profiles(output,target)
        path=output/'qualified-profiles.json'
        if path.exists():
            if json.loads(path.read_text())!=collected:raise ValueError('Existing qualification collection changed')
        else:write_new_json(path,collected)
    return result


def _committed_observation(value,root,job,source,runtime):
    ref=value.get('resource_observation',{});path=Path(ref.get('path','')).resolve()
    if (not path.is_relative_to(root) or path.is_symlink() or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest()!=ref.get('sha256')):
        raise ValueError('Missing/authentication-failed native resource observation')
    o=json.loads(path.read_text())
    if (o.get('schema')!='research/v2-M3-resource-observation' or o.get('job_id')!=job['job_id']
            or o.get('job_sha256')!=canonical_hash(job) or o.get('source_sha256')!=source
            or o.get('execution_identity_hash')!=runtime or o.get('procedure_status')!=value['status']
            or o.get('production_admission') is not False or o.get('T_accessed') is not False):
        raise ValueError('Native resource observation source/job/status differs')
    return o


def _native_model(value,job,view,root):
    report=value['report'];kind=job['kind'];p=job['payload']
    if kind=='parent':
        path=Path(value['artifact_path'])/report['selected']['path']
    else:path=Path(report['models']['selected']['path'])
    path=path.resolve()
    if not path.is_relative_to(root):raise ValueError('Qualified native checkpoint escaped its queue')
    _,descriptor,_=_descriptor(path);meta=descriptor['state']
    state=load_checkpoint(path,expected_hashes=meta['identity_hashes'],device='cpu')
    if (state.identity_hashes!=train._identities(view,state.config,state.preprocessing)
            or state.data_hashes!=train._fg_hashes(view) or not state.stage['complete']
            or state.model.layer.d!=16):raise ValueError('Qualified native checkpoint source/data/completion differs')
    role=p.get('role');ordinary=role in ('warm_upper','fresh_upper')
    expected_width=95 if ordinary else 64
    gates=0 if ordinary or role=='keep' or kind=='parent' else 1 if kind=='probe' else 16
    coordinates='widened' if role=='warm_upper' else 'selective'
    if state.config.model.coordinates!=coordinates:
        raise ValueError('Qualified M3 native parameterization differs from its declared control')
    actual_gates=0 if coordinates=='widened' else len(state.model.layer.grown_ids)
    if coordinates=='widened' and state.model.layer.parent_width!=64:
        raise ValueError('Qualified ordinary widening must retain its64-unit parent')
    if state.model.layer.h!=expected_width or actual_gates!=gates:
        raise ValueError('Full native M3 topology is not qualified by a reduced/ineligible model')
    expected_parameters=1+expected_width*18+2*gates*17
    if sum(v.numel() for v in state.model.parameters())!=expected_parameters:
        raise ValueError('Qualified M3 parameter count differs from its exact native topology')
    policy=state.stage['policy'];updates=state.stage['updates_completed']
    if (policy['max_updates']!=p['max_updates'] or policy['checkpoint_interval']!=25
            or policy['validation_interval']!=25 or policy['logical_batch_size']!=128
            or policy['stop_by_patience']!=(kind=='parent')
            or kind!='parent' and updates!=p['max_updates'] or updates<=0):
        raise ValueError('Full native M3 training/stopping/cadence was shortened or changed')
    prediction=state.predict(view.G.X)
    if prediction.shape!=(577,) or not np.isfinite(prediction).all():raise ValueError('Incomplete/invalid native full-G prediction')
    return updates


def collect_joint_profiles(directory,target):
    _threads()
    q=build_joint_qualification_plan(target);root=Path(directory).resolve();source=target['identity']['source_sha256']
    runtime=canonical_hash(target['identity']['execution_identity']);profiles=[];native_models=0
    with authenticated_queue(root,expected_plan_hash=q['plan_hash']) as (plan,ledger,results):
        if canonical_json(plan)!=canonical_json(q):raise ValueError('Qualification queue differs from the complete native spine')
        latest={event['attempt_id']:event for event in ledger.events}
        if set(results)!=set(j['job_id'] for j in q['attempts']):raise ValueError('Qualification profiles require every terminal job')
        data=[r['value'] for name,r in results.items() if next(j for j in q['attempts'] if j['job_id']==name)['kind']=='data']
        if len(data)!=1 or data[0].get('status')!='completed_eligible':raise ValueError('Completed native qualification data is missing')
        view=load_fitting_data(data[0]['artifact_path'])
        if view.F.X.shape!=(2306,16) or view.G.X.shape!=(577,16) or view.validation_features.shape!=(577,16):
            raise ValueError('Native qualification full-data coverage differs')
        for job in q['attempts']:
            name=job['job_id'];record=results[name];value=record['value'];status=value['status']
            if job['kind']=='qualification_fixture':
                if (status!='completed_eligible' or value.get('evidence_kind')!='synthetic_resource_decision_fixture'
                        or value.get('scientific_admission') is not False):raise ValueError('Synthetic decision fixture was misqualified')
                continue
            if (value.get('evidence_kind')!=('synthetic_resource_decision' if job['kind']=='extension_decision'
                    else 'native_engineering_resource_workload') or value.get('scientific_admission') is not False
                    or value.get('T_accessed') is not False):raise ValueError('Native qualification evidence status/scope is missing')
            if status=='completed_ineligible' and job['kind']=='probe':continue
            if status!='completed_eligible' and not (status=='completed_ineligible' and job['kind']=='extension_decision'):
                raise ValueError('Failed/ineligible native workload cannot qualify the complete M3 resource track')
            o=_committed_observation(value,root,job,source,runtime)
            if job['kind'] in ('parent','fit','probe'):
                updates=_native_model(value,job,view,root);native_models+=1
                if updates!=o['executed_updates']:raise ValueError('Observed and authenticated native update counters differ')
            event=latest[name]
            running=[e for e in ledger.events if e['attempt_id']==name and e['status']=='running'][-1]
            elapsed=event['elapsed_ns']-running['elapsed_ns']+event.get('preparation_ns',0)
            profiles.append(dict(workload_key=workload_key(job),source_sha256=source,execution_identity_hash=runtime,
                status=status,elapsed_ns=elapsed,charged_total_ns=event['elapsed_ns'],executed_updates=o['executed_updates'],
                parent_selected_update=o['parent_selected_update'],retained_bytes=sum(f['bytes'] for f in record['files'])
                    +Path(event['result']['path']).stat().st_size,peak_rss_bytes=o['peak_process_rss_bytes'],
                max_checkpoint_transaction_bytes=o['estimated_checkpoint_transaction_bytes'],
                provenance=dict(qualification_plan_hash=q['plan_hash'],job_id=name,result=event['result'],
                    evidence_kind=value['evidence_kind'],timing='Measured current full queue attempt; cumulative paid attempts separate')))
        # This also refuses a missing active singleton or any other unmeasured class.
        forecast_joint_resources(target,profiles)
        return dict(schema='research/v2-M3-qualified-profiles',status='profiles_complete',source_sha256=source,
            target_plan_hash=target['plan_hash'],qualification_plan_hash=q['plan_hash'],profiles=profiles,
            native_model_checkpoints_authenticated=native_models,qualification_directory=str(root),
            ledger_event_hash=ledger.events[-1]['event_hash'],T_accessed=False,scientific_admission=False)


def make_joint_admission(target,qualification_directory):
    collected=collect_joint_profiles(qualification_directory,target)
    forecast=forecast_joint_resources(target,collected['profiles'])
    return dict(schema='research/v2-M3-resource-admission',source_sha256=target['identity']['source_sha256'],
        target_plan_hash=target['plan_hash'],qualification_directory=str(Path(qualification_directory).resolve()),
        qualification_plan_hash=collected['qualification_plan_hash'],collection_sha256=canonical_hash(collected),forecast=forecast)


def run_joint(study,output,*,resume=False):
    target=compile_joint_graph(study);locator=study['config'].get('resource_admission')
    if not locator:raise ValueError('Measured M3 resource admission is required before scientific run')
    path=Path(locator)
    if path.is_symlink() or not path.is_file():raise ValueError('Measured M3 resource admission file is missing')
    raw=path.read_bytes();record=json.loads(raw)
    _threads()
    actual=make_joint_admission(target,record.get('qualification_directory',''))
    if record!=actual:raise ValueError('M3 native profile/source/forecast admission changed')
    _runtime(target['identity']);_threads();output=Path(output).resolve();forecast=record['forecast']
    digest=hashlib.sha256(raw).hexdigest();binding=dict(schema='research/v2-M3-admission-binding',
        path=str(path.resolve()),sha256=digest,record=record)
    binding_path=output/'resource-admission-binding.json'
    def check(value):
        if value!=target['identity']:raise ValueError('M3 scientific queue identity changed')
        _runtime(value)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Original M3 admission bytes changed')
        if binding_path.exists() and (binding_path.is_symlink() or json.loads(binding_path.read_text())!=binding):
            raise ValueError('Saved M3 resource admission binding changed')
        if not binding_path.exists() and (output/'events.jsonl').exists() and (output/'events.jsonl').stat().st_size:
            raise ValueError('Missing M3 admission binding after paid work')
    def before(job,events):
        if not binding_path.exists():write_new_json(binding_path,binding)
        return joint_resource_decision(target,forecast,job,events,live_resources(output))
    guard=lambda:_guard(output,forecast['max_checkpoint_transaction_bytes'])
    return run_queue(target,output,ObservedJointWorkers(target,guard=guard),check_identity=check,
        resume=resume,before_start=before)
