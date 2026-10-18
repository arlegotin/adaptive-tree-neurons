"""A fixed engineering spine measures M2 workloads without scientific admission."""
import copy
import hashlib
from pathlib import Path

from treeneuron.config import canonical_hash,canonical_json
from .config import ResearchConfig,expand
from .opportunity_graph import compile_bounded_graph
from .opportunity_resources import workload_manifest
from .workqueue import make_queue_plan


def build_qualification_plan(target):
    expected=compile_bounded_graph(expand(ResearchConfig.from_dict(target['identity']['study_config'])).to_dict())
    if canonical_json(expected)!=canonical_json(target):raise ValueError('Qualification target differs from the frozen current-source graph')
    manifest=workload_manifest(target);original={j['job_id']:j for j in target['attempts']}
    chosen=[]
    for key,entry in manifest['classes'].items():
        name=entry['representative_job_id']
        candidate=name.replace('/actual_250/','/late_G_best/')
        if candidate in original and manifest['jobs'][candidate]['workload_hash']==key:name=candidate
        chosen.append(name)
    extension=next(n for n in chosen if original[n]['kind']=='extension_decision')
    eager=next(n for n in chosen if original[n]['kind']=='parent'
        and n.endswith('/fixed_parent_budget_fallback'))
    jobs={};fixture_ids=[]
    for dependency in original[extension]['depends_on']:
        name='qualification_fixture:'+dependency
        jobs[name]=dict(job_id=name,kind='qualification_fixture',depends_on=[],
            payload=dict(target_evaluator_id=dependency,evidence_kind='synthetic_resource_fixture',
                procedure_value=dict(status='completed_ineligible') if '/fallback/' in dependency
                    or 'fallback' in dependency else dict(status='completed_eligible',
                        metrics=dict(selected=dict(V=dict(normalized_mse=1.))))))
        fixture_ids.append(name)

    def include(name):
        if name in jobs:return
        j=copy.deepcopy(original[name]);j.pop('attempt_id')
        if name==eager:j['depends_on']=[d for d in j['depends_on'] if original[d]['kind']=='data']
        if name==extension:j['depends_on']=fixture_ids
        jobs[name]=j
        for dep in j['depends_on']:include(dep)
    for name in chosen:include(name)
    # Preserve target topological order for native jobs; synthetic prerequisites
    # precede their one engineering extension. make_queue_plan rechecks closure.
    ordered=[jobs[n] for n in fixture_ids]+[jobs[j['job_id']] for j in target['attempts'] if j['job_id'] in jobs]
    mapping={name:dict(target_job_id=name,target_job_sha256=canonical_hash(original[name]),
        workload_hash=manifest['jobs'][name]['workload_hash']) for name in jobs if name in original}
    contract=Path(__file__).resolve().parents[3]/'docs/research/v2/M2-qualification.md'
    identity=dict(admission='engineering_resource_qualification_only',scientific_admission=False,T_accessed=False,
        source_sha256=target['identity']['source_sha256'],execution_identity=target['identity']['execution_identity'],
        study_config=target['identity']['study_config'],target_plan_hash=target['plan_hash'],
        target_mapping=mapping,target_classes=manifest['classes'],contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
        engineering_exceptions=['eager_fixed_budget_parent','synthetic_extension_inputs'],
        target_maximum_planned_updates=manifest['maximum_planned_updates'],
        maximum_qualification_updates=sum(manifest['jobs'][n]['maximum_updates'] for n in mapping),
        fixed_cases='primary teacher0/repeat0; positive_small teacher0/repeat0; no outcome-based replacement',
        scope='Resource evidence only; eager fallback and synthetic extension never enter scientific M2 or Gate A')
    return make_queue_plan(identity,ordered)


class QualificationWorkers:
    """Native spine plus the one explicitly synthetic extension decision."""
    def __init__(self,target,qualification,*,guard=lambda:None):
        from .opportunity_workers import OpportunityWorkers
        if canonical_json(build_qualification_plan(target))!=canonical_json(qualification):
            raise ValueError('Changed engineering qualification mapping or policy')
        self.jobs={j['job_id']:copy.deepcopy(j) for j in qualification['attempts']}
        self.target_jobs={j['job_id']:copy.deepcopy(j) for j in target['attempts']}
        self.native=OpportunityWorkers(qualification,guard=guard)
        self.target=OpportunityWorkers(target,guard=guard)
        self.guard=guard
        self.mapping=qualification['identity']['target_mapping']
        self.classes=qualification['identity']['target_classes']

    def __call__(self,job,directory,dependencies,attempt):
        from .ledger import write_new_json
        if job.get('job_id') not in self.jobs or canonical_hash(job)!=canonical_hash(self.jobs[job['job_id']]):
            raise ValueError('Changed engineering qualification job')
        self.guard()
        if job['kind']=='qualification_fixture':
            value=dict(status='completed_eligible',evidence_kind='synthetic_resource_fixture',
                target_evaluator_id=job['payload']['target_evaluator_id'],
                procedure_value=copy.deepcopy(job['payload']['procedure_value']),
                scientific_admission=False,T_accessed=False)
            write_new_json(Path(directory)/'fixture.json',value)
        elif job['kind']=='extension_decision':
            if set(dependencies)!=set(job['depends_on']):raise ValueError('Synthetic extension dependency inventory differs')
            inputs={}
            for name,record in dependencies.items():
                value=record['value'];p=self.jobs[name]['payload']
                if (value.get('evidence_kind')!='synthetic_resource_fixture'
                        or value.get('target_evaluator_id')!=p['target_evaluator_id']
                        or value.get('procedure_value')!=p['procedure_value']):
                    raise ValueError('Changed synthetic extension fixture')
                inputs[p['target_evaluator_id']]=dict(value=copy.deepcopy(value['procedure_value']))
            original=self.target_jobs[job['job_id']]
            value=self.target._extension_decision(original,Path(directory),inputs)
            value.update(evidence_kind='synthetic_resource_decision',scientific_admission=False,T_accessed=False,
                synthetic_dependency_ids=list(original['depends_on']))
            write_new_json(Path(directory)/'engineering-disposition.json',value)
        else:
            value=self.native(job,directory,dependencies,attempt)
            value=dict(value,evidence_kind='native_engineering_resource_workload',scientific_admission=False,T_accessed=False)
            if value['status']=='completed_eligible':
                descriptor=self.classes[self.mapping[job['job_id']]['workload_hash']]['descriptor']
                if job['kind'] in ('parent','fit'):value['resource_topology']=_saved_topology(value,descriptor)
                elif job['kind'] in ('head','pair_oracle','evaluate'):
                    parents=[r['value'] for r in dependencies.values() if 'resource_topology' in r['value']]
                    if job['kind']=='pair_oracle':parents=[dependencies[value['selected_procedure_job_id']]['value']]
                    if len(parents)!=1:raise ValueError('Engineering retained topology prerequisite differs')
                    value['resource_topology']=copy.deepcopy(parents[0]['resource_topology'])
        self.guard()
        return value


def observed_qualification_workers(target,qualification,*,guard=lambda:None):
    from .queue_observation import ObservedOpportunityWorkers
    worker=ObservedOpportunityWorkers(qualification,guard=guard)
    worker.handler=QualificationWorkers(target,qualification,guard=guard)
    return worker


def topology_matches(topology,descriptor):
    layer=topology.get('layer',{});shape=descriptor['shape']
    width=descriptor.get('hidden_width',shape['parent_h']);count=descriptor.get('gate_count',0)
    if (layer.get('d'),layer.get('h'))!=(shape['d'],width):return False
    if topology.get('kind')=='widened_relu_scalar':
        return count==0 and layer.get('parent_width')==shape['parent_h']
    return topology.get('kind')=='selective_relu_scalar' and len(layer.get('grown_ids',[]))==count


def _saved_topology(value,descriptor):
    """Load an actual validated native checkpoint, preserving caller RNG."""
    from treeneuron import train
    from treeneuron.checkpoint import load_checkpoint
    report=value['report'];root=Path(value['artifact_path'])
    if descriptor['kind']=='parent':
        reference=report['selected'];path=(root/reference['path']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Qualification checkpoint escaped artifact')
    else:
        reference=report['models']['selected'];path=Path(reference['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']:
        raise ValueError('Qualification checkpoint pointer differs')
    before=train.capture_rng()
    try:
        state=load_checkpoint(path,expected_hashes=report['identity_hashes'],device='cpu')
        topology=state.model.export_config()
        return dict(topology=topology,matches_workload=topology_matches(topology,descriptor),
            checkpoint=dict(path=str(path),sha256=reference['sha256']),
            total_parameters=sum(p.numel() for p in state.model.parameters()),
            verification='Actual source/runtime/metadata/tensor validated selected checkpoint; caller RNG restored')
    finally:train.restore_rng(before)


def collect_qualification_profiles(directory,target,*,expected_plan_hash):
    """Authenticate engineering observations without changing procedure statuses."""
    import json
    from treeneuron import train
    from .queue_profiles import _profile_from_result
    from .workqueue import authenticated_queue
    expected=build_qualification_plan(target);manifest=workload_manifest(target)
    if expected_plan_hash!=expected['plan_hash']:raise ValueError('Engineering qualification commitment differs')
    with authenticated_queue(directory,expected_plan_hash=expected_plan_hash) as (plan,ledger,results):
        if canonical_json(plan)!=canonical_json(expected):raise ValueError('Engineering qualification plan differs')
        jobs={j['job_id']:j for j in plan['attempts']};mapping=plan['identity']['target_mapping']
        latest={e['attempt_id']:e for e in ledger.events};profiles=[];unqualified=[]
        for name,result in results.items():
            if name not in mapping:continue  # Synthetic inputs are never workload profiles.
            job=jobs[name];value=result['value'];status=value['status']
            pure=job['kind'] in ('fallback_gate','snapshot_gate','extension_decision')
            if status!='completed_eligible' and not (pure and status=='completed_ineligible'):
                unqualified.append(dict(job_id=name,reason='Procedure did not execute an eligible workload',procedure_status=status));continue
            entry=mapping[name];workload=manifest['jobs'][entry['target_job_id']]
            profile,reason=_profile_from_result(plan,job,latest[name],result,workload)
            if profile is None:
                unqualified.append(dict(job_id=name,reason=reason,procedure_status=status));continue
            descriptor=manifest['classes'][entry['workload_hash']]['descriptor']
            if job['kind'] in ('parent','fit','evaluate','head','pair_oracle'):
                topology=value.get('resource_topology')
                topology_descriptor=descriptor
                if job['kind']=='pair_oracle':
                    winner=value['selected_procedure_job_id']
                    if winner not in job['depends_on']:raise ValueError('Oracle topology winner differs')
                    topology_descriptor=manifest['classes'][manifest['jobs'][winner]['workload_hash']]['descriptor']
                if topology is None or not topology_matches(topology['topology'],topology_descriptor):
                    unqualified.append(dict(job_id=name,reason='Actual checkpoint topology does not qualify workload',procedure_status=status));continue
            profile['procedure_status']=status
            profile['resource_status']='completed_eligible'
            profile['scientific_admission']=False
            profile['provenance'].update(entry,evidence_kind=value['evidence_kind'],
                target_descriptor=descriptor,actual_topology=value.get('resource_topology'),
                dependency_result_commitments={d:latest[d]['result'] for d in job['depends_on']})
            profiles.append(profile)
        covered={p['workload_hash'] for p in profiles};missing=sorted(set(manifest['classes'])-covered)
        invocations=[]
        for path in sorted(Path(directory).glob('invocation-*.json')):
            if path.is_symlink():raise ValueError('Unsafe engineering invocation receipt')
            raw=path.read_bytes();row=json.loads(raw)
            if type(row.get('invocation_ns')) is not int or row['invocation_ns']<=0:
                raise ValueError('Invalid engineering invocation clock')
            invocations.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),**row))
        if ledger.events and not invocations:raise ValueError('Missing engineering invocation clock')
        return dict(schema='research/v2-opportunity-engineering-collection',queue_plan_hash=expected_plan_hash,
            target_plan_hash=target['plan_hash'],source_sha256=plan['identity']['source_sha256'],
            collector_source_sha256=train.source_hash(),profiles=profiles,missing_workload_hashes=missing,
            unqualified_jobs=unqualified,status='profiles_complete' if not missing else 'incomplete_qualification',
            ledger_event_hash=ledger.events[-1]['event_hash'] if ledger.events else '0'*64,
            charged_cumulative_job_ns=ledger.charged_ns,invocations=invocations,
            worker_loss_upper_bounds_present=any(e.get('cost_precision')=='wall_time_upper_bound_after_worker_loss' for e in ledger.events),
            time_scope='Nested queue clocks; external process startup/import/publication timer remains separately required',
            scientific_admission=False,production_admission=False,T_accessed=False)
