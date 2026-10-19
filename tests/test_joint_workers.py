"""Real short-budget fixture composes M3 selectors, paid probes and native models."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_train import cpu_training_evidence
from test_opportunity_workers import tiny_plan


def test_multi_digit_candidate_trace_survives_real_publication_and_still_rejects_changes(tmp_path):
    """Numeric candidate-key ordering must not change the committed trace hash."""
    import hashlib
    from treeneuron.research.joint_allocation import SelectionTrace
    from treeneuron.research.joint_workers import JointWorkers
    from treeneuron.research.ledger import write_new_json
    from treeneuron.research.workqueue import make_queue_plan
    plan=make_queue_plan({'study_config':{}},[
        dict(job_id='placement',kind='placement',payload={},depends_on=[])])
    worker=JointWorkers(plan)
    directory=tmp_path/'queue/jobs/local/attempt-0';directory.mkdir(parents=True)
    path=tmp_path/'queue/jobs/placement/attempt-0/selection.json';path.parent.mkdir(parents=True)
    original=SelectionTrace(json.dumps({'parent_span_novelty_fraction':{j:j/64 for j in range(64)}},
        sort_keys=True,separators=(',',':')))
    write_new_json(path,original.to_dict())
    value=dict(selection_path=str(path),selection_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        trace_sha256=original.sha256)
    loaded=worker._trace(directory,{'placement':{'value':value}})
    assert loaded.sha256==original.sha256
    assert loaded.to_dict()['parent_span_novelty_fraction']['10']==.15625
    changed=json.loads(path.read_text());changed['parent_span_novelty_fraction']['10']=.9
    path.write_text(json.dumps(changed))
    value['selection_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='trace digest'):
        worker._trace(directory,{'placement':{'value':value}})


def joint_fixture(receipt):
    from treeneuron.research.workqueue import make_queue_plan
    original=tiny_plan(receipt);jobs=[{k:v for k,v in j.items() if k!='attempt_id'}
        for j in original['attempts'] if j['kind'] in ('data','parent','score')]
    def add(name,kind,payload,deps,policy='all_eligible'):
        jobs.append(dict(job_id=name,kind=kind,payload=payload,depends_on=deps,dependency_policy=policy))
    common=dict(snapshot='late_G_best',requested_count=2,allocation_seed=55,max_updates=6,
        stop_by_patience=False,treatment='continued_stream_retained_adam',fresh_optimizer_seed=883)
    for role in ('local_singleton','joint','unconditioned','pure_span'):
        p=dict(common,role=role,parent_ridge=0. if role=='pure_span' else .01,gate_ridge=.01,
            condition_on_parent=role!='unconditioned',selection_strategy='singleton' if role=='local_singleton' else 'conditional')
        add('placement-'+role,'placement',p,['data','parent','score'])
        add('local-'+role,'local_model',p,['data','parent','score','placement-'+role])
    probes=[]
    for unit in (None,*range(8)):
        name='probe-keep' if unit is None else 'probe-'+str(unit);probes.append(name)
        add(name,'probe',dict(common,role='keep' if unit is None else 'subset',
            unit_ids=[] if unit is None else [unit],requested_count=0 if unit is None else 1),['data','parent','score'])
    add('short','short_selection',dict(snapshot='late_G_best',requested_count=2),['data','parent','score',*probes],'all_terminal')
    for role in ('keep','top','random','short_singleton','local_singleton','joint','unconditioned','pure_span','warm_upper','fresh_upper'):
        dependencies=['data','parent','score']
        if role in ('local_singleton','joint','unconditioned','pure_span'):dependencies.append('placement-'+role)
        if role=='short_singleton':dependencies.append('short')
        add('fit-'+role,'fit',dict(common,role=role),dependencies)
        add('evaluate-'+role,'evaluate',dict(teacher_index=0,repetition=0,role=role),['data','fit-'+role])
    return make_queue_plan(original['identity'],jobs)


def test_native_joint_workers_use_F_only_selection_and_pay_all_G_probes(tmp_path,cpu_training_evidence,monkeypatch):
    from treeneuron.research.joint_workers import JointWorkers
    from treeneuron.research.workqueue import run_queue,AttemptLedger
    from treeneuron.research import opportunity_workers
    from treeneuron.research.opportunity_workers import standalone_costs
    plan=joint_fixture(cpu_training_evidence);worker=JointWorkers(plan);events=[]
    original=opportunity_workers.load_evaluation_data
    def load(path):events.append('V/E');return original(path)
    monkeypatch.setattr(opportunity_workers,'load_evaluation_data',load)
    def handler(job,*args):events.append(job['kind']);return worker(job,*args)
    root=tmp_path/'queue';result=run_queue(plan,root,handler,check_identity=lambda _:None)
    assert result['status']=='completed'
    assert result['statuses']['probe-7']=='completed_ineligible'
    assert all(events[i-1]=='evaluate' for i,event in enumerate(events) if event=='V/E')
    ledger=AttemptLedger.resume(root,plan);latest={e['attempt_id']:e for e in ledger.events}
    def value(name):return json.loads(Path(latest[name]['result']['path']).read_text())['value']
    short=value('short')
    assert short['effective_count']==2 and len(short['candidate_records'])==8
    assert short['selection_labels']=='G' and short['probe_attempts']==9
    for role in ('keep','top','random','short_singleton','local_singleton','joint','unconditioned','pure_span','warm_upper','fresh_upper'):
        fitted=value('fit-'+role)
        assert fitted['report']['executed_stage_updates']==6
        assert fitted['method_role']==role
        assert value('evaluate-'+role)['T_accessed'] is False
    assert value('fit-fresh_upper')['report']['inherit_parent_parameters'] is False
    assert value('fit-fresh_upper')['report']['total_parameters']==45
    assert value('local-joint')['diagnostic']['fraction_selected_on_G'] is False
    costs=standalone_costs(plan,ledger)
    assert set('probe-'+str(i) for i in range(8)).issubset(costs['fit-short_singleton']['prerequisite_job_ids'])
    assert costs['fit-short_singleton']['charged_ns']>costs['short']['charged_ns']
    before=len(ledger.events)
    run_queue(plan,root,lambda *_:pytest.fail('Completed scientific work reran'),check_identity=lambda _:None,resume=True)
    assert len(AttemptLedger.resume(root,plan).events)==before
    # A failed eligible probe remains a missing comparison even if another candidate won.
    job=next(j for j in plan['attempts'] if j['job_id']=='short')
    deps={d:json.loads(Path(latest[d]['result']['path']).read_text()) for d in job['depends_on']}
    deps['probe-0']['value']['status']='numerical_failure'
    directory=root/'missing-probe';directory.mkdir()
    missing=worker(job,directory,deps,0)
    assert missing['status']=='irrecoverable_missing' and missing['selected_ids'] is None
    deps={d:json.loads(Path(latest[d]['result']['path']).read_text()) for d in job['depends_on']}
    deps['probe-keep']['value']['report']['executed_stage_updates']-=1
    directory=root/'shortened-keep';directory.mkdir()
    with pytest.raises(ValueError,match='budget|horizon'):
        worker(job,directory,deps,0)


def test_short_selection_retains_requested_budget_with_fewer_eligible_units(tmp_path,cpu_training_evidence):
    from treeneuron.research.joint_workers import JointWorkers
    from treeneuron.research.workqueue import make_queue_plan,run_queue,AttemptLedger
    full=joint_fixture(cpu_training_evidence)
    jobs=[{k:v for k,v in row.items() if k!='attempt_id'} for row in full['attempts']
        if row['kind'] in ('data','parent','score','probe','short_selection') or row['job_id']=='fit-short_singleton']
    for row in jobs:
        if row['kind']=='short_selection' or row['kind']=='fit':row['payload']['requested_count']=8
    plan=make_queue_plan(full['identity'],jobs);root=tmp_path/'reduced'
    result=run_queue(plan,root,JointWorkers(plan),check_identity=lambda _:None)
    assert result['status']=='completed'
    events={e['attempt_id']:e for e in AttemptLedger.resume(root,plan).events}
    row=json.loads(Path(events['fit-short_singleton']['result']['path']).read_text())['value']['report']
    assert row['requested_count']==8 and row['effective_count']==7
    assert len(row['selected_ids'])==7
