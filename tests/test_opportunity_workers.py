"""Real small workers compose through the immutable prerequisite queue."""
import json

import numpy as np
import pytest

from test_train import cpu_training_evidence


def tiny_plan(receipt):
    from treeneuron.config import resolve_experiment
    from treeneuron.research.workqueue import make_queue_plan
    base=resolve_experiment(dict(study_id='M2-worker-fixture',model=dict(d=2,h=8),
        data=dict(n_fit=512,n_growth=256,n_validation=256),
        training=dict(base_updates=6,continuation_updates=6,batch_size=128,
            validation_interval=2,patience=20),
        backend=dict(cpu_threads=1,interop_threads=1),
        execution=dict(backend_report=str(receipt)))).to_dict()
    jobs=[]
    def add(name,kind,payload,depends=(),policy='all_eligible'):
        jobs.append(dict(job_id=name,kind=kind,payload=payload,depends_on=list(depends),dependency_policy=policy))
    spec=dict(population='positive_small',teacher_index=0,teacher_seed=123,data_seed=456,
        reference_seed=789,evaluation_seed=1011,evaluation_rows=16384)
    add('data','data',spec)
    parent=dict(prerequisite_id='small',parent_recipe='raw_positive_control',**spec,
        repetition=0,optimizer_seed=44,allocation_seed=55,contrast_seed=66,
        candidate_sample_seed=77,snapshot_updates=[],stop_by_patience=True)
    add('parent','parent',parent,['data'])
    add('score','score',dict(snapshot='late_G_best'),['data','parent'])
    add('binding','horizon_binding',dict(seed=77),['score'])
    add('local','local_diagnostic',{},['data','parent','score'])
    common=dict(parent_key='small',snapshot='late_G_best',allocation_seed=55,
        max_updates=6,stop_by_patience=True,treatment='continued_stream')
    for name,ids in (('good',[0,2]),('dead',[0,7])):
        add(name,'fit',dict(matrix='M2d',attempt_id=name,role='pair',unit_ids=ids,
            requested_count=2,**common),['data','parent','score'])
    add('keep','fit',dict(matrix='M2b',attempt_id='keep',role='keep',unit_ids=[],requested_count=0,
        **{**common,'stop_by_patience':False,'selection_prefixes':[0,2,6]}),['data','parent','score'])
    add('head','head',dict(matrix='M2e_head',attempt_id='head',**common),['data','parent'])
    add('oracle','pair_oracle',dict(select_on='G'),['good','dead'],'all_terminal')
    for name in ('keep','head','oracle'):
        add('evaluate-'+name,'evaluate',dict(matrix='fixture'),['data',name])
    return make_queue_plan(dict(study_config=dict(base_experiment=base)),jobs)


def test_native_workers_fit_select_then_evaluate_without_refitting(tmp_path,cpu_training_evidence,monkeypatch):
    from treeneuron.research.workqueue import run_queue,AttemptLedger
    from treeneuron.research import opportunity_workers as workers
    plan=tiny_plan(cpu_training_evidence);root=tmp_path/'queue'
    actual=workers.load_evaluation_data;events=[]
    def evaluator(path):events.append('evaluator');return actual(path)
    monkeypatch.setattr(workers,'load_evaluation_data',evaluator)
    handler=workers.OpportunityWorkers(plan)
    def traced(job,*args):
        events.append(job['kind']);return handler(job,*args)
    result=run_queue(plan,root,traced,check_identity=lambda _:None)
    assert result['status']=='completed'
    assert result['statuses']['dead']=='completed_ineligible'
    assert len([e for e in events if e=='evaluator'])==3
    for i,e in enumerate(events):
        if e=='evaluator':assert events[i-1]=='evaluate'
    ledger=AttemptLedger.resume(root,plan)
    latest={e['attempt_id']:e for e in ledger.events}
    def value(name):return json.loads(open(latest[name]['result']['path']).read())['value']
    oracle=value('oracle');assert oracle['selection']['unit_ids']==[0,2]
    assert oracle['selection']['attempted_pair_count']==2
    local=value('local');assert local['calibration']['selection_labels']=='F_only'
    assert local['calibration']['minibatches']==64
    assert local['calibration']['gate_gradient_max_absolute_error']<1e-10
    keep=value('evaluate-keep')
    assert set(keep['metrics'])=={'selected','prefix_0','prefix_2','prefix_6'}
    assert keep['T_accessed'] is False
    with np.load(keep['predictions_path'],allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['selected_E'],saved['prefix_6_E'])
        assert len(saved['E_row_id'])==16384
    from treeneuron.research.opportunity_workers import standalone_costs
    costs=standalone_costs(plan,ledger)
    assert costs['evaluate-oracle']['charged_ns']>=costs['good']['charged_ns']
    assert len(costs['evaluate-oracle']['prerequisite_job_ids'])==6
    assert costs['evaluate-head']['charged_ns']>=costs['head']['charged_ns']
    # A valid same-shaped dataset from a different teacher cannot be substituted
    # at the final evaluation boundary, even when both artifacts authenticate.
    from treeneuron.research.opportunity_data import prepare_data_artifact
    wrong_spec={**plan['attempts'][0]['payload'],'teacher_seed':987,'data_seed':654}
    prepare_data_artifact(plan['identity']['study_config']['base_experiment'],wrong_spec,root/'other-data')
    for name in ('keep','head'):
        job=next(j for j in plan['attempts'] if j['job_id']=='evaluate-'+name)
        deps={dep:json.loads(open(latest[dep]['result']['path']).read()) for dep in job['depends_on']}
        deps['data']['value']['artifact_path']=str(root/'other-data')
        directory=root/'jobs'/('wrong-'+name)/'attempt-0';directory.mkdir(parents=True)
        with pytest.raises(ValueError,match='data identity'):
            handler(job,directory,deps,0)
    before=len(ledger.events)
    resumed=run_queue(plan,root,lambda *_:pytest.fail('completed worker reran'),check_identity=lambda _:None,resume=True)
    assert resumed['charged_job_ns']==result['charged_job_ns']
    assert len(AttemptLedger.resume(root,plan).events)==before


def test_missing_candidate_closes_oracle_and_unknown_worker_cannot_fit(tmp_path,cpu_training_evidence):
    from treeneuron.research.opportunity_workers import OpportunityWorkers
    plan=tiny_plan(cpu_training_evidence);worker=OpportunityWorkers(plan)
    deps={'good':dict(value=dict(status='numerical_failure')),'dead':dict(value=dict(status='completed_ineligible'))}
    job=next(j for j in plan['attempts'] if j['job_id']=='oracle')
    out=worker(job,tmp_path/'oracle',deps,0)
    assert out['status']=='irrecoverable_missing' and out['selection']['unit_ids'] is None
    with pytest.raises(NotImplementedError,match='unimplemented_fixture'):
        worker(dict(kind='unimplemented_fixture'),tmp_path/'unknown',{},0)


def test_worker_rejects_a_changed_job_before_its_side_effect(tmp_path,cpu_training_evidence):
    import copy
    from treeneuron.research.opportunity_workers import OpportunityWorkers
    plan=tiny_plan(cpu_training_evidence);worker=OpportunityWorkers(plan)
    changed=copy.deepcopy(plan['attempts'][0]);changed['payload']['teacher_seed']+=1
    with pytest.raises(ValueError,match='frozen'):
        worker(changed,tmp_path/'changed',{},0)
    assert not (tmp_path/'changed').exists()


def test_scoring_numerical_failure_keeps_numerical_domain(tmp_path,cpu_training_evidence,monkeypatch):
    from treeneuron.growth.score import InvalidScoreRun
    from treeneuron.research import opportunity_workers as workers
    plan=tiny_plan(cpu_training_evidence);worker=workers.OpportunityWorkers(plan)
    job=next(j for j in plan['attempts'] if j['kind']=='score')
    monkeypatch.setattr(worker,'_context',lambda *_:(type('Fit',(),dict(F=object()))(),object()))
    def invalid(*_):raise InvalidScoreRun('nonfinite prediction')
    monkeypatch.setattr(workers,'score_parent',invalid)
    with pytest.raises(FloatingPointError,match='nonfinite'):
        worker(job,tmp_path/'score',{},0)


def test_cost_accounting_rejects_changed_or_different_plan(tmp_path):
    import copy
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import make_queue_plan,run_queue,AttemptLedger
    from treeneuron.research.opportunity_workers import standalone_costs
    jobs=[dict(job_id='parent',kind='fixture',payload={},depends_on=[]),
        dict(job_id='child',kind='fixture',payload={},depends_on=['parent'])]
    plan=make_queue_plan({},jobs);root=tmp_path/'queue'
    run_queue(plan,root,lambda *_:dict(status='completed_eligible'),check_identity=lambda _:None)
    ledger=AttemptLedger.resume(root,plan)
    altered=copy.deepcopy(plan);altered['attempts'][1]['depends_on']=[]
    with pytest.raises(ValueError,match='plan'):standalone_costs(altered,ledger)
    altered['plan_hash']=canonical_hash({k:v for k,v in altered.items() if k!='plan_hash'})
    with pytest.raises(ValueError,match='plan'):standalone_costs(altered,ledger)
