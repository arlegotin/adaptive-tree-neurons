"""Measurements require committed real worker observations; incomplete is explicit."""
import copy
import json
from pathlib import Path

import pytest


def graph():
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.opportunity_graph import compile_bounded_graph
    return compile_bounded_graph(expand(load_research_config(Path('configs/research/v2/M2-opportunity.json'))).to_dict())


def partial_queue(tmp_path,transform=lambda value,directory:value):
    from treeneuron.research.queue_observation import ObservedOpportunityWorkers
    from treeneuron.research.workqueue import run_queue
    plan=graph();worker=ObservedOpportunityWorkers(plan);first=plan['attempts'][0]['job_id']
    def handler(job,directory,deps,attempt):
        if job['job_id']!=first:raise KeyboardInterrupt('Declared engineering qualification boundary')
        return transform(worker(job,directory,deps,attempt),directory)
    with pytest.raises(KeyboardInterrupt):run_queue(plan,tmp_path/'queue',handler,check_identity=lambda _:None)
    return plan,tmp_path/'queue',first


def collector():
    from treeneuron.research import queue_profiles
    return queue_profiles.collect_resource_profiles


def test_real_data_worker_measurement_retains_queue_costs_and_missing_classes(tmp_path):
    plan,root,first=partial_queue(tmp_path)
    report=collector()(root,expected_plan_hash=plan['plan_hash'])
    assert report['production_admission'] is False and report['T_accessed'] is False
    assert report['status']=='incomplete_qualification' and len(report['missing_workload_hashes'])==52
    assert len(report['profiles'])==1 and report['profiles'][0]['provenance']['job_id']==first
    row=report['profiles'][0]
    assert row['elapsed_ns']>=row['provenance']['handler_and_observation_ns']>0
    assert row['executed_updates']==0 and row['retained_bytes']>0
    assert report['invocations'][0]['invocation_ns']>=row['elapsed_ns']
    assert report['charged_cumulative_job_ns']>=row['elapsed_ns']


def test_committed_observation_with_false_execution_counter_is_rejected(tmp_path):
    from treeneuron.research.ledger import write_new_json
    import hashlib
    def wrong(value,directory):
        reference=value['resource_observation'];path=Path(reference['path']);row=json.loads(path.read_text())
        row['executed_updates']=99;path.unlink();write_new_json(path,row)
        value['resource_observation']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();return value
    plan,root,_=partial_queue(tmp_path,wrong)
    with pytest.raises(ValueError,match='counter'):collector()(root,expected_plan_hash=plan['plan_hash'])


def test_completed_job_without_observation_is_unqualified_not_imputed(tmp_path):
    def missing(value,directory):value.pop('resource_observation');return value
    plan,root,first=partial_queue(tmp_path,missing)
    result=collector()(root,expected_plan_hash=plan['plan_hash'])
    assert not result['profiles'] and len(result['missing_workload_hashes'])==53
    assert result['unqualified_jobs']==[dict(job_id=first,reason='No committed resource observation')]


def test_mutation_of_retained_raw_observation_fails_outer_file_authentication(tmp_path):
    plan,root,_=partial_queue(tmp_path)
    file=next(root.rglob('resource-observation.json'));file.write_bytes(file.read_bytes()+b' ')
    with pytest.raises(ValueError,match='inventory'):collector()(root,expected_plan_hash=plan['plan_hash'])
