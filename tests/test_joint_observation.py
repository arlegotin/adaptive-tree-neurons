"""Real source-bound probes report executed work even if their selected offset is0."""
import json
from pathlib import Path

from test_train import cpu_training_evidence
from test_joint_workers import joint_fixture


def test_observed_probe_records_actual_updates_files_and_runtime_without_qualifying_them(tmp_path,cpu_training_evidence):
    from treeneuron import train
    from treeneuron.research.joint_observation import ObservedJointWorkers
    from treeneuron.research.identity import execution_identity
    from treeneuron.research.workqueue import make_queue_plan,run_queue,AttemptLedger
    p=joint_fixture(cpu_training_evidence)
    jobs=[{k:v for k,v in j.items() if k!='attempt_id'} for j in p['attempts']
        if j['kind'] in ('data','parent','score') or j['job_id']=='probe-0']
    identity={**p['identity'],'source_sha256':train.source_hash(),'execution_identity':execution_identity(p['identity']['study_config'])}
    plan=make_queue_plan(identity,jobs);root=tmp_path/'observed'
    result=run_queue(plan,root,ObservedJointWorkers(plan),check_identity=lambda _:None)
    assert result['status']=='completed'
    events={e['attempt_id']:e for e in AttemptLedger.resume(root,plan).events}
    value=json.loads(Path(events['probe-0']['result']['path']).read_text())['value']
    path=Path(value['resource_observation']['path']);o=json.loads(path.read_text())
    assert o['schema']=='research/v2-M3-resource-observation'
    assert o['executed_updates']==6
    assert o['parent_selected_update']==0
    assert o['retained_bytes_before_observation']>0 and o['peak_process_rss_bytes']>0
    assert o['estimated_checkpoint_transaction_bytes']>4096
    assert o['source_sha256']==train.source_hash()
    assert o['production_admission'] is False and o['T_accessed'] is False
    assert o['handler_and_observation_ns']>0
