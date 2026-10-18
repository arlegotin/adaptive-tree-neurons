"""Resource observations preserve native state and retain honest timing scopes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_train import cpu_training_evidence
from test_opportunity_workers import tiny_plan


def plan(receipt):
    from treeneuron import train
    from treeneuron.config import canonical_hash
    from treeneuron.research.identity import execution_identity
    value=tiny_plan(receipt)
    value['identity'].update(source_sha256=train.source_hash(),
        execution_identity=execution_identity(value['identity']['study_config']))
    value['plan_hash']=canonical_hash({k:v for k,v in value.items() if k!='plan_hash'})
    return value


def test_observer_preserves_native_predictions_and_counts_actual_file_bytes(tmp_path,cpu_training_evidence):
    from treeneuron.research.queue_observation import ObservedOpportunityWorkers
    from treeneuron.research.opportunity_workers import OpportunityWorkers
    from treeneuron.research.workqueue import run_queue,authenticated_queue
    value=plan(cpu_training_evidence);roots=[tmp_path/'plain',tmp_path/'observed']
    for root,worker in zip(roots,[OpportunityWorkers(value),ObservedOpportunityWorkers(value)]):
        assert run_queue(value,root,worker,check_identity=lambda _:None)['status']=='completed'
    outputs=[]
    for root in roots:
        with authenticated_queue(root,expected_plan_hash=value['plan_hash']) as (_,ledger,records):
            outputs.append(records)
    for name in ('evaluate-keep','evaluate-head','evaluate-oracle'):
        a,b=(np.load(rows[name]['value']['predictions_path'],allow_pickle=False) for rows in outputs)
        with a,b:
            assert set(a.files)==set(b.files)
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    observed=outputs[1]
    for name in ('parent','score','keep','head','evaluate-keep','dead'):
        reference=observed[name]['value']['resource_observation'];path=Path(reference['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['sha256']
        row=json.loads(path.read_text())
        assert row['job_id']==name and row['source_sha256']==value['identity']['source_sha256']
        assert row['procedure_status']==observed[name]['value']['status']
        assert row['retained_bytes_before_observation']==sum(f['bytes'] for f in row['raw_inventory']['files'])
        assert row['handler_and_observation_ns']>0 and row['peak_process_rss_bytes']>0
        assert row['production_admission'] is False
    keep=json.loads(Path(observed['keep']['value']['resource_observation']['path']).read_text())
    assert keep['executed_updates']==6
    dead=json.loads(Path(observed['dead']['value']['resource_observation']['path']).read_text())
    assert dead['procedure_status']=='completed_ineligible' and dead['executed_updates']==0


def test_observer_rejects_source_drift_before_native_work(tmp_path,cpu_training_evidence,monkeypatch):
    from treeneuron import train
    from treeneuron.research.queue_observation import ObservedOpportunityWorkers
    value=plan(cpu_training_evidence);worker=ObservedOpportunityWorkers(value)
    monkeypatch.setattr(train,'source_hash',lambda:'changed')
    with pytest.raises(ValueError,match='source'):
        worker(value['attempts'][0],tmp_path/'not-created',{},0)
    assert not (tmp_path/'not-created').exists()
