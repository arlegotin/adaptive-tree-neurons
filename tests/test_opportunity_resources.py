"""M2 resource admission covers the bounded DAG and keeps paid retry evidence."""
import copy
from pathlib import Path

import pytest


@pytest.fixture
def graph():
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.opportunity_graph import compile_bounded_graph
    return compile_bounded_graph(expand(load_research_config(Path('configs/research/v2/M2-opportunity.json'))).to_dict())


def test_workloads_share_only_seed_independent_shape_and_treatment_classes(graph):
    from treeneuron.research.opportunity_resources import workload_manifest
    manifest=workload_manifest(graph)
    assert set(manifest['jobs'])=={j['job_id'] for j in graph['attempts']}
    assert sum(c['job_count'] for c in manifest['classes'].values())==3853
    def key(fragment):return next(v['workload_hash'] for k,v in manifest['jobs'].items() if k==fragment)
    first='procedure:M2a/teacher-0/repeat-0/late_G_best/continued_stream/'
    other='procedure:M2a/teacher-1/repeat-1/late_G_best/continued_stream/'
    assert key(first+'top')==key(other+'top')
    assert key(first+'top')!=key(first+'warm_upper')
    assert key(first+'warm_lower')!=key(first+'warm_upper')
    assert key(first+'top')!=key(first.replace('continued_stream','new_stream_retained_adam')+'top')
    assert manifest['maximum_planned_updates']>0
    assert manifest['includes_all_conditional_slots'] is True


def profiles(graph):
    from treeneuron.research.opportunity_resources import workload_manifest
    manifest=workload_manifest(graph)
    # Explicit analytic software profiles, never scientific workload receipts.
    return [dict(schema='research/v2-queue-resource-profile',source_sha256=graph['identity']['source_sha256'],
        execution_identity_hash=__import__('treeneuron.config',fromlist=['canonical_hash']).canonical_hash(graph['identity']['execution_identity']),
        workload_hash=key,status='completed_eligible',executed_updates=value['maximum_updates']//2,
        elapsed_ns=1000000,retained_bytes=1000,peak_rss_bytes=100000000,max_checkpoint_transaction_bytes=128,
        provenance={'software_fixture':True}) for key,value in manifest['classes'].items()]


def test_forecast_covers_every_conditional_job_at_full_ceilings_and_one_rolling_retry(graph):
    from treeneuron.research.opportunity_resources import forecast_resources
    record=forecast_resources(graph,profiles(graph),available_worker_seconds=86400,free_disk_margin_bytes=1024)
    assert len(record['job_retained_bytes'])==3853
    assert record['forecast_remaining_bytes']==sum(record['job_retained_bytes'].values())+record['retry_reserve_bytes']
    assert record['retry_reserve_bytes']==max(record['job_retained_bytes'].values())
    assert record['forecast_type']=='uncalibrated_full_ceiling_projection'
    assert record['forecast_slow_seconds']>=record['forecast_typical_seconds']>0
    assert record['forecast_remaining_bytes']>3853*1000


@pytest.mark.parametrize('mutation',['source','class','missing','failed','zero'])
def test_forecast_rejects_unqualified_or_missing_native_workload_classes(graph,mutation):
    from treeneuron.research.opportunity_resources import forecast_resources
    rows=profiles(graph)
    if mutation=='source':rows[0]['source_sha256']='changed'
    elif mutation=='class':rows[0]['workload_hash']='unknown'
    elif mutation=='missing':rows.pop()
    elif mutation=='failed':rows[0]['status']='completed_ineligible'
    else:rows[0]['elapsed_ns']=0
    with pytest.raises(ValueError):
        forecast_resources(graph,rows,available_worker_seconds=86400,free_disk_margin_bytes=1024)


def test_remaining_storage_only_credits_whole_terminal_jobs_and_unused_retry_reserve(graph):
    from treeneuron.research.opportunity_resources import forecast_resources,remaining_storage
    record=forecast_resources(graph,profiles(graph),available_worker_seconds=86400,free_disk_margin_bytes=1024)
    names=list(record['job_retained_bytes']);a,b=names[:2]
    pending=remaining_storage(graph,record,{a:'completed_eligible',b:'interrupted_recoverable'},attempted={a,b})
    expected=sum(v for k,v in record['job_retained_bytes'].items() if k!=a)
    reserve=max(v for k,v in record['job_retained_bytes'].items() if k not in (a,b))
    assert pending['required_additional_bytes']==expected+reserve
    assert pending['terminal_job_count']==1
    assert pending['additional_retry_reserve_bytes']==reserve
    done=remaining_storage(graph,record,{n:'completed_ineligible' for n in names},attempted=set(names))
    assert done['required_additional_bytes']==0


def test_remaining_forecast_cannot_be_edited_or_moved_to_another_source(graph):
    from treeneuron.research.opportunity_resources import forecast_resources,remaining_storage
    record=forecast_resources(graph,profiles(graph),available_worker_seconds=86400,free_disk_margin_bytes=1024)
    for key in ('source_sha256','execution_identity_hash'):
        changed=copy.deepcopy(record);changed[key]='changed'
        with pytest.raises(ValueError):remaining_storage(graph,changed,{},attempted=set())
    changed=copy.deepcopy(record);changed['job_retained_bytes'][next(iter(changed['job_retained_bytes']))]=1
    with pytest.raises(ValueError):remaining_storage(graph,changed,{},attempted=set())


def test_statusless_attempted_ids_cannot_erase_retry_reserve(graph):
    from treeneuron.research.opportunity_resources import forecast_resources,remaining_storage
    record=forecast_resources(graph,profiles(graph),available_worker_seconds=86400,free_disk_margin_bytes=1024)
    with pytest.raises(ValueError,match='inventory|status'):
        remaining_storage(graph,record,{},attempted=set(record['job_retained_bytes']))
