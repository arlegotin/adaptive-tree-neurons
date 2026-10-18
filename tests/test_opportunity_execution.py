"""Production admission cannot be replaced by engineering or analytic profiles."""
import copy
from pathlib import Path

import pytest


def target():
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.opportunity_graph import compile_bounded_graph
    return compile_bounded_graph(expand(load_research_config(Path('configs/research/v2/M2-opportunity.json'))).to_dict())


def forecast(p):
    from test_opportunity_resources import profiles
    from treeneuron.research.opportunity_resources import forecast_resources
    return forecast_resources(p,profiles(p),available_worker_seconds=86400,free_disk_margin_bytes=1024)


def test_resource_decision_retains_full_forecast_and_pauses_before_next_transaction():
    from treeneuron.research.opportunity_execution import resource_decision
    p=target();f=forecast(p);job=p['attempts'][0]
    good=resource_decision(p,f,job,[],dict(free_disk_bytes=10**12,available_memory_bytes=10**12))
    assert good['admitted'] and good['full_remaining_forecast_bytes']==f['forecast_remaining_bytes']
    needed=good['required_disk_bytes'];memory=good['required_memory_bytes']
    assert needed>f['job_retained_bytes'][job['job_id']]
    low=resource_decision(p,f,job,[],dict(free_disk_bytes=needed-1,available_memory_bytes=10**12))
    assert not low['admitted'] and low['reason']=='insufficient_live_disk'
    assert resource_decision(p,f,job,[],dict(free_disk_bytes=needed,available_memory_bytes=memory))['admitted']
    low=resource_decision(p,f,job,[],dict(free_disk_bytes=10**12,available_memory_bytes=memory-1))
    assert not low['admitted'] and low['reason']=='insufficient_live_memory'
    unknown=resource_decision(p,f,job,[],dict(free_disk_bytes=10**12,available_memory_bytes=None))
    assert not unknown['admitted'] and unknown['reason']=='unknown_live_memory'


def test_changed_forecast_or_unknown_current_job_cannot_be_admitted():
    from treeneuron.research.opportunity_execution import resource_decision
    p=target();f=forecast(p);bad=copy.deepcopy(f);bad['peak_rss_bytes']=1
    with pytest.raises(ValueError):resource_decision(p,bad,p['attempts'][0],[],dict(free_disk_bytes=10**12,available_memory_bytes=10**12))
    job=copy.deepcopy(p['attempts'][0]);job['payload']['teacher_seed']+=1
    with pytest.raises(ValueError,match='job'):
        resource_decision(p,f,job,[],dict(free_disk_bytes=10**12,available_memory_bytes=10**12))


def test_live_probe_reports_real_bytes_and_preserves_machine_details(tmp_path):
    from treeneuron.research.opportunity_execution import live_resources
    value=live_resources(tmp_path)
    assert type(value['free_disk_bytes']) is int and value['free_disk_bytes']>0
    assert type(value['available_memory_bytes']) is int and value['available_memory_bytes']>0
    assert value['memory_basis'] and value['platform']


def test_m2_production_requires_authenticated_complete_resource_admission_before_queue(tmp_path):
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.runner import run
    plan=expand(load_research_config(Path('configs/research/v2/M2-opportunity.json')))
    with pytest.raises(ValueError,match='admission'):
        run(plan,tmp_path/'not-started')
    assert not (tmp_path/'not-started').exists()


def test_unavailable_memory_probe_pauses_instead_of_starting_unmeasured_worker(tmp_path,monkeypatch):
    from treeneuron.research import opportunity_execution as module
    import subprocess
    monkeypatch.setattr(module.sys,'platform','darwin')
    def unavailable(*args,**kwargs):raise subprocess.CalledProcessError(1,['vm_stat'])
    monkeypatch.setattr(module.subprocess,'run',unavailable)
    value=module.live_resources(tmp_path)
    assert value['available_memory_bytes'] is None and value['free_disk_bytes']>0
    p=target();decision=module.resource_decision(p,forecast(p),p['attempts'][0],[],value)
    assert decision['admitted'] is False and decision['reason']=='unknown_live_memory'


def test_resource_time_uses_latest_cumulative_paid_cost_once():
    from treeneuron.research.opportunity_execution import resource_decision
    p=target();f=forecast(p);first=p['attempts'][0]['job_id'];next_job=p['attempts'][1]
    events=[dict(attempt_id=first,status='running',elapsed_ns=1000000000),
        dict(attempt_id=first,status='completed_eligible',elapsed_ns=3000000000)]
    value=resource_decision(p,f,next_job,events,dict(free_disk_bytes=10**12,available_memory_bytes=10**12))
    assert value['charged_worker_seconds']==3 and value['admitted']
    assert value['full_remaining_forecast_bytes']<f['forecast_remaining_bytes']


@pytest.mark.parametrize('record',[{}, {'schema':'research/v2-opportunity-admission','production_admission':True}])
def test_unqualified_handwritten_admission_cannot_create_a_queue(tmp_path,record):
    import json
    from treeneuron.research.opportunity_execution import validate_opportunity_admission
    path=tmp_path/'admission.json';path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='schema'):
        validate_opportunity_admission(target(),path)


@pytest.mark.parametrize('status,code',[('paused_before_job',3),('completed_with_failures',4),('completed',0)])
def test_cli_reports_m2_queue_disposition_and_exit_code(tmp_path,monkeypatch,capsys,status,code):
    from treeneuron.research import cli,runner
    monkeypatch.setattr(runner,'run',lambda *a,**k:dict(status=status,statuses={'x':'completed_eligible'},charged_job_ns=1))
    result=cli.main(['run','--config','configs/research/v2/M2-opportunity.json','--output',str(tmp_path/'queue')])
    assert result==code
    import json
    assert json.loads(capsys.readouterr().out)['status']==status
