"""Finite native qualification orchestration never grants predictive authority."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from treeneuron.config import canonical_hash
from test_practical_workload import fixtures


def api():
    from treeneuron.research import qualify
    return qualify


def config(tmp_path,*,all_cases=False):
    module=api();datasets={}
    for name,value in fixtures().items():
        if not all_cases and name!='energy':continue
        value.update(feature_names=['x'],categorical_features=[])
        value['manifest_hash']=canonical_hash({k:v for k,v in value.items() if k!='manifest_hash'})
        root=tmp_path/name;root.mkdir();path=root/'manifest.json';path.write_text(json.dumps(value))
        datasets[name]=dict(root=str(root),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    value=dict(schema='research/v2-native-qualification',study_id='qualification-fixture',
        source_sha256=module.qualification_source_sha256(),locks_sha256=module.qualification_locks_sha256(),
        interpreter=sys.executable,datasets=datasets,families=[f'P{i:02}' for i in range(14)] if all_cases else ['P13'],overlays={})
    path=tmp_path/'qualification.json';path.write_text(json.dumps(value));return path,value


def test_all_declared_anchors_expand_to_exact_120_primary_jobs(tmp_path):
    module=api();path,value=config(tmp_path,all_cases=True);plan=module.qualification_plan(path)
    assert len(plan['attempts'])==120 and len({row['job_id'] for row in plan['attempts']})==120
    assert plan['identity']['full_native_anchor_inventory_planned'] is True
    assert plan['identity']['quality_eligible'] is plan['identity']['Gate_D'] is False
    for row in plan['attempts']:
        case=row['payload'];assert case['outer_fold']==(4 if case['dataset']=='bike' else 0)
        if case['family']=='P12':assert case['config']['execution']=='bounded_memory_v1'
    assert plan['plan_hash']==canonical_hash({k:v for k,v in plan.items() if k!='plan_hash'})


@pytest.mark.parametrize('change',['extra','source','locks','duplicates','ordering','secondary','overlay_escape','manifest'])
def test_invalid_or_changed_inputs_fail_before_any_worker(tmp_path,change):
    module=api();path,value=config(tmp_path)
    if change=='extra':value['unknown']=True
    elif change=='source':value['source_sha256']='0'*64
    elif change=='locks':value['locks_sha256']='0'*64
    elif change=='duplicates':value['families']=['P13','P13']
    elif change=='ordering':value['families']=['P13','P00']
    elif change=='secondary':value['outer_fold']=1
    elif change=='overlay_escape':value['overlays']={'P13':['../other-project']}
    else:
        manifest=Path(value['datasets']['energy']['root'])/'manifest.json';manifest.write_text('{}')
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):module.qualification_plan(path)


def test_resource_pause_keeps_exact_queue_without_starting_fit(tmp_path,monkeypatch):
    module=api();path,value=config(tmp_path)
    monkeypatch.setattr(module,'_resources',lambda output:dict(available_memory_bytes=0,free_disk_bytes=0))
    def forbidden(*args,**kwargs):raise AssertionError('Resource pause must precede the worker')
    monkeypatch.setattr(module,'_execute_anchor',forbidden)
    out=tmp_path/'output';result=module.qualify(path,out)
    assert result['status']=='paused_before_job' and result['quality_eligible'] is False
    before=(out/'queue/plan.json').read_bytes()
    again=module.qualify(path,out,resume=True)
    assert again['status']=='paused_before_job' and before==(out/'queue/plan.json').read_bytes()
    assert (out/'queue/events.jsonl').read_bytes()==b''


def test_cli_can_plan_explicit_qualification_schema_without_fitting(tmp_path):
    module=api();path,value=config(tmp_path)
    from treeneuron.research.cli import main
    out=tmp_path/'plan.json'
    assert main(['plan','--config',str(path),'--output',str(out)])==0
    plan=json.loads(out.read_text());assert len(plan['attempts'])==1
    assert plan['identity']['full_native_anchor_inventory_planned'] is False


def test_cli_qualify_dispatch_preserves_resume_without_implicit_promotion(tmp_path,monkeypatch):
    module=api();path,value=config(tmp_path);calls=[]
    def run(path,output,*,resume=False):
        calls.append((path,output,resume));return dict(status='paused_before_job',statuses={},quality_eligible=False)
    monkeypatch.setattr(module,'qualify',run)
    from treeneuron.research.cli import main
    out=tmp_path/'out'
    assert main(['qualify','--config',str(path),'--output',str(out),'--resume'])==3
    assert calls==[(path,out,True)]


@pytest.mark.parametrize('change',['interpreter','dataset_root','overlay_type'])
def test_malformed_paths_are_actionable_configuration_errors(tmp_path,change):
    module=api();path,value=config(tmp_path)
    if change=='interpreter':value['interpreter']=None
    elif change=='dataset_root':value['datasets']['energy']['root']=None
    else:value['overlays']={'P13':[[]]}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):module.load_qualification_config(path)


def test_cli_plan_nonmapping_json_is_reported_without_traceback(tmp_path,capsys):
    path=tmp_path/'invalid.json';path.write_text('[]')
    from treeneuron.research.cli import main
    assert main(['plan','--config',str(path),'--output',str(tmp_path/'out')])==2
    assert 'research/v2:' in capsys.readouterr().err


def test_paused_queue_can_complete_and_resume_without_repeating_a_terminal_fixture(tmp_path,monkeypatch):
    module=api();path,value=config(tmp_path);out=tmp_path/'output';calls=[]
    monkeypatch.setattr(module,'_resources',lambda output:dict(available_memory_bytes=0,free_disk_bytes=0))
    assert module.qualify(path,out)['status']=='paused_before_job'
    monkeypatch.setattr(module,'_resources',lambda output:dict(available_memory_bytes=8*1024**3,free_disk_bytes=8*1024**3))
    def fixture(job,directory,identity):
        calls.append(job['job_id'])
        (directory/'fixture-only.txt').write_text('No native model is fitted in this orchestration fixture')
        return dict(status='completed_ineligible',fixture_only=True)
    monkeypatch.setattr(module,'_execute_anchor',fixture)
    result=module.qualify(path,out,resume=True)
    assert result['status']=='completed' and result['native_anchor_executions_complete'] is True
    assert result['full_native_anchor_inventory_planned'] is result['Gate_D'] is result['quality_eligible'] is False
    assert module.qualify(path,out,resume=True)['status']=='completed' and len(calls)==1


def test_committed_manifest_without_semantic_features_is_rejected_before_plan(tmp_path):
    module=api();path,value=config(tmp_path)
    reference=value['datasets']['energy'];manifest=Path(reference['root'])/'manifest.json'
    data=json.loads(manifest.read_text());data.pop('feature_names')
    data['manifest_hash']=canonical_hash({k:v for k,v in data.items() if k!='manifest_hash'})
    manifest.write_text(json.dumps(data));reference['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):module.qualification_plan(path)
