"""Managed official acquisition -> sealed roles, with no modeling authority."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest


ROOT=Path(__file__).resolve().parents[1]


def api():
    assert importlib.util.find_spec('treeneuron.research.preparation_execution_v2') is not None,'Missing managed preparation execution'
    from treeneuron.research import preparation_execution_v2
    return preparation_execution_v2


def copied_acquisition(tmp_path):
    original=ROOT/'artifacts/research-v2/WP07/acquisition/acquisition-manifest.json'
    raw=original.read_bytes();manifest=json.loads(raw);destination=tmp_path/'acquisition';destination.mkdir()
    (destination/original.name).write_bytes(raw)
    for row in manifest['datasets']:
        for ref in row['matched_members']:
            target=destination/ref['path'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original.parent/ref['path'],target)
    cfg=json.loads((ROOT/'configs/research/v2/P0-data.json').read_text())
    cfg['acquisition_manifest']=str(destination/original.name)
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg)+'\n')
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def actual_preparation(tmp_path_factory):
    module=api();root=tmp_path_factory.mktemp('managed-preparation');path,digest=copied_acquisition(root)
    result=module.run_managed_preparation(path,expected_config_sha256=digest,output=root/'execution',
        interpreter=sys.executable,data_overlay=ROOT/'environment/research-v2-data-site')
    assert result['status']=='completed_preparation',result
    return result,root,path,digest


def test_actual_six_table_preparation_has_closed_managed_provenance_without_reader_label_decoding(actual_preparation,monkeypatch):
    result,root,path,digest=actual_preparation;module=api()
    def forbidden(*a,**kw):raise AssertionError('Preparation proof reader must not decode archive arrays or rerun preparation')
    monkeypatch.setattr(np,'load',forbidden)
    from treeneuron.research import transfer
    monkeypatch.setattr(transfer,'prepare_data',forbidden)
    proof=module.load_preparation_execution(result['reference'])
    assert proof['preparation_execution_authenticated'] is True
    assert proof['preparation_runtime_authenticated'] is True
    assert proof['raw_to_sealed_generation_chain_authenticated'] is True
    assert len(proof['datasets'])==6 and proof['role_archives_hashed']==180
    assert proof['protected_label_archives_hashed']==30
    assert proof['arrays_or_labels_decoded'] is False
    assert proof['data_content_independently_rederived'] is False
    assert proof['models_fit']==0 and proof['protected_labels_read_for_modeling'] is False
    assert proof['native_workload_qualified'] is proof['quality_eligible'] is proof['protected_access_authorized'] is False


def test_unknown_preparation_configuration_rejects_before_worker_admission(tmp_path,monkeypatch):
    module=api();path,digest=copied_acquisition(tmp_path)
    value=json.loads(path.read_text());value['choose_favorable_split']=True;path.write_text(json.dumps(value))
    def forbidden(**kw):raise AssertionError('Unknown preparation setting reached a worker')
    monkeypatch.setattr(module,'supervise_process',forbidden)
    with pytest.raises(ValueError,match='schema|configuration'):
        module.run_managed_preparation(path,expected_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            output=tmp_path/'rejected',interpreter=sys.executable,data_overlay=ROOT/'environment/research-v2-data-site')


def test_mutated_external_preparation_reference_cannot_redirect_the_reader(actual_preparation):
    result,root,path,digest=actual_preparation;reference=copy.deepcopy(result['reference'])
    reference['inventory_sha256']='0'*64
    with pytest.raises(ValueError,match='inventory|commitment|bytes'):
        api().load_preparation_execution(reference)


@pytest.mark.parametrize('fault',['import_hash','model_count','rss_count','protected_bytes',
    'result_schema','result_authority','suite_argv'])
def test_consistently_recommitted_preparation_faults_cannot_authorize_official_data(actual_preparation,fault):
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    result,root,path,digest=actual_preparation;execution=Path(result['reference']['root'])
    terminal=execution/'worker/terminal.json';process=execution/'process/process.json';outer=execution/'result.json'
    protected=next((execution/'worker/data').glob('*/protected/outer-0/T-labels.npz'))
    suite=execution/'worker/data/suite.json';preparation_status=execution/'worker/data/preparation-status.json'
    changed={p:p.read_bytes() for p in (terminal,process,outer,protected,suite,preparation_status)}
    try:
        value=json.loads(terminal.read_text());record=json.loads(outer.read_text())
        if fault=='import_hash':value['runtime']['imports']['numpy']['sha256']='0'*64
        elif fault=='model_count':value['models_fit']=1
        elif fault=='rss_count':
            observed=json.loads(process.read_text());observed['rss_sample_count']+=1;process.write_text(json.dumps(observed))
        elif fault=='result_schema':record['schema']='research/v99-unrecognized-preparation'
        elif fault=='result_authority':record['quality_eligible']=True
        elif fault=='suite_argv':
            prepared=json.loads(suite.read_text());prepared['argv']=['an-unrelated-preparation-command']
            prepared['suite_hash']=canonical_hash({k:v for k,v in prepared.items() if k!='suite_hash'})
            suite.write_text(json.dumps(prepared));preparation_status.write_text(json.dumps(prepared))
            value['suite_sha256']=hashlib.sha256(suite.read_bytes()).hexdigest()
        else:
            raw=bytearray(protected.read_bytes());raw[0]^=1;protected.write_bytes(raw)
        terminal.write_text(json.dumps(value))
        record['worker_terminal_sha256']=hashlib.sha256(terminal.read_bytes()).hexdigest()
        record['process_sha256']=hashlib.sha256(process.read_bytes()).hexdigest();outer.write_text(json.dumps(record))
        reference=dict(root=str(execution),inventory_sha256=canonical_hash(_inventory(execution)))
        with pytest.raises(ValueError,match='import|runtime|authority|RSS|artifact|bytes|schema|invocation'):
            api().load_preparation_execution(reference)
    finally:
        for p,raw in changed.items():p.write_bytes(raw)
        assert canonical_hash(_inventory(execution))==result['reference']['inventory_sha256']
