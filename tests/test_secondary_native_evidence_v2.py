"""Actual toy native fits exercise role cores without faking global qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from test_released_transfer_v2 import toy


def api():
    assert importlib.util.find_spec('treeneuron.research.secondary_native_evidence_v2'), 'Missing secondary native evidence route'
    from treeneuron.research import secondary_native_evidence_v2
    return secondary_native_evidence_v2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def fitted(toy,tmp_path):
    api()
    from treeneuron.research.released_transfer_v2 import _ReleasedRoles
    from treeneuron.research.method_worker import scientific_source_sha256
    from treeneuron.research.search_supports_v2 import family_search_spec
    from treeneuron.research.native_fit_v2 import run_native_fit
    root,digest,legacy=toy;events=[]
    store=_ReleasedRoles(root,expected_manifest_sha256=digest,record=lambda **kw:events.append(kw))
    inputs=tmp_path/'generation/input';inputs.mkdir(parents=True);files={}
    for role,name in (('F','F.npz'),('G','G.npz'),('V_features','V-features.npz')):
        source=root/'outer-1'/name;target=inputs/(role+'.npz');shutil.copyfile(source,target)
        files[role]=dict(path=target.name,sha256=sha(target))
    config=family_search_spec('P13',dict(columns=legacy.manifest['feature_names'],categorical=legacy.manifest['categorical_features']))['anchors'][0]['config']
    request=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',job_id='toy-secondary/mean',family_id='P13',
        fit_seed=17,scientific_source_sha256=scientific_source_sha256(),config=config,budget={'native_stopping':True},files=files,
        observation_identity=dict(dataset='energy',split_kind='grouped',split_sha256=legacy.manifest['splits'][1]['split_hash']))
    path=inputs/'request.json';path.write_text(json.dumps(request)+'\n')
    native=run_native_fit(path,expected_request_sha256=sha(path),output=tmp_path/'generation/native',interpreter=sys.executable)
    assert native['status']=='completed_ineligible'
    args=dict(request_path=path,expected_request_sha256=sha(path),native_reference=native['recovery_reference'],prepared_root=root,
        expected_manifest_sha256=digest,outer_fold=1,expected_fit_seed=17,expected_configuration=config)
    return args,store,events,tmp_path/'generation'


def test_real_secondary_mean_fit_workload_replay_V_and_reopen_without_primary_or_T_access(fitted,tmp_path,monkeypatch):
    module=api();args,store,events,generation=fitted
    from treeneuron.research.workqueue import _inventory
    from treeneuron.config import canonical_hash
    original=np.load
    def no_T(path,*a,**kw):
        assert '/protected/' not in str(path),'Secondary evidence cannot open T'
        return original(path,*a,**kw)
    monkeypatch.setattr(np,'load',no_T)
    workload=module._workload(store,args)
    inspection=module._candidate(store,generation_arguments=args,immutable_generation_root=generation,
        output=tmp_path/'inspection',interpreter=sys.executable)
    assert workload['training']['procedure']=='closed_form'
    assert inspection['status']=='completed_candidate_inspection'
    assert inspection['replay']['prediction_reload_exact'] is True and inspection['replay']['labels_read'] is False
    with original(args['prepared_root']/'outer-1/F.npz',allow_pickle=False) as data:F=data['y'].copy()
    V=store.coordinator_validation(1);mean=float(np.mean(F))
    expected=float(np.mean([np.mean((V.y[V.group_id==g]-mean)**2) for g in np.unique(V.group_id)])/np.var(F,ddof=0))
    assert inspection['validation']['score']==pytest.approx(expected)
    ref=dict(root=str(tmp_path/'inspection'),inventory_sha256=canonical_hash(_inventory(tmp_path/'inspection')))
    loaded=module._inspection(store,ref,generation_arguments=args,immutable_generation_root=generation)
    assert loaded['inspection_authenticated'] is True and loaded['new_fit_or_inference'] is False
    assert loaded['validation']==inspection['validation']
    assert all(row['outer_fold']==1 for row in events) and not (args['prepared_root']/'access.jsonl').exists()
    assert any(row['actor']=='frozen_secondary_V_coordinator' for row in events)
    for record in (workload,inspection,loaded):
        assert record['quality_eligible'] is record['native_workload_qualified'] is record['T_accessed'] is False


def test_legacy_generation_still_refuses_the_real_secondary_fit(fitted):
    args,_,_,_=fitted
    from treeneuron.research.native_generation_v2 import audit_native_generation
    with pytest.raises((ValueError,PermissionError)):audit_native_generation(**args)


@pytest.mark.parametrize('fault',['seed','query','role'])
def test_secondary_chain_rejects_changed_seed_or_original_roles_before_V(fitted,fault):
    module=api();args,store,events,_=fitted
    if fault=='seed':args=dict(args,expected_fit_seed=18)
    else:
        name='V-features.npz' if fault=='query' else 'F.npz'
        (args['prepared_root']/'outer-1'/name).write_bytes(b'changed authoritative role')
    with pytest.raises(ValueError):module._workload(store,args)
    assert not any(e['actor']=='frozen_secondary_V_coordinator' for e in events)


def test_public_secondary_workload_rejects_false_global_gate_before_role_metadata_or_output(tmp_path,monkeypatch):
    module=api()
    from treeneuron.research.transfer import TransferStore
    def forbidden(*a,**kw):raise AssertionError('False global gate cannot inspect data')
    monkeypatch.setattr(TransferStore,'__init__',forbidden)
    with pytest.raises((ValueError,FileNotFoundError)):
        module.inspect_secondary_native_workload(dataset='energy',barrier_reference=dict(path=str(tmp_path/'absent-barrier'),sha256='a'*64),
            qualification_reference=dict(root=str(tmp_path/'absent-registry'),inventory_sha256='b'*64),
            generation_arguments=dict(prepared_root=str(tmp_path/'forbidden-data')))
    assert list(tmp_path.iterdir())==[]
