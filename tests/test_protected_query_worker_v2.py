"""Owned cold leaf fixtures are numerical/process evidence, never native gates."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_protected_query_v2 import saved_query


def api():
    assert importlib.util.find_spec('treeneuron.research.protected_query_worker_v2'),'Missing owned saved-query leaf'
    from treeneuron.research import protected_query_worker_v2
    return protected_query_worker_v2


def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def delegated(tmp_path,saved_query):
    module=api();_,_,core=saved_query
    from treeneuron.research import protected_journal_v2 as journal
    protocol=tmp_path/'toy-protocol.json';protocol.write_text('{"unqualified_fixture":true}\n')
    # Rebind the private core's dummy protocol identity to these actual metadata bytes.
    core['protocol_sha256']=ref(protocol)['sha256'];query=core['query'];study=tmp_path/'study';owners=tmp_path/'owners'
    binding=journal._prepare_metadata_study(directory=study,owner_root=owners,protocol_reference=ref(protocol),source_sha256=core['source_sha256'],
        prediction_manifest_sha256=core['prediction_manifest_sha256'],queries=[query],assets=[dict(dataset='toy',raw_sha256='d'*64,target='known-target')])
    journal._claim_metadata_owners(study)
    journal._append_metadata_event(study,kind='query_started',details=dict(query_id=query['query_id'],attempt_index=0,resume_kind='initial'))
    state=journal._read_metadata_study(study);episode=study/'queries'/query['query_id']/'episode-0000';episode.mkdir(parents=True)
    value=dict(schema='research/v2-owned-saved-query-delegate',source_sha256=core['source_sha256'],protocol_reference=ref(protocol),
        study_directory=str(study),study_binding_reference=ref(study/'binding.json'),genesis_sha256=binding['genesis_sha256'],
        query_id=query['query_id'],attempt_index=0,journal_sha256=state['journal_sha256'],query_started_event_hash=state['last_event_hash'],
        core_request=core,core_request_sha256=journal._hash(core),worker_output=str(episode/'worker'))
    request=episode/'request.json';request.write_text(journal._canonical(value)+'\n')
    return module,request,value,ref(request)['sha256']


def test_delegate_metadata_matches_exact_active_query_without_importing_method(saved_query,tmp_path):
    m,path,value,digest=delegated(tmp_path,saved_query)
    actual=m._checked_delegate(path,digest,output=Path(value['worker_output']))
    assert actual==value


@pytest.mark.parametrize('fault',['hash','source','protocol','binding','genesis','query','attempt','journal','event','core_hash','output','unknown'])
def test_changed_delegate_rejects_before_feature_archive_access(saved_query,tmp_path,fault,monkeypatch):
    m,path,value,digest=delegated(tmp_path,saved_query)
    if fault=='hash':digest='0'*64
    elif fault=='source':value['source_sha256']='0'*64
    elif fault=='protocol':value['protocol_reference']['sha256']='0'*64
    elif fault=='binding':value['study_binding_reference']['sha256']='0'*64
    elif fault=='genesis':value['genesis_sha256']='0'*64
    elif fault=='query':value['query_id']='0'*64
    elif fault=='attempt':value['attempt_index']=True
    elif fault=='journal':value['journal_sha256']='0'*64
    elif fault=='event':value['query_started_event_hash']='0'*64
    elif fault=='core_hash':value['core_request_sha256']='0'*64
    elif fault=='output':value['worker_output']=str(tmp_path/'escaped')
    else:value['native_qualified']=True
    if fault!='hash':path.write_text(json.dumps(value)+'\n');digest=ref(path)['sha256']
    original=Path.read_bytes
    def no_arrays(p):
        assert p.suffix!='.npz','Delegate admission cannot open any feature/model arrays'
        return original(p)
    monkeypatch.setattr(Path,'read_bytes',no_arrays)
    with pytest.raises((ValueError,PermissionError)):m._checked_delegate(path,digest,output=Path(value['worker_output']))
    assert not Path(value['worker_output']).exists()


def test_unowned_cli_cannot_read_delegate_or_any_feature_array(saved_query,tmp_path):
    m,path,value,digest=delegated(tmp_path,saved_query)
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    env.pop('ATN_SEARCH_CONTROL_ROOT',None);env.pop('ATN_SEARCH_CONTROL_START_SHA256',None)
    done=subprocess.run([sys.executable,'-P','-B','-m','treeneuron.research.protected_query_worker_v2','--request',str(path),
        '--expected-request-sha256',digest,'--output',value['worker_output']],env=env,capture_output=True,text=True,timeout=20)
    assert done.returncode!=0 and not Path(value['worker_output']).exists()


def test_actual_owned_fixture_closes_and_binds_finite_output_without_torch(saved_query,tmp_path):
    m,path,value,digest=delegated(tmp_path,saved_query)
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process,inspect_cooperative_process
    control=path.parent/'process';project=Path(__file__).resolve().parents[1]
    env={key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}
    env['PYTHONPATH']=str(project/'src');env['KMP_DUPLICATE_LIB_OK']=None
    identity=m.protected_query_worker_identity(path,digest)
    result=supervise_cooperative_process(argv=[sys.executable,'-P','-B','-m','treeneuron.research.protected_query_worker_v2',
        '--request',str(path),'--expected-request-sha256',digest,'--output',value['worker_output']],cwd=project,env_overrides=env,
        output=control,identity=identity,admission_identity=dict(purpose='unqualified_fixture_only'),
        admission_check=lambda:dict(admitted=True,reason='private numerical/process fixture; no actual qualification'),
        rss_pause_bytes=1024**3,admit_before_spawn=True)
    assert result['returncode']==0,result
    proof=inspect_cooperative_process(root=control,expected_receipt_sha256=ref(control/'process.json')['sha256'],
        expected_started_sha256=ref(control/'started.json')['sha256'])
    assert proof['worker_protocol_closed'] is True and proof['returncode']==0
    body=json.loads((control/'body-outcome.json').read_text())
    assert body['status']=='closed_saved_query_body'
    assert body['native_qualification_authenticated'] is body['protected_access_authorized'] is body['T_accessed'] is False
    assert body['query_id']==value['query_id'] and body['core_request_sha256']==value['core_request_sha256']
    terminal=json.loads(Path(body['terminal_reference']['path']).read_text())
    assert terminal['loaded_heavy_modules']==[]
    readback=json.loads(Path(body['readback_reference']['path']).read_text())
    assert readback['finite_aligned_prediction_artifact_checked'] is True
    assert hasattr(m,'inspect_owned_saved_query'),'Missing owned saved-query proof join'
    process_reference=dict(process_root=str(control),process_receipt_sha256=ref(control/'process.json')['sha256'],
        process_started_sha256=ref(control/'started.json')['sha256'])
    joined=m.inspect_owned_saved_query(process_reference)
    assert joined['owned_saved_query_body_checked'] is True and joined['finite_aligned_prediction_artifact_checked'] is True
    assert joined['protected_access_authorized'] is joined['native_qualification_authenticated'] is False
    from treeneuron.research import protected_journal_v2 as journal
    stub=tmp_path/'untrusted-later-commit.json';stub.write_text('{}\\n')
    journal._append_metadata_event(Path(value['study_directory']),kind='query_committed',details=dict(query_id=value['query_id'],attempt_index=0,
        prediction_reference=ref(stub),elapsed_ns=3))
    assert m.inspect_owned_saved_query(process_reference)['owned_saved_query_body_checked'] is True
    pred=Path(body['prediction_reference']['path'])
    with pred.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError):m.inspect_owned_saved_query(process_reference)


def test_historical_delegate_prefix_remains_readable_after_a_valid_query_commit(saved_query,tmp_path):
    m,path,value,digest=delegated(tmp_path,saved_query)
    assert hasattr(m,'_checked_delegate_prefix'),'Missing retained delegate prefix reader'
    from treeneuron.research import protected_journal_v2 as journal
    proof=tmp_path/'untrusted-commit.json';proof.write_text('{"metadata_only":true}\n')
    journal._append_metadata_event(Path(value['study_directory']),kind='query_committed',details=dict(query_id=value['query_id'],attempt_index=0,
        prediction_reference=ref(proof),elapsed_ns=3))
    checked=m._checked_delegate_prefix(path,digest,output=Path(value['worker_output']))
    assert checked==value
    with pytest.raises(ValueError):m._checked_delegate(path,digest,output=Path(value['worker_output']))


def test_owned_reader_requires_actual_closed_worker_not_metadata_flags(saved_query,tmp_path):
    m,path,value,digest=delegated(tmp_path,saved_query)
    assert hasattr(m,'inspect_owned_saved_query'),'Missing owned saved-query proof join'
    reference=dict(process_root=str(path.parent/'missing-process'),process_receipt_sha256='a'*64,process_started_sha256='b'*64)
    with pytest.raises((ValueError,FileNotFoundError)):m.inspect_owned_saved_query(reference)


@pytest.mark.parametrize('case',['prespawn_denial','forbidden_feature_labels'])
def test_owned_leaf_retains_denied_or_failed_execution_without_prediction(saved_query,tmp_path,case):
    if case=='forbidden_feature_labels':
        import numpy as np
        from test_protected_query_v2 import rebind
        from test_prediction_replay_v2 import rewrite_archive
        _,_,core=saved_query;feature=Path(core['query']['feature_reference']['path'])
        rewrite_archive(feature,lambda a:a.update(y=np.zeros(37)))
        core['query']['feature_reference']['sha256']=ref(feature)['sha256'];rebind(core)
    m,path,value,digest=delegated(tmp_path,saved_query)
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process,inspect_cooperative_process
    control=path.parent/'process';project=Path(__file__).resolve().parents[1]
    env={key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}
    env.update(PYTHONPATH=str(project/'src'),KMP_DUPLICATE_LIB_OK=None)
    identity=m.protected_query_worker_identity(path,digest)
    result=supervise_cooperative_process(argv=[sys.executable,'-P','-B','-m','treeneuron.research.protected_query_worker_v2',
        '--request',str(path),'--expected-request-sha256',digest,'--output',value['worker_output']],cwd=project,env_overrides=env,
        output=control,identity=identity,admission_identity=dict(purpose='unqualified_fixture_only'),
        admission_check=lambda:dict(admitted=case!='prespawn_denial',reason='private fixture denial/negative numerical input'),
        rss_pause_bytes=1024**3,admit_before_spawn=True)
    process_reference=dict(process_root=str(control),process_receipt_sha256=ref(control/'process.json')['sha256'],
        process_started_sha256=ref(control/'started.json')['sha256'])
    assert not (Path(value['worker_output'])/'prediction.npz').exists()
    if case=='prespawn_denial':
        assert result['child_started'] is False and result['grant_publication_started'] is False
    else:
        assert result['returncode']==2 and result['worker_protocol_closed'] is True
        terminal=json.loads((Path(value['worker_output'])/'terminal.json').read_text())
        assert terminal['adapter_import_started'] is False
        body=json.loads((control/'body-outcome.json').read_text())
        assert body['status']=='failed_saved_query_body' and body['protected_access_authorized'] is False
    with pytest.raises(ValueError):m.inspect_owned_saved_query(process_reference)
