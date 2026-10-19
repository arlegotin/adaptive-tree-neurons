"""No observations: exercise pre-generation commitments in temporary Git roots."""
import json
from pathlib import Path
import shutil

import pytest

from test_protocol import repository,write
from test_protocol_v2 import create,method,sha
from treeneuron.config import canonical_hash


def api():
    from treeneuron.research import confirmation_seeds
    return confirmation_seeds


def setup(repository,tmp_path,monkeypatch,*,claim='placement',excluded=()):
    assert hasattr(api(),'commit_confirmation_seeds')
    import secrets
    # Fixed entropy is needed only to exercise exclusion/collision and replay.
    monkeypatch.setattr(secrets,'randbits',lambda bits:123)
    p=repository/'src/treeneuron/research/confirmation_seeds.py'
    p.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__).resolve().parents[1]/p.relative_to(repository),p)
    locked,_=create(repository,tmp_path,method=method(claim))
    registry=write(repository,'artifacts/excluded-seeds.json',
        dict(schema='research/v2-teacher-seed-registry',teacher_seeds=sorted(excluded)))
    kwargs=dict(method_path=locked.path,expected_method_sha256=locked.file_sha256,
        repository_root=repository,excluded_registry=dict(path=str(registry.relative_to(repository)),sha256=sha(registry)),
        teacher_count=32,public_output=tmp_path/'public.json',private_output=tmp_path/'private.json')
    return locked,kwargs


def test_committed_root_replays_distinct_new_teacher_seeds_without_generating_observations(repository,tmp_path,monkeypatch):
    _,kwargs=setup(repository,tmp_path,monkeypatch,excluded=[261018411,918401])
    result=api().commit_confirmation_seeds(**kwargs)
    public=json.loads(result.path.read_text())
    assert 'root' not in public['payload'] and 'teacher_seeds' not in public['payload']
    assert public['payload']['protected_access_authorized'] is False
    assert public['payload']['teacher_id_exclusions_verified'] is False
    assert kwargs['private_output'].stat().st_mode&0o077==0
    plan=api().load_confirmation_seeds(result.path,expected_file_sha256=result.file_sha256,
        private_path=kwargs['private_output'],method_path=kwargs['method_path'],
        expected_method_sha256=kwargs['expected_method_sha256'],repository_root=repository)
    assert plan.teacher_seeds[:2]==(8253235515464078001,5397323325650482004)
    assert len(plan.teacher_seeds)==len(set(plan.teacher_seeds))==32
    assert not set(plan.teacher_seeds)&{261018411,918401}
    assert plan.protected_access_authorized is False
    assert plan.claim=='placement'
    assert sorted(p.name for p in tmp_path.iterdir())==['method-lock.json','private.json','public.json','repo']


@pytest.mark.parametrize('claim',['placement','pruning_selection','residual_capacity'])
def test_claim_scopes_bind_the_candidate_before_root_commitment(claim,repository,tmp_path,monkeypatch):
    locked,kwargs=setup(repository,tmp_path,monkeypatch,claim=claim)
    result=api().commit_confirmation_seeds(**kwargs);payload=json.loads(result.path.read_text())['payload']
    assert payload['method_lock_hash']==locked.lock_hash
    assert payload['matrix']==('C2' if claim=='residual_capacity' else 'C1')
    assert payload['claim']==claim


def test_excluded_teacher_collision_aborts_before_publishing_any_seed_file(repository,tmp_path,monkeypatch):
    _,kwargs=setup(repository,tmp_path,monkeypatch,excluded=[8253235515464078001])
    with pytest.raises(ValueError,match='[Cc]ollision|[Ee]xcluded'):api().commit_confirmation_seeds(**kwargs)
    assert not kwargs['public_output'].exists() and not kwargs['private_output'].exists()


@pytest.mark.parametrize('change',['same_path','occupied_public','occupied_private','invalid_count','execution_output'])
def test_invalid_publication_cannot_overwrite_or_modify_execution(change,repository,tmp_path,monkeypatch):
    _,kwargs=setup(repository,tmp_path,monkeypatch)
    if change=='same_path':kwargs['private_output']=kwargs['public_output']
    elif change=='occupied_public':kwargs['public_output'].write_text('keep this')
    elif change=='occupied_private':kwargs['private_output'].write_text('keep this')
    elif change=='invalid_count':kwargs['teacher_count']=33
    else:kwargs['public_output']=repository/'src/forbidden.json'
    before={p:p.read_bytes() for p in (kwargs['public_output'],kwargs['private_output']) if p.is_file()}
    with pytest.raises(ValueError):api().commit_confirmation_seeds(**kwargs)
    assert all(p.read_bytes()==v for p,v in before.items())
    assert not (repository/'src/forbidden.json').exists()


@pytest.mark.parametrize('change',['private_bytes','private_permissions','registry','authority','method_binding','external_digest'])
def test_loading_rejects_drift_and_cannot_promote_seed_commitment_to_protected_authority(change,repository,tmp_path,monkeypatch):
    _,kwargs=setup(repository,tmp_path,monkeypatch)
    result=api().commit_confirmation_seeds(**kwargs)
    digest=result.file_sha256;method_digest=kwargs['expected_method_sha256']
    if change=='private_bytes':kwargs['private_output'].write_text('{}')
    elif change=='private_permissions':kwargs['private_output'].chmod(0o644)
    elif change=='registry':(repository/kwargs['excluded_registry']['path']).write_text('{}')
    elif change=='external_digest':digest='0'*64
    else:
        value=json.loads(result.path.read_text())
        if change=='authority':value['payload']['protected_access_authorized']=True
        else:value['payload']['method_file_sha256']='0'*64
        value['commitment_hash']=canonical_hash(value['payload']);result.path.write_text(json.dumps(value));digest=sha(result.path)
    with pytest.raises(ValueError):api().load_confirmation_seeds(result.path,expected_file_sha256=digest,
        private_path=kwargs['private_output'],method_path=kwargs['method_path'],
        expected_method_sha256=method_digest,repository_root=repository)


def test_failed_public_write_retains_private_root_without_a_false_public_commitment(repository,tmp_path,monkeypatch):
    _,kwargs=setup(repository,tmp_path,monkeypatch)
    def disk_full(path,value):raise OSError(28,'controlled publication failure')
    monkeypatch.setattr(api(),'write_new_json',disk_full)
    with pytest.raises(OSError):api().commit_confirmation_seeds(**kwargs)
    private=kwargs['private_output']
    assert private.is_file() and json.loads(private.read_text())['root']==123
    assert private.stat().st_mode&0o077==0 and not kwargs['public_output'].exists()
    with pytest.raises(ValueError):api().commit_confirmation_seeds(**kwargs)


@pytest.mark.parametrize('change',['registry','source','method'])
def test_inputs_changed_during_root_draw_fail_before_any_publication(change,repository,tmp_path,monkeypatch):
    _,kwargs=setup(repository,tmp_path,monkeypatch)
    import secrets
    def changing_entropy(bits):
        if change=='registry':target=repository/kwargs['excluded_registry']['path']
        elif change=='source':target=repository/'src/model.py'
        else:target=kwargs['method_path']
        target.write_text('{}')
        return 123
    monkeypatch.setattr(secrets,'randbits',changing_entropy)
    with pytest.raises(ValueError):api().commit_confirmation_seeds(**kwargs)
    assert not kwargs['public_output'].exists() and not kwargs['private_output'].exists()


def test_failed_publication_resumes_the_same_committed_private_root_without_redrawing(repository,tmp_path,monkeypatch):
    _,kwargs=setup(repository,tmp_path,monkeypatch)
    assert hasattr(api(),'resume_confirmation_seed_publication')
    writer=api().write_new_json
    def disk_full(path,value):raise OSError(28,'controlled publication failure')
    monkeypatch.setattr(api(),'write_new_json',disk_full)
    with pytest.raises(OSError):api().commit_confirmation_seeds(**kwargs)
    private=kwargs['private_output'];before=private.read_bytes();digest=sha(private)
    import secrets
    def no_new_entropy(bits):raise AssertionError('A resume must retain the existing root')
    monkeypatch.setattr(secrets,'randbits',no_new_entropy)
    monkeypatch.setattr(api(),'write_new_json',writer)
    resume=dict(method_path=kwargs['method_path'],expected_method_sha256=kwargs['expected_method_sha256'],
        repository_root=repository,excluded_registry=kwargs['excluded_registry'],private_path=private,
        expected_private_sha256=digest,public_output=kwargs['public_output'])
    result=api().resume_confirmation_seed_publication(**resume)
    assert private.read_bytes()==before
    plan=api().load_confirmation_seeds(result.path,expected_file_sha256=result.file_sha256,
        private_path=private,method_path=kwargs['method_path'],
        expected_method_sha256=kwargs['expected_method_sha256'],repository_root=repository)
    assert plan.teacher_seeds[:2]==(8253235515464078001,5397323325650482004)
    with pytest.raises(ValueError):api().resume_confirmation_seed_publication(**resume)


def test_seed_publication_recovery_rejects_uncommitted_private_bytes(repository,tmp_path,monkeypatch):
    _,kwargs=setup(repository,tmp_path,monkeypatch)
    assert hasattr(api(),'resume_confirmation_seed_publication')
    def disk_full(path,value):raise OSError(28,'controlled publication failure')
    monkeypatch.setattr(api(),'write_new_json',disk_full)
    with pytest.raises(OSError):api().commit_confirmation_seeds(**kwargs)
    with pytest.raises(ValueError):api().resume_confirmation_seed_publication(
        method_path=kwargs['method_path'],expected_method_sha256=kwargs['expected_method_sha256'],
        repository_root=repository,excluded_registry=kwargs['excluded_registry'],
        private_path=kwargs['private_output'],expected_private_sha256='0'*64,public_output=kwargs['public_output'])
    assert not kwargs['public_output'].exists()
