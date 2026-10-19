"""Pure identity expansion on fixed-entropy fixtures; no observation generation."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from test_protocol import repository
from test_confirmation_seed_commitment import setup
from treeneuron.research import confirmation_seeds
from treeneuron.research.seeds import seed64,seed32


def api():
    assert importlib.util.find_spec('treeneuron.research.confirmation_namespaces') is not None, 'Missing confirmation role namespace expansion'
    from treeneuron.research import confirmation_namespaces
    return confirmation_namespaces


def prepare(repository,tmp_path,monkeypatch,claim='placement'):
    project=Path(__file__).resolve().parents[1];relative='src/treeneuron/research/confirmation_namespaces.py'
    target=repository/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/relative,target)
    _,args=setup(repository,tmp_path,monkeypatch,claim=claim)
    committed=confirmation_seeds.commit_confirmation_seeds(**args)
    return dict(path=committed.path,expected_file_sha256=committed.file_sha256,private_path=args['private_output'],
        method_path=args['method_path'],expected_method_sha256=args['expected_method_sha256'],repository_root=repository)


@pytest.mark.parametrize('claim',['placement','pruning_selection','residual_capacity'])
def test_expansion_reuses_the_committed_teacher_identity_and_exact_pairing_namespaces(claim,repository,tmp_path,monkeypatch):
    module=api();args=prepare(repository,tmp_path,monkeypatch,claim);plan=module.expand_confirmation_namespaces(**args)
    matrix='C2' if claim=='residual_capacity' else 'C1';used32=set();independent=[]
    assert plan.claim==claim and plan.matrix==matrix and plan.teacher_count==32
    assert plan.protected_access_authorized is plan.observation_generation_authorized is False
    assert plan.registry_completeness_qualified is plan.teacher_id_exclusions_verified is False
    for i,row in enumerate(plan.teachers):
        labels=(matrix,'primary',f'teacher-{i}');assert row['teacher_index']==i
        assert row['teacher_seed']==seed64(123,*labels,'teacher')
        assert row['reference_seed']==seed64(123,*labels,'reference')
        assert row['observation_seeds']=={role:seed64(123,*labels,role) for role in 'FGVT'}
        independent += [row['teacher_seed'],row['reference_seed'],*row['observation_seeds'].values()]
        assert len(row['repetitions'])==2
        for r,repeat in enumerate(row['repetitions']):
            assert repeat['repetition']==r
            for role in ('optimizer','continuation_stream','allocation','contrast'):
                assert repeat[role]==seed64(123,*labels,'repeat',str(r),role)
                independent.append(repeat[role])
            value,counter=seed32(123,*labels,'repeat',str(r),'optimizer',used=used32)
            assert (repeat['optimizer_library32'],repeat['optimizer_library32_counter'])==(value,counter)
    assert len(set(independent))==len(independent)==32*14
    assert len(used32)==64
    first=plan.teachers[0]['teacher_seed'];changed=plan.teachers;changed[0]['teacher_seed']=0
    assert plan.teachers[0]['teacher_seed']==first and str(first) not in repr(plan)
    assert module.expand_confirmation_namespaces(**args).namespace_sha256==plan.namespace_sha256


def test_expansion_never_draws_entropy_or_invokes_a_teacher_or_observation_generator(repository,tmp_path,monkeypatch):
    module=api();args=prepare(repository,tmp_path,monkeypatch)
    import secrets
    from treeneuron.data import synthetic
    def forbidden(*args,**kwargs):raise AssertionError('Namespace expansion must create no entropy, teacher or data')
    monkeypatch.setattr(secrets,'randbits',forbidden)
    for name in ('make_teacher','make_development','sample_teacher','make_from_config'):monkeypatch.setattr(synthetic,name,forbidden)
    before={p:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    result=module.expand_confirmation_namespaces(**args)
    assert result.teacher_count==32 and {p:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}==before


def test_an_independent_namespace_collision_fails_without_replacing_the_committed_root(repository,tmp_path,monkeypatch):
    module=api();args=prepare(repository,tmp_path,monkeypatch);before=args['private_path'].read_bytes()
    original=module.seed64
    monkeypatch.setattr(module,'seed64',lambda root,*labels:original(root,*labels) if labels[-1]=='teacher' else 42)
    with pytest.raises(ValueError,match='[Cc]ollision'):module.expand_confirmation_namespaces(**args)
    assert args['private_path'].read_bytes()==before


@pytest.mark.parametrize('change',['private_bytes','public_hash','dirty_source','running_module'])
def test_changed_commitment_or_namespace_code_cannot_be_used(change,repository,tmp_path,monkeypatch):
    module=api();args=prepare(repository,tmp_path,monkeypatch)
    if change=='private_bytes':args['private_path'].write_text('{}')
    elif change=='public_hash':args['expected_file_sha256']='0'*64
    elif change=='dirty_source':(repository/'src/model.py').write_text('# drift\n')
    else:monkeypatch.setattr(module,'__file__',str(repository/'src/model.py'))
    with pytest.raises(ValueError):module.expand_confirmation_namespaces(**args)
