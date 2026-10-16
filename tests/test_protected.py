"""Trap label readers test validation ordering; fixtures are not scientific evidence."""
import json
from unittest.mock import Mock

import numpy as np
import pytest

from test_protocol import repository,prepared,candidate,write,ref,git

ROLES=('keep','top','random','greedy','widen_upper','ordinary_lower','ordinary_upper','static_partial')

def predictors():
    return {name:lambda X: np.zeros(len(X)) for name in ROLES}


@pytest.fixture
def grant(candidate):
    from treeneuron.protocol import freeze
    from treeneuron.evaluation.protected import register_run
    root,decision,sources,manifests=candidate
    frozen=freeze(decision,sources,root/'protocol/frozen.yaml')
    model=write(root,'evidence/model.json',{'toy':1})
    register_run(frozen,manifests[0],run_id='unit-0-rep-0',models={name:ref(root,model) for name in ROLES},f_target_variance=1.)
    return root,frozen,manifests[0]


def test_protected_access_is_logged_before_label_read_and_writes_once(grant,monkeypatch):
    from treeneuron.evaluation import protected
    root,frozen,manifest=grant
    original=protected._test_labels
    def labels(*args):
        events=protected.access_events(frozen)
        assert events[-1]['event']=='labels_access'
        return original(*args)
    monkeypatch.setattr(protected,'_test_labels',labels)
    access=protected.open_test(frozen,manifest,run_id='unit-0-rep-0')
    result=access.evaluate(predictors())
    assert result['run_id']=='unit-0-rep-0' and set(result['metrics'])==set(ROLES)
    assert len(result['predictions']['top'])==manifest.assignments.count('T')
    assert len(list(root.rglob('predictions.json')))==1
    trap=Mock(side_effect=AssertionError('duplicate T label read'))
    monkeypatch.setattr(protected,'_test_labels',trap)
    again=protected.open_test(frozen,manifest,run_id='unit-0-rep-0')
    assert again.evaluate(predictors())==result
    trap.assert_not_called()


@pytest.mark.parametrize('change',['source','lock','config','analysis','manifest','run','teacher','self_construct','after_grant'])
def test_rejected_access_never_reads_test_labels(grant,monkeypatch,change):
    from dataclasses import replace
    from treeneuron.evaluation import protected
    root,frozen,manifest=grant
    trap=Mock(side_effect=AssertionError('T labels reached on rejection'))
    monkeypatch.setattr(protected,'_test_labels',trap)
    access=None
    if change=='after_grant':
        access=protected.open_test(frozen,manifest,run_id='unit-0-rep-0')
        (root/'src/model.py').write_text('drift')
    elif change in ('source','lock','config','analysis'):
        p={'source':'src/model.py','lock':'requirements.lock','config':'configs/scientific.yaml','analysis':'analysis/infer.py'}[change]
        (root/p).write_text('drift')
    elif change=='manifest': manifest=replace(manifest,dataset_id='different')
    elif change=='teacher': (root/'evidence/confirmation/teacher-0.json').write_text('{}')
    with pytest.raises((ValueError,TypeError)):
        if change=='self_construct': protected.ProtectedEvaluation(frozen,manifest,run_id='unit-0-rep-0')
        else:
            access=access or protected.open_test(frozen,manifest,run_id='missing' if change=='run' else 'unit-0-rep-0')
            access.evaluate(predictors())
    trap.assert_not_called()
    assert not list(root.rglob('predictions.json'))


def test_operational_resume_requires_existing_link_and_rejects_scientific_overrides(grant):
    from treeneuron.evaluation.protected import register_run
    root,frozen,manifest=grant
    model=ref(root,root/'evidence/model.json')
    for operational in ({'seed':999},{'resume_run_id':'absent'},{'verbosity':'verbose'}):
        with pytest.raises(ValueError):
            register_run(frozen,manifest,run_id='unit-0-rep-0',models={name:model for name in ROLES},
                f_target_variance=1.,operational=operational)
    register_run(frozen,manifest,run_id='unit-0-rep-0',models={name:model for name in ROLES},
        f_target_variance=1.,operational={'resume_run_id':'unit-0-rep-0','verbosity':'quiet'})


def test_scientific_amendment_preserves_old_hash_and_revokes_access(grant):
    from treeneuron.protocol import amend
    from treeneuron.evaluation.protected import open_test
    root,frozen,manifest=grant
    old=frozen.path.read_bytes()
    amend(frozen,reason='Changed inference policy',replacement_study_id='toy-v2')
    assert frozen.path.read_bytes()==old
    with pytest.raises(ValueError,match='amend'): open_test(frozen,manifest,run_id='unit-0-rep-0')
    amendment=list((root/'protocol/amendments').glob('*.json'))
    assert len(amendment)==1
    assert json.loads(amendment[0].read_text())['old_results']=='exploratory'


def test_authorized_saved_rows_reload_without_regeneration_and_reject_tamper(grant,monkeypatch):
    from treeneuron.evaluation import protected
    root,frozen,manifest=grant
    access=protected.open_test(frozen,manifest,run_id='unit-0-rep-0')
    access.evaluate(predictors())
    trap=Mock(side_effect=AssertionError('saved evidence must not regenerate T'))
    monkeypatch.setattr(protected,'_test_labels',trap)
    rows=protected.load_evaluation(frozen,run_id='unit-0-rep-0')
    assert len(rows['observations']['y'])==len(rows['row_id'])
    assert len(rows['observations']['noise_free_target'])==len(rows['row_id'])
    assert len(rows['observations']['group_id'])==len(rows['row_id'])
    trap.assert_not_called()
    output=next(root.rglob('predictions.json')); output.write_text('{}')
    with pytest.raises(ValueError,match='drift'): protected.load_evaluation(frozen,run_id='unit-0-rep-0')


def test_incomplete_label_access_never_retries_under_same_run(grant,monkeypatch):
    from treeneuron.evaluation import protected
    root,frozen,manifest=grant
    trap=Mock(side_effect=RuntimeError('simulated generation failure'))
    monkeypatch.setattr(protected,'_test_labels',trap)
    access=protected.open_test(frozen,manifest,run_id='unit-0-rep-0')
    with pytest.raises(RuntimeError): access.evaluate(predictors())
    assert trap.call_count==1
    with pytest.raises(ValueError,match='incomplete'):
        protected.open_test(frozen,manifest,run_id='unit-0-rep-0')
    assert trap.call_count==1


def test_registration_accepts_all_frozen_controls_but_rejects_unobserved_F_scale(candidate):
    from treeneuron.protocol import freeze
    from treeneuron.evaluation.protected import register_run
    root,decision,sources,manifests=candidate
    frozen=freeze(decision,sources,root/'protocol/frozen.yaml')
    model=write(root,'evidence/model.json',{'toy':1})
    roles=('keep','top','random','greedy','widen_upper','ordinary_lower','ordinary_upper','static_partial')
    path=register_run(frozen,manifests[0],run_id='unit-0-rep-0',
        models={name:ref(root,model) for name in roles},f_target_variance=1.)
    assert path.exists()
    with pytest.raises(ValueError,match='F'):
        register_run(frozen,manifests[0],run_id='unit-0-rep-1',
            models={name:ref(root,model) for name in roles},f_target_variance=999.)
