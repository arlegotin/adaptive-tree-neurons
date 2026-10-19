"""Exact TPE event replay includes pending choices and paid failures."""
import copy
import hashlib
import json

import pytest

from treeneuron.config import canonical_hash


def api():
    from treeneuron.research import search_v2
    return search_v2


def support():
    return dict(x=dict(kind='float',low=.001,high=1.,log=True),
        width=dict(kind='integer',low=64,high=128,step=16,log=False),
        act=dict(kind='categorical',choices=['relu','gelu']))


def make(**overrides):
    kwargs=dict(seed=871,support=support(),identity=dict(study='artificial',source_sha256='a'*64),max_suggestions=32)
    return api().ReplayTPEV2(**(kwargs|overrides))


def objective(params):
    return float((params['x']-.11)**2+(params['width']-96)**2/10000+(0 if params['act']=='gelu' else .02))


def test_interrupted_tpe_after_posterior_startup_has_identical_next_suggestion(tmp_path):
    control=make()
    for index in range(18):
        params=control.ask();control.tell(None if index in (2,9) else objective(params))
    reference=control.save(tmp_path/'state.json')
    restored=api().ReplayTPEV2.load(tmp_path/'state.json',expected_sha256=reference['sha256'],identity=dict(study='artificial',source_sha256='a'*64))
    assert len(restored.history)==18 and restored.pending_suggestion is None
    assert sum(row['value'] is None for row in restored.history)==2
    for _ in range(3):
        a=control.ask();b=restored.ask();assert a==b
        control.tell(objective(a));restored.tell(objective(b))
    assert control.history==restored.history


def test_pending_ask_is_replayed_without_extra_rng_draw_or_free_attempt(tmp_path):
    control=make(max_suggestions=3)
    first=control.ask();control.tell(None)
    pending=control.ask();ref=control.save(tmp_path/'pending.json')
    loaded=api().ReplayTPEV2.load(tmp_path/'pending.json',expected_sha256=ref['sha256'],identity=dict(study='artificial',source_sha256='a'*64))
    assert loaded.pending_suggestion==pending
    with pytest.raises(ValueError,match='pending'):loaded.ask()
    loaded.tell(objective(pending));control.tell(objective(pending))
    assert loaded.ask()==control.ask()
    loaded.tell(None)
    with pytest.raises(ValueError,match='limit'):loaded.ask()
    assert len(loaded.history)==3 and sum(x['value'] is None for x in loaded.history)==2


def test_source_and_file_commitments_are_required_for_resume(tmp_path):
    obj=make()
    obj.ask();obj.tell(1.);ref=obj.save(tmp_path/'state.json')
    with pytest.raises(ValueError):api().ReplayTPEV2.load(tmp_path/'state.json',expected_sha256='0'*64,identity=dict(study='artificial',source_sha256='a'*64))
    with pytest.raises(ValueError):api().ReplayTPEV2.load(tmp_path/'state.json',expected_sha256=ref['sha256'],identity=dict(study='changed',source_sha256='a'*64))
    original=json.loads((tmp_path/'state.json').read_text())
    original['state']['history'][0]['params']['width']=65
    original['sha256']=canonical_hash(original['state'])
    path=tmp_path/'altered.json';path.write_text(json.dumps(original))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):api().ReplayTPEV2.load(path,expected_sha256=digest,identity=dict(study='artificial',source_sha256='a'*64))


def test_state_publication_is_immutable_and_public_values_do_not_mutate_sampler(tmp_path):
    obj=make();params=obj.ask();saved=copy.deepcopy(params);params['width']=-1
    assert obj.pending_suggestion==saved
    obj.tell(objective(saved));history=obj.history;history[0]['value']=999
    assert obj.history[0]['value']!=999
    obj.save(tmp_path/'state.json')
    with pytest.raises((ValueError,FileExistsError)):obj.save(tmp_path/'state.json')


def test_support_key_order_does_not_change_seeded_choices_and_integer_step_is_real():
    a=make();b=make(support=dict(reversed(list(support().items()))))
    for _ in range(15):
        x=a.ask();y=b.ask();assert x==y
        assert x['width'] in (64,80,96,112,128)
        a.tell(objective(x));b.tell(objective(y))


@pytest.mark.parametrize('bad',[
 dict(kind='integer',low=64,high=127,step=16,log=False),
 dict(kind='integer',low=0,high=32,step=1,log=True),
 dict(kind='integer',low=1,high=32,step=2,log=True),
 dict(kind='categorical',choices=[1,True]),
 dict(kind='categorical',choices=[]),
 dict(kind='categorical',choices=[float('nan')]),
 dict(kind='float',low=0.,high=1.,log=True),
 dict(kind='float',low=.1,high=1.,log=False,unknown=True)])
def test_invalid_support_never_silently_adjusts_bounds_or_categories(bad):
    with pytest.raises(ValueError):make(support=dict(x=bad))


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf'),True])
def test_only_nonnegative_finite_V_objectives_or_paid_failure_can_be_told(value):
    obj=make();obj.ask()
    with pytest.raises(ValueError):obj.tell(value)
    assert obj.pending_suggestion is not None
    obj.tell(None);assert len(obj.history)==1


def test_replayed_parameter_types_cannot_change_under_numerical_equality(tmp_path):
    obj=make(support=dict(x=dict(kind='float',low=1.,high=1.,log=False)))
    assert obj.ask()=={'x':1.};obj.tell(1.)
    obj.save(tmp_path/'original.json');payload=json.loads((tmp_path/'original.json').read_text())
    payload['state']['history'][0]['params']['x']=True
    payload['sha256']=canonical_hash(payload['state'])
    path=tmp_path/'typed-corruption.json';path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='replay'):
        api().ReplayTPEV2.load(path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            identity=dict(study='artificial',source_sha256='a'*64))
