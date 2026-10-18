"""Interim reporting must not turn event arithmetic into a raw audit."""
import importlib.util
import json
from pathlib import Path

import pytest

from treeneuron.config import canonical_hash, canonical_json


def reporter():
    path=Path(__file__).parents[1]/'docs/research/v2/scripts/summarize_positive_prefix.py'
    assert path.exists(), 'The read-only positive-prefix reporter is not implemented'
    spec=importlib.util.spec_from_file_location('positive_prefix_report',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.summarize


def fixture():
    attempts=[dict(attempt_id=f'{g}/{s}/{t}/{r}/{role}',population='positive',gamma=g,sigma=s,
        teacher_index=t,repetition=r,role=role) for g in (.5,2.) for s in (0.,.1)
        for t in range(4) for r in range(2) for role in ('keep','top','random','oracle_current','oracle_true')]
    plan=dict(attempts=attempts,source_sha256='1'*64,protected_labels_accessed=False)
    plan['plan_hash']=canonical_hash(plan)
    events=[];previous='0'*64
    for a in attempts:
        for status in ('running','completed_eligible'):
            event=dict(attempt_id=a['attempt_id'],status=status,previous_hash=previous,elapsed_ns=0)
            if status=='completed_eligible':
                value=1. if a['role']!='oracle_true' else (.95 if a['teacher_index']<2 else 1.01)
                event['metrics']={split:dict(normalized_mse=value) for split in ('V','E')}
            event['event_hash']=canonical_hash(event);previous=event['event_hash'];events.append(event)
    raw=('\n'.join(canonical_json(e) for e in events)+'\n').encode()
    return plan,raw


def test_closed_positive_subset_uses_fixed_sensitivity_and_provisional_scope():
    plan,raw=fixture();summary,prefix=reporter()(plan,raw,plan['plan_hash'])
    assert prefix==raw and summary['prefix_event_count']==320
    assert summary['positive_sensitivity']['qualifying_instances']==2
    assert summary['positive_sensitivity']['sensitive'] is True
    assert summary['independent_raw_audit_pending'] is True
    assert summary['Gate_A_decided'] is False and summary['T_accessed'] is False
    assert summary['cells'][0]['mean_risk']['V']['oracle_true']==.98
    assert len(summary['cells'][0]['teachers'])==4


def test_earliest_closed_prefix_excludes_later_live_append():
    plan,raw=fixture();summary,prefix=reporter()(plan,raw+b'{"partial":',plan['plan_hash'])
    assert prefix==raw and summary['prefix_event_count']==320


@pytest.mark.parametrize('change', ['event','plan','missing','duplicate'])
def test_rejects_uncommitted_or_unclosed_positive_subset(change):
    plan,raw=fixture();expected=plan['plan_hash']
    if change=='event':raw=raw.replace(b'"elapsed_ns":0',b'"elapsed_ns":1',1)
    elif change=='plan':plan['source_sha256']='2'*64
    elif change=='missing':raw=b'\n'.join(raw.splitlines()[:-1])+b'\n'
    else:
        plan['attempts'][-1]=plan['attempts'][0]
        plan['plan_hash']=canonical_hash({k:v for k,v in plan.items() if k!='plan_hash'});expected=plan['plan_hash']
    with pytest.raises(ValueError):reporter()(plan,raw,expected)


def test_failed_positive_slot_keeps_cell_incomplete_without_complete_case_mean():
    plan,raw=fixture();events=[json.loads(x) for x in raw.splitlines()]
    events[-1]['status']='numerical_failure';events[-1].pop('metrics')
    event=events[-1];event.pop('event_hash');event['event_hash']=canonical_hash(event)
    raw=('\n'.join(canonical_json(e) for e in events)+'\n').encode()
    summary,_=reporter()(plan,raw,plan['plan_hash'])
    assert summary['cells'][-1]['mean_risk'] is None
    assert summary['positive_sensitivity']['sensitive'] is None
    assert summary['status_counts']=={'completed_eligible':159,'numerical_failure':1}
