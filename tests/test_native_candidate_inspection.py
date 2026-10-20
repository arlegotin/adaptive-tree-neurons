"""Generation, independent reload and V arithmetic need one ordered evidence chain."""
import importlib.util
import json

import pytest

from test_native_generation_chain import actual


def api():
    assert importlib.util.find_spec('treeneuron.research.native_candidate_v2') is not None,'Missing candidate inspection assembly'
    from treeneuron.research.native_candidate_v2 import inspect_primary_native_candidate
    return inspect_primary_native_candidate


def access(args):
    return [json.loads(line) for line in (args['prepared_root']/'access.jsonl').read_text().splitlines()]


def completed_native(tmp_path,monkeypatch):
    args,attempt=actual(tmp_path,monkeypatch)
    assert json.loads((attempt/'native/result.json').read_text())['status']=='completed_ineligible'
    assert json.loads((attempt/'native/process/process.json').read_text())['child_reaped'] is True
    return args,attempt


def inspect(fn,args,attempt,output,**kwargs):
    import sys
    return fn(generation_arguments=args,immutable_generation_root=attempt,
        output=output,interpreter=sys.executable,**kwargs)


def test_real_native_candidate_replays_in_a_fresh_label_free_worker_before_primary_V(tmp_path,monkeypatch):
    from treeneuron.research.workqueue import _inventory
    fn=api();args,attempt=completed_native(tmp_path,monkeypatch);before=_inventory(attempt)
    result=inspect(fn,args,attempt,tmp_path/'inspection')
    assert result['status']=='completed_candidate_inspection'
    assert result['generation']['model_generation_chain_authenticated'] is True
    assert result['replay']['prediction_reload_exact'] is True and result['replay']['labels_read'] is False
    assert result['validation']['score']>=0 and result['V_labels_read_by_coordinator'] is True
    assert result['quality_eligible'] is result['native_workload_qualified'] is result['HPO_journal_authenticated'] is result['T_accessed'] is False
    assert result['process']['timeout_ns'] is None and result['process']['child_reaped'] is True
    assert result['elapsed_ns']>=result['replay_supervisor_span_ns']>0
    assert any('V_labels' in row['roles'] for row in access(args))
    assert _inventory(attempt)==before
    import os
    from pathlib import Path
    source=Path(result['generation']['snapshot']['path']).parent
    files=[p for p in source.rglob('*') if p.is_file()]
    assert files and all(os.path.samefile(p,tmp_path/'inspection/input/snapshot'/p.relative_to(source)) for p in files)
    assert len(result['input_materialization'])==len(files)+2


@pytest.mark.parametrize('change',['seed','overlap','container'])
def test_invalid_generation_or_output_is_rejected_before_replay_and_V(tmp_path,monkeypatch,change):
    fn=api();args,attempt=completed_native(tmp_path,monkeypatch);output=tmp_path/'inspection';container=attempt
    if change=='seed':args['expected_fit_seed']=18
    elif change=='overlap':output=attempt/'inspection'
    else:container=args['prepared_root']
    with pytest.raises(ValueError):inspect(fn,args,container,output)
    assert not output.exists() and all('V_labels' not in row['roles'] for row in access(args))


def test_owned_replay_memory_stop_cannot_read_V_or_promote_candidate(tmp_path,monkeypatch):
    fn=api();args,attempt=completed_native(tmp_path,monkeypatch)
    result=inspect(fn,args,attempt,tmp_path/'inspection',rss_limit_bytes=1)
    assert result['status']=='resource_exhausted' and result['process']['child_reaped'] is True
    assert result['validation'] is None and result['V_labels_read_by_coordinator'] is False
    assert result['quality_eligible'] is result['T_accessed'] is False
    assert all('V_labels' not in row['roles'] for row in access(args))


def test_real_V_score_does_not_hide_late_original_role_replacement(tmp_path,monkeypatch):
    fn=api();args,attempt=completed_native(tmp_path,monkeypatch)
    from treeneuron.research import native_candidate_v2 as module
    original=module.score_primary_validation_prediction
    def change_after_real_arithmetic(**kwargs):
        result=original(**kwargs)
        path=args['prepared_root']/'outer-0/G.npz';replacement=path.with_name('changed.tmp')
        replacement.write_bytes(b'late original role change');replacement.replace(path)
        return result
    monkeypatch.setattr(module,'score_primary_validation_prediction',change_after_real_arithmetic)
    with pytest.raises(ValueError):inspect(fn,args,attempt,tmp_path/'inspection')
    saved=json.loads((tmp_path/'inspection/result.json').read_text())
    assert saved['status']=='input_or_dependency_failure' and saved['quality_eligible'] is False


@pytest.mark.parametrize('changed',['V_labels','replay_terminal'])
def test_final_cross_component_checks_reject_late_validation_or_replay_evidence_change(tmp_path,monkeypatch,changed):
    fn=api();args,attempt=completed_native(tmp_path,monkeypatch)
    from treeneuron.research import native_candidate_v2 as module
    original=module.audit_native_generation;calls=[]
    def change_after_final_generation_check(**kwargs):
        result=original(**kwargs);calls.append(True)
        if len(calls)==3:
            path=(args['prepared_root']/'coordinator/outer-0/V-labels.npz') if changed=='V_labels' else tmp_path/'inspection/replay/terminal.json'
            replacement=path.with_name('late-change.tmp');replacement.write_bytes(path.read_bytes()+b' ');replacement.replace(path)
        return result
    monkeypatch.setattr(module,'audit_native_generation',change_after_final_generation_check)
    with pytest.raises(ValueError):inspect(fn,args,attempt,tmp_path/'inspection')
