"""Repeated prerequisites are shared only within one checked invocation."""
import hashlib
import importlib
import json

import pytest


def api():
    return importlib.import_module('treeneuron.research.validation_invocation_v2')


def reader(module, path, visits):
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    def load():
        visits.append(path.name)
        return json.loads(path.read_bytes())
    def guard(value):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Prerequisite bytes changed')
        assert value == {'native_workload_qualified': False, 'rows': 168}
    return lambda: module._load_once('toy-global', {'path': str(path)}, load, guard=guard)


def test_672_nested_searches_do_not_reconstruct_168_rows_each_time(tmp_path):
    module = api(); path = tmp_path/'metadata.json'; visits = []
    path.write_text(json.dumps({'native_workload_qualified': False, 'rows': 168}))
    load = reader(module, path, visits)
    @module._within_validation
    def secondary():
        first = load(); second = load()
        assert first == second
        return first['rows']
    @module._within_validation
    def complete():
        assert load()['native_workload_qualified'] is False
        return sum(secondary() for _ in range(672))
    assert complete() == 112896
    assert visits == ['metadata.json']
    assert complete() == 112896
    assert visits == ['metadata.json', 'metadata.json']


def test_returned_metadata_cannot_mutate_saved_invocation_context(tmp_path):
    module = api(); path = tmp_path/'metadata.json'; visits = []
    path.write_text(json.dumps({'native_workload_qualified': False, 'rows': 168}))
    load = reader(module, path, visits)
    @module._within_validation
    def run():
        supplied = load(); supplied['rows'] = 0
        return load()
    assert run() == {'native_workload_qualified': False, 'rows': 168}


@pytest.mark.parametrize('raise_body', [False, True])
def test_final_input_guard_runs_after_success_and_body_failure(tmp_path, raise_body):
    module = api(); path = tmp_path/'metadata.json'; visits = []
    path.write_text(json.dumps({'native_workload_qualified': False, 'rows': 168}))
    load = reader(module, path, visits)
    @module._within_validation
    def run():
        load(); path.write_text('changed')
        if raise_body: raise RuntimeError('Known body error')
    with pytest.raises(ValueError, match='Prerequisite bytes changed'): run()


def test_source_change_is_rejected_before_reusing_a_prerequisite(tmp_path, monkeypatch):
    module = api(); path = tmp_path/'metadata.json'; visits = []
    path.write_text(json.dumps({'native_workload_qualified': False, 'rows': 168}))
    load = reader(module, path, visits)
    @module._within_validation
    def run():
        load(); monkeypatch.setattr(module, '_source', lambda: 'changed-source')
        return load()
    with pytest.raises(ValueError, match='Source changed'): run()
    assert visits == ['metadata.json']


def test_failed_validation_is_never_reused(tmp_path):
    module = api(); calls = []
    def fail():
        calls.append(1); raise ValueError('Known failed prerequisite')
    @module._within_validation
    def run():
        for _ in range(2):
            with pytest.raises(ValueError, match='Known failed'):
                module._load_once('toy-failure', {}, fail, guard=lambda value: None)
    run(); assert calls == [1, 1]


def test_private_reuse_cannot_be_started_by_supplied_metadata():
    module = api()
    with pytest.raises(ValueError, match='invocation'):
        module._load_once('caller-flags', {}, lambda: {'qualified': True}, guard=lambda value: None)


def test_publication_guard_finishes_before_output_is_written(tmp_path):
    module = api(); path = tmp_path/'metadata.json'; visits = []
    path.write_text(json.dumps({'native_workload_qualified': False, 'rows': 168}))
    load = reader(module, path, visits); output = tmp_path/'never-published.json'
    @module._within_validation
    def publish():
        load(); path.write_text('changed')
        module._recheck_validation()
        output.write_text('must not appear')
    with pytest.raises(ValueError, match='Prerequisite bytes changed'): publish()
    assert not output.exists()


def test_valid_publication_can_recheck_inputs_before_writing(tmp_path):
    module = api(); path = tmp_path/'metadata.json'; visits = []
    path.write_text(json.dumps({'native_workload_qualified': False, 'rows': 168}))
    load = reader(module, path, visits); output = tmp_path/'published.json'
    @module._within_validation
    def publish():
        value = load(); module._recheck_validation()
        output.write_text(json.dumps(value))
    publish()
    assert json.loads(output.read_bytes()) == {'native_workload_qualified': False, 'rows': 168}
