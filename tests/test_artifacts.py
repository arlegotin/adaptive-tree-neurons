"""Exercise real disk artifacts and rejection before destructive side effects."""
import json
from types import SimpleNamespace

import pytest


def test_run_persists_resolved_config_and_complete_manifest_before_training(tmp_path):
    from treeneuron.artifacts import create_run
    from treeneuron.config import canonical_hash

    paths = create_run(tmp_path, {"study_id": "pilot"})
    assert paths.directory == tmp_path / paths.run_id
    resolved = json.loads((paths.directory / "config.resolved.json").read_text())
    assert resolved["model"]["tau"] == 1.0
    assert resolved["seeds"]["namespace"] == "exploratory"
    manifest = json.loads((paths.directory / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["identity"]["status"] == "created"
    assert manifest["identity"]["config_hash"] == canonical_hash(resolved)
    assert manifest["method"]["resolved_config"] == resolved
    assert set(manifest) >= {"identity", "data", "method", "costs", "resources"}
    assert manifest["data"]["boundaries"] == {
        "F": "fit_weights_and_transforms", "G": "select_checkpoint_and_growth",
        "V": "select_hyperparameters", "T": "protected_evaluator_only"}
    assert manifest["data"]["test_labels_accessed"] is False
    assert manifest["data"]["counts"]["F"]["rows"] == 2306
    assert manifest["data"]["counts"]["T"]["rows"]["value"] is None
    for name in ("rss_bytes", "mps_allocated_bytes", "mps_driver_bytes", "swap_delta_bytes"):
        assert manifest["resources"][name]["value"] is None
        assert manifest["resources"][name]["reason"]
    assert (paths.directory / "events.jsonl").read_text() == ""
    assert not (paths.directory / "metrics.json").exists()


def test_duplicate_run_id_refuses_to_overwrite(tmp_path, monkeypatch):
    from treeneuron import artifacts

    monkeypatch.setattr(artifacts.uuid, "uuid4", lambda: SimpleNamespace(hex="same-id"))
    first = artifacts.create_run(tmp_path, {"study_id": "pilot"})
    before = (first.directory / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        artifacts.create_run(tmp_path, {"study_id": "another"})
    assert (first.directory / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("resolved", [{"study_id": "pilot", "phase": "confirmatory"},
                                      {"study_id": "pilot", "model": {"tau": float("nan")}}])
def test_invalid_or_protected_config_creates_no_run(tmp_path, resolved):
    from treeneuron.artifacts import create_run

    with pytest.raises(ValueError):
        create_run(tmp_path, resolved)
    assert list(tmp_path.iterdir()) == []


def test_interruption_preserves_events_and_null_reasons(tmp_path):
    from treeneuron.artifacts import append_event, create_run, finish_run

    paths = create_run(tmp_path, {"study_id": "pilot"})
    append_event(paths, {"type": "base_started", "status": "running", "update": 0})
    original = (paths.directory / "events.jsonl").read_bytes()
    append_event(paths, {"type": "interruption", "reason": "user interrupt", "update": 3})
    finish_run(paths, {"rss_bytes": {"value": None, "reason": "counter unavailable"}}, "interrupted")
    assert (paths.directory / "events.jsonl").read_bytes().startswith(original)
    events = [json.loads(line) for line in (paths.directory / "events.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events] == ["base_started", "interruption"]
    metrics = json.loads((paths.directory / "metrics.json").read_text())
    assert metrics["rss_bytes"] == {"value": None, "reason": "counter unavailable"}
    manifest = json.loads((paths.directory / "manifest.json").read_text())
    assert manifest["identity"]["status"] == "interrupted"
    assert manifest["identity"]["failure_class"]["value"] is None
    before = (paths.directory / "metrics.json").read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        finish_run(paths, {"changed": True}, "completed")
    assert (paths.directory / "metrics.json").read_bytes() == before


def test_nonfinite_event_and_metrics_leave_valid_artifacts_untouched(tmp_path):
    from treeneuron.artifacts import append_event, create_run, finish_run

    paths = create_run(tmp_path, {"study_id": "pilot"})
    append_event(paths, {"type": "ok"})
    event_before = (paths.directory / "events.jsonl").read_bytes()
    manifest_before = (paths.directory / "manifest.json").read_bytes()
    with pytest.raises(ValueError):
        append_event(paths, {"loss": float("nan")})
    with pytest.raises(ValueError):
        finish_run(paths, {"loss": float("inf")}, "completed")
    assert (paths.directory / "events.jsonl").read_bytes() == event_before
    assert (paths.directory / "manifest.json").read_bytes() == manifest_before
    assert not (paths.directory / "metrics.json").exists()
    assert not list(paths.directory.glob("*.tmp"))


@pytest.mark.parametrize("status", ["created", "running", "scientific_failure", "unknown"])
def test_finish_requires_terminal_operational_status(tmp_path, status):
    from treeneuron.artifacts import create_run, finish_run

    paths = create_run(tmp_path, {"study_id": "pilot"})
    with pytest.raises(ValueError):
        finish_run(paths, {}, status)
    assert not (paths.directory / "metrics.json").exists()


def test_forged_run_paths_cannot_append_or_finish(tmp_path):
    from treeneuron.artifacts import RunPaths, append_event, create_run, finish_run

    paths = create_run(tmp_path, {"study_id": "pilot"})
    forged = RunPaths(run_id="wrong", directory=paths.directory)
    with pytest.raises(ValueError):
        append_event(forged, {"type": "wrong"})
    with pytest.raises(ValueError):
        finish_run(forged, {}, "completed")


def test_draft_has_no_authority_to_create_protected_run(tmp_path):
    import yaml
    from pathlib import Path
    from treeneuron.artifacts import create_run

    draft = yaml.safe_load(Path("protocol/draft.yaml").read_text())
    assert draft["status"] == "draft"
    assert draft["freeze_hash"] is None
    assert draft["test_access_authorized"] is False
    with pytest.raises(ValueError):
        create_run(tmp_path, {"study_id": "pilot", "phase": "confirmatory", "protocol": draft})


def test_observations_update_measured_fields_and_parent_atomically(tmp_path):
    from treeneuron.artifacts import create_run, update_run_observations
    paths = create_run(tmp_path, {'study_id': 'pilot'})
    update_run_observations(paths, {'identity': {'parent_run_id': 'parent', 'code_revision': 'abc'},
                                   'method': {'trainable_parameters': 17},
                                   'data': {'counts': {'F': {'rows': 2306, 'groups': 2306}}}})
    manifest = json.loads((paths.directory / 'manifest.json').read_text())
    assert manifest['identity']['parent_run_id'] == 'parent'
    assert manifest['method']['trainable_parameters'] == 17
    assert manifest['data']['counts']['G']['groups']['value'] is None
    before = (paths.directory / 'manifest.json').read_bytes()
    with pytest.raises(ValueError):
        update_run_observations(paths, {'method': {'trainable_parameters': 99}, 'identity': {'run_id': 'forged'}})
    assert (paths.directory / 'manifest.json').read_bytes() == before


@pytest.mark.parametrize('patch', [
    {'identity': {'protocol_hash': 'forged'}}, {'identity': {'study_id': 'forged'}},
    {'identity': {'status': 'completed'}}, {'method': {'resolved_config': {}}},
    {'data': {'test_labels_accessed': True}}, {'unknown': {}},
    {'method': {'trainable_parameters': -1}}, {'costs': {'base': {'nanoseconds': float('nan')}}},
])
def test_observation_api_rejects_identity_and_invalid_updates(tmp_path, patch):
    from treeneuron.artifacts import create_run, update_run_observations
    paths = create_run(tmp_path, {'study_id': 'pilot'})
    before = (paths.directory / 'manifest.json').read_bytes()
    with pytest.raises(ValueError): update_run_observations(paths, patch)
    assert (paths.directory / 'manifest.json').read_bytes() == before


def test_observation_api_rejects_terminal_and_inconsistent_records(tmp_path):
    from treeneuron.artifacts import create_run, finish_run, update_run_observations
    paths = create_run(tmp_path, {'study_id': 'pilot'})
    finish_run(paths, {}, 'failed')
    with pytest.raises(ValueError): update_run_observations(paths, {'identity': {'parent_run_id': 'p'}})
    other = create_run(tmp_path, {'study_id': 'pilot'})
    (other.directory / 'metrics.json').write_text('{}')
    with pytest.raises(ValueError): update_run_observations(other, {'identity': {'parent_run_id': 'p'}})


def test_observation_update_rejects_tampered_study_identity(tmp_path):
    from treeneuron.artifacts import create_run, update_run_observations
    paths = create_run(tmp_path, {'study_id': 'pilot'})
    path = paths.directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['identity']['study_id'] = 'other'
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='identity'):
        update_run_observations(paths, {'method': {'trainable_parameters': 17}})
    assert path.read_bytes() == before


def test_observation_update_rejects_unknown_nested_backend_field(tmp_path):
    from treeneuron.artifacts import create_run, update_run_observations
    paths = create_run(tmp_path, {'study_id': 'pilot'})
    with pytest.raises(ValueError, match='backend'):
        update_run_observations(paths, {'method': {'backend': {'actual': {
            'device': 'cpu', 'dtype': 'float32', 'fallback': False, 'invented': 'not observed'}}}})


def test_study_observations_are_strict_and_terminal_immutable(tmp_path):
    from treeneuron.artifacts import create_run, update_run_observations, finish_run
    paths=create_run(tmp_path,{'study_id':'study-observations'})
    patch={'identity':{'coordinator_run_id':'coordinator','score_hash':'a'*64},
           'data':{'boundaries':{'V':'descriptive_only'}},
           'method':{'scope':'single_model','prediction_artifacts':{'V':{'path':'predictions/top-V.json','sha256':'b'*64}}},
           'resources':{'memory_accounting':'process lifetime high-water; not additive'}}
    update_run_observations(paths,patch)
    manifest=json.loads((paths.directory/'manifest.json').read_text())
    assert manifest['data']['boundaries']=={'F':'fit_weights_and_transforms','G':'select_checkpoint_and_growth','V':'descriptive_only','T':'protected_evaluator_only'}
    assert manifest['method']['prediction_artifacts']==patch['method']['prediction_artifacts']
    before=(paths.directory/'manifest.json').read_bytes()
    for bad in [{'data':{'boundaries':{'T':'descriptive_only'}}},
                {'data':{'boundaries':{'V':'train'}}},
                {'method':{'scope':'invented'}},
                {'method':{'prediction_artifacts':{'T':{'path':'predictions/T.json','sha256':'b'*64}}}},
                {'method':{'prediction_artifacts':{'V':{'path':'../elsewhere','sha256':'bad'}}}},
                {'identity':{'score_hash':'c'*64}}]:
        with pytest.raises(ValueError):update_run_observations(paths,bad)
        assert (paths.directory/'manifest.json').read_bytes()==before
    finish_run(paths,{},'completed')
    with pytest.raises(ValueError):update_run_observations(paths,patch)
