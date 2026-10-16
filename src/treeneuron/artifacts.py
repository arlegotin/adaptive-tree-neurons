"""Exclusive run directories, atomic JSON snapshots and an append-only ledger.

The ledger supports serial exploratory execution. File locks serialize writers;
terminal snapshots never replace existing metrics. An interrupted run is retained
as evidence; a later resumed attempt must have a new run identity.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

from .config import canonical_hash, canonical_json, resolve_experiment

RUN_STATUSES = frozenset({"created", "running", "interrupted", "failed", "completed"})
TERMINAL_STATUSES = frozenset({"interrupted", "failed", "completed"})


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    directory: Path


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unavailable(reason: str = "not measured") -> dict[str, Any]:
    return {"value": None, "reason": reason}


def _atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> None:
    # Validate all bytes before opening a temporary file or touching a target.
    payload = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # Atomic publication with exclusive destination, unlike replace().
            os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(paths: RunPaths, resolved: dict[str, Any]) -> dict[str, Any]:
    missing = _unavailable
    counts = {}
    for split, field in (("F", "n_fit"), ("G", "n_growth"), ("V", "n_validation"), ("T", "n_test")):
        rows = resolved["data"][field]
        counts[split] = {
            "rows": rows if rows is not None else missing("not specified before pilot freeze"),
            "groups": missing("data not generated"),
        }
    stages = {
        "search": "hyperparameter search; V selection included",
        "preprocess": "F-only transform fitting and split transformation",
        "base": "parent training including G checkpoint selection",
        "score": "CPU float64 scoring and transfer from parent backend",
        "probe": "local gate-only descent check",
        "continuation": "post-growth logical optimizer updates",
        "validation": "G evaluation at update zero and shared intervals",
        "first_prediction": "first permitted prediction call",
        "warm_prediction": "subsequent permitted prediction call",
        "acquisition": "asset acquisition, separate from training costs",
    }
    return {
        "schema_version": 1,
        "identity": {
            "run_id": paths.run_id, "parent_run_id": missing("no parent supplied"),
            "study_id": resolved["study_id"], "phase": resolved["phase"], "status": "created",
            "failure_class": missing("no scientific failure assessed"),
            "config_hash": canonical_hash(resolved),
            **{field: missing("not supplied") for field in (
                "code_revision", "adapter_revision", "protocol_hash", "lock_hash", "hardware_hash", "growth_hash")},
        },
        "data": {
            "dataset": resolved["data"]["dataset"], "population": resolved["data"]["population"],
            **{field: missing("data not generated") for field in (
                "raw_hash", "split_hash", "transform_hash", "teacher_normalization")},
            "seeds": resolved["seeds"], "counts": counts,
            "boundaries": {"F": "fit_weights_and_transforms", "G": "select_checkpoint_and_growth",
                           "V": "select_hyperparameters", "T": "protected_evaluator_only"},
            "test_labels_accessed": False, "full_development_refit": False,
        },
        "method": {
            "resolved_config": resolved,
            **{field: missing("model not initialized") for field in (
                "trainable_parameters", "total_parameters", "serialized_bytes", "weight_checkpoint",
                "weight_hash", "weight_terms", "context_bytes", "cache_bytes")},
            "backend": {"requested": resolved["backend"], "actual": missing("execution not started")},
            "dtype": resolved["backend"]["training_dtype"],
            "threads": {"cpu": resolved["backend"]["cpu_threads"] if resolved["backend"]["cpu_threads"] is not None
                        else missing("runtime threads not observed"),
                        "interop": resolved["backend"]["interop_threads"] if resolved["backend"]["interop_threads"] is not None
                        else missing("runtime threads not observed")},
            "fallback": resolved["backend"]["allow_fallback"],
        },
        "costs": {stage: {"nanoseconds": None, "reason": "not measured", "status": "unmeasured",
                           "boundary": boundary} for stage, boundary in stages.items()},
        "resources": {
            **{field: missing() for field in ("rss_bytes", "mps_allocated_bytes", "mps_driver_bytes",
                                             "pressure", "swap_delta_bytes")},
            "memory_accounting": "RSS and MPS allocated/driver overlap; never sum as disjoint usage",
            "created_utc": _utc(), "finished_utc": missing("run not finished"),
            "interruption_links": missing("no interruption linked"),
            "retry_links": missing("no retry linked"),
            "timing_exclusions": missing("timing not started"),
        },
    }


def create_run(root: Path, resolved: Mapping[str, Any]) -> RunPaths:
    """Resolve exploratory config before exclusively claiming a unique directory."""
    config = resolve_experiment(resolved).to_dict()
    run_id = uuid.uuid4().hex
    paths = RunPaths(run_id=run_id, directory=root / run_id)
    manifest = _manifest(paths, config)
    # Preflight serialization before creating any artifacts.
    canonical_json(manifest)
    root.mkdir(parents=True, exist_ok=True)
    paths.directory.mkdir(exist_ok=False)
    _atomic_json(paths.directory / "config.resolved.json", config)
    _atomic_json(paths.directory / "manifest.json", manifest)
    for name in ("events.jsonl", ".ledger.lock"):
        with (paths.directory / name).open("x", encoding="utf-8"):
            pass
    return paths


@contextmanager
def _locked_manifest(paths: RunPaths):
    # r+ avoids creating anything when a caller forges or mistypes a path.
    with (paths.directory / ".ledger.lock").open("r+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            manifest = json.loads((paths.directory / "manifest.json").read_text(encoding="utf-8"))
            identity = manifest["identity"]
            config = json.loads((paths.directory / "config.resolved.json").read_text(encoding="utf-8"))
            if (identity["run_id"] != paths.run_id or paths.directory.name != paths.run_id
                    or identity["phase"] != "exploratory"
                    or identity["config_hash"] != canonical_hash(config)
                    or identity["study_id"] != config["study_id"]
                    or manifest["method"]["resolved_config"] != config):
                raise ValueError("Run identity, phase or resolved config hash mismatch")
            if identity["status"] not in RUN_STATUSES:
                raise ValueError("Unknown operational run status")
            yield manifest
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def append_event(paths: RunPaths, event: Mapping[str, Any]) -> None:
    """Append one finite JSON record; optional status may advance to running."""
    payload = (canonical_json(event) + "\n").encode("utf-8")
    status = event.get("status")
    if status is not None and status != "running":
        raise ValueError("Event status may only be running; finish_run records terminal status")
    with _locked_manifest(paths) as manifest:
        if manifest["identity"]["status"] in TERMINAL_STATUSES:
            raise ValueError("Cannot append to a finished run")
        with (paths.directory / "events.jsonl").open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if status is not None:
            manifest["identity"]["status"] = status
            _atomic_json(paths.directory / "manifest.json", manifest, replace=True)


def finish_run(paths: RunPaths, metrics: Mapping[str, Any], status: str) -> None:
    """Publish terminal metrics once, keeping scientific failure class separate."""
    canonical_json(metrics)
    if status not in TERMINAL_STATUSES:
        raise ValueError("finish_run requires interrupted, failed or completed status")
    with _locked_manifest(paths) as manifest:
        if manifest["identity"]["status"] in TERMINAL_STATUSES:
            raise ValueError("Run already finished; terminal artifacts cannot be overwritten")
        manifest["identity"]["status"] = status
        manifest["resources"]["finished_utc"] = _utc()
        _atomic_json(paths.directory / "metrics.json", metrics)
        _atomic_json(paths.directory / "manifest.json", manifest, replace=True)


def update_run_observations(paths: RunPaths, observations: Mapping[str, Any]) -> None:
    """Atomically merge only measured fields into a nonterminal run manifest.

    Scientific/run identity and protocol are immutable. ``parent_run_id`` is a
    one-time linkage; code/lock/data identity observations cannot be changed once
    supplied. Serial callers own measurements; this is not a cross-file ledger
    transaction. A metrics-before-manifest interruption is preserved and rejected.
    """
    canonical_json(observations)
    allowed = {
        'identity': {'parent_run_id', 'code_revision', 'adapter_revision', 'lock_hash',
                     'hardware_hash', 'growth_hash', 'failure_class'},
        'data': {'raw_hash', 'split_hash', 'transform_hash', 'teacher_normalization', 'counts', 'seeds'},
        'method': {'trainable_parameters', 'total_parameters', 'serialized_bytes', 'weight_checkpoint',
                   'weight_hash', 'weight_terms', 'context_bytes', 'cache_bytes', 'backend', 'threads',
                   'checkpoint_retention'},
        'costs': {'search', 'preprocess', 'base', 'score', 'probe', 'continuation', 'validation',
                  'first_prediction', 'warm_prediction', 'acquisition'},
        'resources': {'rss_bytes', 'mps_allocated_bytes', 'mps_driver_bytes', 'pressure', 'swap_delta_bytes',
                      'interruption_links', 'retry_links', 'timing_exclusions'},
    }
    with _locked_manifest(paths) as manifest:
        if (manifest['identity']['status'] in TERMINAL_STATUSES
                or (paths.directory / 'metrics.json').exists()):
            raise ValueError('Cannot update terminal or inconsistent metrics-before-manifest record')
        for section, updates in observations.items():
            if section not in allowed or not isinstance(updates, Mapping) or set(updates) - allowed[section]:
                raise ValueError('Unsupported observation field or immutable identity update')
            for key, value in updates.items():
                previous = manifest[section].get(key)
                unavailable = (isinstance(value, dict) and set(value) == {'value', 'reason'}
                               and value['value'] is None and isinstance(value['reason'], str) and bool(value['reason']))
                if section == 'identity' or (section == 'data' and key.endswith('_hash')):
                    if not unavailable and (type(value) is not str or not value):
                        raise ValueError('Identity observations require nonempty strings')
                    if previous != value and not (isinstance(previous, dict) and previous.get('value') is None):
                        raise ValueError('Previously observed identity is immutable')
                elif section == 'data' and key == 'seeds':
                    if value != previous:
                        raise ValueError('Observed seeds must match immutable resolved configuration')
                elif section == 'data' and key == 'counts':
                    if not isinstance(value, Mapping) or set(value) - {'F', 'G', 'V', 'T'}:
                        raise ValueError('Unsupported split counts')
                    merged = {p: dict(counts) for p, counts in previous.items()}
                    for split, counts in value.items():
                        if not isinstance(counts, Mapping) or set(counts) - {'rows', 'groups'}:
                            raise ValueError('Unsupported count observation')
                        for field, count in counts.items():
                            if type(count) is not int or count < 0:
                                raise ValueError('Counts must be nonnegative integers')
                            if field == 'rows' and isinstance(previous[split]['rows'], int) and count != previous[split]['rows']:
                                raise ValueError('Observed row count differs from resolved configuration')
                            merged[split][field] = count
                    value = merged
                elif section == 'method' and key in {'trainable_parameters', 'total_parameters', 'serialized_bytes',
                                                     'context_bytes', 'cache_bytes'}:
                    if not unavailable and (type(value) is not int or value < 0):
                        raise ValueError('Parameter and byte observations must be nonnegative integers')
                elif section == 'method' and key in {'weight_checkpoint', 'weight_hash'}:
                    if not unavailable and (type(value) is not str or not value):
                        raise ValueError('Weight artifact observations must be nonempty strings')
                elif section == 'method' and key == 'backend':
                    if not isinstance(value, Mapping) or set(value) != {'actual'} or not isinstance(value['actual'], Mapping):
                        raise ValueError('Only actual backend observations may be updated')
                    actual = value['actual']
                    backend_fields = {'device', 'dtype', 'execution', 'fallback', 'cpu_threads', 'interop_threads',
                                      'kernel_tolerance', 'resume_tolerance', 'evidence_sha256',
                                      'kernel_implementation_sha256', 'workload', 'performance_calibration_matches',
                                      'performance_scope', 'causal_cpu_evidence'}
                    if set(actual) - backend_fields:
                        raise ValueError('Unsupported actual backend observation field')
                    if actual.get('device') not in ('cpu', 'mps') or actual.get('dtype') != 'float32' or actual.get('fallback') is not False:
                        raise ValueError('Actual backend requires explicit float32 CPU/MPS without fallback')
                    value = {**previous, 'actual': dict(actual)}
                elif section == 'method' and key == 'threads':
                    if (not isinstance(value, Mapping) or set(value) != {'cpu', 'interop'}
                            or any(type(n) is not int or n <= 0 for n in value.values())):
                        raise ValueError('Thread observations require positive CPU and interop counts')
                elif section == 'costs':
                    if (not isinstance(value, Mapping) or not {'nanoseconds', 'status', 'reason'} <= set(value)
                            or set(value) - {'nanoseconds', 'status', 'reason', 'boundary'}
                            or ('boundary' in value and (type(value['boundary']) is not str or not value['boundary']))
                            or type(value['nanoseconds']) is not int or value['nanoseconds'] < 0
                            or value['status'] not in ('measured', 'estimated') or value['reason'] is not None):
                        raise ValueError('Cost observations require finite nonnegative duration and measurement status')
                    value = {**previous, **value}
                elif section == 'resources' and key.endswith('_bytes'):
                    if not unavailable and (type(value) is not int or (key != 'swap_delta_bytes' and value < 0)):
                        raise ValueError('Resource bytes must be integers or unavailable with reason')
                elif not unavailable and not isinstance(value, (Mapping, list, str, int)):
                    raise ValueError('Malformed observation')
                manifest[section][key] = value
        _atomic_json(paths.directory / 'manifest.json', manifest, replace=True)
