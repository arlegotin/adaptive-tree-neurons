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
                    or identity["config_hash"] != canonical_hash(config)):
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
