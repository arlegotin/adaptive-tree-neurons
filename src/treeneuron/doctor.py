"""Privacy-conscious, best-effort observations; unavailable counters stay null."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CORE_PACKAGES = {
    "torch": "torch", "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
    "scikit-learn": "sklearn", "PyYAML": "yaml", "pytest": "pytest",
}
ROOT = Path(__file__).resolve().parents[2]


def _redact(text: str) -> str:
    lines = [line for line in text.splitlines()
             if not re.search(r"serial|uuid|udid", line, re.IGNORECASE)]
    text = "\n".join(lines)
    text = re.sub(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b", "[redacted]", text)
    return re.sub(r"\(id=\d+\)", "(id=redacted)", text)


def _probe(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": args, "value": None, "reason": _redact(str(error)), "returncode": None}
    stdout, stderr = _redact(result.stdout).strip(), _redact(result.stderr).strip()
    return {"command": args, "value": stdout if result.returncode == 0 and stdout else None,
            "reason": stderr or (None if stdout and result.returncode == 0 else "No output"),
            "returncode": result.returncode}


def _integer(observation: dict[str, Any]) -> dict[str, Any]:
    result = dict(observation)
    if result["value"] is not None:
        try:
            result["value"] = int(result["value"])
        except ValueError:
            result.update(value=None, reason="Probe returned a noninteger value")
    return result


def _file_status(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as error:
        return {"sha256": None, "reason": _redact(str(error))}
    return {"sha256": hashlib.sha256(data).hexdigest(), "reason": None}


def collect_system_report() -> dict[str, Any]:
    """Collect observations without assuming architecture or optional probe success.

    Does not run scientific kernels, change environment settings, or allocate an
    MPS tensor. Source evidence/lock are read from this checkout.
    """
    commands = {
        "uname": ["uname", "-m"], "os": ["sw_vers"],
        "hardware": ["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"],
        "physical_bytes": ["sysctl", "-n", "hw.memsize"],
        "logical_cpus": ["sysctl", "-n", "hw.logicalcpu"],
        "swap": ["sysctl", "-n", "vm.swapusage"],
        "vm_stat": ["vm_stat"], "pressure": ["memory_pressure", "-Q"],
        "power": ["pmset", "-g", "batt"], "disk": ["df", "-h", "."],
        "xcode": ["xcode-select", "-p"], "uv": ["uv", "--version"],
        "git": ["git", "--version"],
    }
    probes = {name: _probe(args) for name, args in commands.items()}
    packages = {}
    modules = {}
    for name, module_name in CORE_PACKAGES.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        try:
            modules[name] = importlib.import_module(module_name)
            packages[name] = {"version": version, "import_ok": True, "reason": None}
        except Exception as error:
            packages[name] = {"version": version, "import_ok": False,
                              "reason": _redact(f"{type(error).__name__}: {error}")}
    torch_status: dict[str, Any] = {"kernel_gate": "not_run", "mps_built": None,
                                    "mps_available": None, "reason": packages["torch"]["reason"]}
    torch_module = modules.get("torch")
    if torch_module is not None:
        try:
            torch_status.update(version=torch_module.__version__,
                                mps_built=torch_module.backends.mps.is_built(),
                                mps_available=torch_module.backends.mps.is_available(),
                                cpu_threads=torch_module.get_num_threads(),
                                interop_threads=torch_module.get_num_interop_threads())
        except Exception as error:
            torch_status["reason"] = _redact(f"{type(error).__name__}: {error}")
    counters = {}
    for name in ("recommended_max_memory", "current_allocated_memory", "driver_allocated_memory"):
        if torch_status["mps_available"]:
            try:
                counters[name] = {"value": int(getattr(torch_module.mps, name)()), "reason": None}
            except Exception as error:
                counters[name] = {"value": None, "reason": _redact(f"{type(error).__name__}: {error}")}
        else:
            counters[name] = {"value": None, "reason": "MPS unavailable or not observable"}
    torch_status["memory_bytes"] = counters
    sources = _file_status(ROOT / "provenance" / "sources.json")
    if sources["sha256"]:
        try:
            source_data = json.loads((ROOT / "provenance" / "sources.json").read_text())
            sources.update(status="recorded", novelty_status=source_data.get("novelty_status"))
        except (ValueError, OSError) as error:
            sources.update(status="invalid", reason=_redact(str(error)))
    else:
        sources["status"] = "unavailable"
    try:
        usage = shutil.disk_usage(ROOT)
        disk = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free,
                "reason": None, "observation": probes["disk"]}
    except OSError as error:
        disk = {"total_bytes": None, "used_bytes": None, "free_bytes": None,
                "reason": _redact(str(error)), "observation": probes["disk"]}
    memory = {"physical_bytes": _integer(probes["physical_bytes"]),
              "swap": probes["swap"], "pressure": probes["pressure"], "vm_stat": probes["vm_stat"]}
    physical = memory["physical_bytes"]["value"]
    memory["initial_reservation_bytes"] = {
        "value": max(8 * 1024**3, physical // 4) if physical else None,
        "reason": None if physical else "Physical memory counter unavailable",
    }
    return {
        "schema_version": 1, "utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {"process": platform.machine(), "uname": probes["uname"]},
        "os": {"system": platform.system(), "release": platform.release(), "observation": probes["os"]},
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "cpu": {"logical_cpus": _integer(probes["logical_cpus"]), "hardware": probes["hardware"]},
        "memory": memory, "disk": disk, "power": probes["power"],
        "tools": {name: probes[name] for name in ("xcode", "uv", "git")},
        "packages": packages, "torch": torch_status,
        "cpu_ready": all(packages[name]["import_ok"] for name in CORE_PACKAGES if name != "pytest"),
        "environment": {name: os.environ.get(name) for name in (
            "PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "lock": _file_status(ROOT / "uv.lock"), "sources": sources,
        "command_errors": [p for p in probes.values() if p["returncode"] != 0 or p["reason"]],
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    """Write finite JSON; validate before touching an existing report."""
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
