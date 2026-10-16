"""Run reference evidence in fresh interpreters without changing training state."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CHECKS = {
    "appendix_a": ROOT / "tests" / "reference_math_checks.py",
    "appendix_b": ROOT / "tests" / "reference_score_checks.py",
}


def _run_reference_check(script: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "failed", "script": str(script.relative_to(ROOT))
                              if script.is_relative_to(ROOT) else str(script),
                              "script_sha256": None, "returncode": None,
                              "report": None, "reason": None}
    try:
        result["script_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
        process = subprocess.run([sys.executable, str(script)], capture_output=True,
                                 text=True, timeout=60, check=False)
        result["returncode"] = process.returncode
        if process.returncode != 0:
            result["reason"] = process.stderr.strip() or f"Reference exited {process.returncode}"
            return result
        report = json.loads(process.stdout)
        # Refuse a partial or nonfinite report even when a child returns zero.
        json.dumps(report, allow_nan=False)
        if not isinstance(report, dict) or report.get("status") != "passed":
            result["reason"] = "Reference did not report passed"
            return result
        result.update(status="passed", report=report)
    except (ValueError, TypeError) as error:
        result["reason"] = f"Invalid reference JSON: {error}"
    except (OSError, subprocess.TimeoutExpired) as error:
        result["reason"] = f"{type(error).__name__}: {error}"
    return result


def collect_math_report() -> dict[str, Any]:
    """Aggregate current CPU references; unfinished package/backend gates stay explicit.

    A successful exit from verify-math means these reference checks passed. It does
    not assert package math parity, CPU training kernels, or MPS validation.
    """
    checks = {name: _run_reference_check(script) for name, script in REFERENCE_CHECKS.items()}
    passed = bool(checks) and all(check["status"] == "passed" for check in checks.values())
    return {
        "schema_version": 1, "scope": "reference_cpu_only", "complete": False,
        "reference_checks_passed": passed, "reference_checks": checks,
        "gates": {
            "reference_cpu": {"status": "passed" if passed else "failed"},
            "package_layer_math": {"status": "not_implemented", "reason": "Pending Core Task 4"},
            "package_score_math": {"status": "not_implemented", "reason": "Pending Core Task 7"},
            "cpu_backend": {"status": "not_implemented", "reason": "Pending Core Task 8"},
            "mps_backend": {"status": "not_implemented", "reason": "Pending Core Task 8; CPU references do not validate MPS"},
        },
    }
