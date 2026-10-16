"""Run reference evidence in fresh interpreters without changing training state."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CHECKS = {
    "appendix_a": ROOT / "tests" / "reference_math_checks.py",
    "appendix_b": ROOT / "tests" / "reference_score_checks.py",
}

PACKAGE_LAYER_CHECK = ROOT / "src" / "treeneuron" / "layer_checks.py"

# Required fields of the actual standalone A/B reports, plus the package check.
_SCHEMAS = {
    'appendix_a': (1602026, (37, 7, 5),
        {'epsilon': 0.05, 'tau': 1, 'unit': 2, 'error_threshold': 1e-12},
        ('forward_max_abs', 'shared_gradient_max_abs', 'collapse_max_abs',
         'growth_preservation_max_abs', 'gate_gradient_formula_max_abs',
         'contrast_gradient_norm', 'gate_gradient_norm', 'loss_before', 'loss_after',
         'gate_step_size', 'soft_hard_gap_max_abs', 'relu_bound_max_violation')),
    'appendix_b': (1602027, (41, 5, 4),
        {'epsilon': 0.05, 'tau': 1, 'adam_epsilon': 1e-8, 'learning_rate': 1e-6,
         'weight_decay': 0.0, 'error_threshold': 1e-12, 'adam_relative_error_threshold': 1e-4},
        ('rescaled_gradient_max_abs', 'rescaled_score_max_abs', 'permuted_score_max_abs',
         'gate_gradient_formula_max_abs', 'gate_gradient_norm', 'contrast_norm',
         'selected_unit', 'loss_before', 'loss_after', 'predicted_loss_decrease',
         'observed_loss_decrease', 'adam_prediction_relative_error')),
    'package_layer_math': (1602028, (13, 3, 4), {'tau': 0.7, 'error_threshold': 1e-10},
        ('forward_max_abs', 'bypass_max_abs', 'shared_gradient_max_abs', 'collapse_max_abs',
         'growth_preservation_max_abs', 'gate_gradient_formula_max_abs', 'gate_gradient_norm',
         'contrast_gradient_norm', 'dead_gradient_norm', 'soft_hard_gap_max_abs',
         'relu_bound_max_violation', 'post_mixture_relu', 'leafwise_relu',
         'base_parameter_count', 'grown_parameter_count')),
}


def _validate_report(report: dict[str, Any], expected_check: str | None) -> None:
    """Require real identities, CPU float64 environment and complete numeric evidence."""
    identity = expected_check if expected_check is not None else report.get('check')
    if identity not in _SCHEMAS:
        raise ValueError('Reference schema: unknown or missing check identity')
    seed, (n, d, h), constants, scalars = _SCHEMAS[identity]
    if (type(report.get('schema_version')) is not int or report['schema_version'] != 1
            or report.get('check') != identity or type(report.get('seed')) is not int
            or report['seed'] != seed or report.get('shapes') != dict(n=n, d=d, h=h)
            or report.get('constants') != constants):
        raise ValueError('Reference schema: invalid identity, dimensions or constants')
    env = report.get('environment')
    if (not isinstance(env, dict) or env.get('dtype') != 'torch.float64'
            or env.get('device') != 'cpu' or type(env.get('threads')) is not int
            or env['threads'] != 1 or env.get('python') != platform.python_version()
            or any(not isinstance(env.get(k), str) or not env[k]
                   for k in ('torch', 'platform', 'architecture'))):
        raise ValueError('Reference schema: invalid or missing environment')
    results = report.get('results')
    def scalar(value):
        return type(value) in (int, float) and math.isfinite(value)
    if not isinstance(results, dict) or any(not scalar(results.get(k)) for k in scalars):
        raise ValueError('Reference schema: missing or invalid numeric results')
    if identity == 'appendix_b':
        for key, columns in (('contrasts', d + 1), ('gate_gradients', d + 1), ('adam_scores', None)):
            values = results.get(key)
            if not isinstance(values, list) or len(values) != h:
                raise ValueError(f'Reference schema: invalid {key} shape')
            if columns is None:
                valid = all(scalar(v) for v in values)
            else:
                valid = all(isinstance(row, list) and len(row) == columns
                            and all(scalar(v) for v in row) for row in values)
            if not valid:
                raise ValueError(f'Reference schema: invalid {key} values')
        if type(results['selected_unit']) is not int or not 0 <= results['selected_unit'] < h:
            raise ValueError('Reference schema: invalid selected unit')
    threshold = constants['error_threshold']
    for key in scalars:
        if (key.endswith('_max_abs') or key == 'relu_bound_max_violation') and not results[key] < threshold:
            raise ValueError(f'Reference schema: numerical contract failed for {key}')
    if identity in ('appendix_a', 'appendix_b') and not results['loss_after'] < results['loss_before']:
        raise ValueError('Reference schema: local descent failed')
    if identity == 'appendix_a':
        if (results['growth_preservation_max_abs'] != 0 or results['contrast_gradient_norm'] != 0
                or results['gate_gradient_norm'] <= 0 or results['gate_step_size'] <= 0):
            raise ValueError('Reference schema: insertion gradient contract failed')
    if identity == 'appendix_b':
        if (results['observed_loss_decrease'] <= 0 or results['predicted_loss_decrease'] <= 0
                or not 0 <= results['adam_prediction_relative_error'] < 1e-4):
            raise ValueError('Reference schema: Adam descent contract failed')
    if identity == 'package_layer_math':
        if (any(results[k] != 0 for k in ('growth_preservation_max_abs', 'contrast_gradient_norm',
                                         'dead_gradient_norm', 'post_mixture_relu'))
                or results['gate_gradient_norm'] <= 0 or results['leafwise_relu'] != 1
                or results['base_parameter_count'] != 1153 or results['grown_parameter_count'] != 1697):
            raise ValueError('Reference schema: package insertion/count contract failed')


def _run_reference_check(script: Path, expected_check: str | None = None) -> dict[str, Any]:
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
        _validate_report(report, expected_check)
        result.update(status="passed", report=report)
    except (ValueError, TypeError) as error:
        result["reason"] = f"Invalid reference JSON: {error}"
    except (OSError, subprocess.TimeoutExpired) as error:
        result["reason"] = f"{type(error).__name__}: {error}"
    return result


def collect_math_report() -> dict[str, Any]:
    """Check references and package layer in isolation; score/backends stay pending."""
    checks = {name: _run_reference_check(script, name) for name, script in REFERENCE_CHECKS.items()}
    passed = bool(checks) and all(check["status"] == "passed" for check in checks.values())
    layer = _run_reference_check(PACKAGE_LAYER_CHECK, 'package_layer_math')
    return {
        "schema_version": 1, "scope": "reference_and_package_layer_cpu_only", "complete": False,
        "reference_checks_passed": passed, "reference_checks": checks,
        "implemented_checks_passed": passed and layer["status"] == "passed",
        "gates": {
            "reference_cpu": {"status": "passed" if passed else "failed"},
            "package_layer_math": layer,
            "package_score_math": {"status": "not_implemented", "reason": "Pending Core Task 7"},
            "cpu_backend": {"status": "not_implemented", "reason": "Pending Core Task 8"},
            "mps_backend": {"status": "not_implemented", "reason": "Pending Core Task 8; CPU references do not validate MPS"},
        },
    }
