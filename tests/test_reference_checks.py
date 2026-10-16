"""Reference checks exercise subprocess math, isolation, and honest gate status."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=[("math", 1602026, (37, 7, 5)),
                                         ("score", 1602027, (41, 5, 4))])
def reference_json(request):
    name, seed, shape = request.param
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / f"reference_{name}_checks.py")],
        capture_output=True, text=True, check=False,
    )
    # Catches missing scripts, failed assertions, and non-JSON diagnostic output.
    assert result.returncode == 0, result.stderr
    return name, seed, shape, json.loads(result.stdout)


def test_reference_runtime_and_finite_evidence(reference_json):
    # Catches running a reference in training precision, on MPS, or with no evidence.
    name, seed, (n, d, h), report = reference_json
    assert report["status"] == "passed"
    assert report["seed"] == seed
    assert report["shapes"] == {"n": n, "d": d, "h": h}
    env = report["environment"]
    assert env["dtype"] == "torch.float64"
    assert env["device"] == "cpu"
    assert env["threads"] == 1
    assert env["python"] == sys.version.split()[0]
    assert env["torch"] and env["architecture"] and env["platform"]
    assert report["results"]
    json.dumps(report, allow_nan=False)


def test_reference_numerical_contract(reference_json):
    # Catches wrong gradient normalization, broken identities, and wrong Adam update.
    name, _, _, report = reference_json
    results = report["results"]
    if name == "math":
        errors = ("forward_max_abs", "shared_gradient_max_abs", "collapse_max_abs",
                  "gate_gradient_formula_max_abs", "soft_hard_gap_max_abs",
                  "relu_bound_max_violation")
        assert results["growth_preservation_max_abs"] == 0.0
        assert results["contrast_gradient_norm"] == 0.0
        assert results["gate_gradient_norm"] == pytest.approx(0.4451935795, abs=1e-9)
        assert results["loss_before"] == pytest.approx(18.7354818901, abs=1e-9)
        assert results["loss_after"] < results["loss_before"]
    else:
        errors = ("rescaled_gradient_max_abs", "rescaled_score_max_abs",
                  "permuted_score_max_abs", "gate_gradient_formula_max_abs")
        assert results["observed_loss_decrease"] > 0
        assert results["adam_prediction_relative_error"] < 1e-4
        assert results["predicted_loss_decrease"] == pytest.approx(8.941316578e-7, rel=1e-6)
        assert results["loss_after"] < results["loss_before"]
        assert len(results["contrasts"]) == 4
        assert len(results["gate_gradients"]) == 4
        assert len(results["adam_scores"]) == 4
        # Historical first-step prediction / (learning rate * 2 * (d+1)).
        assert results["adam_scores"][0] == pytest.approx(0.0745109714833, rel=1e-6)
        for contrast in results["contrasts"]:
            pivot = max(range(6), key=lambda k: abs(contrast[k]))
            assert contrast[pivot] > 0
    for error in errors:
        assert results[error] < 1e-12, error


def test_reference_assertion_exits_nonzero():
    # A no-op optimizer violates real observed descent; its assertion must escape.
    script = ROOT / "tests" / "reference_score_checks.py"
    code = ("import runpy, torch; "
            "torch.optim.AdamW.step = lambda self, *a, **kw: None; "
            f"runpy.run_path({str(script)!r}, run_name='__main__')")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "AssertionError" in result.stderr
    assert result.returncode != 0
    assert not result.stdout.strip()


def test_verification_preserves_caller_dtype_rng_and_threads():
    # Catches replacing fresh child execution with in-process reference imports.
    import torch
    from treeneuron.verification import collect_math_report

    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    rng = torch.random.get_rng_state().clone()
    report = collect_math_report()
    assert report["reference_checks_passed"] is True
    assert torch.get_default_dtype() == dtype
    assert torch.get_num_threads() == threads
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert report["scope"] == "reference_package_and_backend_gates"
    layer = report["gates"]["package_layer_math"]
    assert layer["status"] == "passed"
    assert layer["report"]["check"] == "package_layer_math"
    assert layer["report"]["results"]["forward_max_abs"] < 1e-10
    assert layer["report"]["results"]["growth_preservation_max_abs"] == 0
    assert report["implemented_checks_passed"] is True
    assert report["complete"] is True
    score_gate = report['gates']['package_score_math']
    assert score_gate['status'] == 'passed'
    assert score_gate['report']['check'] == 'package_score_math'
    assert score_gate['report']['results']['gate_gradient_max_abs'] < 1e-10
    assert report["gates"]["cpu_backend"]["status"] == "passed"
    assert report["gates"]["mps_backend"]["status"] in ("passed", "unavailable")
    assert report["backend_probes"]["decision"]["selected_backend"] in ("cpu", "mps")
    for check in report["reference_checks"].values():
        assert check["status"] == "passed"
        assert check["returncode"] == 0
        assert len(check["script_sha256"]) == 64


@pytest.mark.parametrize("output_file", [False, True])
def test_verify_math_cli_emits_reference_scope(tmp_path, output_file):
    # Catches requiring --output or presenting CPU references as all math/MPS gates.
    args = [sys.executable, "-m", "treeneuron.cli", "verify-math"]
    output = tmp_path / "nested" / "math.json"
    if output_file:
        args += ["--output", str(output)]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["reference_checks_passed"] is True
    assert report["complete"] is True
    if output_file:
        assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("body, reason", [
    ("raise AssertionError('numerical failure')", "AssertionError"),
    ("print('not json')", "Invalid reference JSON"),
    ("print('{\"status\": \"failed\"}')", "did not report passed"),
])
def test_aggregate_failure_reaches_cli(tmp_path, monkeypatch, capsys, body, reason):
    # Catches swallowing assertions, malformed output, or a failed child status.
    from treeneuron import cli, verification

    script = tmp_path / "broken_check.py"
    script.write_text(body + "\n")
    monkeypatch.setattr(verification, "REFERENCE_CHECKS", {"broken": script})
    output = tmp_path / "report.json"
    assert cli.main(["verify-math", "--output", str(output)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == report
    assert report["reference_checks_passed"] is False
    assert report["reference_checks"]["broken"]["status"] == "failed"
    assert reason in report["reference_checks"]["broken"]["reason"]


@pytest.mark.parametrize('partial', [
    {'status': 'passed'},
    {'schema_version': 1, 'check': 'appendix_a', 'status': 'passed',
     'seed': 1602026, 'shapes': {'n': 37, 'd': 7, 'h': 5}, 'results': {}},
])
def test_partial_success_report_is_rejected(tmp_path, partial):
    from treeneuron.verification import _run_reference_check

    script = tmp_path / 'partial.py'
    script.write_text(f'print({json.dumps(partial)!r})\n')
    result = _run_reference_check(script)
    assert result['status'] == 'failed'
    assert 'schema' in result['reason'].lower()


@pytest.mark.parametrize('mutation', ['identity', 'environment', 'missing_result', 'wrong_result_shape'])
def test_reference_schema_rejects_corrupted_real_report(tmp_path, reference_json, mutation):
    from treeneuron.verification import _run_reference_check

    name, _, _, original = reference_json
    report = json.loads(json.dumps(original))
    if mutation == 'identity':
        report['check'] = 'some_other_check'
    elif mutation == 'environment':
        report['environment']['dtype'] = 'torch.float32'
    elif mutation == 'missing_result':
        del report['results']['loss_after']
    elif name == 'score':
        report['results']['contrasts'][0].pop()
    else:
        report['results']['forward_max_abs'] = [0.0]
    script = tmp_path / 'corrupt.py'
    script.write_text(f'print({json.dumps(report)!r})\n')
    result = _run_reference_check(script, 'appendix_a' if name == 'math' else 'appendix_b')
    assert result['status'] == 'failed'
    assert 'schema' in result['reason'].lower()


def test_package_layer_failure_reaches_cli(tmp_path, monkeypatch, capsys):
    from treeneuron import cli, verification

    script = tmp_path / 'broken_layer.py'
    script.write_text("raise AssertionError('package mismatch')\n")
    monkeypatch.setattr(verification, 'PACKAGE_LAYER_CHECK', script)
    assert cli.main(['verify-math']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['reference_checks_passed'] is True
    assert report['implemented_checks_passed'] is False
    assert report['gates']['package_layer_math']['status'] == 'failed'


def test_package_score_failure_reaches_cli(tmp_path, monkeypatch, capsys):
    from treeneuron import cli, verification

    script = tmp_path / 'broken_score.py'
    script.write_text("raise AssertionError('package score mismatch')\n")
    monkeypatch.setattr(verification, 'PACKAGE_SCORE_CHECK', script)
    assert cli.main(['verify-math']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['reference_checks_passed'] is True
    assert report['implemented_checks_passed'] is False
    assert report['gates']['package_score_math']['status'] == 'failed'


def test_backend_status_only_cannot_complete_math_gate(monkeypatch):
    from treeneuron import verification

    monkeypatch.setattr(verification, 'collect_backend_report', lambda: {
        'status': 'passed', 'probes': {'cpu': {'status': 'passed', 'device': 'cpu'}},
        'decision': {'selected_backend': 'cpu'}})
    report = verification.collect_math_report()
    assert report['complete'] is False
    assert report['implemented_checks_passed'] is False
    assert report['gates']['cpu_backend']['status'] == 'failed'
