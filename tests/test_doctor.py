"""Doctor boundary tests: failures must survive, identities must not."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_doctor_preserves_unavailable_fields(tmp_path, monkeypatch):
    # Catches replacing denied counters with zero, dropping errors, or leaking IDs.
    from treeneuron import doctor

    actual_run = subprocess.run

    def denied_sysctl(args, **kwargs):
        if args[0] == "sysctl":
            return subprocess.CompletedProcess(args, 1, "", "Operation not permitted")
        if args[0] == "system_profiler":
            return subprocess.CompletedProcess(args, 0, """Hardware:
    Chip: Apple M3 Max
    Serial Number (system): SECRET-SERIAL
    Hardware UUID: 11111111-2222-3333-4444-555555555555
    Provisioning UDID: SECRET-UDID
""", "")
        return actual_run(args, **kwargs)

    monkeypatch.setattr(doctor.subprocess, "run", denied_sysctl)
    monkeypatch.setattr(doctor.platform, "machine", lambda: "arm64")
    report = doctor.collect_system_report()
    assert report["architecture"]["process"] == "arm64"
    assert report["memory"]["physical_bytes"]["value"] is None
    assert "Operation not permitted" in report["memory"]["physical_bytes"]["reason"]
    assert report["memory"]["physical_bytes"]["returncode"] == 1
    assert any(error["returncode"] == 1 for error in report["command_errors"])
    output = tmp_path / "nested" / "report.json"
    doctor.write_report(report, output)
    assert json.loads(output.read_text()) == report
    raw = output.read_text()
    assert "SECRET" not in raw
    assert "11111111-2222" not in raw
    assert "NaN" not in raw and "Infinity" not in raw


def test_report_rejects_nonfinite_json_without_damaging_existing_output(tmp_path):
    # Catches accidental permissive NaN serialization or truncation on rejection.
    from treeneuron.doctor import write_report

    output = tmp_path / "report.json"
    output.write_text('{"previous": true}\n')
    with pytest.raises(ValueError):
        write_report({"bad": float("nan")}, output)
    assert json.loads(output.read_text()) == {"previous": True}


def test_cli_writes_report_and_fails_for_broken_cpu_import(tmp_path, monkeypatch):
    # Catches false success on a broken indispensable package import.
    from treeneuron import cli, doctor

    real_import = doctor.importlib.import_module

    def missing_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("CPU runtime unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(doctor.importlib, "import_module", missing_torch)
    output = tmp_path / "report.json"
    assert cli.main(["doctor", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["cpu_ready"] is False
    assert report["packages"]["torch"]["import_ok"] is False
    assert "CPU runtime unavailable" in report["packages"]["torch"]["reason"]


def test_doctor_module_cli_reports_real_process_and_lock(tmp_path):
    # Catches unregistered CLI dispatch and fabricated package/lock observations.
    output = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, "-m", "treeneuron.cli", "doctor", "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["python"]["version"] == sys.version.split()[0]
    assert report["cpu_ready"] is True
    assert len(report["lock"]["sha256"]) == 64
    assert report["sources"]["status"] == "recorded"
    assert report["torch"]["kernel_gate"] == "not_run"
