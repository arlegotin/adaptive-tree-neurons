"""Harmless-child lifecycle and pre-import identity checks; no native fits."""
import sys
sys.dont_write_bytecode = True

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name('qualify_native_trees.py')
spec = importlib.util.spec_from_file_location('native_tree_fixture_harness', SCRIPT)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
OUTPUT = None


class FixtureLifecycleChecks(unittest.TestCase):
    def monitor_exception(self, failure, name):
        directory = OUTPUT/name
        directory.mkdir()
        launched = []
        original_popen = subprocess.Popen
        original_run = subprocess.run

        def harmless_child(argv, **kwargs):
            if '--worker' in argv:
                child = original_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
                launched.append(child)
                return child
            return original_popen(argv, **kwargs)

        def failing_monitor(argv, **kwargs):
            if argv[0] == '/bin/ps':
                raise failure
            return original_run(argv, **kwargs)

        try:
            with patch.object(fixture, 'import_auditor', return_value=types.SimpleNamespace(
                    validate_environment=lambda overlay: {'harmless_lifecycle_test_only': True})), \
                    patch.object(fixture, 'resource_guard', return_value={'admitted': True}), \
                    patch.object(subprocess, 'Popen', side_effect=harmless_child), \
                    patch.object(subprocess, 'run', side_effect=failing_monitor):
                with self.assertRaises(type(failure)) as caught:
                    fixture.run(directory/'study', directory/'unused-overlay')
            self.assertIs(caught.exception, failure, 'Original supervisor exception was replaced')
            self.assertEqual(len(launched), 1, 'Supervisor failure must not launch a replacement child')
            child = launched[0]
            destination = directory/'study/case-0-catboost-numeric/attempt-0'
            command_path = destination/'command.json'
            error_path = destination/'supervisor-error.json'
            command = json.loads(command_path.read_text()) if command_path.exists() else {}
            error = json.loads(error_path.read_text()) if error_path.exists() else {}
            observed = dict(child_reaped_before_test_cleanup=child.returncode is not None,
                command_receipt=command_path.exists(), error_receipt=error_path.exists(),
                original_error_type=error.get('exception'), original_error_message=error.get('message'),
                child_reaped_in_receipt=command.get('child_reaped'),
                cost_retained=command.get('complete_process_wall_ns', 0) > 0)
            fixture.write(directory/'observed.json', observed)
            self.assertEqual(observed, dict(child_reaped_before_test_cleanup=True,
                command_receipt=True, error_receipt=True,
                original_error_type=type(failure).__name__, original_error_message=str(failure),
                child_reaped_in_receipt=True, cost_retained=True))
            with self.assertRaises(ChildProcessError):
                os.waitpid(child.pid, os.WNOHANG)
        finally:
            # The red run must not leave behind the orphan it demonstrates.
            for child in launched:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=5)

    def test_monitor_oserror_reaps_child_publishes_error_and_preserves_exception(self):
        self.monitor_exception(PermissionError('mock RSS monitor permission denied'), 'monitor-oserror')

    def test_keyboard_interrupt_reaps_child_publishes_error_and_preserves_exception(self):
        self.monitor_exception(KeyboardInterrupt('mock coordinator interrupt'), 'keyboard-interrupt')

    def guard_identity(self, field):
        directory = OUTPUT/f'identity-{field}'
        directory.mkdir()
        (directory/'fixture.npz').write_bytes(b'not-loaded-if-guard-is-correct')
        manifest = dict(script_sha256=fixture.file_hash(SCRIPT),
            executable_path=sys.executable, executable_sha256=fixture.file_hash(sys.executable),
            fixture_sha256=fixture.file_hash(directory/'fixture.npz'),
            import_auditor_sha256=fixture.file_hash(fixture.ROOT/'locks/research-v2/tree-imports.py'),
            tree_lock_sha256=fixture.file_hash(fixture.ROOT/'locks/research-v2/tree-requirements.txt'),
            core_lock_sha256=fixture.file_hash(fixture.ROOT/'uv.lock'),
            cases=[dict(method='catboost', categorical=False)])
        manifest[field] = '0'*64
        fixture.write(directory/'manifest.json', manifest)
        def forbidden_import():
            raise AssertionError('Auditor imported before changed identity was rejected')
        with patch.object(fixture, 'import_auditor', side_effect=forbidden_import):
            with self.assertRaisesRegex(ValueError, 'changed|differs'):
                fixture.worker(directory, 0, directory/'unused-attempt', directory/'unused-overlay')

    def test_changed_auditor_rejected_before_dynamic_import(self):
        self.guard_identity('import_auditor_sha256')

    def test_changed_tree_lock_rejected_before_dynamic_import(self):
        self.guard_identity('tree_lock_sha256')

    def test_changed_core_lock_rejected_before_dynamic_import(self):
        self.guard_identity('core_lock_sha256')

    def test_changed_interpreter_rejected_before_dynamic_import(self):
        self.guard_identity('executable_sha256')

    def test_changed_fixture_rejected_before_dynamic_import(self):
        self.guard_identity('fixture_sha256')

    def test_changed_script_rejected_before_dynamic_import(self):
        self.guard_identity('script_sha256')

    def test_after_fit_auditor_record_has_explicit_process_context(self):
        self.assertTrue(hasattr(fixture, 'audit_phase'), 'Missing phase-aware runtime audit wrapper')
        destination = OUTPUT/'phase-context'
        destination.mkdir()
        def raw_audit(package, path, overlay):
            fixture.write(path/'import.json', dict(method=package,
                models_constructed=False, fits_executed=False))
        fixture.audit_phase(types.SimpleNamespace(worker=raw_audit), 'catboost',
            destination, OUTPUT/'unused-overlay', completed_fits=2)
        value = json.loads((destination/'import.json').read_text())
        raw = json.loads((destination/'raw-auditor-call/import.json').read_text())
        self.assertTrue(value['models_constructed'])
        self.assertTrue(value['fits_executed'])
        self.assertEqual(value['completed_native_fits'], 2)
        self.assertEqual(value['auditor_call_phase'], 'after-fitting')
        self.assertFalse(raw['models_constructed'])
        self.assertFalse(raw['fits_executed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT/'tests.log').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(FixtureLifecycleChecks))
    fixture.write(OUTPUT/'result.json', dict(tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors),
        fits_executed=False, fallback_test_cleanup_enabled=True,
        lifecycle_assertion='Children must be reaped by the supervisor before fallback test cleanup',
        fixture_script_sha256=fixture.file_hash(SCRIPT),
        test_script_sha256=fixture.file_hash(__file__)))
    print((OUTPUT/'tests.log').read_text())
    raise SystemExit(0 if result.wasSuccessful() else 1)
