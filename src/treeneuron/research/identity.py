"""Execution dependencies bound by a development plan and every resume."""
import hashlib
import importlib.metadata
from pathlib import Path

from treeneuron.backend import lock_hash, runtime_hash


def execution_identity(config):
    receipt = Path(config['base_experiment']['execution']['backend_report'])
    return dict(lock_sha256=lock_hash(), runtime_sha256=runtime_hash(),
        dependencies={name: importlib.metadata.version(name) for name in
                      ('torch', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'PyYAML')},
        backend_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest() if receipt.is_file() else None)
