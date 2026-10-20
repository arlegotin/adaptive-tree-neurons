"""Stage byte-bound, read-only archives; training recovery uses separate copies."""
import errno
import hashlib
import os
from pathlib import Path
import shutil


def _sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def stage_immutable_file(source,target,*,expected_sha256):
    """Share an immutable ordinary file, or copy only across filesystems.

    Callers must recheck original/staged commitments after their read-only worker.
    Do not use this function for writable training checkpoints or mutable files.
    """
    source=Path(source);target=Path(target)
    if (source.is_symlink() or not source.is_file() or target.exists() or target.is_symlink()
            or not target.parent.is_dir() or type(expected_sha256) is not str or len(expected_sha256)!=64
            or _sha(source)!=expected_sha256):raise ValueError('Exact ordinary immutable source and fresh destination required')
    try:os.link(source,target);method='hard_link'
    except OSError as error:
        if error.errno!=errno.EXDEV:raise
        with source.open('rb') as original,target.open('xb') as staged:
            shutil.copyfileobj(original,staged);staged.flush();os.fsync(staged.fileno())
        method='verified_copy'
    if _sha(source)!=expected_sha256 or _sha(target)!=expected_sha256:raise ValueError('Immutable source/staged bytes changed')
    return dict(source_path=str(source.resolve()),target_path=str(target.resolve()),sha256=expected_sha256,
        logical_bytes=target.stat().st_size,method=method,
        shared_inode_at_staging=os.path.samefile(source,target),
        policy='Declared immutable read-only archive; byte hashes are rechecked by the containing worker/inspector; inode sharing is not a standalone cost or free-volume measurement')
