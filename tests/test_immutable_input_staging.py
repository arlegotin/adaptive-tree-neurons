"""Read-only staging shares immutable bytes without overwriting old evidence."""
import errno
import hashlib
import importlib.util
import os

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.immutable_inputs') is not None,'Missing verified immutable input staging'
    from treeneuron.research import immutable_inputs
    return immutable_inputs


def test_same_filesystem_staging_shares_bytes_and_retains_the_original(tmp_path):
    module=api();source=tmp_path/'closed.npz';source.write_bytes(b'committed closed bytes')
    digest=hashlib.sha256(source.read_bytes()).hexdigest();target=tmp_path/'staged.npz'
    row=module.stage_immutable_file(source,target,expected_sha256=digest)
    assert row['method']=='hard_link' and row['sha256']==digest and row['logical_bytes']==len(source.read_bytes())
    assert os.path.samefile(source,target) and target.read_bytes()==source.read_bytes()==b'committed closed bytes'


def test_cross_device_fallback_is_an_independent_exact_copy(tmp_path,monkeypatch):
    module=api();source=tmp_path/'source';source.write_bytes(b'original immutable bytes');target=tmp_path/'target'
    def cross_device(*a,**kw):raise OSError(errno.EXDEV,'cross-device')
    monkeypatch.setattr(module.os,'link',cross_device)
    row=module.stage_immutable_file(source,target,expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert row['method']=='verified_copy' and not os.path.samefile(source,target)
    assert source.read_bytes()==target.read_bytes()==b'original immutable bytes'


@pytest.mark.parametrize('kind',['digest','existing','symlink'])
def test_bad_source_or_existing_destination_cannot_replace_original_evidence(tmp_path,kind):
    module=api();source=tmp_path/'source';source.write_bytes(b'original');target=tmp_path/'target'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if kind=='digest':digest='0'*64
    elif kind=='existing':target.write_bytes(b'earlier output')
    else:
        linked=tmp_path/'linked';linked.symlink_to(source);source=linked
    with pytest.raises(ValueError):module.stage_immutable_file(source,target,expected_sha256=digest)
    assert source.read_bytes()==b'original'
    if kind=='existing':assert target.read_bytes()==b'earlier output'
    else:assert not target.exists()
