"""Real fixture launches must leave private parent process state behind."""
import importlib.metadata
import os
from pathlib import Path
import sys


def test_fresh_gated_fixture_drops_inherited_openmp_default_and_preserves_caller(tmp_path,monkeypatch):
    from test_gated_native_fit import run_native
    monkeypatch.setenv('KMP_DUPLICATE_LIB_OK','True')
    result=run_native(tmp_path,"""
assert __import__('os').environ.get('KMP_DUPLICATE_LIB_OK') is None
model=fit('P00','selected')
assert np.isfinite(model.predict(G)).all()
print('{}')
""")
    assert result=={}
    assert os.environ['KMP_DUPLICATE_LIB_OK']=='True'


def test_NODE_deployment_fixture_fits_in_a_fresh_core_without_changing_parent_metadata(tmp_path,monkeypatch):
    from test_deployment_worker_v2 import prepare
    core_site=Path(importlib.metadata.distribution('numpy')._path).parent
    monkeypatch.syspath_prepend(str(core_site))
    before=list(sys.path)
    names=[d.metadata['Name'] for d in importlib.metadata.distributions()]
    assert len(names)>len(set(names)), 'Controlled parent metadata duplication must actually exist'
    path,request,digest=prepare(tmp_path,'P12')
    assert path.is_file() and request['family_id']=='P12' and len(digest)==64
    assert sys.path==before
    assert [d.metadata['Name'] for d in importlib.metadata.distributions()]==names
