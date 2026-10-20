"""Compressed size cannot authorize allocation from a contradictory NPY header."""
import hashlib
import io
import zipfile

import numpy as np
import pytest


def test_native_workload_rejects_inflated_array_header_before_numpy_materialization(tmp_path,monkeypatch):
    from treeneuron.research.native_workload_v2 import _epoch_payload
    path=tmp_path/'worker/snapshot/weights.npz';path.parent.mkdir(parents=True)
    data=io.BytesIO()
    np.lib.format.write_array_header_1_0(data,dict(descr='<f4',fortran_order=False,shape=(500_000_000,)))
    data.write(b'\0'*4)
    with zipfile.ZipFile(path,'w') as archive:archive.writestr('head.bias.npy',data.getvalue())
    state=dict(payload={'head.bias':dict(dtype='float32',shape=[500_000_000])},weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def forbidden(*a,**kw):raise AssertionError('Contradictory tensor header reached NumPy materialization')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError,match='allocation|header|geometry'):
        _epoch_payload(tmp_path,state,{},'P03')
