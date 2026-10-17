import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from treeneuron import backend, train

root = Path('environment/long-probes-20261018')
raw = backend.probe_backend('cpu', seed=1602030, batch_sizes=(128,))
raw_bytes = (json.dumps(raw, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
(root/'cpu128-raw.json').write_bytes(raw_bytes)
assert raw['status'] == 'passed', raw.get('reason')
backend.validate_backend_report(raw)
aggregate = {'backend_probes': {'probes': {'cpu': raw}, 'comparison_blocks': [],
                              'decision': backend.select_backend({'cpu': raw})}}
receipt_bytes = (json.dumps(aggregate, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
(root/'cpu128-receipt.json').write_bytes(receipt_bytes)
provenance = dict(source_sha256=train.source_hash(), implementation_sha256=backend.implementation_hash(),
    lock_sha256=backend.lock_hash(), runtime_sha256=backend.runtime_hash(), runtime=backend.runtime_identity(),
    commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
    raw_sha256=hashlib.sha256(raw_bytes).hexdigest(), receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
    arguments=dict(device='cpu',seed=1602030,batch_sizes=[128]),
    threads=dict(cpu=torch.get_num_threads(),interop=torch.get_num_interop_threads()),
    environment={key:os.environ[key] for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK')},
    validated=True)
(root/'receipt-provenance.json').write_text(json.dumps(provenance,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=raw['status'],validated=True,receipt_sha256=provenance['receipt_sha256'])))
