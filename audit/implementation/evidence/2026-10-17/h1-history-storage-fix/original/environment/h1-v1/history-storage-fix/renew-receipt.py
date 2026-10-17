"""One final-package receipt renewal through the existing numerical producers."""
import json
from pathlib import Path
import subprocess
import time
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from treeneuron.backend import probe_backend, select_backend
from treeneuron.cpu_envelope import probe_cpu_envelope, validate_cpu_envelope
from treeneuron.experiments.paired import write_json, file_hash
from treeneuron.protocol import execution_snapshot
from treeneuron.train import source_hash

out = Path(__file__).parent
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
snapshot = execution_snapshot(Path.cwd(), commit)
write_json(out/'strict-snapshot.json', snapshot)
started = time.perf_counter_ns()
probe = probe_backend('cpu', seed=1602030, batch_sizes=(128,))
write_json(out/'kernel.json', probe)
assert probe['status'] == 'passed', probe.get('reason')
envelope = probe_cpu_envelope()
write_json(out/'envelope.json', envelope)
assert envelope['status'] == 'passed', envelope.get('reason')
validate_cpu_envelope(envelope)
probes = {'cpu': probe}
report = {'backend_probes': {'probes': probes, 'comparison_blocks': [],
                            'decision': select_backend(probes)}, 'cpu_envelope': envelope}
write_json(out/'backend.json', report)
assert execution_snapshot(Path.cwd(), commit) == snapshot
result = dict(status='passed', commit=commit, source_sha256=source_hash(),
              execution_sha256=snapshot['execution_sha256'], elapsed_ns=time.perf_counter_ns()-started,
              backend_sha256=file_hash(out/'backend.json'), kernel_sha256=file_hash(out/'kernel.json'),
              envelope_sha256=file_hash(out/'envelope.json'), wrapper_sha256=file_hash(Path(__file__)),
              scope='existing CPU128 kernel and full finite envelope once; no fits or pilot')
write_json(out/'receipt-measurement.json', result)
print(json.dumps(result), flush=True)
