# Native NODE environment

The adapter uses the existing `uv.lock` native arm64 Python 3.12.13 environment.
`environment.json` records and enforces exact core lock, Python, architecture and
numerical package versions. No core packages were changed. `qhoptim==1.1.0` is
vendored from its hash-verified official wheel/source, not installed globally.
All numerical NODE files execute unchanged; see the vendor patch inventory.

Independent reference tests additionally import unchanged upstream `utils.py`,
which imports requests/tqdm despite using neither for the tested numerical path.
Install those into the isolated reference target from the hash lock:

```sh
uv pip install --python .venv/bin/python --target environment/node-reference-site --require-hashes -r locks/node/reference-requirements.txt
.venv/bin/python -m pytest tests/test_node.py -q
```

This is a repository-checkout adapter and verification worker, not a standalone
wheel distribution. The worker runs in the native environment, uses only local
synthetic arrays, and exchanges JSON and nonpickled NPZ checkpoint arrays. No
YEAR data, split indices, pretrained weights, or external agreements are used.
`dependency-licenses.json` and `licenses/` preserve installed dependency notices.
The extra entmax MIT notice has a pinned attribution source; the NODE repository
does not identify the exact earlier entmax revision it incorporated.
