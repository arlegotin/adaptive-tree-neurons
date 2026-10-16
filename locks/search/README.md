# Search dependency overlay

Optuna 4.5.0 is pinned for the serial exploratory TPE coordinator. `requirements.txt`
contains the complete hash-verified resolution. `environment.json` validates the
versions actually imported, including dependencies reused from the native core.
The search module appends `environment/search-site` after core site-packages; it
does not replace the core environment or modify `uv.lock`. NODE therefore retains
its genuine native environment lock. Its package source hash still changes when
search/timing Python files change, requiring new checkpoints.

Install the local ignored overlay:

```sh
uv pip install --target environment/search-site --require-hashes -r locks/search/requirements.txt --python .venv/bin/python --cache-dir .uv-cache
```

The resolution was produced with `uv pip compile locks/search/requirements.in
--generate-hashes --output-file locks/search/requirements.txt --python
.venv/bin/python --cache-dir .uv-cache`. Optuna and its dependencies' installed
license files are archived under `licenses/`, with exact hashes and metadata in
`dependency-licenses.json`. Optuna is MIT licensed. No weights are involved.

The sampler archive is finite hashed JSON, with every ordered ask/tell, anchor,
parameter support, seed, failure and objective. Loading replays the pinned TPE
trajectory and checks every proposal. It never unpickles RNG or study objects.
Study state uses an in-memory Optuna store; SQLAlchemy/Alembic are mandatory Optuna
import dependencies, not a persistent database or parallel scheduler.

`tqdm==4.67.1` also matches the existing pinned NODE reference-only environment.
The first full-suite attempt exposed that reference-site precedence selected4.67.1
against an initial search resolution4.70.1. The strict check failed correctly;
the search lock was re-resolved to4.67.1, with both fresh-process and combined
NODE/search verification. Core/NODE locks and historical receipts were not edited.
