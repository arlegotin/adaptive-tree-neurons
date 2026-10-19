# Development prediction artifact audit

This Task 10/11 slice connects retained, explicitly authorized development arrays
to the canonical reporter. It does not implement protected confirmation. Its
input schema is `research/v2-development-audit-input`; E or V are the only
evaluation roles. A T role, confirmation scope or supplied efficacy decision is
rejected before opening prediction or target files.

The caller supplies an external SHA256 for one immutable study manifest. That
manifest names the study status, exact committed execution snapshot and scientific
package hash, declared independent units, every evaluation case and planned
method/status, the full declared attempt inventory and limitations. Each case
contains its task/repetition/role/grouping and a byte reference to original-unit
F targets, evaluation targets and row/group IDs. Temporal cases also carry actual
timestamps. Each eligible method has a separate referenced NPZ of predictions and
their row IDs; failed/missing methods have no prediction file. No raw producer
metric or supplied interval is accepted as a substitute for these arrays.

The auditor verifies current Git execution bytes against the committed snapshot,
including the running audit, arithmetic and reporter modules. It hashes each
file's bytes before decoding those same bytes with pickle disabled, checks exact
array keys and delegates independent original-unit arithmetic to
`audit_prediction_arrays`. All referenced evidence is rechecked before returning. Traversal,
symlink components, drift, duplicate/unplanned identities and reordered prediction
rows fail. One case's methods share exactly its declared rows, grouping and F
normalizer. Every planned result remains visible even when no prediction exists.

The output uses the canonical renderer's development envelope. Source and raw
prediction bytes are authenticated at this level; model-generation provenance,
the completeness of an independently produced attempt ledger, protected access,
population inference and cost authentication remain unqualified. Costs are
explicitly unavailable and comparison decisions are empty. The manifest's stated
independent-unit description is retained as a declaration, not newly proved by
the auditor. No result can become a confirmed win through this API. Production
confirmation requires the separate final protocol, protected evaluator, complete
ledger/model/data binding and their acceptance tests.

Retired and blocked studies may have no evaluation cases; this permits an honest
report without inventing a confirmation directory. This schema is an additive
artifact interface. Existing M1/M2 collectors and historical records keep their
original identities and dedicated independent audits.

The report CLI supports this explicit development branch:

```sh
python -m treeneuron.research.cli report --study PATH_TO_STUDY --output report.md \
  --repository-root PATH_TO_COMMITTED_CHECKOUT \
  --development-manifest-sha256 EXTERNALLY_RECORDED_SHA256
```

The study directory contains `development-manifest.json`. The CLI audits its raw
arrays before publishing a fresh report, refuses an existing output, and rejects
outputs that add/change the checkout's executable scientific inputs. Omitting the
development commitment leaves the general/protected reporting command unavailable
until its remaining WP11 implementation and qualification. This is not the
production confirmation report command becoming complete.

The component and development CLI are implemented and fixture-tested: 25 missing
component failures preceded 73 passing affected checks. A file-change-during-
audit regression then failed and led to rechecking every referenced file before
returning. Four missing CLI-option failures preceded integration. The final
affected run passed 89 checks, including the existing scientific queue's CLI
disposition tests. The evidence uses artificial arrays and temporary Git trees;
it supplies no new trained-model result. Exact source, commands, failures and
raw inventory are in [the receipt](WP11-development-raw-audit-receipt.json).
