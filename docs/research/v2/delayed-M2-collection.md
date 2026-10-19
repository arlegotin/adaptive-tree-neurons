# Delayed M2 collection

The operator [finish_m2.py](scripts/finish_m2.py) waits for the entire frozen M2
queue to finish and release its worker lease. It imports only the standard
library while waiting and never starts fitting, resumes a queue, changes a seed,
or reads partial predictive outcomes. An inactive incomplete queue stays pending;
a failed queue stops without analysis. A finite watch window can expire without
changing the study.

The external configuration pins the producer package, producer configuration,
queue plan, independent auditor package/helper, and M1 control audit. The operator
checks these commitments on every poll. At readiness it holds a read-only shared
lease through both subprocess stages: collection under the original producer
source, followed by independent raw-prediction arithmetic under a separately held
auditor source. Both must agree with the same complete ledger endpoint. It retains
argv, environment, clocks, errors and child logs outside the producer queue.
An existing output is refused; a failed stage is not retried automatically.

The production producer remains detached at `e1b1018`, package `e322da64…`.
The auditor checkout `/private/tmp/atn-research-v2-next` remains at `b848726`,
package `6e81099c…`, until this operator finishes. Further implementation uses
the main research branch or another checkout. Changing the auditor would stop
the operator rather than silently change its evidence identity.

Verification observed nine missing-helper failures, then eleven passing checks.
Five additional cases exercised shared-lease exclusion, endpoint mismatch,
symlink output containment, a real finite CLI wait, and real sequential child
processes using explicitly artificial actors. The symlink test exposed one
output-containment defect; resolving the output path corrected it. The final
affected run passed 44 checks in 3.30s, including the existing collector and raw
auditor fixtures. These checks grant no efficacy, Gate A, or protected-data claim.
See [the preparation receipt](WP03-delayed-collection-preparation-receipt.json).

The retained launch configuration is
`artifacts/research-v2/WP03/M2-delayed-collection-f87b387/config.json`, with SHA-256
`4e9287922d165f22138731fe7b9d8e135f90096abf9951d37fea390c525113aa`.
Use the qualified interpreter with `-B`, the operator, `--config` that path,
`--expected-config-sha256` that digest, and fresh `--output` and `--watch-record`
paths. The production launch uses a 30-second poll and a finite 24-hour watch.
Its exact argv and terminal result are retained in the watch record. Success
still requires inspecting both authenticated results and the collector's actual
Gate A decision before admitting any new selector.
