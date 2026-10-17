# Ruling 16 — compact duplicated checkpoint row history

Ruling: Represent ordered minibatch row IDs in duplicated in-memory/checkpoint update histories by their count and the existing ordered-int64 SHA256 commitment, while preserving complete ordered IDs in every original stage event log and preserving exact sampler/RNG checkpoint state — measured calibration shows repeated parent histories in all65 greedy trials make the minimum independent study exceed local storage; the stage logs already supply the independently checked full row streams — what it costs if wrong: checkpoint-only exports require the retained stage logs for full historical row inspection; a consumer depending on the redundant copies needs a versioned export or replay adaptation, and any numerical or audit discrepancy blocks the change and requires corrected source-bound validation/calibration before scientific claims.

This authorizes a narrowly reviewed storage correction before genuine pilot or
protected access. Every update event, raw ordered row ID, loss, failure, selected
state and sampler/RNG state remains retained. It does not authorize truncating
history, discarding losing probes/runs, changing training, or altering the
primary score, population, splits, checkpoint choice, optimizer or allowance.
The compact history representation must be explicit and versioned; raw stage
events and their independent pairing/protocol checks remain authoritative.

The measured4000-update full-batch parent retained551,777,966bytes, with about
183.86MB in each of two metadata files and only58/115KB tensor payloads. Two
smaller-batch4000-update parents still imply over85GB of B6 prefix metadata for
16pilot teachers plus the minimum16confirmation teachers, before final models
and other records. These are measured-prefix projections, not completed study
costs. All original runs and failures remain unchanged under their old source.

Before accepting the correction, compare exact model, optimizer, scheduler,
sampler/RNG, predictions and ordered-row commitments against an old-source
numerical reference extracted while that source is valid. Do not load old
checkpoints under a changed reader or waive source identity. Recheck growth,
resume, raw-event auditing and actual metadata/storage behavior. New package
bytes require fresh applicable receipts, reviewed acceptance bindings and an
actual complete resource sample before the genuine pilot. No protected cohort
exists yet.
