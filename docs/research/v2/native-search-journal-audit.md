# Native quality search journal audit

`research.search_audit_v2.audit_native_search_journal` reads a closed native
quality search under a shared writer lease. Supply external byte hashes for its
plan, journal and result. The plan must bind the current scientific source and
explicitly use `deadline_ns=None`; earlier capped searches retain their own
semantics and cannot supply this proof.

The reader reproduces the exact finite or locked TPE proposal sequence, including
paid failures, pending suggestions and the single identical-candidate recovery.
It checks every referenced artifact, session preparation and cumulative clock,
legal stopping/resumption, sampler feedback, winner and descriptive budget
prefixes against the full result. Strict JSON and final byte/source checks reject
ambiguous or changed evidence. Reading neither mutates the search nor decodes
model/data arrays or labels. Active writers cannot be audited.

Only `search_journal_authenticated` becomes true. A valid paused search remains
incomplete. Native workload/support qualification, model generation, validation
arithmetic, external whole-process time, final freeze and protected access still
need their own evidence. Journal V scores are inputs to this reader, rather than
independently authenticated predictions. This is a component for future production
assembly; no actual practical HPO was launched by these fixtures.

The final affected suite passed67checks in1.09seconds, including real locked TPE
replay across a clean pause and paid recovery. Controlled rehashed journals exposed
missing preparation/exception/stop checks before release. The final exception
status test initially passed because another result discrepancy rejected it;
after rebuilding compatible prefix outputs, it failed for the intended missing
status check and then passed with the exact controller failure contract enforced.
The normal controller already emitted that contract correctly. No study artifact
or historical search was changed. Logs and source bindings are retained in
[the receipt](WP08-native-search-journal-audit-receipt.json).
