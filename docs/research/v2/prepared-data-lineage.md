# Prepared data byte and source consistency

`research.prepared_lineage_v2.audit_prepared_transfer_suite` authenticates the
externally committed preparation/acquisition JSON, the recorded source hash
against an exact retained Git commit, all six fixed raw-file hashes, semantic
schemas and every declared sealed archive byte hash. It checks the five stored
fold identities/order and row-count declarations. It accepts relocated current
asset paths; historical locator strings do not direct filesystem probes.

This is an opaque integrity audit. It hashes protected archive bytes, including
30T-label archives, without decoding labels, arrays, spreadsheets or CSV rows.
It fits no model and computes no protected metric. It does not rerun preparation,
reconstruct raw-to-row contents or independently authenticate the historical
preparation runtime. Source and byte consistency remain conditional on the
trusted original preparation record. This evidence cannot qualify native
workloads, authorize protected evaluation or release secondary development.

The reader hashes and decodes each metadata buffer together, then rechecks all
consumed references and its current implementation at exit. A controlled test
exposed a gap before release: replacing a manifest immediately after decoding
could bind old values to a new file hash. The corrected reader rejects that
replacement. No actual study file was changed; the counterexample used temporary
fixtures. Missing/corrupt data, duplicate acquisitions and escaping paths also
reject rather than silently substituting assets.

Seven missing-interface failures preceded seven passing cases in1.78seconds.
The controlled late-manifest failure then preceded the final52-case affected
pass in2.44seconds. Fixtures use explicitly substituted opaque raw hashes and
metadata, not official acquisition evidence. They forbid NumPy/pandas decoding
while exercising all six identities and180declared role archives. See
[the source/log receipt](WP07-prepared-data-lineage-receipt.json). Any real
retained-suite audit is recorded separately; test counts are not that audit.
