# Known synthetic-problem exclusions

The new [known-seed registry](../../../protocol/research-v2/exclusions/known-teacher-seeds.json)
and [ID candidates](../../../protocol/research-v2/exclusions/known-teacher-id-candidates.json)
make current exclusions durable. They grant no confirmation or protected access.

The structural scan at813dbfc parsed575tracked nonvendor JSON/YAML files without
parse failures. Three separately authenticated development plans supply M0/M1/M2
seed identities; the M2 input is its immutable plan, with no partial predictive
results opened. Direct fields and the handoff supply77unique seed candidates and
74public teacher-ID candidates. D1 seed261018411 and D2 seeds918401–918404 are
explicitly excluded. Input-file hashes are retained and checked again after
processing. A structural scan is not complete semantic review of these records.

Legacy pilot source derives each teacher with SHA256 over
`treeneuron-seeds-v1:exploratory/pilot/teacher-{index}/teacher:{root}`, taking the
first8bytes big-endian and masking to63bits. The inventory conservatively expands
79known seed-map/default/direct-seed candidates over indices0–127. All10,112
derived entries agree with the existing `derive_seed` implementation. Combined
with direct exclusions, the sorted registry has10,171distinct seeds and passes
the existing known-exclusion loader. No teachers, observation arrays or fresh
confirmation root were generated. Extra conservative exclusions are not claims
that those teachers were actually fit.

The initial scan failed on a mismatched identity type: it treated the compacted
note's M0 `31f97567...` as a file SHA, while the retained file identifies it as the
canonical plan hash. Inspection established the actual file SHA as
`7274ac81499bc6be37d2b1887829b2fda53c6596764e68b0386b099275856b7a`.
The failed script/log and corrected canonical recomputation remain retained; no
source plan or expected historical identity was edited to pass.

Registry completeness and ID semantics remain unqualified. The74IDs are public
field candidates rather than independently authenticated historical raw tensors.
Known namespace derivations, missing historical assets, subsequent development/
calibration teachers and actual generated confirmation IDs still need the final
readiness review. Update this versioned registry before confirmation; its byte
hash alone cannot establish that every previous problem was covered. The exact
inputs, scripts, occurrences, closure and limitations are bound in
[the receipt](WP09-teacher-exclusion-inventory-receipt.json).
