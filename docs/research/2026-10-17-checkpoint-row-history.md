# Checkpoint row history

Newly recorded update events in `TrainingState.history` and checkpoint history
replace `row_ids` with `row_ids_commitment`. Its schema version is 1, encoding
is `ordered_int64_array_hash`, count is the number of ordered IDs, and `sha256`
is the existing `data.types.array_hash` of the ordered `numpy.int64` array.
That hash covers the dtype string, shape string, and contiguous ordered bytes;
it is not a hash of a sorted set or JSON. `rows` and every other event field
remain present. Every event remains in order.

The append-only per-stage `events.jsonl` keeps the original full ordered
`row_ids` list. Pairing and audit consumers continue to read those raw events.
Sampler replay uses the saved order, cursor, generator and RNG state. The
checkpoint's representation avoids duplicating historical minibatch IDs in
each snapshot and each fork; it does not replace raw event retention.

Opaque legacy history remains readable subject to its original source-bound
checkpoint contract. Reading or copying it does not convert it to this format,
and the new representation grants no cross-source checkpoint permission.
