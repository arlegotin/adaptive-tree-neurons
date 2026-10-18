# Checkpoint history storage v2

Motivation: the bounded18 October checkpoint-only profile found repeated full
history canonicalization dominating publication. This is an infrastructure change,
not an optimization/selection treatment. Implementation and numerical equivalence
are underway. Preserve raw row-ID events once in their existing stage streams.

Native TrainingState history becomes an append-only sequence of immutable
canonical JSON records. Reads return defensive values; forks copy its immutable
record references and preserve all losing events. Each checkpoint directory owns
a separate append-only JSONL history file. Before publishing any checkpoint
pointer, fsync newly appended records, then store a compact history reference:
safe local basename, committed byte prefix, event count, SHA256 of exactly that
prefix and explicit schema/encoding. Later append does not change any earlier
committed prefix. Full payload/model/Adam/current/best validation stays intact.

Descriptor schema2 carries that reference; schema1's inline-list reader remains
for its supported source identity. Loading validates the complete committed
history prefix before constructing/restoring state. Safe basenames, no symlinks,
exact prefix size/newline/count/JSON/checksum are mandatory. A copied complete
checkpoint directory is portable; no absolute history locator is serialized.
Trailing uncommitted bytes are retained as failed evidence and ignored for an
older reference. A resumed writer starts a fresh journal when the observed file
has a tail inconsistent with its committed append cursor; never truncate it.

Cached immutable records/incremental SHA state make normal publication append
only the new suffix. Append prepares a copied hash and immutable linked tail,
then commits tail/count/bytes/hash through one reference assignment. An interrupt
before that assignment leaves the previous history intact. Forks share immutable
tails and commit independent new heads. No cache certifies caller-mutable dicts.
The owned-file cache binds device/inode/size/mtime_ns/ctime_ns; an unchanged stamp
avoids rereading the old prefix. Any changed stamp triggers complete prefix SHA
validation, and every load always checks the full committed prefix. This fast
path assumes reliable local filesystem change stamps and one writer. Network or
coarse-timestamp filesystems require separate qualification; metadata-preserving
external mutation is outside the fast-path integrity guarantee.
Generic list snapshots can retain inline serialization. Checkpoint pruning must
retain journals referenced by any durable pointer and account for their committed
bytes without repeatedly decoding every historical record. Native update order,
G selection, sampler, optimizer, absolute schedule, RNG and event contents stay
identical. Compare default selected predictions and all model/Adam/update arrays
with the existing pre-change reference under the same runtime before admission.

Acceptance: immutable sequence/fork tests; prefix load after later append; byte
corruption rejected before unpack; interrupted append leaves old pointer valid;
same-state recovery retains its exact complete history; directory-copy portability;
metadata size bounded with history growth; existing checkpoint/history/recovery
tests; profiled new publication; exact valid-run numerical reference. Retain first
failures and avoid any practical efficiency claim from these contended profiles.

Current verification: four original journal tests RED,72 first focused passed,
158 expanded state/history/recovery checks passed. Review identified the partial
append and repeated-prefix-read defects; both reproduced RED. The atomic-head
and file-stamp repairs passed75 focused checks, then11 journal validation tests
including traversal/symlink/count/noncanonical-rehashed input and append/fsync
interruption. Static re-review found no remaining material issue in the stated
local single-writer scope. Exact numerical reference/profile remain pending.

The19-array default reference subsequently matched byte-for-byte (NPZ SHA256
8c072d32850a63e1512e0fa6858363fe3ab5d7df3300914db1e469799ff6bb77).
Ten publications of the same state/history with8571 explicit synthetic diagnostic
marker events took.20427s with journal history versus.97994s inline; empty history
took.18513s. These are contended bounded profile results, not a full-run speedup.

A separate measured retention issue also needs correction before large runs:
ten prunes with4000 synthetic expired publication entries took1.38205s, repeatedly
stat-ing about8000 already absent files per prune. Retention should inspect
currently existing version files against a cached validated publication-name set,
then delete only names known to have been successfully published and unprotected
by any pointer. The append-only publication journal stays intact. Cache valid
journal identity/change stamps, append known entries only after durable journal
write, and rebuild on unexpected stamp change. An interruption can invalidate the
cache and force a rebuild; it must never authorize deletion of an unpublished or
protected file. This uses the same reliable local-stamp/single-writer assumption.
Raw publication history is retained; only redundant filesystem rescans change.
Cache membership is bounded to32 recently used directories; evicted directories
revalidate their full journal when needed. This bounds retained process bookkeeping
across long queues instead of accumulating every past trial's names indefinitely.
