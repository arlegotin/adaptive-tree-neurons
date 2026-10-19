# Storage recovery qualification

Transparent filesystem compression was tested only on newly created copies of
an already completed checkpoint. The live scientific queue and its original
files were left unchanged. No production storage operator was implemented.

The first diagnostic hit an unavailable optional os.listxattr API. The retained
retry then found that ditto --hfsCompression returned success while the copied
checkpoint exposed a zero-byte readable data fork and flags0. A separate ordinary
copy preserved all114,917bytes and the original SHA256. The immutable scientific
checkpoint and its queue-result commitment were reauthenticated afterward.

The underlying compression failure was not isolated. This establishes no general
filesystem defect or permission diagnosis. It does establish that this proposed
storage recovery cannot be used safely in the current environment. The compression
branch is rejected; original evidence remains intact and existing live disk/RAM
pauses govern the experiment. No model, protected outcome or scientific source
was changed. Exact commands, partial copies, failures and hashes remain in the
[receipt](WP03-storage-qualification-receipt.json).

A later, explicitly native execution of the same copy-only idea passed all four
checkpoint/text byte checks. The earlier negative result is retained unchanged;
the exact OS mechanism remains unisolated. Native execution is now a separately
qualified condition, not a reason to trust the earlier empty outputs.

The narrowed storage operator touches only producer-published immutable training
event logs in completed job directories. It authenticates the result marker and
file bytes before copying; verifies a compressed staging file's whole readable
SHA/length and unchanged source; then atomically replaces the name. Scratch and
receipts stay outside the frozen queue, so readers never see an extra transient
file in an inventoried directory. An old inode link survives publication until a
durable matching-byte receipt, while failures retain diagnostic scratch/backup.
No active-writer/global ledger, checkpoint, array, source, model or data bytes are
altered. Changing an open writer's inode would be unsafe; it is never admitted.

Fourteen missing-interface REDs became GREEN;80affected storage/audit/queue tests
passed2.74s. On a disposable copy of a21,715,143-byte actual completed event log,
whole-file hashes, already-open readers, random seeks, readonly mmap and16,324JSON
records all matched. Its stat-reported allocation decreased7,970,816bytes. This is
per-file observation, not attributed whole-volume savings, especially with APFS
clones or other applications changing disk use. The original was reauthenticated.

The native qualification binds the exact operator and ditto bytes/filesystem. A
batch helper refuses a changed/missing qualification. End-to-end disposable queue
verification and actual application are recorded separately in
[the native recovery receipt](WP03-storage-native-recovery-receipt.json). External
storage housekeeping overlaps the contended mechanism run and must not be hidden
as intrinsic method speed or added again to an enclosing physical clock. The
full227GiB logical forecast remains unchanged and live disk/RAM guards still apply.
