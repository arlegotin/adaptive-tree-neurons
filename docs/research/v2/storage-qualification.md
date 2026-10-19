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
