# Immutable input staging

Native fitting and fresh prediction replay now stage byte-verified immutable
archives using hard links on the same filesystem. Cross-device staging uses an
exclusive, fsynced, verified copy. `research.immutable_inputs.stage_immutable_file`
rejects a changed/symlink source or existing destination, records the method,
logical size/hash and inode-sharing fact, and verifies both paths after staging.

This applies only to read-only F/G/query archives and closed inference model/
prediction files. Their containing workers and inspectors still check the
original/staged byte commitments after use. It does not change rows, model
capacity, numerical operations, seeds, training or prediction values. Training
recovery keeps its independent checkpoint-copy contract and exact resume checks.
Earlier held native jobs and their artifacts were not modified.

The fitting input directory records `materialization.json`; replay inspection
records every `input_materialization` entry and its input manifest. Model
inference remains a fresh process with the same closed state. The purpose is to
avoid physically duplicating the same committed archives as the search grows.
Shared cache/storage is disclosed. Every procedure still pays its complete
measured preparation/fitting/inspection cost; logical model size and standalone
cost are not discounted because files share an inode. This is no gated-model
efficiency result or measured whole-volume saving. Full study disk admission
still requires the actual forecast and free-space margin.

Five missing-module failures preceded five passing filesystem cases, including
the real same-device link, simulated cross-device exact fallback, bad hashes,
symlinks and existing-output preservation. Two real native/replay integration
cases then failed on the original duplicated staging. The final44-case affected
suite passed59.93seconds with7retained native vendor warnings. It includes
real fits/replays, unchanged original containers, source/role drift rejection
and exact independently copied NODE checkpoint recovery. See
[the source/log receipt](WP08-immutable-input-staging-receipt.json).
