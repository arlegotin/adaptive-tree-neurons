# Outer search controller preparation

`research/search_controller_v2.py` adds the serial outer queue around the pinned
TPE sampler or an explicitly supplied finite schedule. It accepts a source/runtime/
support/split identity, fixed fitting seed, separate HPO seed, native anchors,
attempt limit and dedicated-worker deadline. These supplied commitments are not
native support qualification or permission to load protected data.

Anchors run first and can win on V. They remain outside the generic TPE surrogate,
as specified in the earlier pre-outcome ruling. Every invocation of a candidate,
including an anchor, failure or recovery, consumes one of at most32paid slots.
Finite supports stop when exhausted. Retrying an interrupted candidate preserves
its configuration, fitting seed and committed checkpoint files. Only one recovery
is allowed. An interruption with no remaining slot/time reports exhausted recovery
budget. It does not advertise a free additional attempt.

The coordinator callback is responsible for qualified F/G fitting, V prediction
and coordinator-only V scoring. The controller receives no labels. It records
every configuration, origin, seed, terminal status, supplied score, file hashes
and cumulative completion/charged clock. Failed and ineligible results cannot
provide a winning score. A result finishing after the enclosing deadline retains
its reported status/score for audit but is marked timeout for selection. Prefix
winners use only results completed within that prefix, with the existing V metric
tie rule. Callback exceptions retain their traceback and paid time and stop for
diagnosis; they cannot silently trigger a new seed.

Clean pauses and controlled interruptions support exact unchanged-plan replay.
The hash-chained journal is bound to an externally retained endpoint on resume.
All prior artifact files and actual TPE dependencies are rechecked. TPE choices,
including a pending choice after more than10observations, are reconstructed and
compared before further work. An OS lease excludes concurrent controllers.
Off-worker downtime is excluded; earlier paid time and each invocation's supplied
preparation cost remain. A process death that leaves an unclosed journal fails
closed: it cannot be resumed by assuming its unmeasured tail cost was zero.

This is callback/queue engineering. It does not implement the native trial worker,
hard child-process deadline/RSS enforcement, process-death supervisor receipts,
independent reconstruction of supplied V scores, automatic source/runtime
qualification, global native support lock or the primary/secondary freeze barrier.
The measured clock covers supplied invocation preparation plus function entry to
its final clock read. Final publication/return and caller process startup still
need an external process receipt. Corresponding qualification/authorization flags
remain false; real practical searches are not admitted by these fixtures.

Verification:21missing-module REDs were followed by four failures caused by using
the mapping-only canonical JSON helper on a None dependency identity. Wrapping the
optional value corrected them. A boundary regression then exposed an interruption
at the final paid slot advertising an unavailable recovery; it now reports
`recovery_budget_exhausted`. Three additional boundary checks passed on first run
(preparation accounting, external endpoint tampering and unclosed process death).
The full affected verification command/result and artificial replay proof are
recorded in `WP08-search-controller-receipt.json`.

`scripts/check_search_controller.py` runs18artificial paid attempts: one anchor,
one numerical failure, an interrupted TPE choice and its same-seed recovery. It
compares all16adaptive observations against a separate actual pinned TPE engine.
All scores, result files and charged clock units in that proof are artificial.
It fits zero models and reads no real V or T targets.
