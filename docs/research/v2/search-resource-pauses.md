# Pausing a search before the next trial

The outer controller can retain an optional, identity-bound live-admission
decision before it asks for the next configuration or starts fitting. Denial
closes the current episode as `resource_pause`. It consumes no model-attempt
slot and creates no attempt directory. Admission/checking time is still paid
by the enclosing search clock. This is a search pause, not a failed prediction
or permission to change the model, data, tuning space or seed.

Resume replays the exact journal and continues its original cursor. A pause
before the first trial leaves all configurations untouched. A pause before
recovering an interrupted fit preserves that fit's same-seed recovery slot and
checkpoint references. Denial before a TPE proposal does not ask or discard a
suggestion. Native-quality training still has no elapsed-hour cutoff.

The admission identity is copied at entry. Each decision must carry a strict
boolean, the matching identity hash and a valid journal transition. A denied
decision must end in the corresponding resource pause; it cannot authorize an
attempt. Explicit null identity fields cannot masquerade as old ungated
history. Existing sessions without this optional gate retain their schema and
behavior. Both the independent native journal reader and the cold worker's
prior-episode reader recognize closed resource pauses, including zero fits.

The affected five-module suite passed **112 checks in 11.37 seconds**. New
fixtures cover initial denial, same-state recovery, unchanged TPE proposals,
independent journal reopening, malformed/null gates and caller mutation. These
use software scores and supplied test clocks, not new native fits or physical
timing evidence. Seven tests first failed for the missing interface; two
additional schema/mutation regressions failed before their fixes. Two test
commands named nonexistent files and ran no tests; the corrected initial
three-module run passed 61 checks. All attempts are retained in the
[receipt](WP08-search-resource-pauses-receipt.json).

This is the controller hook and replay contract. Binding the actual qualified
resource forecast/host observation into the production worker and launcher is
still required. A caller-supplied admission flag is no native qualification,
resource guarantee, whole-procedure cost or protected-access capability. No
existing scientific result or held-source job changed.
