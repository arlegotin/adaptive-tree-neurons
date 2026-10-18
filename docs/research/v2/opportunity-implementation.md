# WP03 execution details

This supplements the binding spec's M2 matrix. It changes no efficacy threshold,
teacher selection rule, contrast, count or optimizer treatment. It precedes M2
fitting. Gate A is still pending.

An actual update250/1000 snapshot is not a G-selected parent. Introduce an
explicit `research_boundary` stage, with `research/v2-current-boundary` provenance
inside model-coupled growth metadata. It records the original active stage,
requested/actual update, full source snapshot fingerprint and current G risk.
Promoting this boundary clones the actual complete-update tensors, optimizer,
schedule, sampler and RNG; it does not restore the older G-best state. The
boundary is complete as a declared intervention boundary and makes no G-best
claim. Legacy insertion/widening reject this stage. Separate research wrappers
validate the boundary and use the same transactional migration implementation.
Late G-best forks continue through the unchanged canonical interfaces.

Fixed-budget trajectory diagnostics require an explicit policy field
`stop_by_patience=False`, with defaultTrue preserving legacy behavior. The G grid,
strict-best rule and eligible update0 remain unchanged; the trajectory still
executes its full declared budget. Parent fallback uses this named treatment
only when the prespecified actual snapshots were unavailable. Every arm forks
the same fallback parent, and the extra parent work is retained and charged.

The numerical policy also carries a versioned immutable actual-snapshot capture
plan. Active-stage resume rejects an omitted or altered plan; canonical JSON
normalizes tuple/list representation when comparing policies. Callback execution
must remain RNG-neutral. This persists capture requirements alongside the sampler
and fixed-budget stopping policy rather than trusting a later command to remember
which snapshots were requested.

For M2c one8000-update trajectory per chosen singleton/keep supplies all three
horizons. At each requested stage prefix0/300/1200/8000 the observer retains its
current G-best complete snapshot, labeled with the prefix as well as the selected
absolute update. This is distinct from actual-update parent capture. The
coordinator predicts V/E for each prefix-selected snapshot after fitting;
workers never receive V/E labels. No additional G evaluation point is introduced.
An interruption must restore the original fixed-budget policy and snapshot plan.

This observer is now implemented with capture-schema2. A distinct on_prefix
callback receives an isolated copy of the G-best complete-update payload and
restores the actual trajectory RNG afterward. PrefixSnapshotSink validates
source/model/Adam state, selected offset<=prefix, exact persisted prefix policy,
and common source/data/config/topology/intervention/stage-start lineage. Its
atomic immutable plan, binding and per-prefix records permit same-boundary replay
with either a new or the same sink object. A previously published pointer is
verified and reused after interruption. Existing-record paths fsync the directory
before training may advance. Final reports explicitly retain unavailable prefixes.
These changes create a new source identity; they do not migrate older checkpoints.

The frozen-head helper now implements the M2e six-alpha diagnostic. Hidden
features retain the native parent's float32 computation; F-only feature centering,
RMS, intercept and all ridge solves use float64. Positive-alpha solves minimize
mean squared error plus alpha times squared coefficients. G chooses alpha, with
smallest-alpha ties; only the selected predictor returns original-unit outputs.
Recorded G solve scores use the supplied target coordinates (F-standardized in
the native wrapper). The coordinator must recompute original-unit V/E metrics.
Per-alpha timers cover solve/G scoring; the enclosing procedure must additionally
pay feature extraction, preparation, parent fitting and final prediction costs.
The helper never consumes V/E labels or changes parent state/RNG. Its controller,
durable serialization and full cost/failure integration remain pending.

M2d enumerates the common eligible singleton/pair pool; every nominal h8 singleton
or pair remains visible, including ineligible dead-unit subsets. G chooses the
trained pair oracle, and its complete candidate cost is separate from each
single selected procedure. M2a includes keep/top/random/warm lower/warm upper for
each snapshot and stream treatment; insertion/widening precedes the new-stream
reset. M2b fixed300 uses all eligible singleton IDs plus keep. M2e retains all
six F-hidden-feature ridge solves, G-selects smallest-alpha ties, and evaluates
V/E only for the selected head. Known noise-free teacher predictions are
information-advantaged capacity/noise diagnostics, not a fair fitted competitor.

Keep all initially declared teachers. The four-to-eight extension and Gate A
screening follow the operational rules already recorded in the main spec and
ledger. No selector implementation or population confirmation is admitted by
these engineering changes alone.
# Diagnostic fork and F-score artifacts

`research/diagnostics.py` now supplies a source-bound worker for keep, current
cheap/random growth, exact nominal singleton/pair subsets, and lower/upper warm
ordinary brackets. It accepts F/G-only worker data and an immutable parent/full
snapshot fingerprint. F-only scores are bound by every table field and tensor
byte. Insertion precedes any immediate minibatch-stream regeneration; the caller's
parent and RNG are preserved. Fixed-budget prefixes and deployable patience
training carry their actual resolved policies. Nominal ineligible subsets remain
explicit records. Unscored keep reports common eligibility as unknown.

`research/opportunity_inputs.py` saves all ScoreTable arrays/spectra without
pickle, verifies raw file/table/parent commitments on read, binds the nine M2c
nominal slots from cheap top-four and a seeded sample of up to four remaining
eligible IDs, and selects the small pair oracle on G with the declared stable
tie rule. Missing candidate slots remain ineligible records.

The outer M2 controller, conditional admission, immutable prerequisite queue,
recovery orchestration, V/E coordinator, complete standalone costs and source-bound
resource admission are still pending. A worker's standalone cost remains null
until those prerequisites are accounted for. Public M2 run remains disabled.

`research/parent_artifacts.py` supplies the corresponding single-attempt parent
prerequisite. It honors the configured backend receipt, saves actual snapshots
and the complete G-selected parent separately, and restores caller RNG/receipt
state. A planted-control prerequisite is explicitly unfitted at update0.
`load_parent_input` verifies the exact F/G arrays and complete manifest identity
before any unavailable-state result; corrupt/moved/changed commitments fail.
Only a recorded unreached requested update returns unavailable, which the outer
controller may use for the separately named fixed-budget fallback. Available
actual snapshots become declared current boundaries, rather than G-best states.
The outer queue still owns one declared same-seed infrastructure retry and must
bind each returned result's bytes in its append-only ledger.
