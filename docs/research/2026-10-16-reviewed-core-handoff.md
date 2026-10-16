# Reviewed core handoff

The core implementation is accepted after task reviews, a whole-core review and one combined fix wave. The independent research study remains pending. Work stays on `main`.

The first exploratory problem remains an early negative result for score-based placement: normalized validation MSE was **0.147531** for top-scored growth, **0.143840** for matched random growth and **0.160662** for continued ordinary training. The separate noiseless 2D example was already learned well by the ordinary network. These are one primary teacher and a different diagnostic, with provisional training limits, G-selected checkpoints and descriptive V results. They establish no population superiority or practical advantage.

The predictive runs retain their original `46d537c` checkout and `ec52c0…` source identity. Their [evidence](2026-10-16-exploratory-paired-evidence.json), failed engineering attempt and metadata correction overlays are preserved. The final fixes do not retroactively add diagnostics or change those results.

## Accepted implementation

The final code commit is `a3561c55aed4f561a27e2a588a22b4faedb66cec`; engineering evidence is committed in `df21950`. Package source SHA256 is `39f697d80eba928e5bbcce019beed912c358a20e3b62995d71c9577517e6a032`; the unchanged core lock is `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5`.

The combined fixes reject unsupported growth settings, export ineligible score diagnostics safely, register children before training can fail, bind paired streams to actual events, restore process settings after publication errors and complete F-derived gate/leaf diagnostics. Verification records **224 focused tests**, **629 full-suite tests with 48 sandbox MPS skips**, and **13 causal correctness recipes**. A separate tiny current-source execution verifies 17 methods, 22 manifests and 21 early child registrations in 5.780 seconds; it is engineering evidence, not another predictive experiment. The [fix evidence](2026-10-16-core-final-fix-evidence.json) records commands and hashes.

The [whole-core review](../../audit/implementation/core/2026-10-16/final-review.md) and [scoped acceptance](../../audit/implementation/core/2026-10-16/final-fix-review.md) close all six findings. The [archive index](../../audit/implementation/core/2026-10-16/index.json) binds reports and the chronological ledger. [All ten rulings](../../audit/implementation/core/2026-10-16/rulings-for-final.md) retain their reasons and costs if wrong. This archive records implementation review; it does not replace the future scientific audit.

## Handoff for the next control

Parents contain complete G-selected weights, optimizer ages/moments, schedule, RNGs, sampler position and F-only preprocessing. Every probe must reconstruct an independent copy of that same parent. Added parameters receive fresh Adam state. Source/config/lock/data identity guards remain mandatory; adding package files requires fresh parents rather than loading old checkpoints under a new source hash.

`CausalPredictor.fit(request, *, on_run_created=None)` invokes its registration callback after creating a run and preparing immutable adapter metadata, before training. `fit_parent(data, config, paths, *, growth_metadata=None)` installs supplied metadata before initial observations and checkpoints. Known method/coordinator/parent/score links, split roles/counts and normalization must be published before failure can terminalize a run. Unknown state-dependent values remain null with reasons. Short direct-training probes must preserve this ordering. The [fix report](../../audit/implementation/core/2026-10-16/final-fix-report.md) describes the exact interfaces.

`continue_fit` permits a shorter new continuation stage within the configured allowance while preserving batch, checkpoint cadence and G-selection policy. The normal causal adapter requires its full resolved allowance, so B6 needs an explicit short-probe path. Its allocation must truthfully identify validation-probed selection, preserving common eligibility, per-unit contrasts and fixed effective growth count.

## Remaining gates

Next implement Evidence Task 2: spline ridge, capability records and the charged B6 control that independently probes each eligible neuron on F and ranks on G. B7 remains pending until the faithful NODE adapter passes its own gate. All losing probes and failed attempts count toward cost.

Then complete search/timing and all H1 pilot, freeze, confirmation, analysis and audit/report code before definitive calibration and the real source freeze. Only afterward execute the fixed independent cohort and scientific audit. The [evidence plan](../superpowers/plans/2026-10-16-selective-tree-neurons-evidence.md) and [cohort amendment](../../protocol/amendments/2026-10-16-minimum-confirmation-cohort.md) govern that sequence. Raw transfer data, strong practical adapters and extensions have separate gates. No T access, frozen protocol, calibrated cohort or completed confirmation is claimed here.
