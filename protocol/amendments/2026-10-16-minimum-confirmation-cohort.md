# Pre-freeze correction: independent confirmation cohort size

Adopted 16 October 2026, before any protected test access or protocol freeze.
Hypothesis H1, its population, metric, pairing and seed separation are unchanged.

The normal approximation can yield one teacher: with planning effect 0.05 and
conservative pilot SD 0.01, `ceil((z_0.975+z_0.8)^2*0.01^2/0.05^2)=1`.
A singleton whole-teacher bootstrap is a point interval, and Student-t
sensitivity has zero degrees of freedom. Neither estimates population uncertainty.

Archive the approximate result as `K_calc`. Set the actual independent cohort to
`K=max(K_calc, K_pilot)`, where `K_pilot` is the completed accepted
variance-calibration pilot's number of independent teachers, validated against
its existing initial minimum of eight. Pilot teachers inform the design only;
all confirmatory teachers and observations remain fresh.

Recalculate runtime and retained-storage forecasts for the actual count,
including repetitions, anchors, retries and artifacts. Unsupported or infeasible
designs use the existing estimation/resource-limited branch, without silently
reducing the cohort or retaining a powered-superiority claim.

Reject singleton and zero/undefined-variance inference as support for superiority.
Report the frozen finite-sample/resampling diagnostics, assumptions and limits.
This study-specific floor does not guarantee nominal 95% coverage and introduces
no application utility threshold. If wrong, it spends extra runtime/storage
relative to `K_calc`; an estimation-only interpretation may still be necessary.

The [preflight review](../../docs/research/2026-10-16-h1-design-preflight.md)
records the defect and the technical assessment. This correction changes no
historical exploratory result or source identity.
