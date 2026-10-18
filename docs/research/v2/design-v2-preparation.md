# Prospective design arithmetic preparation

Task9 is partially implemented before any real variance calibration, practical
primary search or protected evaluation. This is software/simulation evidence,
not a completed WP09 design or permission to freeze.

`analysis/design_v2.py` first averages exactly two eligible risk observations per
teacher/role, then takes the teacher log-risk ratio. Missing/failed observations
and undefined zero risks fail rather than disappearing. Ordered attempts supply
teacher order; lexical IDs do not determine the initial16 prefix.

Calibration accepts16 or32 independent teacher effects, uses10,000 fixed seeded
whole-teacher resamples, central90% SD interval, bootstrap90th SD percentile and
the one-sided90% normal-theory SD upper bound. Their maximum is s_U. Only the
predeclared SD interval/runtime precision failures trigger one16-to32 extension;
effect mean/sign never selects the alternative or extension. A32-teacher size
decision must reconstruct the triggering initial16 receipt from its exact ordered
float64 effect-byte commitment and original runtime forecast/seed. Raw teacher,
method, source and protocol authenticity remain external requirements.

The useful margin is log(1/.98), distinct from the5% alternative log(1/.95).
The normal approximation starts the fixed32,40,…,128 grid. Each admitted K must
also pass finite-t power and100,000 simulated cohorts under each declared centered
empirical, matched-variance normal and t5 law. Empirical atoms have population SD
s_U; pilot mean is removed. Exact two-sided95% Clopper–Pearson bounds require
power lower>=.90 and margin-null rejection upper<=.03. Failed fixed-K validations
stay retained while the prescribed grid advances; infeasible/exhausted designs
become estimation-only. No alternative analysis is searched. Every output remains
freeze-ineligible until the remaining source/protocol/resource gates exist.

A deliberately sparse no-op toy fails the32-teacher t check:12,794of100,000
empirical margin-null cohorts reject (12.794%, interval12.588–13.003%). A symmetric
small-variance toy passes its declared gate. Both are analytic software fixtures,
with zero independent scientific teachers. They demonstrate why passing a formula
alone cannot authorize confirmation; they say nothing about the actual future
teacher-effect distribution. Original v1 and corrected v2 outputs have separate
source hashes and are both retained.

The single independent review found two material issues. Original-dtype centering
corrupted adjacent-float32 and adjacent-float64 empirical nulls; reference-shifted
float64 centering now governs calibration/bootstrap/simulation. The32-teacher
decision lacked initial16 authorization; exact prefix commitments and triggering
rule reconstruction now enforce it. Three regressions were observed RED then
GREEN. Declared teacher order and raw zero-effect labeling received separate RED
checks during the same pass. Final affected verification:86passed in1.87s,
including62 unchanged legacy design checks. No re-review was commissioned.

Deferred minor: direct selector regressions for a failed first K advancing and
complete-grid exhaustion. Full protocol/fresh-root/source/resource/protected
validation, multiplicity, actual calibration and Gate F remain pending. The old
significance-conditioned planning mode is unchanged.

Reproduce software fixtures only:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src "$ATN_PY" docs/research/v2/scripts/check_design_v2.py \
  --output artifacts/research-v2/WP09/design-fixtures-fresh
```

[Receipt](WP09-design-preparation-receipt.json) binds the tests, retained failures,
helper source and both toy execution files. This component cannot freeze a study.
