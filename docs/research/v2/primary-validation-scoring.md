# Primary validation prediction scoring

This Task8 component binds one original-unit prediction archive to a sealed
primary transfer split, then computes the fixed V metric in the coordinator.
It is not a fitting worker, a native qualification or eligibility promotion.
It does not select a winner or open T/secondary-development labels. Its first
verification uses artificial sealed grouped/temporal tables only.

`score_primary_validation_prediction` takes external manifest/prediction byte
commitments and a declared primary outer fold. It verifies the sealed manifest,
primary split and feature-only V row/group/time identities before opening V
labels. Prediction archives contain exactly prediction, row_id, group_id and,
for Bike, timestamps. Unexpected fields, duplicate/reordered rows, wrong groups,
dates, nonfinite predictions and byte drift fail. All referenced bytes used by
the scorer are checked again before returning.

Only the coordinator calls the existing logged `coordinator_validation` reader.
F targets supply ddof0 variance; all errors use original-unit float64 arithmetic.
Stored float32 native predictions are explicitly promoted for metric arithmetic,
with their original dtype recorded. Groups receive equal weight, then rows
within a group; Bike uses equal days, then hours within a day. Row RMSE/MAE are
also retained. Zero F variance gives unavailable normalized risk, without a floor
or an invented adverse score. A secondary fold remains forbidden until the
separate global-freeze release is implemented.

The existing Bike store numbers observed dates from zero; the independent metric
uses calendar-day ordinals. The scorer checks an exact bijection, retains the
sealed group-ID hash and mapping, and supplies calendar ordinals only to metric
arithmetic. Sealed data identities remain unchanged. The logged V reader is
followed by strict validation of the original label payload, so its legacy
ObservationSet conversion cannot conceal a float32 target or an extra field.
Worker prediction paths must be outside the sealed store, including its protected
directory. No protected file is accepted as a candidate archive.

The result authenticates the opened split/prediction bytes and arithmetic. It
cannot authenticate model generation, raw acquisition or qualification from
those hashes, and its quality/selection/protected flags remain false. The future
production callback must independently validate its model/qualification before
using this score in an eligible search result. Reading real V is deliberately
not part of the component's fixture verification. Startup, data loading, hashing
and scoring must remain charged to the enclosing procedure when integrated.

Verification retained 22 initial missing-component failures. The first combined
run passed 40 checks and exposed the Bike ID-convention mismatch; the corrected
run passed 41. Two additional malformed-label cases then failed before their
strict payload checks were implemented. The final affected run passed 63 checks
in 1.22 s, using artificial Energy/Bike tables, prediction arithmetic, transfer
roles and practical metrics. It covers hash/row/group/time/field/dtype/finite
sentinels, zero F variance, explicit prediction upcast and late byte drift in
each of the five references. No actual transfer V/T labels, production search,
eligibility promotion or scientific result was produced. The complete receipt is
[WP08-primary-validation-scoring-receipt.json](WP08-primary-validation-scoring-receipt.json).
