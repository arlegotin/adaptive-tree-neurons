# Independent M2 prediction arithmetic

`research/opportunity_prediction_audit.py` checks an inactive producer queue under
a read-only shared lease, using externally supplied plan and final ledger hashes.
The shared queue reader authenticates all retained paid-attempt files. The new
auditor accepts the frozen producer's source identity; it does not require that
identity to equal the later auditor source. It loads no model, training module,
producer data loader, producer metric helper, Torch or sklearn.

For every planned evaluation job it preserves the terminal, missing or ineligible
status. Eligible jobs must have one eligible data dependency and one eligible
model-producer dependency. It checks data requests/source/split hashes, file and
array commitments, float64 targets/features and int64 row/group IDs, declared
F/G/V assignments and every V/E prediction/target/row/group alignment. Both the
selected model and every declared horizon-prefix model must have exactly the
expected prediction vectors and metric records.

The auditor computes F population target variance, MSE/NMSE/RMSE/MAE, linear95th
absolute-error quantile, maximum error and noise-free-target MSE independently.
It divides before summation, uses Python fsum and an explicit sorted quantile,
then compares with retained producer metrics at rtol1e-12/atol1e-14. Undefined
zero-variance normalization remains None with its reason. Unexpected model/vector/
metric inventories fail; failed or never-started cases cannot silently disappear.

This establishes raw arithmetic conditional on retained bytes. It does not
regenerate a teacher, reexecute model predictions or validate checkpoint tensor
semantics. It does not decide Gate A or establish predictive superiority. A fully
audited engineering-qualification queue remains engineering evidence. Its data
and fits cannot become additional independent scientific teachers.

Fourteen missing-module REDs became GREEN. Tests cover authenticated but wrong
metrics, variance, target/row/group alignment, array headers, missing/extra model
vectors, strict external endpoints, active-worker exclusion, incomplete inventory
and finite means whose totals overflow. Two additional checks (horizon vectors
and import isolation) passed first run. The affected auditor/queue/collection/
control-audit/metric suite passed66tests in2.58s. Actual production M2 remains
active; the auditor has not opened its predictions.

Use the standalone helper only once the producer queue is inactive, and write
outside it:

```sh
PYTHONPATH=src "$ATN_PY" -B docs/research/v2/scripts/audit_m2_predictions.py \
  --study /absolute/immutable/producer-queue \
  --expected-plan-hash EXTERNALLY_RETAINED_PLAN_SHA256 \
  --expected-ledger-hash EXTERNALLY_RETAINED_FINAL_EVENT_SHA256 \
  --output artifacts/research-v2/WP03/independent-prediction-audit
```

The helper records separate producer/auditor identities, exact commands/runtime,
output hashes and failures. It does not modify the frozen queue. Frozen-source
`collect_opportunity` still owns the prespecified opportunity/Gate A decision;
this later arithmetic audit supplies a separate consistency check.
