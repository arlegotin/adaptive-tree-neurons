# Native tree qualification brief

Read-only official-source investigation on18 October2026. These are installation
and adapter candidates, not installed/qualified baselines or predictive results.
No prepared dataset labels were accessed. Gate D remains pending.

| Family | Candidate | Official source commit | macOS/Python3.12 wheel SHA256 |
|---|---|---|---|
| P08 | CatBoost1.2.10 | b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084 | b27115d5b443048f710001c8ac666892dfe03498492310b00466203c91cc30a5 |
| P09 | XGBoost3.4.1 | 6fe8c547bdd21c73e4555d85b087d9260595d30d | a7afd7dbace0951c93aa85ffe046e54bc40893f5b51cd3e7991eb157bf9c7c7c |
| P10/P11 | LightGBM4.7.0 | 8f7036f03627054d5a54a6f965b13f4b9ff2cb63 | 129535462686f274df179133643118c5c5c5667167fe6c3a28d955f0b3c8e868 |

Wheel metadata: versioned official PyPI records for
[CatBoost](https://pypi.org/pypi/catboost/1.2.10/json),
[XGBoost](https://pypi.org/pypi/xgboost/3.4.1/json), and
[LightGBM](https://pypi.org/pypi/lightgbm/4.7.0/json).
The XGBoost floating stable documentation describes3.4.2 while this wheel is3.4.1;
use the immutable source above. No compatibility claim follows from wheel tags.

CatBoost anchor uses native CPU RMSE/default policy, one thread, max10000,
G stopping200 and use_best_model=True. Retain get_all_params(), automatic LR and
conditional bootstrap policy before freezing HPO. Explicit SkipTest excludes G
objects from Counter statistics. Categories use F dictionaries and canonical
strings/integers, not raw float category values. See official
[parameters](https://catboost.ai/docs/en/references/training-parameters/common),
[fit](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit),
[category rules](https://catboost.ai/docs/en/concepts/faq#why-is-it-forbidden-to-use-float-values-and-nan-values-for-categorical-features), and
[pinned defaults](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/options/cat_feature_options.cpp#L234).

XGBoost native train uses hist/reg:squarederror/CPU/nthread1/max10000. F builds
QuantileDMatrix; G references F quantization. Inject native-default anchor depth6,
eta.3/child1/subsample1/colsample1/L2=1/L1=0. Preserve native automatic intercept.
Create a fresh EarlyStopping(100,data_name='G',metric_name='rmse',maximize=False,
save_best=True) per trial: plain early_stopping_rounds returns the last booster.
Retain executed rounds separately from retained rounds and full evaluation history.
F-only category dictionaries need an explicit unknown-to-missing rule, not a
claim that native recoding handles unseen categories. Serialize UBJSON/JSON and
verify category metadata. See pinned
[parameters](https://github.com/dmlc/xgboost/blob/6fe8c547bdd21c73e4555d85b087d9260595d30d/doc/parameter.rst),
[training](https://github.com/dmlc/xgboost/blob/6fe8c547bdd21c73e4555d85b087d9260595d30d/python-package/xgboost/training.py),
[callback](https://github.com/dmlc/xgboost/blob/6fe8c547bdd21c73e4555d85b087d9260595d30d/python-package/xgboost/callback.py), and
[categories](https://github.com/dmlc/xgboost/blob/6fe8c547bdd21c73e4555d85b087d9260595d30d/doc/tutorials/categorical.rst).

LightGBM native Dataset/train uses CPU serial/one thread/regression L2, anchor
LR.1/leaves31/min_leaf20/L2=0; linear_lambda=0 for P11. Native maxima10000/5000,
G stopping100. P10 bagging_freq=1, anchor fraction1. A declared deterministic
histogram policy can use deterministic=True/force_col_wise=True; record its cost.
P10/P11 and every trial require fresh F Dataset objects. linear_tree must be in
Dataset construction parameters; G references that F Dataset. Rebuilding avoids
stale feature prefiltering. Scale P11 continuous predictors using F; native
categorical codes can split but do not enter leaf regressions. Inspect actual
nonempty leaf coefficients after multiple trees; the first tree is constant.
See pinned [defaults](https://github.com/lightgbm-org/LightGBM/blob/8f7036f03627054d5a54a6f965b13f4b9ff2cb63/include/LightGBM/config.h),
[construction guard](https://github.com/lightgbm-org/LightGBM/blob/8f7036f03627054d5a54a6f965b13f4b9ff2cb63/src/c_api.cpp#L354), and
[linear tree semantics](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters.html#linear_tree).

Native row-weighted G stopping is allowed; the coordinator uses group/day-weighted
V risk for HPO. Adapters must be lazy/import-isolated, consume only F/G and query
features, persist F transforms/categories, and return original-unit predictions.
Keep legacy adapters/guards intact. Use synthetic192F/64G fixtures for native
direct-call equality, save/load, unknown categories, F-only quantization/scaling,
applied-versus-recorded G consistency and actual linear leaves. Short fixtures
are quality-ineligible; real anchors must reach their native eligible endpoint.
Interrupted native callbacks need full verified resume or one deterministic same
seed operational rerun, retaining both costs and attempts.

Resolve a full optional dependency lock against the frozen core before install.
Direct candidates (not a complete transitive lock): narwhals2.27.1
`d057df13f5852b8e157596e82eb5e955fad267425df5e420e0ee9863da483b31`,
graphviz.21 `54f33de9f4f911d7e84e4191749cac8cc5653f815b06738c54db9a15ab8b1e42`,
matplotlib3.11.2 `ef31985c4dedb5f1424e1aec6849a47dd37689cb7fa3c20b1b82187f26806261`,
plotly7.1.0 `dbb7fa18afce40d0a8e80d1bf162eceb3faa0ce5a77fe741ad09a74cf78f53f3`.
CatBoost additionally declares six (core1.17.0 already present). Inspect import
locations, duplicate distributions, wheel RECORDs and native libraries. Existing
Homebrew libomp20.1.7 was read-only hashed at
`/opt/homebrew/Cellar/libomp/20.1.7/lib/libomp.dylib`,740896bytes,
SHA256 `956f48d4d7abf8edc01ed0111694aa7df0fe00ca6824790188af95daaacb8502`.
Verify actual loaded libraries in fresh method-specific processes; do not suppress
OpenMP conflicts with KMP_DUPLICATE_LIB_OK. A changed library needs a new receipt.

Remaining: full transitive wheel lock, install/import qualification, adapters,
synthetic fixtures, genuine native anchor runs, workload/cost receipts, then Gate D.
