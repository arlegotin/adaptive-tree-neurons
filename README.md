# Adaptive tree-neurons

Give a neuron two specialists and a learned switch between them. Keep the network's predictions unchanged when they arrive, then let training decide whether the extra flexibility helps.

This project implements that growth mechanism and tests a harder question: **can we choose which neurons deserve the extra capacity better than choosing at random?** It includes a compact PyTorch layer, a gradient-derived allocation score, matched ordinary and gated controls, and experiments that retain the cost of finding a model as well as training it.

**The mechanism works. A reliable placement advantage hasn't emerged.** The useful result is a working implementation and a reproducible account of what the proposed selector achieves, where it loses, and how much the search costs.

[Results](#results) · [Try the layer](#try-the-layer) · [Reproduction](REPRODUCE.md) · [Research design](docs/superpowers/specs/2026-10-16-selective-tree-neurons-design.md) · [Citation](CITATION.cff)

The [research v2 continuation](docs/research/v2/execution.md) is in progress;
the [current cross-study report](reports/research-v2.md) brings its results,
negative decisions and unfinished comparisons together.
Its [spec](docs/superpowers/specs/2026-10-18-research-v2-design.md) and
[implementation plan](docs/superpowers/plans/2026-10-18-research-v2.md) preserve
the finite experiments and claim gates. Checkpoint semantic hardening, diagnostic
foundations and transfer-data split preparation are verified. Native diagnostic
controls now include a complete256-fit study with independent raw-prediction
arithmetic. Growth helps the planted tasks, while the present chooser has not
established an advantage over random; see the [plain status](docs/research/v2/status.md)
and [M1 results](reports/research-v2-M1.md). The strong-baseline benchmark and protected confirmation remain unfinished.
The completed [M2 diagnostic](reports/research-v2-M2.md) finds substantial ordinary
training headroom and admits a bounded test of joint allocation. It does not
establish a win for the present chooser.
The completed [M3 study](reports/research-v2-M3.md) is more discouraging: across
16fresh primary synthetic problems, ordinary continuation beat the new joint-grown
model on all16problem averages, with about15.6%lower mean error. Joint placement
averaged1.25%higher error than random. The tested selector branch is closed as a
negative development result. The earlier [planted-control lead](reports/research-v2-small-joint.md)
remains a separate, narrower finding; the practical comparison is unfinished.

The [first native-anchor survey](reports/research-v2-primary-anchors.md) also now
reports primary validation scores from115existing fits. Adaptive placement has
one favorable real-data coordinate and one reversal; existing anchors lead on
all five observational validation splits. A static fully gated model leads the
separate Energy simulation. These single-seed development results are distinct
from the unfinished tuned benchmark and protected test evaluation.

The [Airfoil NODE follow-up](reports/research-v2-native-anchors-follow-up.md)
adds its completed native anchor: validation error0.107493 versus
the grown anchor's0.092186 and RealMLP's0.073302. All20Airfoil points remain
visible. The [Concrete tranche](reports/research-v2-native-anchors-concrete.md)
adds NODE at0.161727 versus grown0.209199 and CatBoost0.129860. These are single
development splits/seeds. The [Parkinsons tranche](reports/research-v2-native-anchors-parkinsons.md)
adds NODE at 1.311088 versus grown 1.246646 and CatBoost 0.738719.
**118 of 120 native engineering anchors have completed**; CCPP is running, with
Bike queued afterward. Native fitting uses each method's stopping rule.
The full tuned benchmark and protected evaluation remain unfinished.

The [B0 native-backbone screen](reports/research-v2-B0.md) completed 72 fits on
four new synthetic problems. A small ordinary network beat the validation-selected
backbone family on all four problem averages. The fixed screen therefore admits
no stronger-parent or residual-growth follow-up. This development result does
not rank fully tuned methods or reverse the separate real-data findings.

## The idea

An ordinary hidden neuron applies ReLU to one affine response. A grown neuron adds a gated contrast before that same ReLU:

```text
ordinary:  ReLU(base(x))
grown:     ReLU(base(x) + tanh(gate(x) / 2τ) × contrast(x))
```

Its two affine specialists are `base − contrast` and `base + contrast`. The gate starts at zero, so insertion preserves the old function exactly. The contrast starts nonzero, allowing the gate to learn immediately. Only selected neurons allocate the new parameters.

For the main one-hidden-layer experiment, growing 16 of 64 neurons changes the complete model from **1,153 to 1,697 parameters**. Random placement receives the same added capacity, inherited parent, contrast construction and continuation opportunity. Ordinary widening and static gated networks test whether simpler ways of spending those parameters explain the result.

Soft affine trees and gated units are established model families. The research target here is **capacity allocation during training**. Soft routing evaluates both specialists; it supplies no automatic inference saving. The [prior-art review](docs/research/2026-10-17-prior-art-revisit.md) records that boundary.

## Results

The published long-probe diagnostic tested **8 fresh independently generated regression problems, with 2 paired training repetitions per problem**. Each problem had 262,144 held-out evaluation observations. Short search tried every single-neuron insertion for 320 updates; long search allowed up to 8,000 updates with the normal validation stopping rule. Both then restarted the joint grown model from the original parent. The newer v2 studies are tracked separately in the [current status](docs/research/v2/status.md); their recipes and teachers differ from these historical results.

Lower normalized evaluation MSE is better: each problem's squared prediction error is divided by its fitting-target variance. Repetitions are averaged within each problem, then all 8 problems receive equal weight.

| Method | Parameters | Mean normalized MSE |
| --- | ---: | ---: |
| Continue the original network | 1,153 | 0.064867 |
| Our cheap allocation score | 1,697 | 0.064205 |
| Random placement | 1,697 | 0.063850 |
| Short validation search | 1,697 | **0.063125** |
| Long validation search | 1,697 | 0.063378 |
| Widen the existing ordinary network | 1,711 | 0.064071 |
| Fresh ordinary network, lower bracket | 1,693 | 0.067278 |
| Fresh ordinary network, upper bracket | 1,711 | 0.069375 |
| Static partially gated network | 1,697 | 0.100520 |

Long search's mean benefit over random was **+0.000471**, with a descriptive whole-problem bootstrap 95% interval of **[−0.001037, +0.002285]**. It won on 3 problem means and lost on 5. Short search had the best average error, but the comparisons do not establish a clear placement advantage. Long search beat the fresh upper-bracket ordinary model in this recipe; ordinary warm widening was much closer. That distinction matters.

The complete cohort took **2 hours 10 minutes** on one M3 Max, including **about 70 minutes of additional long-search work**. All 144 final model instances completed without numerical failures. An independent calculation from saved predictions agreed exactly with the reported metrics; both representative rankings replayed exactly across 130 trials, 53,300 updates and 133 complete saved-state comparisons. A long-probe checkpoint restore also matched.

An earlier, separate **16-problem × 2-repeat pilot** reached the same unresolved allocation question: cheap-score MSE **0.061188**, random **0.060652**, paired benefit **−0.000536**, interval **[−0.001205, +0.000156]**. These studies use different fresh problems; their error levels aren't a before/after comparison.

Read the [latest findings](docs/research/2026-10-18-long-probe-diagnostic.md), [complete diagnostic tables](reports/long-probe-diagnostic.md) and [earlier pilot report](reports/final.md). These are exploratory synthetic results. Independent confirmation, real-data transfer and competitiveness against strong tabular baselines remain unestablished.

## What you can use

- **Grow selected units without a prediction jump.** Compact active tensors, explicit topology export and an optimizer migration manifest.
- **Test an allocation rule against matched controls.** Shared parents, paired minibatches, fixed capacity, separate placement and contrast-direction comparisons, and exhaustive validation probes with their losing candidates retained.
- **Recover the actual training state.** Model, Adam moments, scheduler, RNGs, sampler and growth state survive checkpoints; raw predictions and source identities support later auditing.
- **Measure the complete procedure.** Charge scoring, candidate search and continuation, keep failures, and separate fitting, checkpoint selection and evaluation data.

The [source](src/treeneuron), [tests](tests) and [research decisions](DECISIONS.md) are included. Small synthetic engineering fixtures and extensive selected audit metadata are versioned. Full experiment archives and environments remain external; [publication scope](docs/PUBLICATION.md) explains what a clone can verify.

## Try the layer

The measured environment is **native macOS ARM64, Python 3.12 and PyTorch 2.14.1**. The supplied `uv.lock` records that environment; other platforms need separate verification. With [uv](https://docs.astral.sh/uv/) installed:

```sh
git clone https://github.com/arlegotin/adaptive-tree-neurons.git
cd adaptive-tree-neurons
uv sync --frozen --python 3.12
```

Run `uv run --frozen python`, then paste this example. The layer returns preactivations; its caller applies ReLU.

```python
import torch
from treeneuron.models.units import SelectiveAffineLayer

torch.manual_seed(42)
layer = SelectiveAffineLayer(d=16, h=64)
x = torch.randn(32, 16)
before = layer(x).detach()

# Contrast rows contain 16 weights and a bias. New gates start at zero.
manifest = layer.grow(torch.arange(16), 0.05 * torch.randn(16, 17))
after = layer(x)

torch.testing.assert_close(after, before, rtol=0, atol=0)
print(manifest.added_parameter_count)  # 544
print(after.relu().shape)             # torch.Size([32, 64])
```

Run the standalone algebra/gradient references and focused CPU checks:

```sh
uv run --frozen python tests/reference_math_checks.py
uv run --frozen python tests/reference_score_checks.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
uv run --frozen python -m pytest -q \
  tests/test_algebra.py tests/test_growth.py tests/test_score.py tests/test_probe_diagnostic.py
```

These checks need no retained research archive or external dataset. For training receipts, fresh experiments, full-suite status and authenticating historical results, follow [REPRODUCE.md](REPRODUCE.md).

## License and citation

Copyright © 2026 Artem Legotin. Original software is **AGPL-3.0-only**; original writing and result presentation are **CC BY 4.0**. [Licensing](LICENSING.md) defines the split and [third-party notices](THIRD_PARTY_NOTICES.md) retain the NODE, entmax and QHOptim exceptions.

Please cite [CITATION.cff](CITATION.cff) and record the commit you used. Implementation, experiments and documentation used AI assistance under the project owner's direction. README editing used [write-like-me](https://github.com/arlegotin/write-like-me).
