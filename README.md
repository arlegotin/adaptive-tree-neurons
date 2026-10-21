# Adaptive tree-neurons

Grow a neural network without changing what it already predicts. Give selected neurons two affine rules and a learned switch between them, then train the added flexibility.

The bet behind this project is that **where you add capacity should matter**. The repository makes that bet testable: a compact PyTorch layer, exact training-state recovery, and paired experiments that compare informed placement with random placement and ordinary training.

The gates learn useful planted structure. The tested placement rules haven't earned their extra complexity. What came out of the work is a reusable growth implementation and a fairly concrete result about where a plausible allocation idea breaks down.

[Results](#results) · [Try the layer](#try-the-layer) · [Research report](reports/research-v2.md) · [Reproduction](REPRODUCE.md) · [Citation](CITATION.cff)

## What you can reuse

- **Function-preserving growth.** Add gates to selected units with exactly unchanged initial predictions. Existing parameters keep their Adam history; new parameters get fresh state.
- **A controlled allocation experiment.** Fork the same parent, keep directions and capacity fixed, and change the placement rule. Compare with ordinary continuation, warm widening and fresh networks. Retain losing candidates and failed attempts.
- **Joint selection with exact ridge scores.** Score sets of gate directions while accounting for what the parent can already explain, then choose units greedily. The regularized objective has direct least-squares references and explicit checks for redundant directions.
- **Recoverable research runs.** Save model, optimizer, sampler, schedule and RNG together. Keep source identities, data roles, predictions and cost records attached to the experiment.

The [native baseline adapters](docs/research/v2/native-qualification-runner.md) cover ordinary MLPs, ResNet, RealMLP, TabM, boosted trees, NODE and simple references. Their qualification records distinguish short engineering checks from completed native fits.

## How it works

An ordinary neuron computes an affine rule followed by ReLU. A grown neuron blends two affine rules **before** ReLU. In centered form, with the bias included in $X=[x,1]$:

$$
z_j=b_j^\top X+\tanh\!\left(\frac{a_j^\top X}{2\tau}\right)c_j^\top X,
\qquad h_j=\operatorname{ReLU}(z_j).
$$

Start the gate at $a_j=0$, with a nonzero contrast $c_j$. The added term is exactly zero, so insertion preserves the parent function. The gate can receive a gradient immediately; the contrast starts learning after the gate moves.

The implemented research setting has one hidden layer and one growth event. Both soft branches are evaluated. Hard routing, recursive trees and automatic inference savings would require further work.

## Results

**The strongest completed comparison favors ordinary training.** On 16 fresh synthetic problems, with two paired optimizer repeats per problem, continuing the ordinary parent beat the new joint-growth selector on all 16 problem averages.

| Method | Parameters | Mean normalized evaluation MSE ↓ |
| --- | ---: | ---: |
| Continue ordinary training | 1,153 | **0.050774** |
| Warm ordinary widening | 1,711 | 0.054791 |
| Random placement, informed directions | 1,697 | 0.059415 |
| Joint ridge placement, same directions | 1,697 | 0.060158 |

Ordinary continuation had **15.6% lower mean error** than joint growth. Joint growth had **1.25% higher mean error** than matched random placement; its descriptive paired interval spans gains and losses. These are development results on the specified synthetic population, with repeats averaged within each problem. The [M3 report](reports/research-v2-M3.md) retains all ten methods, uncertainty, training treatment and costs.

There are useful positive findings around that result. In [256 planted and negative-control fits](reports/research-v2-M1.md), gates learned the planted conditional structure; random growth helped too. The [opportunity diagnostics](reports/research-v2-M2.md) exposed a stopping-rule effect: restoring the best complete state can make unchanged training replay an already unsuccessful tail. Giving ordinary continuation a fair opportunity matters before attributing a gain to new architecture. We also constructed and checked cases where the best individual additions form a poor pair, which motivated testing joint selection directly.

The earlier [16-problem pilot](reports/final.md) and [eight-problem longer-search diagnostic](reports/long-probe-diagnostic.md) likewise didn't establish a reliable placement advantage. The new joint-selector branch closed after its declared experiment. It wasn't extended until a favorable result appeared.

On real data, **118 of 120 native engineering anchors completed** across six datasets. Existing methods lead the grown anchor on all five observational validation splits examined. A static fully gated model leads the separate Energy simulation, a narrow representation result with few geometry groups. This [survey](reports/research-v2-primary-anchors.md) and its NODE follow-ups use one seed per split. The full tuned benchmark and protected test evaluation are unfinished. Research jobs are paused, with checkpoints retained; see [current status](docs/research/v2/status.md).

## Try the layer

The locked core targets **Python 3.12**, measured on macOS ARM64 with PyTorch 2.14.1. Other platforms need their own runtime checks. With [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/arlegotin/adaptive-tree-neurons.git
cd adaptive-tree-neurons
uv sync --frozen --python 3.12
uv run --frozen python
```

Paste this into Python. The layer returns preactivations; its caller applies ReLU.

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

For growth during training, the [insertion interface](src/treeneuron/growth/insert.py) also handles optimizer state. Run the algebra and score references without downloading datasets or restoring an experiment archive:

```sh
uv run --frozen python tests/reference_math_checks.py
uv run --frozen python tests/reference_score_checks.py
```

[REPRODUCE.md](REPRODUCE.md) gives focused test commands, optional dependencies and source-specific test status. The complete suite is not green; its failures and scoped recoveries are retained. Source, tests, compact result tables and selected audit records are public. Full raw training archives remain external, so a clone alone cannot authenticate every saved prediction. [Publication scope](docs/PUBLICATION.md) defines that boundary.

## License and citation

Copyright © 2026 Artem Legotin. Original code is **AGPL-3.0-only**; original writing and result presentation are **CC BY 4.0**. See [licensing](LICENSING.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

Please cite [CITATION.cff](CITATION.cff) and record the commit you used. Implementation, experiments and documentation used AI assistance under the owner's direction. README editing used [write-like-me](https://github.com/arlegotin/write-like-me).
