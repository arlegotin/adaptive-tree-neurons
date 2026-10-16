# Evidence and machine record

Recorded: 16 October 2026. This file separates the supplied research from observations made while writing the spec and plans. It is a planning record, not a completed WP00 report.

## Evidence status

The user supplied a researched proposal titled “Research and proof-of-concept plan: selectively growing tree-neurons,” with a research cut-off of 16 October 2026. Its equations, experiments, comparison families, data suite, statistical rules, and reference identifiers are the requirements baseline for this repository.

The research author reports executed Linux CPU algebra/autograd checks, but no predictive benchmark, downloaded public dataset, or M3 Max experiment. This planning session has not independently reproduced those checks, inspected the linked papers, verified current package releases or licenses, installed dependencies, or trained a model. Descriptions of prior work and software below are attributed to the supplied proposal. WP00 must verify consequential source, version, license, and asset claims before relying on them. No methodological novelty is established here.

## Local discovery performed during planning

Commands ran in the current repository on 16 October 2026. Outputs below omit machine serial numbers and hardware UUIDs. Storage and power are point-in-time observations; rediscover them before installation or execution.

| Check | Observed value |
|---|---|
| `uname -m` | `arm64` |
| `sw_vers` | macOS `26.7.1`, build `25G241` |
| `system_profiler SPHardwareDataType -detailLevel mini` | MacBook Pro `Mac15,11`; Apple M3 Max; 14 cores, 10 performance and 4 efficiency; memory displayed as 36 GB |
| `sysctl -n hw.memsize` | `38654705664` bytes (36 GiB) |
| `sysctl -n hw.logicalcpu` | `14` |
| `df -h .` | Data volume: size 867 GiB; used 817 GiB; available 21 GiB; displayed capacity 98% |
| `xcode-select -p` | `/Applications/Xcode.app/Contents/Developer` |
| `python3` | Python `3.14.6`, Clang 21.0.0; native `arm64` |
| Python module discovery | `torch` absent from the default Python |
| Tool discovery | `python3`, `pip3`, `uv`, `git`, `xcodebuild`, and `rg` found |
| `pmset -g batt` | AC power, battery 100%, charged |
| Repository at start | Empty working tree; unborn current branch `main`; no existing project files or `AGENTS.md` found |

The sandbox denied the first `sysctl` reads; a permitted read outside the sandbox returned the memory and logical-CPU values. `system_profiler` emitted an `hw.cpufamily` access error but returned the hardware summary above. Neither condition establishes an MPS limitation. MPS availability and kernel behavior remain untested because PyTorch is absent.

The initial OS/framework memory reservation is `max(8 GiB, 0.25 × 36 GiB) = 9 GiB`. This is a starting reservation, not observed free RAM or permission to allocate the remaining 27 GiB. WP00 must measure current pressure, swap, and the MPS working-set recommendation. The 21 GiB disk observation warrants a download/install forecast before creating optional environments or acquiring weights; no cleanup or system modification is authorized by this document.

## Reported checks from the supplied research

These are historical author-reported results, not measurements on this laptop.

Appendix A environment: Linux x86-64, Python 3.13.5, PyTorch 2.10.0+cpu, CPU float64, one thread, seed 1602026, `N=37, d=7, h=5`.

| Check | Reported result |
|---|---:|
| Tree versus affine-plus-gate forward max absolute error | `8.882e-16` |
| Shared-coordinate gradient max absolute error | `1.776e-15` |
| Scalar-head affine collapse max absolute error | `2.665e-15` |
| Centered insertion prediction change | `0.0` |
| Analytic gate-gradient max absolute error | `5.551e-17` |
| Initial contrast-gradient norm | `0.0` |
| Initial gate-gradient norm | `0.4451935795` |
| Loss before / after a tiny gate-only step | `18.7354818901 / 18.7354681759` |
| Soft–hard gap identity max absolute error | `8.882e-16` |
| ReLU bound maximum violation | `6.661e-16` |

Appendix B environment: same stack, seed 1602027, `N=41, d=5, h=4`. Reported errors: rescaled gate gradient `1.388e-16`, rescaled score `1.388e-17`, permuted score `2.776e-17`, normalized analytic gradient `5.551e-17`. Predicted/observed gate-only first-Adam-step loss decrease: `8.941316578e-7 / 8.941315848e-7`; relative approximation error `8.168e-8`.

These checks motivate local reproduction. They establish neither allocation quality nor generalization, convergence, current dependency support, or local performance.

## Reference ledger inherited from the brief

The identifiers below preserve traceability to the user’s research. Links are source locations supplied by the user, not fresh verification in this planning session. Snapshot retrieval date, content hash/revision, license, and inspected scope during WP00; revisit the closest overlap at WP09.

| ID | Source | Relevance / execution obligation |
|---|---|---|
| R01 | Jordan and Jacobs, [Hierarchical Mixtures of Experts and the EM Algorithm](https://www.cs.toronto.edu/~hinton/absps/hme.pdf), 1993/1994 | Established hierarchical gating and affine experts. |
| R02 | Dauphin et al., [Language Modeling with Gated Convolutional Networks](https://proceedings.mlr.press/v70/dauphin17a/dauphin17a.pdf), ICML 2017 | Gated-linear primitive; the stump identity is a direct algebraic deduction. |
| R03 | Srivastava et al., [Highway Networks](https://arxiv.org/abs/1505.00387), 2015 | Gated carry/transform paths. |
| R04 | İrsoy and Alpaydın, [Distributed Decision Trees](https://arxiv.org/html/1412.6388v1), 2014 | Soft/distributed and budding-tree precedent. |
| R05 | Tanno et al., [Adaptive Neural Trees](https://proceedings.mlr.press/v97/tanno19a.html), ICML 2019 | Validation-driven structural growth. |
| R06 | [Official ANT repository](https://github.com/rtanno21609/AdaptiveNeuralTrees) | Supplied research reports MIT and Python 2.7-era dependencies; verify before copying. |
| R07 | Wu et al., [Splitting Steepest Descent for Growing Neural Architectures](https://arxiv.org/abs/1910.02366), NeurIPS 2019 | Derivative-informed allocation precedent; different splitting matrix. |
| R08 | Martinelli et al., [Flat Channels to Infinity in Neural Loss Landscapes](https://arxiv.org/html/2506.14951v2), 2025 | Splitting, parameterization, gated limits. |
| R09 | University of Siegen, [Local Model Network Toolbox / LOLIMOT](https://www.mb.uni-siegen.de/mrt/lmn-tool/?lang=de) | Incremental local affine models; supplied noncommercial terms need checking. |
| R10 | Han et al., [Parameter efficient dendritic-tree neurons outperform perceptrons](https://arxiv.org/html/2207.00708v1), 2022 | Adjacent scalar-unit complexity; workshop evidence, different wiring. |
| R11 | Ojha and Nicosia, [Backpropagation Neural Tree](https://doi.org/10.1016/j.neunet.2022.02.003), 2022 | Computational trees differ from decision routing. |
| R12 | Yan et al., [Arboreal Neural Network listing](https://icml.cc/Downloads/2026), [OpenReview](https://openreview.net/forum?id=NDM1ltvBOf), [PDF route](https://openreview.net/pdf/5800594ecbf7f51db4b3b98499665c0100a30ba4.pdf) | Supplied research could access metadata only; unresolved novelty hold. Try official/author copies without bypassing access controls. |
| R13 | Kontschieder et al., [Deep Neural Decision Forests](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf), ICCV 2015 | Differentiable neural/tree integration. |
| R14 | Popov et al., [NODE](https://arxiv.org/abs/1909.06312), ICLR 2020 | Differentiable oblivious-tree baseline. |
| R15 | [Official NODE repository](https://github.com/Qwicen/node), [odst.py](https://raw.githubusercontent.com/Qwicen/node/master/lib/odst.py) | Pin actual choice/bin functions and F-only initialization. |
| R16 | Marton et al., [GRANDE](https://arxiv.org/abs/2309.17130), [repository](https://github.com/s-marton/GRANDE), ICLR 2024 | Optional hard-tree extension comparator. |
| R17 | Liu et al., [KAN](https://arxiv.org/abs/2404.19756), 2024 | Univariate function baseline motivation. |
| R18 | Grinsztajn et al., [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815), NeurIPS 2022 | Historical context, no imported scores. |
| R19 | Holzmüller et al., [Better by Default](https://arxiv.org/html/2407.04491v3), [pytabkit](https://github.com/dholzmueller/pytabkit), NeurIPS 2024 | Faithful RealMLP defaults/preprocessing. |
| R20 | Gorishniy et al., [TabM](https://arxiv.org/abs/2410.24210), [official code](https://github.com/yandex-research/tabm), ICLR 2025 | Member-loss training, averaged prediction, official defaults; supplied Apache-2.0 claim needs checking. |
| R21 | Erickson et al., [TabArena](https://arxiv.org/html/2506.16791v4), 2025 | Tuning/ensembling context, not this project’s benchmark. |
| R22 | Prior Labs, [TabPFN-3.5 report](https://arxiv.org/html/2609.17895v1), September 2026 | Supplied current model context; verify asset/version claims. |
| R23 | [TabPFN repository](https://github.com/PriorLabs/tabpfn), [3.5 model card](https://huggingface.co/Prior-Labs/tabpfn_3_5) | Separate code/weight terms; supplied Apple PyTorch 2.13+ recommendation and 5,000-row CPU allowance require rechecking. |
| R24 | Qu et al., [TabICLv2](https://arxiv.org/abs/2602.11139), [repository](https://github.com/soda-inria/tabicl) | Supplied regression checkpoint `tabicl-regressor-v2-20260212.ckpt`; verify weights/license/default ensemble. |
| R25 | [CatBoost](https://github.com/catboost/catboost), [release files](https://pypi.org/project/catboost/) | Supplied snapshot 1.2.10 and universal2 wheels; resolve native compatibility. |
| R26 | [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html) | Native CPU histogram path and OpenMP verification. |
| R27 | UCI [Airfoil Self-Noise, 291](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) | 1,503 × 5 numerical inputs; proposed configuration grouping. |
| R28 | UCI [Concrete Compressive Strength, 165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) | 1,030 × 8; mixture grouping across ages. |
| R29 | UCI [Energy Efficiency, 242](https://archive.ics.uci.edu/dataset/242/energy+efficiency) | 768 × 8; simulated; heating only, geometry groups. |
| R30 | UCI [Combined Cycle Power Plant, 294](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) | 9,568 × 4; one of five shuffled copies, verify multisets. |
| R31 | UCI [Parkinsons Telemonitoring, 189](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) | 5,875 rows, 42 subjects; exclude identifier and total UPDRS. |
| R32 | Cawley and Talbot, [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html), JMLR 2010 | Separate selection from outer evaluation. |
| R33 | Bouthillier et al., [Accounting for Variance in Machine Learning Benchmarks](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html), MLSys 2021 | Independent units and variance decomposition. |
| R34 | UCI [Bike Sharing, 275](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | Hourly 17,379 rows; chronological day splits, target-component exclusions. |
| R35 | Apple, [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) | Version-specific requirements need local verification. |
| R36 | PyTorch, [MPS backend notes](https://docs.pytorch.org/docs/main/notes/mps.html) | Kernel support must be exercised, not inferred. |
| R37 | PyTorch, [MPS environment](https://docs.pytorch.org/docs/main/mps_environment_variables.html), [synchronize](https://docs.pytorch.org/docs/main/generated/torch.mps.synchronize.html), [working-set memory](https://docs.pytorch.org/docs/main/generated/torch.mps.recommended_max_memory.html), [allocated memory](https://docs.pytorch.org/docs/main/generated/torch.mps.current_allocated_memory.html) | Memory counters overlap; synchronize timed regions. |
| R38 | Microsoft et al., [Neural trees, US20200005148A1](https://patents.google.com/patent/US20200005148A1/en) | Technical overlap screen only; no legal conclusion. |
| R39 | LightGBM, [parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html), [installation](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html) | Supplied docs snapshot 4.7.0; linear leaves, native macOS/OpenMP path. |

## Reproduction recipes for the two reference checks

WP01 implements standalone scripts under `tests/reference_math_checks.py` and `tests/reference_score_checks.py`. The numerical recipe is preserved here so execution does not depend on access to the chat.

For Appendix A, set CPU float64, one Torch thread, seed 1602026, `N=37,d=7,h=5`. In RNG draw order create `x=randn(N,d)` requiring gradients, `L,R,G=randn(h,d+1)` requiring gradients, and augmented `X=[x,1]`. Form `g=sigmoid(X@G.T)`, `l=X@L.T`, `r=X@R.T`, `tree=(1-g)*l+g*r`, `skip=l+g*(r-l)`. Draw `probe=randn_like(tree)`; compare gradients of both probe sums with respect to `(x,L,R,G)` using the shared tensors. Draw `v=randn(h)`, `c=randn(())`; compare `tree@v+c` to `X@(v@L)+(g*(X@(v[:,None]*(R-L)).T)).sum(1)+c`.

Continue the same RNG stream: draw `B=randn(h,d+1)`, `head=randn(h)`, `y=randn(N)`. Set `base=X.detach()@B.T`, `pre=base.clone().requires_grad_()`, `old_pred=pre.relu()@head`, `old_loss=mean((old_pred-y)^2)`. Obtain `delta=grad(sum((old_pred-y)^2),pre)`, detached. For `j=2`, `M=X0.T@(delta[:,j:j+1]*X0)/N`, where `X0=X.detach()`. Use `eigh(M)` and its largest-absolute-eigenvalue eigenvector `q`; the original Appendix A has no sign normalization. Set `D=(0.05*q).detach().requires_grad_()`, `A=zeros(d+1,requires_grad=True)`, `tau=1`. Add `tanh(X0@A/2)*(X0@D)` only to preactivation `j`, apply ReLU and the same head, and differentiate mean squared error with respect to `(A,D)`. Compare the gate gradient to `M@D.detach()/2`. Take `step=1e-4/(1+norm(gA))`, `A1=A-step*gA`, and verify the loss decreases. Compare `abs(tree-hard)` with `min(g,1-g)*abs(r-l)` using `hard=where(g>=0.5,r,l)`; check the ReLU discrepancy bound.

Appendix A assertions: forward/gradient/collapse/analytic/gap errors `<1e-12`, insertion error and initial contrast-gradient norm `==0`, ReLU maximum bound violation `<1e-12`, final tiny-step loss `<old_loss`. Emit environment, shapes, seed, all error maxima/norms/losses as JSON. Numerical values can vary within tolerance with a different linear algebra implementation.

For Appendix B, start a fresh process: CPU float64, one thread, seed 1602027, `N=41,d=5,h=4`. Draw `X=[randn(N,d),1]`, then `W=randn(h,d+1)`, `head=randn(h)`, `y=randn(N)`. Implement the normalized score in the spec with `epsilon=0.05,tau=1,adam_epsilon=1e-8`; compute `e` from the sum of row losses, symmetrize each moment, apply the largest-coordinate-positive sign convention, and return stacked contrasts `V`, gradients `U`, scores `S`. All denominators are nondegenerate on this reference draw.

Recompute after positive row rescaling `[0.3,2.0,7.0,1.1]` of `W` with reciprocal head rescaling; compare gradients and scores. Recompute after permutation `[2,0,3,1]`; compare to the permuted original scores. For `j=argmax(S)`, train only a fresh `A=zeros(d+1)` with AdamW, learning rate `1e-6`, epsilon `1e-8`, weight decay `0.0`, keeping parent and `V[j]` fixed. Assert the autograd gate gradient matches `U[j]`. Predicted loss decrease is `1e-6*sum(U[j]^2/(abs(U[j])+1e-8))`. Step once and compare the observed decrease. Assert all invariance/gradient errors `<1e-12`, observed decrease `>0`, and relative prediction error `<1e-4`; emit JSON. The primary project acceptance tolerance for its wider float64 test suite remains `<1e-10`.
