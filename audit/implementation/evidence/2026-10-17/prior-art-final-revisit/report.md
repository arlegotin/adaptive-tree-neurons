# WP09 bounded closest-prior-art revisit

Recorded 2026-10-17. Scope: R12 architecture, initialization, optimization and appendix; R07/R08 insofar as they bear on insertion, spectral scores and allocation. The comparison uses the project specification §§4–5, not pilot outcomes. No training, tests, numerical imports, code audit, vendor-code copying or Git operations were performed. This is source comparison, not independent peer review or proof of absence.

**Decision:** these inspected sources do not establish a prior implementation of the project's exact combined rule, so this revisit does not trigger mandatory reclassification as an exact replication. They do establish substantial precedent, including spectral selection of learned neurons, function-preserving duplication, GLU emergence, and—in R07 Appendix B.2—random placement with the same per-neuron splitting direction. Neither the architecture family nor that allocation-control idea warrants a novelty claim. Remaining differentiation is a narrowly specified allocation/training characterization, with exact-rule priority unresolved outside this bounded comparison.

## Comparison target

The selected unit computes `ReLU(b_j + tanh(a_j^T X/(2τ)) v_j^T X)`, with augmented input `X=[x,1]`. Inserting `a_j=0` with nonzero `v_j` preserves predictions. At the learned parent, `M_j=mean_i[e_ij X_i X_i^T]`, where `e_ij` is the whole-loss derivative with respect to that unit's preactivation. The contrast uses a sign-fixed dominant **absolute** eigenvector, scaled to `0.05 RMS_F(b_j)/max(RMS_F(Xq_j),1e-12)`. The score ranks `sum_k u_jk²/(|u_jk|+Adam_epsilon)/[2(d+1)]`, with `u_j=M_j v_j/(2τ)`. One top-versus-uniform-random allocation event shares the parent, eligible set, growth count, each unit's contrast, minibatches and continuation updates.

Algebraically the preactivation is `(b_j-v_j^T X)+sigmoid(a_j^T X/τ)(2v_j^T X)`: affine bypass plus GLU. The final ReLU and the insertion/optimizer coordinates matter to the comparison.

## R12 — Arboreal Neural Network

**Inspected:** official full PDF, pp. 3–6 (Figure 1, §§3.1–3.4, Eqs. 1–11), p. 7 training details, pp. 8–9 (§4.3, Table 4, §5), and complete Appendix A/Algorithm 1, p. 12. Appendix B is dataset documentation. [Official entry](https://proceedings.mlr.press/v306/yan26o.html), [official PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/yan26o/yan26o.pdf).

**Confirmed overlap:** differentiable tree-derived units, tanh routing, inherited trained structure and gradient-based parameter refinement. Its prediction sums ArborCells initialized from a pretrained decision-tree ensemble. Eqs. 3–6 use feature-threshold projections, tanh decisions, topology-weighted affinities, a softmax over leaves, and scalar leaf values. Appendix A compiles existing tree topology by traversal; it is not an allocation algorithm for ordinary learned neurons.

**Narrower difference:** the preferred configuration fixes feature assignments/topology and refines thresholds and scalar leaf values; Table 4 also studies unfreezing other parameters. This is different from the project's affine-contrast preactivation, zero-gate insertion and moment/Adam allocation. AdamW training in §4.1 does not itself imply the project's Adam-derived selection score. Section 5 explicitly leaves learning discrete topology from scratch as future work. No exact project rule was located in the inspected mechanism/training/appendix material.

**Unresolved scope / wording:** paper-level comparison does not certify every author-code path or later revision. No code was acquired. Safe wording: “ArborCell supplies a related differentiable tree family, while this experiment studies conditional affine insertions into learned ReLU preactivations.” Access is resolved; it must not remain an asserted blocker. This comparison does not endorse the paper's broader fidelity or performance claims.

## R07 — Splitting Steepest Descent for Growing Neural Architectures

**Inspected:** arXiv v3, §§2.1–2.3, pp. 3–7, Eqs. 1–9, Theorems 2.2–2.5 and Algorithm 1; Appendix A.1, pp. 12–14; §3 RBF comparison, p. 8; Appendices B.1/B.2, pp. 18–19, and B.3 training details, p. 21. [Primary paper](https://arxiv.org/pdf/1910.02366v3).

**Confirmed overlap:** exact preservation when offspring are identical and their positive combination weights sum to one (Eq. 2); loss-derivative-informed per-unit symmetric matrices; spectral directions; ranking which units to grow under a count budget. Its splitting matrix is `E[Φ′(σ) ∇²_θ σ]` (Eqs. 3, 7). Algorithm 1 selects the most negative minimum eigenvalues and creates equally weighted offspring displaced in opposite minimum-eigenvector directions. Broad “training-derived spectral neuron allocation” is already established.

The experimental overlap is closer than the main-text random baseline alone suggests: the RBF comparison randomizes neuron and direction, whereas Appendix B.2 randomly chooses a prototype and uses **that prototype's splitting-matrix direction**. Holding a per-unit informed direction while varying allocation is therefore also established. Those passages do not establish the project's full same-parent, paired-stream, single-event experiment.

**Narrower difference:** the project's `M_j` is the mixed gate/contrast derivative structure, not generally the curvature of the original activated neuron used by R07. For `σ(θ,X)=φ(θ^T X)`, R07 includes `φ″(θ^T X)XX^T` and a derivative with respect to the activated output; the project uses the preactivation derivative directly. Appendix B.3 replaces ReLU with Softplus to calculate splitting curvature. R07 ranks signed minimum eigenvalues and perturbs duplicated neurons; the project uses absolute spectral magnitude to choose a contrast, RMS scales it, inserts an exactly zero gate, and ranks the fresh-Adam gate-gradient descent heuristic. R07's local optimality theorem does not transfer to this score.

**Unresolved scope / wording:** the general spectral-splitting framework is broad enough that differently parameterized specializations deserve attribution; different symbols alone are not differentiation. No exact combined RMS/Adam/centered-insertion rule was identified in the inspected sections. Safe wording: “We characterize a particular normalized gate-gradient allocation heuristic within established derivative-informed growth, using a controlled allocation comparison.” Do not claim first learned-versus-random placement with shared directions.

## R08 — Flat Channels to Infinity in Neural Loss Landscapes

**Inspected:** arXiv v2, §2/Eq. 2, pp. 3–4; §4/Eqs. 3–6, pp. 7–9; discussion, pp. 9–10; Appendix A.1, p. 15; B.1–B.2, p. 19; C.1–C.2/Eqs. 10–13, p. 26. [Primary paper](https://arxiv.org/pdf/2506.14951v2), [official HTML](https://arxiv.org/html/2506.14951v2).

**Confirmed overlap:** function-preserving neuron duplication and GLU-like functionality arising from differences between nearby neurons. Eq. 4 gives `cσ(w^T X)+(v^T X)σ′(w^T X)` as a channel limit. Softplus yields a sigmoid-gated affine term. Appendix A.1 also uses a loss-derivative/activation-curvature weighted `XX^T` matrix to characterize duplication stability.

**Narrower difference:** Eq. 4's bypass is `cσ(w^T X)` with the same `w` that controls the derivative gate. The project has an independently learned affine bypass and applies ReLU after combining it with the contrast. R08 obtains its limit from diverging, oppositely signed output weights and converging input weights; the project directly inserts finite gate/contrast coordinates. Appendix C.1's jump halves the channel separation and resumes gradient flow; it is not the project's exact preservation-plus-allocation step. Minimum-Hessian-eigenvector perturbations in B.1 are also different from its normalized moment/Adam rank.

**Unresolved scope / wording:** these sections establish the mechanistic connection, not an implementation of the project's score. No code comparison or general equivalence theorem was attempted. Safe wording: “The block belongs to an established gated family, with a related GLU limit in neuron-duplication geometry; the experiment concerns its explicit insertion and allocation behavior.”

## Permitted final framing and evidence limits

Suggested framing: “Within an established affine-bypass/GLU family and derivative-informed growth literature, we test one specified RMS-normalized moment/Adam allocation rule against uniformly random eligible units with matched per-unit directions and continuation. This bounded comparison did not identify the exact combined rule in R07, R08 or R12; it does not establish priority. The contribution sought is measured allocation/training behavior, including a negative result.”

R12's original PDF hash was rechecked and matches the supplied provenance. New R07/R08 public PDFs, extracted text, acquisition URLs, UTC recording date, sizes and SHA-256 hashes are indexed in `environment/h1-v2/prior-art-revisit/evidence-index.json`. Downloads required ordinary network permission after sandbox DNS resolution failed; retrieval then succeeded. No assets, weights, agreements or vendor implementation were acquired. Later versions, uninspected author code and other papers remain outside this bounded revisit. No experimental outcome was read or used to shape the comparison.
