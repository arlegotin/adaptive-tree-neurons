# Provisional prior-art delta

Recorded 16 October 2026. Soft gating, affine experts, gated units, and structural growth are established families; no new primitive is claimed. The proposed empirical target is training-derived placement of conditional affine specialists in learned units at fixed added capacity. Any differentiation of its normalized moment/Adam score and centered insertion remains provisional until a focused source comparison and experiments are complete.

R12 access has improved since planning. The official [PMLR entry](https://proceedings.mlr.press/v306/yan26o.html) links an [official PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/yan26o/yan26o.pdf). Ordinary direct retrieval succeeded on 2026-10-16: 2,756,816 bytes, SHA-256 `f1628581fc83f79a4c32d4013299e56f2cf3693eb50821cb355b8c2c5926cd65`. Local untracked artifact: `environment/sources/R12-yan26o.pdf`; text extracted with `pdftotext -layout`. OpenReview forum and supplied PDF routes still present browser-verification challenges; no bypass attempted.

Inspected scope: abstract/introduction, Section 3.1/3.2 setup and Appendix A tree parsing algorithm. These describe an ensemble initialized from decision trees, feature-to-node and topology matrices, differentiable routing/aggregation, and a DFS compilation procedure. This limited inspection does not establish absence or presence of the project's exact allocation score or insertion rule elsewhere. A bounded follow-up inspection by the parent agent covered Sections 3.1–3.3, Equations (1), (3)–(6): ArborCell starts from a pretrained tree ensemble, applies tanh to scaled feature-threshold differences, forms topology-weighted affinities, uses softmax leaf weights, and aggregates scalar leaf values. This differs from this project’s affine-contrast preactivation and centered insertion. Searches for growth/moment/eigenvector and these inspected sections located no implementation of this project’s training-moment allocation rule; that bounded absence is not proof of novelty. A complete comparison remains required before broad novelty claims. Newly retrieved full text resolves a retrieval obstacle, not the overlap decision.

The PDF states author copyright on page 1; no author code license or separate data/weight rights were established. No vendor code was copied, dataset or weight downloaded, or agreement accepted. Package wheel terms, retained license-file hashes, primary documentation retrieval and failed source routes are in `provenance/sources.json`. Other R01–R39 items remain attributed planning references until their consequential claims are inspected in the relevant work package. WP09 must revisit the complete comparison. Exact prior implementation would change the project framing to replication/refinement.

## Focused revisit — 17 October 2026

The bounded comparison of R07, R08 and the acquired R12 full text is now complete;
the [comparison and provenance](../docs/research/2026-10-17-prior-art-revisit.md)
record its exact scope. R07 Appendix B.2 already compares random neuron selection
while keeping each neuron's informed splitting direction. The general idea of
that allocation control is therefore prior art. R08 further establishes related
duplication geometry and a gated-linear limit.

The inspected R12 architecture, training, ablations and complete parsing appendix
use pretrained tree topology, scalar leaves and topology/softmax routing. The
specific centered affine-contrast insertion, absolute residual-moment direction,
RMS scaling and fresh-Adam ranking were not located as one combined rule in these
three sources. This bounded distinction is not proof of priority or absence
elsewhere. R12's former access obstacle is resolved. The contribution sought
remains empirical characterization of a specified allocation/training heuristic,
including costs and negative outcomes; no model-family novelty is claimed.
