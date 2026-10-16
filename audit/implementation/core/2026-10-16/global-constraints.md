
- Work in the current branch. Do not create a worktree or switch branches unless a later user instruction changes this constraint.
- Training uses float32 eager PyTorch. Mathematical checks and score construction use CPU float64.
- Primary neural execution uses no reduced precision and no implicit MPS fallback.
- No dropout, BatchNorm, learned normalization, embeddings, or ensemble in the causal experiment.
- Initial synthetic coordinate: `d=16,h=64,m_requested=16,τ=1,epsilon=0.05`.
- IDs are a 1-D CPU int64 tensor; contrasts have shape `[m,d+1]`, the base parameter dtype/device, explicit bias in the last coordinate, and finite values.
- T labels are read only by the protected evaluator after protocol validation.
- No primary refit on F+G+V.
- Zero-growth teachers remain in all results.
- RSS, MPS allocated/driver memory overlap in unified memory; record separately, never sum as disjoint usage.
- No `frozen.yaml`, confirmatory result or completed-WP record is created by this documentation task.

All checkboxes start unchecked because these are future implementation steps. Update `DECISIONS.md` at every WP gate with evidence paths, hashes, decision, blockers and next command. Commit only task-owned files; preserve unrelated work. Tests in this plan are requirements to implement, not claimed existing fixtures or executed results.
