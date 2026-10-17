# Causal numerical probe regression

`causal_seed42_probe.pt` contains only an actual toy model's topology and float32
state dictionary, not protected labels or a scientific result. SHA256:
`34e2d6653f14764df60ada5cdd5f4730b5c6123268d1174470df209c191b92a1`.

It was reconstructed from the existing `tests/test_protocol.py::prepared` toy
recipe using `confirmation_seeds('toy', 'negative-fixture', 42, 6, 2)`, repetition
1, after unrelated pytest temp retention removed the original failed run. The
original failed log and isolated reconstruction remain in
`/tmp/task13-seed42-diagnostic/` for the implementation evidence handoff. The
reconstruction reproduced the exact original exception before any source change.
Original package SHA256:
`a6495f78e8ea46785559ae5e277262834124a053d0521ade8b304c595e5ceeb5`.

The failing control is B6/greedy, selective d=16, h=64, m=16, probe batch=8,
probe seed=1602041. `layer.gate.weight[0,11]` starts at zero. Float32 actual and
independent leaf-reference gradients were -1.400608429236172e-8 and
-1.3866156223230064e-8. First AdamW parameters were 0.0005834389012306929 and
0.0005809965659864247, difference 2.442335244268179e-6. Both forward and gradient
maximum discrepancies were 1.1920928955078125e-7 and passed unchanged tolerances.

For the identical float32 operands promoted to float64, actual/reference
gradients were -1.3995804324633955e-8 and -1.3995804324399873e-8; updated values
were 0.0005832604790107387 and 0.0005832604790066732, difference
4.065432886168896e-15. Forward and gradient maximum differences were
1.1102230246251565e-16. This isolates first-Adam conditioning near epsilon,
without evidence of a wrong model equation.

For first-step zero moments, bias correction yields
`p1=p0*(1-lr*weight_decay)-lr*g/(abs(g)+eps)`. The numerical check now verifies
both updates against this formula using their own gradients, evaluated in
float64, with gamma_20 float32 rounding error on the sum of term magnitudes.
Cross-reference allowance follows the exact componentwise difference of
`g/(abs(g)+eps)`: same sign gives
`eps*abs(g-h)/((abs(g)+eps)*(abs(h)+eps))`; opposite signs give
`abs(g)/(abs(g)+eps)+abs(h)/(abs(h)+eps)`. It includes both rounding bounds.
Forward/gradient tolerances, recorded optimizer discrepancy, overall 1e-4 receipt
limit and exact serialization/next-update checks are unchanged. An injected
incorrect common update is independently rejected by the formula check.
