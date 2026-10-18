# What fixed-count placement can improve

For the current one-hidden-layer model, every fixed set of m grown units has the
same unrestricted function family as any other set of m units, when all model
weights may vary freely. This follows by permuting the hidden units, their base
coefficients and head coefficients, and moving each selected gate/contrast block
with its unit. Every term in the output sum is retained. Any two equal-size
subsets admit such a bijection; the inverse permutation gives the converse.

This is a mathematical scope clarification, not a measured explanation of the
historical placement result or a new growth principle. Capacity increases when m
increases. At a fixed m, informed placement can improve the finite learning path,
regularization or quality–cost behavior from the inherited parent. That is the
research question of the parent-conditioned selector and matched random control.

The relabeling moves the entire parent. It does not equate two different
allocations on one unchanged learned parent: those begin with different local
Jacobians and inherited optimizer states. They may learn very differently under
finite minibatch training and G selection. No global optimization/convergence
claim follows. Freezing parent weights, changing layers/depth or changing the
growth count requires a separate scope analysis.

A source-recorded numerical check exercised32fixed random float64 d16/h64/m16
fixtures with nonzero gates,257input rows each. It permuted base/head and selected
blocks together, checking predictions and all mapped parameter gradients against
the actual model. Both maximum absolute discrepancies were8.88e-16; all models
had1,697parameters. This is algebra/software evidence, zero training updates and
zero independent research teachers. Optimizer trajectories were not tested.

The [script](scripts/check_fixed_count_permutation.py) and
[receipt](fixed-count-permutation-receipt.json) retain exact seeds, permutations,
source identity, command and limitations. It changes no matrix, threshold,
eligibility rule or conditional admission. M1/M2/Gate A remain the scientific tests.
