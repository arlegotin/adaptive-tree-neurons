# Search sampler replay preparation

ReplayTPEV2 uses the existing pinned Optuna4.5.0 loader without changing legacy
search behavior. It supports frozen float, stepped integer and categorical
coordinates. Invalid/log-incompatible bounds, silently rounded integer endpoints,
ambiguous duplicate categories and unknown fields fail before sampling. Parameter
keys have canonical ordering; category order stays as declared.

Every ask/tell is retained, including failures. None marks a paid failed choice;
it consumes the same suggestion slot as a completed result. A pending choice
cannot be asked again, but it can be published with the prior history and restored
exactly after interruption. The restored worker first reproduces every past choice
and objective, then the pending choice, without an extra RNG draw. Public values
are copies. States are new immutable JSON files, with file and payload commitments,
caller execution identity and dependency identity checked on load. No pickle is
used. Numerically equal values with different JSON types cannot replace an archived
parameter during replay.

Published model-specific anchors remain outer-controller attempts and can win V,
but sit outside this generic TPE surrogate. Some required anchors have nonuniform
architectures or coordinates outside the uniform support, so they cannot be
represented faithfully as generic TPE observations. This policy applies to every
TPE family. The outer controller must deduct every paid anchor/failure/retry from
its32attempt limit and charge all startup/preparation. P06 uses its native random
schedule and P13 its exhaustive29configuration support. The new policy is a
pre-outcome implementation choice; legacy ReplayTPE remains unchanged.

The private search overlay was cloned without changing the original or core
installation. All1,923original regular files/38,747,241logical bytes matched;208
inherited generated pyc files were omitted only from the new copy. The loader
checks the historical requirements/environment lock and versions. Sampler states
also bind the Optuna module origin/digest, NumPy and Python version. This is not
an independent audit of every installed core/native byte. Whole-runtime/worker
qualification and a validated caller execution identity remain prerequisites to
real searches.

Seventeen missing-interface REDs became GREEN. A semantic corruption regression
then exposed Python's equality between1.0andTrue; canonical typed replay corrected
it. The final affected new/legacy search and analysis suite passed144tests in23.64s.
The tests exercise the actual TPE posterior after more than10completed artificial
observations, pending replay, paid failures, type/support/file/identity tampering,
immutable publication and suggestion limits.

scripts/check_tpe_replay.py retains an artificial20choice sequence, including two
failed choices, a pending state and its exact next-suggestion replay. No model,
real validation target or protected outcome is involved. The adapter worker,
full search clock/deadline, outer attempt queue/cost lineage, frozen supports,
anchor eligibility, primary/secondary freeze release and practical production
searches remain pending. Sampler state replay alone does not establish those
contracts or authorize protected access.
