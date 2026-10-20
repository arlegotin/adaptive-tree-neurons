# Continuing the native inventory after disk exhaustion

Parkinsons resumed its original full NODE state at update 4,000 and has published
later checkpoints. The old failed attempt remains intact, including its missing
terminal records. The original source, seed17, effective batch512/microbatch32,
4 GiB owned RSS guard and native stopping rule are unchanged.

The original completion auditor assumes one successful attempt. The separately
versioned `audit_resumed_node_completion.py` additionally handles this exact
two-attempt storage-interruption case. It requires externally hashed error,
checkpoint-validation and resume records; validates retained predecessor bytes;
then checks the named successful endpoint, native stopping and saved-prediction
replay. Its predecessor-only mode reads no active model or query. It records
the original resource-time lower bound separately and keeps the missing terminal
tail unavailable. It does not probe an old PID, fabricate a terminal receipt or
claim complete procedure cost.

Qualification replayed the already completed Concrete model and matched all
33 shared result fields, excluding operator identity and audit duration. The
actual closed predecessor check passed, and an active producer was rejected.
An initial `-P` sibling-import failure occurred before model access; an explicit
file import fixed it. All failures and retries are in the
[operator receipt](WP07-NODE-storage-continuation-operator-receipt.json).

A new hash-pinned serial controller waits for Parkinsons attempt001, audits it,
then runs CCPP and Bike one at a time. Each native launcher and each audit keeps
its resource admission and owned-process evidence. Any admission, fit or audit
failure stops the queue for explicit recovery. The controller never hard-kills
an outer process group as a substitute for cleaning up nested workers.
Byte-identical operator copies live in the retained control namespace, keeping
this queue isolated from later implementation edits.

This remains engineering continuation at the held c33 source. Completions still
number 117 of 120 at admission. No new validation score, HPO outcome, production
qualification or protected result follows from these operator checks.
