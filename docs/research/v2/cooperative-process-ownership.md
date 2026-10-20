# Cooperative search-process ownership

The new `cooperative_process_v2` component measures a cold root process and
retains its startup, logs, resource samples, ownership handshake and closure.
It is ready for composition with the practical search worker; it does not
execute or qualify a production search by itself.

A trusted worker enters `cooperative_worker` before model imports or nested
process creation. It publishes its identity and waits for a matching grant.
The parent validates the exact command, identity, nonce, owner and resource
admission before granting execution. The handshake imports only the standard
library. A rejected or missing handshake never receives a grant.

Before grant publication begins, the parent may stop its own root directly:
the trusted worker contract prohibits descendants at that point. Once grant
publication begins, the worker may already have created a separate training
session, even if the publisher raises an error before recording success.
Cleanup therefore sends at most one SIGINT to the authorized root and waits
for its nested supervisors to finish. There is no automatic hard kill of the
outer coordinator. A stop-record publication failure still requests cleanup;
missing or partial evidence remains explicit.

The root clock covers supervisor entry through interpreter startup, worker
execution, root reap, log closure and file hashing. Its own final receipt
publication and caller overhead are excluded. Model-specific nested work,
search journal closure, paid retry lineage and source/data qualification still
need their own authenticated records. The reader never spawns a process or
probes a saved PID. It grants no model, native-recipe, search-resume, protected
access or full training-procedure cost authority.

RSS observations cover the root only. A threshold requests cooperative cleanup;
it is neither a strict peak guarantee nor aggregate descendant memory. Existing
native supervisors keep their own hard worker guards. The default outer clock
has no wall cap. A trusted worker that fails to cooperate can take arbitrarily
long to exit; the component makes no bounded-cleanup promise.

The affected three-module suite passed **71 checks in 5.52 seconds**. Actual tiny
workers exercised a separately sessioned nested child, orderly interruption,
admission denial, incorrect ready identity, handshake timeout, RSS pause,
nonzero exit and missing worker closure. Injected publication failures before
and after the grant and stop records verified conservative cleanup across the
disk-full boundary. Raw-counter and ownership tampering rejected. These checks
fit no predictive models and inspect no labels.

The [receipt](WP08-cooperative-process-ownership-receipt.json) pins source, tests,
commands and initial failures. The [capture script](scripts/capture_cooperative_process_fixture.py)
reopens thirteen actual closed fixture episodes while process creation is
forbidden. Whole practical worker execution, outer-cost reconciliation with
the search journal, the all-primary-model barrier and final protocol remain
unfinished. Existing held-source native benchmarks continue unchanged.
