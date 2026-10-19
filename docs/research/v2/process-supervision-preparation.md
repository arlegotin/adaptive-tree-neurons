# Trial process supervision preparation

`research/process_supervision_v2.py` runs one supplied command without a shell in
a new process group. It retains argv/cwd/environment overrides, caller identity,
startup receipt, stdout/stderr, root-worker RSS samples, exit/termination status,
elapsed time and exact-file hashes. It does not retry, change seeds, load labels,
score predictions or make a model eligible. Zero exit status is process execution
evidence; native checkpoint/quality qualification remains separate.

The supervisor enforces a supplied dedicated-worker deadline by terminating its
own newly created process group, allowing bounded TERM grace, then using KILL if
needed and reaping its child. Spawn/import time and RSS/log checks count against
the deadline. Polling/probe/termination latency can overshoot the deadline; the
receipt records that overshoot and the outer selector still excludes late results.
RSS exceeding the supplied limit or excessive stdout/stderr stops the worker.
Unknown RSS observation capability fails before starting a fit; losing the probe
after startup stops the child. A process leaving live members of its group is an
unclean completion. Only groups created by this invocation are signaled.

Memory scope is deliberately limited: samples observe root-worker RSS. They do
not measure aggregate descendants or unobserved peaks. The strict peak-memory
qualification flag remains false. Native methods using extra processes need
separate measurement/qualification. Current whole-process billing also needs the
external supervisor/publication tail: the internal receipt ends its clock after
log closure/file hashing, before its own final JSON publication. A killed
supervisor cannot be recovered from this receipt alone. Integration with the outer
queue, native role-safe workers, live resource admission and process-death receipts
remains pending. No practical search is authorized by these checks.

Verification began with22missing-module REDs. The sandbox then blocked `/bin/ps`,
including a probe of the test's own PID: seven process tests failed closed before
child startup, while15other checks passed. The same tests under automatic
escalation passed22/22 in1.09s. Escalation permits only our short-lived toy
process/PID checks; it does not alter source or bypass native resource guards.

A further regression caught a fast-exiting child evading the log-volume guard;
moving that check before the exit branch fixed it. A combined run then retained
one process error,158passes and nine legacy setup errors. The legacy errors were
the unchanged backend refusing host memory-pressure level2. The process error was
macOS reporting EPERM for a zombie-only group before child reaping. A direct
own-child diagnostic observed group existence before TERM, EPERM after TERM and
ESRCH after wait. A failing regression now requires reaping the owned child and
rechecking the group; EPERM is never interpreted as absence. The final scoped
new search/analysis/process suite passed148tests in2.32s. The nine guarded legacy
setups remain pending a level1 resource window; no resource guard was weakened.

The raw archive includes395regular test files/968,207bytes from the native,
fast-exit and combined attempts, with57pytest aliases mapped rather than copied.
The initial sandbox attempt's temporary receipts expired through pytest cleanup
before archival; its failure log and captured diagnostic remain. The scoped final
run's raw inventory is added separately. These are software fixtures, with no
research teachers, native predictive fits, transfer experiments or T access.
