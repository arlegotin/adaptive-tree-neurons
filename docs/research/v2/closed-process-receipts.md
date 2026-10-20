# Closed operational process evidence

`inspect_closed_process(reference, *, expected_argv, expected_cwd,
expected_environment_overrides, expected_identity)` reopens a closed managed
process directory. It checks exact invocation, start/ownership records, closed
child or pre-spawn failure, file commitments, deadline counters and raw RSS
observations. It probes and signals no live PID; a historical PID may have been
reused.

Success, nonzero exit, timeout, RSS/log stops and input failure retain their
operational meanings. A legitimate observed RSS excess is accepted only as the
corresponding resource failure, never as a successful workload. Missing RSS
observations remain unavailable rather than becoming zero memory. The stricter
successful-native-workload reader remains unchanged.

All 67 affected checks passed at the recorded source. Five actual tiny process
outcomes cover normal/nonzero exit, an owned RSS stop, timeout and an expected
nonexecutable-file start failure. Controlled changed argv, unreaped status,
authority, counters, deadlines and termination flags reject. Three malformed
field types first exposed unstructured exceptions; they now reject explicitly.
The [receipt](WP08-closed-process-receipt.json) retains commands, failure logs
and independently reopened observations.

This is trusted managed-receipt consistency, not independent absolute-clock
authentication, method quality, aggregate memory or whole external job cost.
All model/native/protected authority flags remain false. It supplies the paid
failure-evidence component for the remaining qualified-search audit. Failures
never become arbitrary predictive risks.

Whole-search ownership still needs care: current nested native supervisors
start their own process groups. Killing only an outer coordinator group cannot
be assumed to stop those workers. No new outer hard-kill controller is supplied
by this reader. Exact interruption/resume disposition and full job costs remain
separate required work.
