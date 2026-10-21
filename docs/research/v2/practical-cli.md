# Practical command boundary

The research CLI now connects the implemented practical operations. It does not
turn a control file, metadata plan, Boolean or hash into native qualification.
Actual public loaders run before native execution and protected evaluation.
The complete qualified positive remains unrun. Production readiness and the
fresh whole-branch review still precede final-source qualification and HPO.

Existing M0–M3, data preparation, engineering qualification and development
report commands retain their previous interfaces. Practical controls use this
separate exact schema:

```json
{
  "schema": "research/v2-practical-command",
  "source_sha256": "<actual current core-source SHA256>",
  "operation": "support_lock",
  "arguments": {}
}
```

This is a schema example, not a qualified configuration. Unknown fields,
duplicate JSON keys, changed source, unknown operations and caller-provided
output arguments fail. The output comes only from `--output`. No arbitrary
module, function, callback or label path can be selected by the dispatcher.
Each public operation retains its own stricter reference, data, source and
publication rules.

For example, from the checkout root, the following creates a fresh metadata
control and runs the real immutable support-lock publisher. It fits no model
and grants no execution or protected access:

```sh
ATN_PY="$PWD/.venv-research-v2/bin/python"
"$ATN_PY" -P -B - <<'PY'
from pathlib import Path
from treeneuron.research.deployment_worker_v2 import _source
from treeneuron.research.ledger import write_new_json
write_new_json(Path('artifacts/research-v2/commands/support.json'), {
    'schema': 'research/v2-practical-command',
    'source_sha256': _source(), 'operation': 'support_lock', 'arguments': {}})
PY
"$ATN_PY" -P -B -m treeneuron.research.cli plan \
  --config artifacts/research-v2/commands/support.json \
  --output artifacts/research-v2/support-command-plan.json
"$ATN_PY" -P -B -m treeneuron.research.cli freeze \
  --config artifacts/research-v2/commands/support.json \
  --output artifacts/research-v2/practical-supports.json
```

Set the qualified one-thread environment documented in
[reproduce.md](reproduce.md) for numerical work. Control files for later stages
contain the actual immutable references returned by their prerequisites.
`plan` validates their metadata argument contract and writes an explicitly
unqualified command plan; it does not open role arrays or run those stages.

| Operation | Existing implementation used by `run` |
| --- | --- |
| `support_lock` | `practical_support_lock_v2.seal_practical_support_lock` |
| `practical_plan` | `workload.expand_practical`, all840 native-quality jobs |
| `procedure` | `practical_procedure_v2.seal_practical_procedure` |
| `managed_preparation` | `preparation_execution_v2.run_managed_preparation` |
| `qualification_registry` | `native_qualification_registry_v2.seal_native_qualification_registry` |
| `resource_forecast` | `native_resource_forecast_v2.forecast_native_practical_resources` |
| `primary_inventory` | `primary_selection_v2.seal_primary_selection_inventory` |
| `primary_models` | `qualified_primary_models_v2.seal_qualified_primary_models` |
| `global_barrier` | `practical_global_freeze_v2.seal_global_primary_barrier` |
| `practical_models` | `qualified_practical_models_v2.seal_qualified_practical_models` |
| `preT_costs` | `practical_preT_costs_v2.seal_practical_preT_costs` |
| `T_protocol` | `practical_T_protocol_v2.seal_practical_T_protocol` |
| `primary_program` / `secondary_program` | Existing public programme through `program_process_v2.run_program_process` |

Arguments have the named public function's exact signature, with output omitted.
For a Python `practical_plan` or `plan` parameter, provide the corresponding
`practical_plan_reference` or `plan_reference` file commitment; the dispatcher
materializes that immutable JSON only after the actual native gate when one
is required. `practical_plan` takes six `data_manifest_references`, an actual
`support_reference`, the qualification `lock_sha256`, and optional `study_id`.
It always retains the native-quality policy and840 jobs. It cannot introduce
a training wall-time cutoff or a new research selector.

`freeze --config ... --output ...` accepts only publication operations, not
training or resource forecasting. `run --resume` is restricted to the primary
and secondary programme operations. Use the same programme `directory` and
immutable inputs with a fresh observer output for each invocation. Existing
programme journals enforce identity and recovery; the CLI does not reset them.
Consumed work and unsuccessful trials stay recorded.

The owned programme starts serially after fresh RAM/disk admission. The default
root RSS threshold is4GiB, with a5.5GiB host-memory proxy and512MiB disk proxy;
the model, F rows and effective batch remain unchanged. These proxies are
operational guards, not allocation guarantees. Inner per-job measured admission
still applies. Training has no two-hour cutoff. Worker completion, safe pause
and blocked execution return exit codes0,3 and4 respectively; invalid inputs
return2.

After the actual final protocol exists:

```sh
"$ATN_PY" -P -B -m treeneuron.research.cli confirm \
  --protocol protocol/research-v2/frozen.json \
  --output artifacts/research-v2/confirmation
"$ATN_PY" -P -B -m treeneuron.research.cli report \
  --study artifacts/research-v2/confirmation \
  --output reports/research-v2-practical.md
```

These commands are implemented; the required genuine production inputs do not
yet exist. Confirmation reads protocol JSON commitments to invoke the full
actual loader before creating output. Its seven operations and paths are fixed
by the existing programme. Resume uses the same study/programme and the next
managed observer slot. A complete retained cost seal is revalidated without
starting another observer or another target read. Incomplete or corrupt evidence
is not silently restarted. The report reads the completed immutable publication
references and invokes the actual raw/metric/native/cost/readiness report gate.
It does not fit, predict or decode targets. A blocked execution cannot fall
through to a successful report. Explicit development reports for completed,
estimation, retired or blocked studies retain the existing
`--development-manifest-sha256` contract.

Integration also exposed two preproduction wiring defects: confirmation worker
argument shadowing after a successful return, and a JSON-only publication guard
used for directories and Markdown. Regression checks now exercise successful
and partial **inert metadata** returns, typed outputs and protected-root
rejection. The native gates were not replaced with successful mocks. No actual
protected benchmark has run through those paths.

The final focused command passed **133 checks in92.25seconds** across eight
affected modules and the legacy CLI planning case. The
[receipt](WP10-practical-cli-receipt.json) retains every RED run and recovery.
An earlier broader command reached144 passes and seven legacy numerical-producer
setup errors from live memory pressure level2. Those seven checks remain
incomplete; their guard was preserved. The first sandbox run also retained
owned-process inspection denials and a live Torch interop mismatch, recovered
with scoped owned-PID permission and declared thread initialization. These
counts overlap and do not constitute a whole-suite pass.

The capture executes real immutable metadata publishers on artificial six-task
manifests:840 metadata jobs,168 primary/672 secondary, no role arrays. It retains
overwrite rejection and actual unqualified confirmation denial before any study
output. It fits/predicts no model and reads no benchmark labels. All scientific
results and the held native worker's source/environment remain unchanged.
