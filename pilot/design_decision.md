# Pilot design decision

Purpose: engineering. Branch: **blocked**. Freeze eligible: false.

Completed independent teachers: 1; repetitions: 2. K_calc: None; K: 1; n_T: None; planning effect: None.

- Missing measured inputs: available_memory_bytes, bootstrap_seed, bytes_per_evaluation_row, endpoint_tolerance, estimation_rows, evidence_hashes, inference_supported, max_seconds, max_storage_bytes, peak_memory_bytes, power_iterations, power_mc_tolerance, required_controls, retries, seconds_per_evaluation_row, seconds_per_repetition, setup_seconds_per_teacher, storage_bytes_per_repetition, storage_bytes_per_teacher, policies.parent_updates, policies.continuation_updates, policies.patience, policies.validation_interval, policies.logical_batch, policies.physical_batch, policies.bootstrap_iterations, policies.probe_updates, policies.hpo_seconds, policies.hpo_trials, policies.timing_blocks, policies.timing_relative_precision
- Required baseline/control coverage missing
- Engineering fixtures are not genuine pilot/design evidence
- Fresh independent confirmation teachers and observations required
- Pilot-based count floor does not guarantee 95% coverage

Raw ledger SHA-256: ef6805e9c079d3a54cf44f267a1890160b848653659fdd3bcb38528dd03a3a5a

Genuine WP05 execution waits for Tasks 9/10/13 implementation, review and committed source.
