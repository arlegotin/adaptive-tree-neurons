"""Declare exact practical semantics before search; this grants no execution.

The post-search global freeze must separately join actual qualified models,
whole-cost evidence and protected access controls. A declaration is no token.
"""
import copy
from pathlib import Path
import sys
import time

from treeneuron.config import canonical_hash,canonical_json
from treeneuron.analysis.practical_comparisons_v2 import MARGIN
from treeneuron.analysis.practical_v2 import OBSERVATIONAL,FAMILIES,CHAMPION
from .deployment_worker_v2 import _source
from .ledger import write_new_json
from .method_worker import _ROOT,_sha
from .practical_support_lock_v2 import load_practical_support_lock
from .primary_selection_v2 import _primary,_ref
from .qualify import qualification_locks_sha256
from .seeds import seed64


_FALSE=('execution_authorized','native_workload_qualified','protected_access_authorized',
    'secondary_release_authorized','global_postsearch_freeze_authenticated',
    'whole_training_procedure_cost_authenticated','T_accessed')


PROTECTED_AUDIT_POLICY=dict(actor='auditor',targets='Same frozen F/T bytes after witnessed metric completion',
    purpose='Independent raw metric verification only',model_updates=False,configuration_selection=False,
    access_records='Immutable role access intent and closed outcome; never reset once-only test study')


def _policy(study_id):
    return dict(
        method=dict(P00_placement='legacy_cheap_adam',P01_placement='informed_direction_random',
            matched_factors='Same admitted contrasts/count/parent/treatment/architecture and HPO supports; only placement differs',
            new_joint_pruning_residual_candidate_admitted=False,
            practical_scope='Unchanged-reference whole-policy benchmark; no placement-only inference from independently V-tuned policies'),
        mechanism_confirmation=dict(admitted=False,calibration_admitted=False,
            reason='M3 default joint closed negative; B0 residual screen negative. No C0/C1/C2 or synthetic protected root authorized'),
        program=dict(family_ids=list(FAMILIES),primary_jobs=168,secondary_jobs=672,search_repetitions=2,
            primary_fold=0,bike_primary_fold=4,primary_models_before_secondary=168,
            secondary_barrier='Same global semantic procedure/source and all actual qualified primary selected models frozen before secondary F/G/V',
            secondary_feedback_to_primary_or_global_policy=False),
        information=dict(transforms_fit_on='F',fit_worker_labels=['F','G'],fit_worker_query_features=['V'],
            V_labels_reader='declared_coordinator',checkpoint_selection='strict_best_G_complete_state',
            final_model='retained_F_fit_G_selected',refit_F_G_V=False,
            retrieval_labels='F_only',foundation_context='declared_F_only',
            protected='Once-only frozen evaluator; no model or configuration selection from T',
            protected_audit=copy.deepcopy(PROTECTED_AUDIT_POLICY)),
        search=dict(official_anchors_first=True,max_attempts=32,P13_unique_configurations=29,
            training_wall_time_limit_ns=None,quality_view='native_completion',
            descriptive_prefixes_ns=[600_000_000_000,1800_000_000_000,7200_000_000_000],
            failures_consume_attempts=True,training_seeds_are_candidates=False,
            recovery='Same-seed/state single declared recovery with all attempts/cost retained; no numerical seed replacement',
            native_qualified_minimum_required=True,unavailable_is_predictive_loss=False),
        selection=dict(metric='V_group_or_day_weighted_F_variance_normalized_MSE',
            tie_relative_tolerance=1e-12,trial_tie='earliest_eligible_completed_trial',
            family_tie='lowest_fixed_family_ID',champion_cost='All12P02-P13 searches, not one family budget'),
        analysis=dict(observational_tasks=list(OBSERVATIONAL),energy='separate_simulation_descriptive_ranges_only',
            repetitions='Mean paired R2 risks before log ratio',task_weight='equal_five_log_risk_ratios',
            within_task_weight='Equal held_out_group_or_day then equal rows within unit',
            useful_log_risk_margin=MARGIN,primary_comparator='V_selected_P02_P13_portfolio',
            champion_family_ids=list(CHAMPION),champion_selection='V_only_separately_per_dataset_repeat',
            primary_interval='two_sided_95_percentile_paired_group_bootstrap',bootstrap_iterations=20000,
            bootstrap_seed=seed64(2026101802,study_id,'practical','protected-analysis','conditional-bootstrap'),
            pairing='Same resampled units for every family and both repeats; datasets independently resampled',
            bike_primary_block_days=7,bike_sensitivity_block_days=[3,14],
            bike_law='overlapping_noncircular_calendar_blocks_uniform_starts_final_truncation',
            group_sensitivity='leave_one_group_out',
            individual_family_ids=list(FAMILIES[1:]),individual_alpha=.025,
            individual_marginal='centered_bootstrap_error_location_plus_one_upper_tail_ties_against_rejection',
            individual_correction='strict_Holm_over_all13',
            individual_regions='Guilbaud_Choice2_discrete_marginal_inversion_closed_marginal_open_rejected_margin',
            insufficient_bootstrap_resolution='whole_family_unavailable',
            primary_conditions=['lower_strictly_above_margin','positive_aggregate_in_each_repeat'],
            consistency_conditions=['at_least_four_of_five_task_effects_nonnegative','no_task_relative_risk_degradation_above_5_percent'],
            missing_predictions='incomplete_primary_claim; observed_pairs_descriptive_only; no favorable or infinite imputation',
            assumptions=['fixed_fitted_models','group_exchangeability_or_declared_Bike_local_dependence',
                'Parkinsons_few_clusters_not_repaired_by_many_resamples','CCPP_exchangeability_unverified',
                'marginal_bootstrap_location_approximation_not_exact_randomization'],
            uncertainty_scope='Conditional held_out_sampling on five named tasks; no retraining or dataset-population inference',
            hypothesis_prefixes='10/30/120minutes descriptive; native-quality endpoint separate'),
        resources=dict(worker_concurrency=1,live_memory_and_disk_admission=True,
            no_fixed_20GiB_free_RAM_requirement=True,root_RSS_ceiling_is_not_aggregate_memory=True,
            standalone='Charge prerequisites to each procedure once; all failures/probes/HPO/preprocessing/prediction retained',
            physical='Separate containing cohort clock; never add nested/shared timers twice',
            caller_endpoint='Observed caller completion before final receipt publication/return; physical study clock covers excluded bookkeeping',
            missing_interval='unavailable_not_zero',
            portfolio='12_family_quality_comparison_not_equal_total_cost',
            deployment=dict(primary='CPU_float32_one_declared_thread_batch1_with_native_preprocessing',
                secondary_batch=1024,paired_blocks_minimum=30,
                primary_models=168,paired_comparisons=156,cold_measurement_jobs=9360,
                comparisons='Primary6tasks_R2_P00_versus_each_P01_P13; Energy_separate',
                query='F_only_original_order_cycle_to1024_float32; workloadIDs0_to1023; no_target_decoding',
                warmup_calls=3,resident_calls_per_batch=32,plan_frozen_before_test_labels=True,
                order_seed='seed64(2026101802,protocolSHA,deployment,dataset,str(repeat),baselinefamily)',
                precision_qualification='Nativefloat64_quality_and_float32_input_timing_not_automatically_equivalent',
                savings_gate='Lower95bound_savings_fraction_strictly_above_.20_and_positive_paired_savings_interval',
                quality_gate='Upper_risk_ratio_bound_strictly_below_1.02',
                cold_imports='method_specific_separate_from_common_harness')))


def _closures():
    paths=dict(M3=_ROOT/'audit/research-v2/M3/gate-B.json',B0=_ROOT/'docs/research/v2/WP07-B0-completion-receipt.json')
    refs={k:dict(path=str(p),sha256=_sha(p)) for k,p in paths.items()}
    m3=_ref(refs['M3'],{},decode=True);b0=_ref(refs['B0'],{},decode=True)
    if (m3.get('status')!='closed_negative_development' or m3.get('confirmation_admitted') is not False
            or m3.get('T_accessed') is not False or b0.get('status')!='completed_development_screen_negative'
            or b0.get('decision',{}).get('residual_branch_admitted') is not False or b0.get('T_accessed') is not False):
        raise ValueError('Practical unchanged-reference declaration requires the retained negative branch dispositions')
    return refs


def _inputs(plan,expected,support):
    value,jobs=_primary(plan,expected);law=load_practical_support_lock(support)
    if (value.get('budget_policy')!='native_quality' or value['source_sha256']!=law['source_sha256']
            or value['runtime_sha256']!=law['core_manifest_sha256'] or value['support_lock_sha256']!=support['sha256']
            or value['lock_sha256']!=qualification_locks_sha256()):
        raise ValueError('Exact native-quality plan/source/runtime/support/adapter pins required')
    return value,_closures()


def _content(plan,expected,support,created):
    if type(created) is not int or created<=0:raise ValueError('Exact declaration publication time required')
    value,closures=_inputs(plan,expected,support)
    body=dict(schema='research/v2-presearch-practical-procedure',created_utc_ns=created,
        study_id=value['study_id'],plan_hash=expected,source_sha256=value['source_sha256'],
        runtime_sha256=value['runtime_sha256'],adapter_lock_sha256=value['lock_sha256'],
        support_reference=dict(support),data_manifest_hashes=value['data_manifest_hashes'],
        ordered_job_inventory_sha256=canonical_hash(dict(jobs=value['jobs'])),
        development_closures=closures,policy=_policy(value['study_id']),**{k:False for k in _FALSE},
        scope='Pre-search complete semantic declaration only. Actual native qualification/resources/execution, qualified primary models/whole costs, post-search global freeze and protected evaluator remain separate required gates.')
    if _inputs(plan,expected,support)!=(value,closures) or _source()!=value['source_sha256']:
        raise ValueError('Practical procedure inputs or source changed during declaration')
    body['declaration_hash']=canonical_hash(body)
    return body


def seal_practical_procedure(*,practical_plan,expected_practical_plan_hash,support_reference,output):
    output=Path(output);protected=[Path(sys.prefix).resolve(),Path(support_reference['path']),
        *((_ROOT/n).resolve() for n in ('src','tests','configs','locks','environment','.git','.agents','.codex','.aws'))]
    if (not output.is_absolute() or output.resolve()!=output or output.is_symlink() or output.exists()
            or output.suffix!='.json' or any(output.is_relative_to(p) for p in protected)):
        raise ValueError('Fresh canonical procedure JSON must preserve executable and support inputs')
    value=_content(practical_plan,expected_practical_plan_hash,support_reference,time.time_ns())
    write_new_json(output,value)
    return dict(path=str(output),sha256=_sha(output))


def load_practical_procedure(reference,*,practical_plan,expected_practical_plan_hash,support_reference):
    saved=_ref(reference,{},decode=True)
    if type(saved) is not dict:raise ValueError('Exact declared practical procedure required')
    rebuilt=_content(practical_plan,expected_practical_plan_hash,support_reference,saved.get('created_utc_ns'))
    if canonical_json(saved)!=canonical_json(rebuilt) or _sha(Path(reference['path']))!=reference['sha256']:
        raise ValueError('Practical procedure semantics/source/plan or bytes changed')
    return rebuilt
