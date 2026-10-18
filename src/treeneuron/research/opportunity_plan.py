"""Finite M2 nominal slots, paired prerequisites and bounded contingent work."""
from collections import Counter
from itertools import combinations
import hashlib
from pathlib import Path

from treeneuron.config import resolve_experiment

from .seeds import seed64, seed32

RECIPES = ('legacy_005', 'constant_001', 'cosine_005', 'wd_1e4', 'wd_1e3')
RECIPE_OVERRIDES = dict(legacy_005={},constant_001={'learning_rate':.001},
    cosine_005={'scheduler':'cosine','min_learning_rate':0.},
    wd_1e4={'weight_decay':1e-4},wd_1e3={'weight_decay':1e-3})
ALPHAS = (0., 1e-6, 1e-4, 1e-2, 1., 100.)
WARM_ROLES = ('keep', 'top', 'random', 'warm_lower', 'warm_upper')


def opportunity_decision_contract(config):
    contrasts = []
    for snapshot in ('actual_250','actual_1000','late_G_best'):
        for treatment in ('continued_stream','new_stream_retained_adam'):
            for role in ('top','random','warm_lower','warm_upper'):
                recipes = list(RECIPES) if (snapshot,treatment) == ('late_G_best','continued_stream') else []
                contrasts.append(dict(contrast_id=f'{snapshot}/{treatment}/{role}',
                    snapshot=snapshot,treatment=treatment,role=role,ordinary_recipes=recipes))
    return dict(schema='research/v2-opportunity-decision',contrasts=contrasts,
        minimum_relative_reduction=.02,minimum_qualifying_teachers=2,
        bootstrap_iterations=20000,bootstrap_seed=seed64(config['seed_root'],config['study_id'],
            'M2-extension-interval/bootstrap'),tie_relative_tolerance=1e-12,
        acquisition='original_if_available_else_required_fixed_fallback',
        aggregation='mean_R2_before_reference_choice_then_equal_teacher_relative_benefit',
        interval='fixed_best_contrast_whole_teacher_percentile; linear quantile; no reselection',
        inclusive_boundaries='risk inequality for counts; same bootstrap indices on risk differences centered at zero and margin',
        extension='maximum_single_fixed_arm_qualifying_count<2 or best_fixed_interval_covers_0_and_.02')


def expand_opportunity(config):
    c, b = config, config['base_experiment']
    anchor = resolve_experiment(dict(study_id=c['study_id'], model=dict(d=16,h=64,tau=1.),
        growth=dict(m_requested=16,epsilon=.05),
        data=dict(n_fit=2306,n_growth=577,n_validation=577,noise_sigma=.1),
        training=dict(base_updates=8000,continuation_updates=8000,batch_size=128,
            learning_rate=.005,weight_decay=0.,scheduler='constant',validation_interval=25,
            patience=20,checkpoint_interval=1),
        backend=dict(device='cpu',cpu_threads=1,interop_threads=1))).to_dict()
    if (c['teachers'] != 4 or c['repetitions'] != 2 or c['evaluation_rows'] != 65536
            or c['seed_root'] != 2026101801
            or any(b[section] != anchor[section] for section in ('model','growth','data','training','backend'))):
        raise ValueError('M2 requires its fixed primary4-teacher/R2 d16/h64/m16 recipe; variants need a named new design')
    recipes = {name:{**b['training'],**RECIPE_OVERRIDES[name]} for name in RECIPES}
    attempts, conditional, parents, used32 = [], [], [], set()
    root, study = c['seed_root'], c['study_id']

    def shared(population, teacher, repeat):
        labels = (study, 'M2-shared-primary' if population == 'primary' else 'M2d-fresh-small',
                  'teacher', str(teacher))
        optimizer, counter = seed32(root, *labels, 'repeat', str(repeat), 'optimizer', used=used32)
        return dict(population=population, teacher_index=teacher, repetition=repeat,
            **{name+'_seed': seed64(root, *labels, name) for name in ('teacher','data','reference','evaluation')},
            optimizer_seed=optimizer, optimizer_collision_counter=counter,
            allocation_seed=seed64(root, *labels, 'repeat', str(repeat), 'allocation'),
            contrast_seed=seed64(root, *labels, 'repeat', str(repeat), 'contrast'),
            new_stream_seed=seed64(root, *labels, 'repeat', str(repeat), 'new_stream'),
            candidate_sample_seed=seed64(root, *labels, 'repeat', str(repeat), 'M2c-candidate-sample'))

    def add(matrix, common, suffix, *, admission='initial', **fields):
        teacher, repeat = common['teacher_index'], common['repetition']
        identity = f'{matrix}/teacher-{teacher}/repeat-{repeat}/{suffix}'
        case = dict(attempt_id=identity, matrix=matrix, **common, admission=admission, **fields)
        (attempts if admission == 'initial' else conditional).append(case)

    for teacher in range(8):
        admission = 'initial' if teacher < 4 else 'primary_extension'
        for repeat in range(2):
            common = shared('primary', teacher, repeat)
            key = f'primary/teacher-{teacher}/repeat-{repeat}'
            for recipe in RECIPES:
                parents.append(dict(prerequisite_id=key+'/'+recipe, **common, admission=admission,
                    parent_recipe=recipe, max_updates=8000, stop_by_patience=True,
                    training=recipes[recipe],absolute_schedule_horizon=16000,
                    snapshot_updates=[250,1000] if recipe == 'legacy_005' else []))
            for snapshot in ('actual_250', 'actual_1000', 'late_G_best'):
                for treatment in ('continued_stream', 'new_stream_retained_adam'):
                    for role in WARM_ROLES:
                        add('M2a', common, f'{snapshot}/{treatment}/{role}', admission=admission,
                            parent_key=key+'/legacy_005', snapshot=snapshot, treatment=treatment,
                            role=role, max_updates=8000, stop_by_patience=True, requested_count=16)
            fallback = 'missing_snapshot_fallback' if teacher < 4 else 'extension_missing_snapshot_fallback'
            parents.append(dict(prerequisite_id=key+'/fixed_parent_budget_fallback', **common,
                admission=fallback, parent_recipe='legacy_005', max_updates=8000,
                stop_by_patience=False, snapshot_updates=[250,1000],
                training=recipes['legacy_005'],absolute_schedule_horizon=16000,
                trigger='original training never reached a requested actual update; never artifact corruption'))
            for snapshot in ('actual_250', 'actual_1000'):
                for treatment in ('continued_stream', 'new_stream_retained_adam'):
                    for role in WARM_ROLES:
                        add('M2a', common, f'fallback/{snapshot}/{treatment}/{role}', admission=fallback,
                            parent_key=key+'/fixed_parent_budget_fallback', snapshot=snapshot,
                            treatment=treatment, role=role, max_updates=8000,
                            stop_by_patience=True, requested_count=16,
                            requires_admissions=(['original_snapshot_unavailable'] if teacher < 4 else
                                                 ['primary_extension','original_snapshot_unavailable']),
                            trigger='this original actual snapshot was not reached')
            for recipe in RECIPES:
                add('M2e_neural', common, recipe, admission=admission, recipe=recipe,
                    parent_key=key+'/'+recipe, parent_recipe=recipe, role='ordinary_keep',
                    max_updates=8000, stop_by_patience=True, absolute_schedule_horizon=16000,
                    snapshot='late_G_best',training=recipes[recipe],requested_count=0)
            add('M2e_head', common, 'ridge_head', admission=admission, role='ridge_head',
                parent_key=key+'/legacy_005', parent_recipe='legacy_005', alphas=list(ALPHAS),
                snapshot='late_G_best',requested_count=0)
            if teacher >= 4:
                continue
            for unit in (None, *range(64)):
                add('M2b', common, 'keep' if unit is None else f'unit-{unit}',
                    parent_key=key+'/legacy_005', role='keep' if unit is None else 'singleton',
                    unit_ids=[] if unit is None else [unit], max_updates=300, stop_by_patience=False,
                    snapshot='late_G_best',requested_count=0 if unit is None else 1,
                    selection_prefixes=[0,300], eligibility_policy='common_current_scorer_retain_nominal_skips')
            for slot in ('keep', *(f'top-{i}' for i in range(4)), *(f'sampled-{i}' for i in range(4))):
                add('M2c', common, slot, parent_key=key+'/legacy_005', slot=slot,
                    max_updates=8000, stop_by_patience=False, selection_prefixes=[0,300,1200,8000],
                    snapshot='late_G_best',requested_count=0 if slot=='keep' else 1,
                    slot_binding='F-only cheap top4 then fixed-seed sample of remaining4; missing slots ineligible')
    for teacher in range(4):
        for repeat in range(2):
            common = shared('positive_small', teacher, repeat)
            key = f'positive_small/teacher-{teacher}/repeat-{repeat}/ungrown'
            parents.append(dict(prerequisite_id=key, **common, admission='initial',
                parent_recipe='raw_positive_control', trained=False, d=2,h=8,
                n_fit=512,n_growth=256,n_validation=256,gamma=2.,sigma=.1))
            for subset in ((), *((i,) for i in range(8)), *combinations(range(8),2)):
                suffix = 'keep' if not subset else 'units-'+'-'.join(map(str,subset))
                add('M2d',common,suffix,parent_key=key,unit_ids=list(subset),gamma=2.,sigma=.1,
                    role='keep' if not subset else 'singleton' if len(subset)==1 else 'pair',
                    max_updates=8000,stop_by_patience=True,
                    requested_count=len(subset),evaluation_rows=16384,
                    eligibility_policy='common_current_scorer_retain_nominal_skips')
    counts = dict(Counter(a['matrix'] for a in attempts))
    if counts != dict(M2a=240,M2b=520,M2c=72,M2d=296,M2e_neural=40,M2e_head=8):
        raise AssertionError('M2 nominal matrix arithmetic differs from its frozen contract')
    documents = Path(__file__).resolve().parents[3]
    bindings = {key:hashlib.sha256((documents/path).read_bytes()).hexdigest() for key,path in
        (('binding_spec_sha256','docs/superpowers/specs/2026-10-18-research-v2-design.md'),
         ('M2_execution_sha256','docs/research/v2/M2-execution.md'))}
    return dict(attempts=attempts,conditional_attempts=conditional,prerequisites=parents,probe_trials=0,
        matrix_counts=counts,execution_implemented=True,
        opportunity_decision_contract=opportunity_decision_contract(c),
        design_bindings=bindings,ordinary_recipes=recipes,
        selection_contract=dict(primary_count='min(requested_count,common_current_eligible_count)',
            ordinary_brackets='ordinary_width_brackets(d,parent_parameter_count+2*m_effective*(d+1))',
            zero_count='all arms retain zero added capacity; ordinary width remains parent width',
            pair_oracle=dict(select_on='G',eligible_roles=['pair'],metric='original_unit_mse',
                ties='1e-12*max(1,abs(best)); smallest lexicographically sorted unit-ID pair',
                no_eligible_pair='completed_ineligible; never filter or replace teacher',
                cost='all attempted pair procedures and shared prerequisites charged'),
            positive_control=dict(mean_repeats_first=True,metric='V_NMSE',gamma=2.,sigma=.1,
                comparison='true_contrast_oracle_over_random',minimum_relative_reduction=.02,
                minimum_instances=2,total_instances=4),
            opportunity=dict(metric='V_NMSE',mean_repeats_first=True,minimum_relative_reduction=.02,
                minimum_independent_teachers=2,reference='best competent same-snapshot/treatment ordinary keep')),
        cost_inventory=dict(initial_ridge_solves=48,initial_ordinary_neural_stages=80,
            initial_fitted_small_trajectory_ceiling=232,initial_shared_primary_parents=8,
            fallback_parent_ceiling=16,conditional_fallback_fits=320,
            primary_extension_fits=288,required_accounting='pay all attempted work; shared physical and standalone costs separate'),
        extension_policy=dict(max_extensions=1,initial_primary_teachers=list(range(4)),
            next_primary_teachers=list(range(4,8)),independent_of_effect_sign=True,
            trigger_any=[dict(qualifying_first_four_teachers_less_than=2),
                dict(best_fixed_contrast_whole_teacher_descriptive_interval=dict(confidence=.95,
                    overlaps_relative_benefit_points=[0.,.02]))],
            fallback_requires_all=['primary_extension_if_teacher_index_ge_4','original_snapshot_unavailable']),
        eligibility_policy='resolve common F-only current eligibility after data creation; retain all nominal skips')
