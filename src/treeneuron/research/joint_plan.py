"""Finite competent-treatment M3 development design, before teacher generation."""
import hashlib
import math
from pathlib import Path

import numpy as np

from treeneuron.config import resolve_experiment
from .seeds import seed64,seed32


FINAL_ROLES=('keep','top','random','short_singleton','local_singleton','joint',
             'unconditioned','pure_span','warm_upper','fresh_upper')


def extension_contract(config):
    return dict(schema='research/v2-M3-uncertainty-extension',initial_teachers=8,maximum_teachers=16,
        repetitions=2,metric='V_NMSE',contrast='log(mean_R2_random/mean_R2_joint)',
        margin=math.log(1/.98),bootstrap_resamples=20000,
        bootstrap_seed=seed64(config['seed_root'],config['study_id'],'M3-uncertainty-extension/bootstrap'),
        interval='equal-teacher percentile95; linear quantile; fixed joint/random contrast',
        extend_if='interval_lower<=0 and interval_upper>=margin; once only',
        missingness='missing/failed paired V prediction blocks decision; no replacement teacher')


def expand_joint(config):
    c=config;b=c['base_experiment']
    anchor=resolve_experiment(dict(study_id=c['study_id'],model=dict(d=16,h=64,tau=1.),
        growth=dict(m_requested=16,epsilon=.05),
        data=dict(n_fit=2306,n_growth=577,n_validation=577,noise_sigma=.1),
        training=dict(base_updates=8000,continuation_updates=8000,batch_size=128,
            learning_rate=.005,weight_decay=.001,scheduler='constant',validation_interval=25,
            patience=20,checkpoint_interval=1),backend=dict(device='cpu',cpu_threads=1,interop_threads=1))).to_dict()
    if (c['teachers']!=8 or c['repetitions']!=2 or c['evaluation_rows']!=65536
            or c['seed_root']!=2026101801
            or any(b[section]!=anchor[section] for section in ('model','growth','data','training','backend'))):
        raise ValueError('M3 requires its locked8-teacher/R2 competent WD1e-3 d16/h64/m16 recipe')
    attempts=[];probes=[];conditional=[];parents=[];used32=set()
    for teacher in range(16):
        admission='initial' if teacher<8 else 'uncertainty_extension'
        labels=(c['study_id'],'M3-fresh-primary','teacher',str(teacher))
        for repeat in range(2):
            paired=(*labels,'repeat',str(repeat))
            optimizer,counter=seed32(c['seed_root'],*paired,'optimizer',used=used32)
            fresh,fresh_counter=seed32(c['seed_root'],*paired,'fresh-ordinary-optimizer',used=used32)
            common=dict(population='primary',teacher_index=teacher,repetition=repeat,admission=admission,
                **{name+'_seed':seed64(c['seed_root'],*labels,name) for name in ('teacher','data','reference','evaluation')},
                optimizer_seed=optimizer,optimizer_collision_counter=counter,fresh_optimizer_seed=fresh,
                fresh_optimizer_collision_counter=fresh_counter,
                allocation_seed=seed64(c['seed_root'],*paired,'allocation'),
                contrast_seed=seed64(c['seed_root'],*paired,'contrast'),
                candidate_sample_seed=seed64(c['seed_root'],*paired,'diagnostic-inventory'),
                parent_recipe='competent_wd_1e3')
            key=f'primary/teacher-{teacher}/repeat-{repeat}/competent_wd_1e3'
            parents.append(dict(prerequisite_id=key,**common,max_updates=8000,stop_by_patience=True,
                snapshot_updates=[],training=b['training'],absolute_schedule_horizon=16000))
            for role in FINAL_ROLES:
                row=dict(attempt_id=f'M3/teacher-{teacher}/repeat-{repeat}/{role}',matrix='M3',**common,
                    parent_key=key,role=role,requested_count=16,max_updates=8000,stop_by_patience=False,
                    snapshot='late_G_best',treatment='continued_stream_retained_adam',selection_grid=list(range(0,8001,25)),
                    parent_ridge=0. if role=='pure_span' else .01,gate_ridge=.01,
                    condition_on_parent=role!='unconditioned',
                    selection_strategy='singleton' if role=='local_singleton' else 'conditional')
                (attempts if admission=='initial' else conditional).append(row)
            for unit in (None,*range(64)):
                row=dict(attempt_id=f'M3_probes/teacher-{teacher}/repeat-{repeat}/'+('keep' if unit is None else f'unit-{unit}'),
                    matrix='M3_probes',**common,parent_key=key,role='keep' if unit is None else 'subset',
                    unit_ids=[] if unit is None else [unit],requested_count=0 if unit is None else 1,
                    max_updates=300,stop_by_patience=False,snapshot='late_G_best',selection_grid=list(range(0,301,25)),
                    eligibility_policy='common_current_scorer; nominal ineligible IDs retained')
                (probes if admission=='initial' else conditional).append(row)
    root=Path(__file__).resolve().parents[3]
    bindings={name:hashlib.sha256((root/path).read_bytes()).hexdigest() for name,path in (
        ('binding_spec_sha256','docs/superpowers/specs/2026-10-18-research-v2-design.md'),
        ('M3_recipe_sha256','docs/research/v2/M3-development-recipe.md'))}
    return dict(attempts=attempts,probe_attempts=probes,conditional_attempts=conditional,prerequisites=parents,
        probe_trials=len(probes),matrix_counts=dict(initial_final_fits=160,initial_probe_entries=1040,
            initial_parent_procedures=16,conditional_final_fits=160,conditional_probe_entries=1040,
            conditional_parent_procedures=16),execution_implemented=False,design_bindings=bindings,
        extension_contract=extension_contract(c),
        recovery_policy=dict(infrastructure_same_seed_reruns=1,numerical_failure_reruns=0,retain_all_costs=True),
        pending=['M3 production dispatcher, measured complete workload/storage admission and held source checkout'],
        selection_contract=dict(F_only_placement=True,unchanged_contrasts=True,common_effective_count=True,
            short_probe_selection='G risk minus common fixed300 keep; stable unit-ID ties; discard weights and restart joint fit',
            final_evaluation='V/E coordinator after fit; F variance ddof0; no refit or seed selection',
            fixed_horizon='all warm arms execute8000 even after losing G checks; complete strict best retained'))


def joint_extension_decision(rows,*,seed):
    """One variance/precision-triggered development expansion, not a claim test."""
    keys={'teacher_index','repetition','method','V_NMSE'}
    if type(rows) is not list or len(rows)!=32:raise ValueError('Complete eight-teacher R2 joint/random V table required')
    values={}
    for row in rows:
        if (type(row) is not dict or set(row)!=keys or type(row['teacher_index']) is not int
                or row['teacher_index'] not in range(8) or type(row['repetition']) is not int
                or row['repetition'] not in (0,1) or row['method'] not in ('joint','random')
                or type(row['V_NMSE']) not in (float,int) or not math.isfinite(row['V_NMSE']) or row['V_NMSE']<=0):
            raise ValueError('Complete finite positive paired teacher risks required')
        key=(row['teacher_index'],row['repetition'],row['method'])
        if key in values:raise ValueError('Duplicate paired teacher risk')
        values[key]=row['V_NMSE']
    effects=np.array([math.log(np.mean([values[t,r,'random'] for r in (0,1)])/
        np.mean([values[t,r,'joint'] for r in (0,1)])) for t in range(8)])
    indices=np.random.default_rng(seed).integers(0,8,size=(20000,8))
    interval=np.quantile(np.mean(effects[indices],axis=1),[.025,.975],method='linear').tolist()
    margin=math.log(1/.98)
    return dict(extend=interval[0]<=0 and interval[1]>=margin,interval=interval,margin=margin,
        teacher_effects=effects.tolist(),teachers=8,repetitions=2,bootstrap_resamples=20000,
        seed=seed,scope='One prospective development precision extension; no efficacy decision')
