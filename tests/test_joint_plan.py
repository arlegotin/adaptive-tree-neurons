"""M3 finite design expands before any teacher exists; no effect-driven sampling."""
from collections import Counter
from copy import deepcopy
import math

import numpy as np
import pytest


def config():
    from treeneuron.research.config import ResearchConfig
    return ResearchConfig.from_dict(dict(schema='research/v2',study_id='M3-joint',kind='m3_joint',seed_root=2026101801,
        teachers=8,repetitions=2,evaluation_rows=65536,base_experiment=dict(study_id='M3-joint',model=dict(d=16,h=64,tau=1.),
        growth=dict(m_requested=16,epsilon=.05),data=dict(n_fit=2306,n_growth=577,n_validation=577,noise_sigma=.1),
        training=dict(base_updates=8000,continuation_updates=8000,batch_size=128,learning_rate=.005,weight_decay=.001,
            scheduler='constant',validation_interval=25,patience=20,checkpoint_interval=25),
        backend=dict(device='cpu',cpu_threads=1,interop_threads=1))))


def test_m3_expands_fixed_replication_controls_and_one_uncertainty_extension():
    from treeneuron.research.config import expand
    p=expand(config()).to_dict()
    assert Counter(row['matrix'] for row in p['attempts'])=={'M3':160}
    assert len(p['probe_attempts'])==1040 and p['probe_trials']==1040
    assert Counter(row['matrix'] for row in p['conditional_attempts'])=={'M3':160,'M3_probes':1040}
    assert len(p['prerequisites'])==32 and all(row['stop_by_patience'] for row in p['prerequisites'])
    assert len({row['attempt_id'] for key in ('attempts','probe_attempts','conditional_attempts') for row in p[key]})==2400
    for row in p['attempts']:
        assert row['max_updates']==8000 and row['stop_by_patience'] is False
        assert row['treatment']=='continued_stream_retained_adam'
        assert row['parent_recipe']=='competent_wd_1e3'
    for row in p['probe_attempts']:
        assert row['max_updates']==300 and row['stop_by_patience'] is False
        assert row['selection_grid']==[0,25,50,75,100,125,150,175,200,225,250,275,300]
    assert p['execution_implemented'] is False
    assert p['protected_labels_accessed'] is False


def test_teacher_and_training_randomness_is_paired_across_methods_and_repeats_are_not_teachers():
    from treeneuron.research.config import expand
    p=expand(config()).to_dict();cases=p['attempts']+p['probe_attempts']+p['conditional_attempts']
    for t in range(16):
        rows=[row for row in cases if row['teacher_index']==t]
        for name in ('teacher_seed','data_seed','evaluation_seed','reference_seed'):
            assert len({row[name] for row in rows})==1
        for repeat in range(2):
            paired=[row for row in rows if row['repetition']==repeat]
            assert len({row['optimizer_seed'] for row in paired})==1
            assert len({row['allocation_seed'] for row in paired})==1
            assert len({row['contrast_seed'] for row in paired})==1
    assert len({row['teacher_seed'] for row in cases})==16
    assert len({row['optimizer_seed'] for row in cases})==32
    for row in p['prerequisites']:assert row['fresh_optimizer_seed']!=row['optimizer_seed']


@pytest.mark.parametrize('change',[
    ('teachers',16),('repetitions',3),('evaluation_rows',16384),('seed_root',1),
    ('training.weight_decay',0.),('training.learning_rate',.001),('growth.m_requested',8),('model.h',128)])
def test_m3_rejects_undeclared_design_changes(change):
    from treeneuron.research.config import ResearchConfig,expand
    raw=config().to_dict();key,value=change
    if '.' in key:section,name=key.split('.');raw['base_experiment'][section][name]=value
    else:raw[key]=value
    with pytest.raises(ValueError):expand(ResearchConfig.from_dict(raw))


def test_extension_uses_R2_teacher_effects_and_only_the_declared_interval_rule():
    from treeneuron.research.joint_plan import joint_extension_decision
    def rows(effects):
        return [dict(teacher_index=t,repetition=r,method=method,V_NMSE=math.exp(effect) if method=='random' else 1.)
            for t,effect in enumerate(effects) for r in range(2) for method in ('random','joint')]
    tied=joint_extension_decision(rows([0.]*8),seed=123)
    assert tied['extend'] is False and tied['interval']==[0.,0.]
    ambiguous=joint_extension_decision(rows([-.1,.1]*4),seed=123)
    assert ambiguous['extend'] is True and ambiguous['teachers']==8
    assert ambiguous['interval'][0]<=0 and ambiguous['interval'][1]>=math.log(1/.98)
    assert joint_extension_decision(rows([.1]*8),seed=123)['extend'] is False
    with pytest.raises(ValueError):joint_extension_decision(rows([0.]*8)[:-1],seed=123)


def test_public_plan_reports_unadmitted_M3_and_run_cannot_start_scientific_work(tmp_path):
    import json
    import subprocess
    import sys
    args=[sys.executable,'-B','-m','treeneuron.research.cli','plan','--config',
        'configs/research/v2/M3-joint.json','--output',str(tmp_path/'plan.json')]
    planned=subprocess.run(args,capture_output=True,text=True)
    assert planned.returncode==0,planned.stderr
    assert len(json.loads((tmp_path/'plan.json').read_text())['attempts'])==160
    args[4]='run';args[-1]=str(tmp_path/'unadmitted')
    run=subprocess.run(args,capture_output=True,text=True)
    assert run.returncode==2 and 'M3' in run.stderr
    assert not (tmp_path/'unadmitted').exists()
