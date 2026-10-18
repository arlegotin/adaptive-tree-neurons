"""Compile the fixed initial M2 inventory; handlers/admission remain separate."""
from collections import Counter

from treeneuron.config import canonical_hash,canonical_json
from .workqueue import make_queue_plan


def compile_initial_graph(study):
    if (study.get('schema')!='research/v2-plan' or study['config']['kind']!='m2_opportunity'
            or study.get('plan_hash')!=canonical_hash({k:v for k,v in study.items() if k!='plan_hash'})):
        raise ValueError('M2 study identity/digest differs')
    from .config import ResearchConfig,expand
    if canonical_json(expand(ResearchConfig.from_dict(study['config'])).to_dict())!=canonical_json(study):
        raise ValueError('M2 study differs from the resolved current-source plan')
    jobs=[];parents={};datasets={};scores={};bindings={}

    def add(name,kind,payload,dependencies=(),*,dependency_policy='all_eligible'):
        jobs.append(dict(job_id=name,kind=kind,payload=payload,depends_on=list(dependencies),
            dependency_policy=dependency_policy))
        return name

    def data_key(case):return f"{case['population']}/teacher-{case['teacher_index']}"

    for parent in study['prerequisites']:
        if parent['admission']!='initial':continue
        key=data_key(parent)
        values={k:parent[k] for k in ('population','teacher_index','teacher_seed','data_seed',
                                    'reference_seed','evaluation_seed')}
        values.update(evaluation_rows=16384 if parent['population']=='positive_small'
                      else study['config']['evaluation_rows'])
        if key not in datasets:
            datasets[key]=add('data:'+key,'data',values)
        else:
            previous=next(j['payload'] for j in jobs if j['job_id']==datasets[key])
            if previous!=values:raise ValueError('Paired prerequisite changed shared teacher/data seeds')
        parent_key=parent['prerequisite_id']
        if parent_key in parents:raise ValueError('Duplicate M2 parent prerequisite')
        parents[parent_key]=add('parent:'+parent_key,'parent',parent,(datasets[key],))
        if parent['parent_recipe'] not in ('legacy_005','raw_positive_control'):continue
        snapshots=['late_G_best']+(['actual_250','actual_1000'] if parent['parent_recipe']=='legacy_005' else [])
        for snapshot in snapshots:
            scores[parent_key,snapshot]=add(f'score:{parent_key}/{snapshot}','score',
                dict(parent_key=parent_key,snapshot=snapshot),
                (datasets[key],parents[parent_key]))
        if parent['parent_recipe']=='legacy_005':
            bindings[parent_key]=add('horizon_binding:'+parent_key,'horizon_binding',
                dict(parent_key=parent_key,seed=parent['candidate_sample_seed']),
                (scores[parent_key,'late_G_best'],))
            add('local_diagnostic:'+parent_key,'local_diagnostic',dict(parent_key=parent_key),
                (datasets[key],parents[parent_key],scores[parent_key,'late_G_best']))
    pairs={}
    for case in study['attempts']:
        parent_key=case['parent_key'];key=data_key(case)
        dependencies=[datasets[key],parents[parent_key]]
        if case['matrix'] not in ('M2e_neural','M2e_head'):
            dependencies.append(scores[parent_key,case.get('snapshot','late_G_best')])
        if case['matrix']=='M2c':dependencies.append(bindings[parent_key])
        fitted=add('procedure:'+case['attempt_id'],'head' if case['matrix']=='M2e_head' else 'fit',case,dependencies)
        add('evaluate:'+case['attempt_id'],'evaluate',dict(attempt_id=case['attempt_id'],matrix=case['matrix']),
            (datasets[key],fitted))
        if case['matrix']=='M2d' and len(case['unit_ids'])==2:
            entry=pairs.setdefault(parent_key,dict(data_key=key,teacher_index=case['teacher_index'],
                repetition=case['repetition'],fits=[]))
            entry['fits'].append(fitted)
    for parent_key,pair in pairs.items():
        oracle=add('pair_oracle:'+parent_key,'pair_oracle',dict(parent_key=parent_key,
            teacher_index=pair['teacher_index'],repetition=pair['repetition'],select_on='G'),pair['fits'],
            dependency_policy='all_terminal')
        add('evaluate:pair_oracle:'+parent_key,'evaluate',dict(diagnostic_outcome_id='pair_oracle:'+parent_key,
            matrix='M2d'),(datasets[pair['data_key']],oracle))
    counts=dict(Counter(j['kind'] for j in jobs))
    expected=dict(data=8,parent=48,score=32,horizon_binding=8,fit=1168,head=8,
        evaluate=1184,local_diagnostic=8,pair_oracle=8)
    if counts!=expected:raise ValueError('M2 compiled job arithmetic differs from its finite inventory')
    identity=dict(study_plan_hash=study['plan_hash'],source_sha256=study['source_sha256'],
        execution_identity=study['execution_identity'],study_config=study['config'],
        design_bindings=study['design_bindings'],matrix_counts=study['matrix_counts'],job_counts=counts,
        opportunity_decision_contract=study['opportunity_decision_contract'],
        admission='initial_only',execution_implemented=False,
        pending=['production dispatcher and resource admission',
                 'conditional fallback/extension admission','Gate A analysis'])
    return make_queue_plan(identity,jobs)


def compile_bounded_graph(study):
    """Precommit conditional slots in the same DAG; admission workers stay explicit.

    A skipped condition remains a terminal ineligible job. No prior queue is
    rewritten, and the unique-ancestor accounting includes fallback training.
    """
    initial = compile_initial_graph(study)
    jobs = [{k:v for k,v in row.items() if k != 'attempt_id'} for row in initial['attempts']]

    def add(name, kind, payload, dependencies=(), *, all_terminal=False):
        jobs.append(dict(job_id=name, kind=kind, payload=payload, depends_on=list(dependencies),
                         dependency_policy='all_terminal' if all_terminal else 'all_eligible'))
        return name

    def data_name(p):
        return f"data:{p['population']}/teacher-{p['teacher_index']}"

    def procedure(case):
        key = case['parent_key']
        dependencies = [data_name(case), 'parent:'+key]
        if case['matrix'] == 'M2a':
            dependencies.append(f"score:{key}/{case['snapshot']}")
        name = add('procedure:'+case['attempt_id'],
                   'head' if case['matrix'] == 'M2e_head' else 'fit', case, dependencies)
        return add('evaluate:'+case['attempt_id'], 'evaluate',
                   dict(attempt_id=case['attempt_id'], matrix=case['matrix']),
                   (data_name(case), name))

    def fallbacks(admission):
        evaluations = []
        for parent in study['prerequisites']:
            if parent['admission'] != admission:
                continue
            key = parent['prerequisite_id']
            original = key.rsplit('/', 1)[0]+'/legacy_005'
            gate = add('fallback_gate:'+key, 'fallback_gate',
                       dict(original_parent='parent:'+original, snapshot_updates=[250,1000]),
                       ('parent:'+original,), all_terminal=True)
            added = add('parent:'+key, 'parent', parent, (data_name(parent), gate))
            for snapshot in ('actual_250', 'actual_1000'):
                original_score = f'score:{original}/{snapshot}'
                snapshot_admission = add(f'snapshot_gate:{key}/{snapshot}', 'snapshot_gate',
                    dict(original_parent='parent:'+original, original_score=original_score,
                         fallback_gate=gate, snapshot=snapshot),
                    ('parent:'+original, original_score, gate), all_terminal=True)
                add(f'score:{key}/{snapshot}', 'score', dict(parent_key=key, snapshot=snapshot),
                    (data_name(parent), added, snapshot_admission))
        for case in study['conditional_attempts']:
            if case['admission'] == admission:
                evaluations.append(procedure(case))
        return evaluations

    fallback_evaluations = fallbacks('missing_snapshot_fallback')
    primary_evaluations = [j['job_id'] for j in initial['attempts']
        if j['kind'] == 'evaluate' and j['payload']['matrix'] in ('M2a','M2e_neural','M2e_head')]
    extension = add('extension_decision:primary', 'extension_decision',
                    dict(policy=study['extension_policy'], selection_contract=study['selection_contract'],
                         opportunity_decision_contract=study['opportunity_decision_contract']),
                    (*primary_evaluations, *fallback_evaluations), all_terminal=True)
    datasets = set()
    for parent in study['prerequisites']:
        if parent['admission'] != 'primary_extension':
            continue
        dataset = data_name(parent)
        if dataset not in datasets:
            values = {k:parent[k] for k in ('population','teacher_index','teacher_seed','data_seed',
                                          'reference_seed','evaluation_seed')}
            values['evaluation_rows'] = study['config']['evaluation_rows']
            add(dataset, 'data', values, (extension,))
            datasets.add(dataset)
        key = parent['prerequisite_id']
        added = add('parent:'+key, 'parent', parent, (dataset,))
        if parent['parent_recipe'] == 'legacy_005':
            for snapshot in ('late_G_best','actual_250','actual_1000'):
                add(f'score:{key}/{snapshot}', 'score', dict(parent_key=key, snapshot=snapshot),
                    (dataset, added))
    for case in study['conditional_attempts']:
        if case['admission'] == 'primary_extension':
            procedure(case)
    fallbacks('extension_missing_snapshot_fallback')
    counts = dict(Counter(j['kind'] for j in jobs))
    expected = dict(data=12,parent=104,score=88,horizon_binding=8,fit=1768,head=16,
                    evaluate=1792,local_diagnostic=8,pair_oracle=8,
                    fallback_gate=16,snapshot_gate=32,extension_decision=1)
    if counts != expected:
        raise ValueError('Bounded M2 job arithmetic differs from its finite inventory')
    identity = dict(initial['identity'], admission='initial_and_bounded_conditions', job_counts=counts,
                    pending=['production dispatcher and resource admission', 'final Gate A analysis'])
    return make_queue_plan(identity, jobs)
