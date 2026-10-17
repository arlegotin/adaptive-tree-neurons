"""Write an exact proposed workload, without source imports or execution."""
import hashlib,json
from pathlib import Path

base=Path('environment/h1-v2/precision')
out=Path('environment/h1-v2/resource-proposal');out.mkdir(exist_ok=True)
mc=json.loads((base/'monte-carlo.json').read_text())
rows=json.loads((base/'loss-timing.json').read_text())
config=json.loads(Path('artifacts/h1-v2/development-8000/h1-v2-primary-a-8000/measurements.json').read_text())['config']
config['study_id']='h1-v2-complete-resource-candidate'
config['execution']['artifacts_root']='artifacts/h1-v2/complete-resource'
config['pilot'].update(purpose='engineering',initial_teachers=1,max_teachers=1,repetitions=2,
    evaluation_rows=rows['E_choice']['candidate_rows'],probe_updates=320,
    bootstrap_iterations=mc['bootstrap_choice']['iterations'],bootstrap_seed=261021106,
    endpoint_tolerance=mc['bootstrap_choice']['absolute_endpoint_tolerance'],
    constraints_path=None,readiness_path=None,max_seconds=14400.0)
config['seeds']={**config['seeds'],**dict(teacher=261021100,data=261021101,optimizer=261021102,
    allocation=261021103,contrast=261021104,reference=261021105,namespace='exploratory')}
recipe={k:config[k] for k in ('model','training','growth','data','backend')}
recipe['pilot_workload']={k:config['pilot'][k] for k in ('repetitions','evaluation_rows','probe_updates','bootstrap_iterations')}
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def write(path,value):
    assert not path.exists(),path
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
write(out/'candidate.json',config)
write(out/'recipe.json',recipe)
proposal=dict(status='proposed_not_authorized_or_measured',recipe_sha256=hashlib.sha256(canonical(recipe)).hexdigest(),
    hash_method='exact stdlib transcription of protocol.recipe_hash canonical JSON; existing API must verify this before the authorized run',
    current_head='fc63de6884ba5d7a911e0228d2c6707c3ce90a11',
    execution_sha256='07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c',
    package_sha256='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c',
    runtime_sha256='4cfb4983fbe8d152decf7e621dc79b2e77b7f4e3825cb379534ce6a4e05e9e42',
    graph=dict(engineering_teachers=1,repetitions=2,fresh_parent_per_repetition=True,
        fixed_teacher_FGV_E_across_repetitions=True,inner_seeds='existing teacher_inputs/repetition_config derivation from independent2610211xx root namespace',
        roles=['keep','top','random','greedy','widen_upper','ordinary_lower','ordinary_upper','static_partial'],
        B6=dict(all_eligible_ids=True,expected_eligible_ids=64,keep_probe=True,expected_probe_count_per_repetition=65,
            probe_updates=320,retain_every_loser=True,ranking='existing G-only ranking; full selected joint allocation receives unchanged continuation allowance'),
        fresh_ordinary_static_allowance='two stages, base8000 plus continuation8000, common G/cp25 and patience20',
        runtime='fresh CPU float32 train / float64 score, intra1/inter1/allBLAS1/fallback0; serialize all fits'),
    numerical_policies=dict(evaluation_rows=262144,E_precision_target_reached=False,
        E_scope='bounded resource/variance-pilot E workload, not final protected T sizing; no expansion inferred from two teacher means',
        bootstrap_iterations=8000,endpoint_tolerance=mc['bootstrap_choice']['absolute_endpoint_tolerance'],
        bootstrap_scope='conditional block-array numeric target met only; actual whole-teacher pilot endpoint diagnostics must pass separately',
        power_iterations=16384,power_mc_tolerance=.01,
        timing=dict(rows=4096,batch=128,min_repetitions=40,max_repetitions=200,relative_precision=.05,warmup_relative_spread=.1),
        timing_scope='existing paired timing outside pilot fitting graph; report distinct added cost and no kernel/host conflation'),
    required_measurements=['external process elapsed including imports and final publication',
        'disjoint setup, fit-graph, E prediction/loss/serialization, aggregation, final-IO regions',
        'fit_repetition nested parent/scoring/B6 and all method clocks, never added to enclosing again',
        'complete raw fit clock copied before run_repetition replaces enclosing_ns; call existing functions exactly once',
        'all stages selected/executed/examples from raw events, hit_ceiling/improving_at_ceiling/final nonimprovements',
        'every65 B6 candidate status/curve/retained files in each repetition, including keep and losers',
        'peak RSS and separate raw_events/checkpoint_metadata/tensor_payloads/predictions/other final bytes',
        'native hardware/current memory pressure/disk headroom and resource/failure policy',
        'strict raw sample binding recipe/runtime/execution SHA, exact final bytes and enclosing seconds',
        'one engineering teacher skips real whole-teacher bootstrap; retain separately labeled measured conditional-block workload at8/16input shapes as cost diagnostic'],
    unresolved=['E conditional H1 SE engineering target was not reached at262144; no adequacy claim',
        'probe320 is largest tested finite horizon and only four sampled IDs; not plateau/rank convergence',
        'bootstrap numerical margin on E-block inputs is not evidence about true between-teacher endpoints or coverage',
        'no full R2 resource sample, readiness, genuine pilot, fixed cohort or T exists'],
    stop='Await root adjudication/authorization before any all8 R2 engineering execution; no further numerical work now')
write(out/'proposal.json',proposal)
print(proposal['recipe_sha256'])
