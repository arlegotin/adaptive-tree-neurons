"""One predeclared saved-scalar MC refinement after exact 8000-draw parity."""
import time
started=time.perf_counter_ns()
import hashlib,json,math,os,re,resource,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.stats import t

base=Path('environment/h1-v2/pilot-saved-mc-supplement');root=Path(os.environ['H1_SAVED_MC_ROOT']);root.mkdir(parents=True,exist_ok=False)
bundle=Path('artifacts/h1-v2/user-pause-continuation/a6b986ed496b4f22be3a43c65943dc63')
saved=bundle/'joined-diagnostics'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
def replace(path,value):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');p.replace(path)
def duration():return (time.perf_counter_ns()-started)/1e9
status=dict(scope='Prospectively declared one fixed512000 saved-scalar whole-teacher MC diagnostic; original8000 and source branch unchanged',
    status='starting',worker_pid=os.getpid(),artifact_root=str(root),observations=[],failures=[],bootstrap_calls=[],supplement_calls=0)
def observe(boundary):
    p=subprocess.run(['memory_pressure'],capture_output=True,text=True);free=re.search(r'System-wide memory free percentage:\s+(\d+)%',p.stdout);swap=re.search(r'Swapouts:\s+(\d+)',p.stdout)
    item=dict(boundary=boundary,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed_seconds=duration(),
        continuation_aggregate_elapsed_seconds=(time.perf_counter_ns()-514707018090000)/1e9,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,disk=dict(zip(('total','used','free'),shutil.disk_usage('.'))),
        memory_pressure=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr),coarse_free_percent=int(free.group(1)) if free else None,
        swapouts=int(swap.group(1)) if swap else None)
    status['observations'].append(item);replace(base/'readout.json',status);return item
def guard(obs):
    assert obs['continuation_aggregate_elapsed_seconds']<7363.856268376,'remaining aggregate elapsed budget exhausted'
    assert obs['peak_rss_bytes']<=6448447488,'process memory cap exceeded'
    assert obs['disk']['free']>=10737418240+1048576,'disk reserve unavailable'
    assert status['accepted_indexed_original_and_continuation_bytes']+1048576<30871367120,'aggregate retained cap unavailable'
    initial=status['observations'][0]
    assert not(obs['coarse_free_percent'] is not None and obs['coarse_free_percent']<=10 and obs['swapouts'] is not None and initial['swapouts'] is not None and obs['swapouts']>initial['swapouts']),'severe availability together with new swapouts'

try:
    env_names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
    assert all(os.environ.get(k)=='1' for k in env_names) and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    bindings={
        str(saved/'joined-scalars-and-risks.json'):'7324942c0ee22b18b3fe76e22ba96c14be2b19d602e096c480ef5bda2a9bbff4',
        str(saved/'joined-summary-diagnostic.json'):'351c2578bde77e225a06478e212ac4a0b2bf60313a7f1daa535b6df5ea578fa1',
        str(saved/'joined-design-diagnostic.json'):'fd6844d27846b871259f5b54f5785fe57a6d99adac885e12e1df5be0eae55c2a',
        str(bundle/'joined-input-members.json'):'7f42ae967496a7e6a36095175617567454b7384bd74a81121d9be23681f7ab92',
        'environment/h1-v2/user-pause-continuation/inventory.json':'d984d8961a59dc5d996862a5361af30f28f5cc9b7c1a45e7be0c8a2fd33ff627',
        'environment/h1-v2/genuine-pilot/inventory.json':'c0a46776b126cd4a30c1f586b25ea3c786155736b362322fccfb428f3c06f4ea',
        '.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/pilot-saved-mc-supplement-predeclaration.md':'20f7d30d536ba1b33e939973e4ef9db0db639ebe197e353980312e7dc54e71fd',
        'src/treeneuron/pilot/design.py':'0c61e83d1e986433953270225833843ce236061582d4cfc6dd329d3580eb842e',
        'environment/h1-v2/backend.json':'4cfb4983fbe8d152decf7e621dc79b2e77b7f4e3825cb379534ce6a4e05e9e42'}
    assert all(sha(Path(p))==h for p,h in bindings.items())
    verification=Path('environment/h1-v2/user-resume-root/continuation-verification.json');assert not json.loads(verification.read_text())['errors']
    bindings[str(verification)]=sha(verification)
    config=json.loads((base/'analysis-config.json').read_text());assert config['counts']==[8000,512000] and config['seed']==261022106
    status.update(bindings=bindings,script_sha256=sha(Path(__file__)),analysis_config_sha256=sha(base/'analysis-config.json'),
        numerical_runtime=dict(numpy=np.__version__,python=sys.version,environment={k:os.environ[k] for k in env_names+('PYTORCH_ENABLE_MPS_FALLBACK',)}))
    indexes=[Path('environment/h1-v2/genuine-pilot/inventory.json'),Path('environment/h1-v2/user-pause-continuation/inventory.json')]
    status['accepted_indexed_original_and_continuation_bytes']=sum(json.loads(p.read_text())['bytes']+p.stat().st_size for p in indexes)
    old=json.loads((saved/'joined-summary-diagnostic.json').read_text());summary=old['summary']
    scalars=json.loads((saved/'joined-scalars-and-risks.json').read_text())['teachers'];assert [x['index'] for x in scalars]==list(range(16))
    assert all(x['repetitions']==2 and x['evaluation_rows']==262144 for x in scalars)
    assert [x['h1_mean'] for x in scalars]==summary['teacher_means']
    values=np.asarray([x['h1_mean'] for x in scalars]);static=np.asarray([x['static_mean'] for x in scalars])
    within=np.asarray([x['within_teacher_H1_variance'] for x in scalars]);rowvars=np.asarray([x['paired_row_H1_variance'] for x in scalars]);rowcounts=np.asarray([x['evaluation_rows'] for x in scalars])
    assert np.array_equal(within,np.asarray(summary['diagnostics']['within_teacher_variances']))
    assert np.array_equal(rowvars,np.asarray(summary['diagnostics']['paired_row_variances']))
    nuisance=within/2+rowvars/rowcounts
    endpoint_names=['sd_lower','sd_upper','h1_lower','h1_upper','static_lower','static_upper','observed_teacher_variance_lower','nuisance_mean_upper']
    def bootstrap(count):
        tick=time.perf_counter_ns();rng=np.random.default_rng(config['seed']);before=rng.bit_generator.state
        indices=rng.integers(len(values),size=(count,len(values)))
        digest=hashlib.sha256();digest.update(str(indices.dtype).encode());digest.update(str(indices.shape).encode());digest.update(indices.tobytes())
        draws=values[indices];sd=draws.std(axis=1,ddof=1)
        hmeans,smeans=draws.mean(axis=1),static[indices].mean(axis=1)
        variance_draws=sd**2;nuisance_draws=nuisance[indices].mean(axis=1);row_draws=rowvars[indices].mean(axis=1)
        component_endpoints=[(variance_draws,.025),(nuisance_draws,.975)]
        vectors=[np.concatenate([np.quantile(a,[.025,.975]) for a in (sd[part],hmeans[part],smeans[part])]+
            [np.atleast_1d(np.quantile(a[part],q)) for a,q in component_endpoints]) for part in np.array_split(np.arange(count),10)]
        raw_sd=np.std(vectors,axis=0,ddof=1);endpoint_se=raw_sd/math.sqrt(10)
        observed_lower=float(np.quantile(variance_draws,.025));nuisance_upper=float(np.quantile(nuisance_draws,.975))
        ci=lambda a:[float(v) for v in np.quantile(a,[.025,.975])]
        margin=float(t.ppf(.975,15)*values.std(ddof=1)/math.sqrt(16))
        result=dict(iterations=count,seed=config['seed'],bootstrap_unit='whole teacher; fixed16 teacher scalars already average both paired repetitions',
            h1_mean=float(values.mean()),static_mean=float(static.mean()),h1_interval=ci(hmeans),static_interval=ci(smeans),sd_interval=ci(sd),
            observed_teacher_variance_lower=observed_lower,nuisance_mean_upper=nuisance_upper,between_teacher_variance_lower=max(0.,observed_lower-nuisance_upper),
            paired_row_variance_upper=float(np.quantile(row_draws,.975)),repetition_variance=float(within.mean()),
            mean_nuisance_allowance=float(nuisance.mean()),nuisance_allowance_per_teacher=nuisance.tolist(),observed_teacher_variance=float(values.var(ddof=1)),
            student_t_interval=[float(values.mean()-margin),float(values.mean()+margin)],
            component_intervals=dict(observed_teacher_variance=ci(variance_draws),nuisance_mean=ci(nuisance_draws),mean_paired_row_variance=ci(row_draws)),
            endpoint_order=endpoint_names,endpoint_SEs=dict(zip(endpoint_names,map(float,endpoint_se))),endpoint_chunk_vectors=[v.tolist() for v in vectors],
            endpoint_mc_error=float(raw_sd.max()/math.sqrt(10)),source_endpoint_method='same8 endpoint vector and max over10 disjoint chunks as design.py; row upper and extra two-sided component quantiles are reported without adding gates',
            sampling_indices_sha256=digest.hexdigest(),rng_state_before=before,rng_state_after=rng.bit_generator.state,
            computation_seconds=(time.perf_counter_ns()-tick)/1e9)
        result['sd_interval_ratio']=result['sd_interval'][1]/result['sd_interval'][0]
        result['endpoint_gate']=config['endpoint_gate'];result['endpoint_gate_passed']=result['endpoint_mc_error']<=config['endpoint_gate']
        status['bootstrap_calls'].append(dict(iterations=count,seed=config['seed'],computation_seconds=result['computation_seconds']));return result
    guard(observe('before8000 parity'))
    parity=bootstrap(8000);save(root/'parity-8000.json',parity)
    expected={key:summary[key] for key in ('h1_mean','static_mean','h1_interval','static_interval','sd_interval','endpoint_mc_error','between_teacher_variance_lower','paired_row_variance_upper','repetition_variance')}
    intrinsic=summary['diagnostics']['intrinsic_variance']
    expected.update({key:intrinsic[key] for key in ('observed_teacher_variance_lower','nuisance_mean_upper','mean_nuisance_allowance','nuisance_allowance_per_teacher')})
    expected.update({key:summary['diagnostics'][key] for key in ('student_t_interval','observed_teacher_variance')})
    errors={key:float(np.max(np.abs(np.asarray(parity[key])-np.asarray(value)))) for key,value in expected.items()}
    comparison=dict(target_summary_sha256=bindings[str(saved/'joined-summary-diagnostic.json')],absolute_tolerance=1e-12,field_max_abs_errors=errors,
        max_abs_error=max(errors.values()),passed=max(errors.values())<=1e-12)
    save(root/'parity-comparison.json',comparison)
    assert comparison['passed'],'8000-draw parity discrepancy;512000 prohibited'
    print('PARITY_PASSED',json.dumps(comparison),flush=True)
    guard(observe('before fixed512000'))
    status['supplement_calls']+=1;assert status['supplement_calls']==1
    supplement=bootstrap(512000);save(root/'supplement-512000.json',supplement)
    status.update(status='completed',original_source_branch='calibrate',original_failed_producer_unchanged=True,
        canonical_freeze_eligible=False,missing_canonical_second_runtime_batch=True,population_coverage_validated=False)
    print('SUPPLEMENT_COMPLETE',json.dumps({k:supplement[k] for k in ('h1_interval','static_interval','sd_interval','sd_interval_ratio','endpoint_mc_error','endpoint_gate_passed','between_teacher_variance_lower')}),flush=True)
except BaseException as error:
    status['status']='failed';status['failures'].append(f'{type(error).__name__}: {error}');raise
finally:
    observe('final');status.update(enclosing_seconds=duration(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(root/'analysis-manifest.json',status);replace(base/'readout.json',status)
