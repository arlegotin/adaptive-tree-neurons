"""Measured development row scaling and existing timing APIs, no refitting."""
import json,time,resource
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.config import resolve_experiment
from treeneuron.data.synthetic import make_from_config,sample_teacher
from treeneuron.data.types import array_hash
from treeneuron.baselines.causal import CausalPredictor
from treeneuron.evaluation.timing import (TimingPolicy,ResidentCausalPredictor,measure_comparison,
    measure_cold_predictor,write_timing,read_timing)
from treeneuron.experiments.paired import file_hash,write_json
from treeneuron.protocol import execution_snapshot

base=Path('environment/h1-v2/precision');root=Path('artifacts/h1-v2/precision');work=json.loads((base/'workload.json').read_text())
assert train.source_hash()=='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c'
snapshot=execution_snapshot(Path.cwd(),'00d6a3a67af47412680ae14944902981aac0c9c8')
report=dict(scope=work['scope'],source_sha256=train.source_hash(),execution_sha256=snapshot['execution_sha256'],
    workload_sha256=file_hash(base/'workload.json'),wrapper_sha256=file_hash(Path(__file__)),problems=[])
all_models={};all_E={};started=time.perf_counter_ns()
for letter in ('a','b'):
    prior_path=Path(f'artifacts/h1-v2/development-8000/h1-v2-primary-{letter}-8000/measurements.json')
    prior=json.loads(prior_path.read_text());c=resolve_experiment(prior['config']);teacher,data=make_from_config(c)
    E=sample_teacher(teacher,max(work['evaluation_rows']),work['E_streams'][letter],c.data.noise_sigma)
    assert not set(E.row_id)&set(data.manifest.row_id)
    np.savez_compressed(root/f'E-inputs-{letter}.npz',X=E.X,y=E.y,row_id=E.row_id,f=E.noise_free_target)
    tick=time.perf_counter_ns();models={name:CausalPredictor.load(Path(prior['methods'][name]['model']['path']),
        expected_hashes=prior['methods'][name]['identity_hashes']) for name in work['models']}
    problem=dict(problem=letter,prior_path=str(prior_path),prior_sha256=file_hash(prior_path),
        E_seed=work['E_streams'][letter],E_rows=len(E.y),E_input_hash=array_hash(E.X),E_row_hash=array_hash(E.row_id),
        model_load_ns=time.perf_counter_ns()-tick,scaling=[])
    variance=models['keep'].state.fit_target_variance
    maximum_predictions=None
    for rows in work['evaluation_rows']:
        calls=[]
        for rep in range(work['prediction_repetitions']):
            tick=time.perf_counter_ns();predictions={name:model.predict(E.X[:rows]) for name,model in models.items()}
            prediction_ns=time.perf_counter_ns()-tick
            tick=time.perf_counter_ns()
            h=((predictions['random']-E.y[:rows])**2-(predictions['top']-E.y[:rows])**2)/variance
            s=((predictions['ordinary_upper']-E.y[:rows])**2-(predictions['static_partial']-E.y[:rows])**2)/variance
            loss_ns=time.perf_counter_ns()-tick
            output=root/f'predictions-{letter}-{rows}-r{rep}.npz';tick=time.perf_counter_ns()
            np.savez_compressed(output,**predictions,y=E.y[:rows],row_id=E.row_id[:rows],h1_rows=h,static_rows=s)
            calls.append(dict(repetition=rep,prediction_ns=prediction_ns,loss_ns=loss_ns,
                serialization_ns=time.perf_counter_ns()-tick,output_bytes=output.stat().st_size,
                output_path=str(output),output_sha256=file_hash(output),prediction_hashes={k:array_hash(v) for k,v in predictions.items()}))
        problem['scaling'].append(dict(rows=rows,calls=calls,h1_mean=float(h.mean()),static_mean=float(s.mean()),
            h1_row_sd=float(h.std(ddof=1)),static_row_sd=float(s.std(ddof=1)),
            conditional_h1_mean_se=float(h.std(ddof=1)/np.sqrt(rows)),conditional_static_mean_se=float(s.std(ddof=1)/np.sqrt(rows))))
        if rows==max(work['evaluation_rows']):
            maximum_predictions=predictions
            np.savez_compressed(root/f'row-losses-{letter}.npz',h1_rows=h,static_rows=s,row_id=E.row_id)
            blocks=len(h)//work['block_rows'];block=work['block_rows']
            problem['block_inputs']=dict(unit='disjoint exploratory E-row block conditional on fixed fitted teacher/models; never independent teachers',
                block_rows=block,blocks=blocks,h1_means=h[:blocks*block].reshape(blocks,block).mean(axis=1).tolist(),
                static_means=s[:blocks*block].reshape(blocks,block).mean(axis=1).tolist())
    all_models[letter]=models;all_E[letter]=E;report['problems'].append(problem)
    print('row scaling completed',letter,flush=True)
separations={contrast:abs(report['problems'][0]['scaling'][-1][contrast+'_mean']-report['problems'][1]['scaling'][-1][contrast+'_mean']) for contrast in ('h1','static')}
candidate=None;checks=[]
for rows in work['evaluation_rows']:
    observed={contrast:max(next(s for s in p['scaling'] if s['rows']==rows)['conditional_'+contrast+'_mean_se'] for p in report['problems']) for contrast in ('h1','static')}
    passes=all(separations[k]>0 and observed[k]<=.1*separations[k] for k in separations)
    checks.append(dict(rows=rows,conditional_SE=observed,engineering_targets={k:.1*v for k,v in separations.items()},passed=passes))
    if passes and candidate is None:candidate=rows
report['E_choice']=dict(candidate_rows=candidate or max(work['evaluation_rows']),engineering_target_reached=candidate is not None,
    A_B_development_mean_separations=separations,checks=checks,limitation='two-problem engineering yardstick, not population variance, coverage or confirmatory row precision; genuinepilot decomposition/T sizing remains required')

rules=work['timing'];common=dict(cpu_threads=1,warmup_window=5,max_warmup=30,min_repetitions=rules['min_repetitions'],
    max_repetitions=rules['max_repetitions'],relative_precision=rules['relative_precision'],warmup_relative_spread=rules['warmup_relative_spread'],seed=rules['seed'])
host=TimingPolicy(batch_sizes=(128,),**common);kernel=TimingPolicy(batch_sizes=(128,),mode='resident_kernel',**common)
observed=[]
class BoundedResident:
    def __init__(self,predictor):self.view=ResidentCausalPredictor(predictor);self.state=predictor.state
    def resource_summary(self):return self.view.resource_summary()
    def prepare_timing_input(self,X):
        assert 128<=min(128,self.state.config.data.n_fit)
        return self.view.prepare_timing_input(X)
    def predict_resident(self,X):
        assert 0<len(X)<=min(128,self.state.config.data.n_fit)
        observed.append(len(X));return self.view.predict_resident(X)
models=all_models['a'];X=all_E['a'].X[:rules['rows']]
timing=measure_comparison({'top-host':(models['top'],host),'random-host':(models['random'],host),
    'top-resident':(BoundedResident(models['top']),kernel),'random-resident':(BoundedResident(models['random']),kernel)},X)
write_timing(base/'paired-timing.json',timing);read_timing(base/'paired-timing.json')
report['timing_summary']=timing['summary'];report['timing_precision_status']=timing['precision_status']
report['ruling15']=dict(requested_resident_batch=128,preparation_assertions=True,F_rows=2306,
    actual_chunk_values=sorted(set(observed)),actual_chunk_count=len(observed),boundary='resident standardized float32 kernel excludes host transforms/inversion; host includes complete per-policy batching')
report['cold']=measure_cold_predictor(models['top'],X,host,root/'cold-top',timeout_seconds=120.)
report.update(enclosing_ns=time.perf_counter_ns()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write_json(base/'loss-timing.json',report);print('completed row/timing precision',report['E_choice']['candidate_rows'],flush=True)
