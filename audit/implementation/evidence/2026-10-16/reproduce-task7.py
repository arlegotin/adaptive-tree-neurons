"""Bounded engineering search/timing receipt; never a quality/default plateau.

Run from the repo with the locked search overlay installed. The explicit
bootstrap watchdog is a process guard, not a proposed study budget. Following
allocations double the actually observed anchor enclosure. A new output root is
required, and all predictors are trained from current-source fresh parents.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from treeneuron.artifacts import create_run,_atomic_json
from treeneuron.backend import _resources,_guard,implementation_hash,lock_hash
from treeneuron.config import resolve_experiment
from treeneuron.data.types import DevelopmentData,ObservationSet
from treeneuron.evaluation.search import SearchBudget,search,produce_search_backend_receipt,file_hash,expansion_diagnostics
from treeneuron.evaluation.timing import TimingPolicy,measure_predictor,measure_cold_predictor,write_timing,read_timing,ResidentCausalPredictor
from treeneuron.train import source_hash


def data_fixture(width=16,fit=128,growth=32,validation=64):
    rng=np.random.default_rng(1602047)
    def rows(n,start):
        X=rng.normal(size=(n,width));ids=np.arange(start,start+n,dtype=np.int64)
        return ObservationSet(X,7+2*X[:,0]-X[:,1]+.1*rng.normal(size=n),ids,ids)
    return DevelopmentData(rows(fit,0),rows(growth,10000),rows(validation,20000),'task7-engineering-no-population-claim')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap-watchdog-seconds',type=float,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    resources=_resources(torch);_atomic_json(root/'resources-before.json',resources);_guard(resources)
    receipt=root/'search-backend.json';produce_search_backend_receipt(receipt)
    data=data_fixture()
    config=resolve_experiment({'study_id':'task7-engineering','model':{'d':16,'h':64},
        'data':{'n_fit':128,'n_growth':32,'n_validation':64},
        'training':{'batch_size':128,'base_updates':2,'continuation_updates':2,'validation_interval':1,'patience':10},
        'backend':{'cpu_threads':1,'interop_threads':1}})
    records=[]
    def run(method,budget,configuration=config,development=data):
        paths=create_run(root/'search-runs',configuration.to_dict())
        result=search(method,development,budget,1602047,paths)
        records.append(dict(method=method,budget=asdict(budget),run_id=paths.run_id,run_directory=str(paths.directory),
            charged_ns=result.charged_ns,winner=result.winner,trials=[{k:t[k] for k in ('number','official_anchor','status','validation_mse','charged_ns','error')} for t in result.trials]))
        _atomic_json(root/f'search-{len(records):02d}.json',records[-1])
        return result,paths
    bootstrap,_=run('Ridge',SearchBudget(seconds=args.bootstrap_watchdog_seconds))
    if bootstrap.winner is None:raise RuntimeError('Bootstrap anchor unavailable; no forecast extrapolation')
    low=bootstrap.charged_ns/1e9*2
    curves=[]
    ridge=None
    for seconds in (low,2*low):
        ridge,paths=run('Ridge',SearchBudget(seconds=seconds,max_trials=8))
        curves.append(dict(seconds=seconds,best_validation_mse=None if ridge.winner is None else ridge.trials[ridge.winner]['validation_mse']))
    diagnostic=expansion_diagnostics(ridge.trials,ridge.winner,{'alpha':{'kind':'float','low':1e-6,'high':1e6,'log':True}},curves)
    last_causal=None
    for method,recipe in [('B1',{}),('B3',{'variant':'partial','count':16,'coordinates':'centered'}),('B5',{'variant':'adam'})]:
        result,paths=run(method,SearchBudget(seconds=2*low,max_trials=2,recipe=recipe,backend_report=str(receipt)))
        if method=='B5' and result.winner is not None:last_causal=paths.directory/f'trial-{result.winner:04d}'/'predictor.json'
    # The short B7 deadline verifies process-timeout exclusion, without starting
    # another full-size training run or extrapolating its quality plateau.
    run('B7',SearchBudget(seconds=min(.01,low/100),max_updates=3))
    from treeneuron.baselines.simple import RidgePredictor
    from treeneuron.baselines.api import FitRequest
    model=RidgePredictor(1.).fit(FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,1602047,{}, {},root))
    X=np.random.default_rng(7).normal(size=(257,16))
    policy=TimingPolicy(batch_sizes=(1,16,32,64,128),warmup_window=5,max_warmup=20,min_repetitions=20,max_repetitions=80,
        relative_precision=.1,block_order='abba')
    reference=model.predict(X)
    ridge_parity={}
    for batch in policy.batch_sizes:
        prediction=np.concatenate([model.predict(X[i:i+batch],batch_size=batch) for i in range(0,len(X),batch)])
        np.testing.assert_allclose(prediction,reference,rtol=1e-12,atol=1e-12)
        ridge_parity[str(batch)]=float(np.max(np.abs(prediction-reference)))
    timing=measure_predictor(model,X,policy);write_timing(root/'ridge-timing.json',timing)
    rebuilt=read_timing(root/'ridge-timing.json')
    _atomic_json(root/'ridge-summary-rebuilt.json',rebuilt['summary'])
    cold=measure_cold_predictor(model,X,policy,root/'ridge-cold')
    causal_timing=None
    if last_causal is not None:
        from treeneuron.baselines.causal import CausalPredictor
        pointer=json.loads(last_causal.read_text());descriptor=json.loads((last_causal.parent/pointer['metadata']).read_text())
        causal=CausalPredictor.load(last_causal,expected_hashes=descriptor['state']['identity_hashes'])
        causal_timing={}
        causal_reference=causal.predict(X)
        for mode,predictor in [('host_to_host',causal),('resident_kernel',ResidentCausalPredictor(causal))]:
            cp=TimingPolicy(**{**asdict(policy),'mode':mode,'block_order':'balanced_random'})
            parity={}
            for batch in cp.batch_sizes:
                if mode=='resident_kernel':
                    prepared=predictor.prepare_timing_input(X)
                    with torch.inference_mode():prediction=torch.cat([predictor.predict_resident(prepared[i:i+batch]) for i in range(0,len(X),batch)]).cpu().numpy()
                    prediction=causal.state.preprocessing.inverse_y(prediction)
                else:prediction=np.concatenate([predictor.predict(X[i:i+batch],batch_size=batch) for i in range(0,len(X),batch)])
                np.testing.assert_allclose(prediction,causal_reference,rtol=2e-5,atol=2e-6)
                parity[str(batch)]=float(np.max(np.abs(prediction-causal_reference)))
            result=measure_predictor(predictor,X,cp);write_timing(root/f'causal-{mode}.json',result)
            causal_timing[mode]=dict(batch_parity_max_abs=parity,parity_tolerance={'rtol':2e-5,'atol':2e-6},summary=result['summary'],precision_status=result['precision_status'],warmup_stable=result['warmup_stable'])
    report=dict(schema_version=1,scope='bounded synthetic engineering search/timing; not H1 pilot, quality plateau or final WP05 policy',
        status='evidence_pending_WP05',source_sha256=source_hash(),implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        core_lock_sha256=lock_hash(),backend_producer_sha256=implementation_hash(),receipt_sha256=file_hash(receipt),
        bootstrap_watchdog_seconds=args.bootstrap_watchdog_seconds,observed_bootstrap_ns=bootstrap.charged_ns,
        exploratory_doubled_seconds=[low,2*low],searches=records,expansion_diagnostics=diagnostic,
        ridge_timing=dict(batch_parity_max_abs=ridge_parity,parity_tolerance={'rtol':1e-12,'atol':1e-12},summary=timing['summary'],warmup_stable=timing['warmup_stable'],precision_status=timing['precision_status']),
        timing_summary_reconstructed=True,cold=cold,causal_timing=causal_timing,
        final_policy=None,remaining_gates=['full-quality/default plateaus','variance pilot after Tasks8/9/10/13 implementation','final numeric WP05 design','source freeze','protected independent cohort'])
    _atomic_json(root/'calibration.json',report)
    _atomic_json(root/'resources-after.json',_resources(torch))
    print(json.dumps({'output':str(root),'searches':len(records),'source':report['source_sha256'],'bootstrap_ns':bootstrap.charged_ns,'doubled_seconds':[low,2*low],'ridge_batch':timing['summary']['selected_batch'],'precision_status':timing['precision_status'],'WP05':'pending'},indent=2))


if __name__=='__main__':main()
