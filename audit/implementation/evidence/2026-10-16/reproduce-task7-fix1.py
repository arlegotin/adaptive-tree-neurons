"""One bounded method-block comparison and costed-resume lineage receipt.

Explicit deadlines exercise availability, not quality/default budget selection.
A new output directory and fresh fitted models are required for every execution.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from treeneuron.artifacts import create_run,_atomic_json
from treeneuron.backend import _resources,_guard,validate_backend_report,implementation_hash,lock_hash,runtime_hash
from treeneuron.config import resolve_experiment
from treeneuron.data.types import DevelopmentData,ObservationSet
from treeneuron.evaluation.search import SearchBudget,search,file_hash,_load_costed_resume
from treeneuron.evaluation.timing import TimingPolicy,measure_comparison,write_timing,read_timing,rebuild_comparison
from treeneuron.baselines.simple import MeanPredictor,RidgePredictor
from treeneuron.baselines.api import FitRequest
from treeneuron.train import source_hash


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--startup-deadline',type=float,required=True);parser.add_argument('--attempt-watchdog',type=float,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    resources=_resources(torch);_atomic_json(root/'resources-before.json',resources);_guard(resources)
    receipt=Path('environment/evidence7-calibration-fix1/search-backend.json')
    numerical=json.loads(receipt.read_text())['backend_probes']['probes']['cpu'];validate_backend_report(numerical)
    assert numerical['environment']['threads']==numerical['environment']['interop_threads']==1
    rng=np.random.default_rng(1602047)
    def rows(n,start):
        X=rng.normal(size=(n,2));ids=np.arange(start,start+n)
        return ObservationSet(X,4+X[:,0],ids,ids)
    data=DevelopmentData(rows(32,0),rows(8,100),rows(16,200),'task7-fix1-engineering')
    config=resolve_experiment({'study_id':'task7-fix1-engineering','model':{'d':2,'h':4},
        'data':{'n_fit':32,'n_growth':8,'n_validation':16},'backend':{'cpu_threads':1,'interop_threads':1}})
    first_paths=create_run(root/'runs',config.to_dict())
    first=search('Ridge',data,SearchBudget(seconds=args.startup_deadline),11,first_paths)
    _atomic_json(root/'first-search.json',asdict(first))
    if len(first.trials)!=1 or first.trials[0]['status']!='unavailable_at_budget':
        raise RuntimeError('Deadline did not exercise an unavailable official trial; retain actual result without claiming failure-lineage evidence')
    original=file_hash(first.sampler_state)
    second_paths=create_run(root/'runs',config.to_dict())
    second=search('Ridge',data,SearchBudget(seconds=args.attempt_watchdog,resume_sampler=first.sampler_state),11,second_paths)
    _atomic_json(root/'resumed-search.json',asdict(second))
    assert second.winner==0 and len(second.prior_trials)==1 and second.prior_trials[0]['official_anchor']
    assert second.cumulative_charged_ns==first.attempt_charged_ns+second.attempt_charged_ns
    assert file_hash(first.sampler_state)==original
    _,prior,cost,attempts=_load_costed_resume(second.sampler_state,second.identity)
    assert list(prior)==list(second.all_trials) and cost==second.cumulative_charged_ns and len(attempts)==2
    try:_load_costed_resume(first_paths.directory/'sampler.final.json',first.identity)
    except ValueError as exc:raw_rejection=str(exc)
    else:raise AssertionError('Uncosted standalone sampler was accepted')
    request=FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,11,{}, {},root)
    mean=MeanPredictor().fit(request);ridge=RidgePredictor(1.).fit(request)
    mean.save(root/'mean.json');ridge.save(root/'ridge.json')
    X=rng.normal(size=(257,2))
    policy=TimingPolicy(batch_sizes=(64,),warmup_window=2,max_warmup=4,min_repetitions=4,max_repetitions=8,block_order='abba')
    result=measure_comparison({'Mean/host':(mean,policy),'Ridge/host':(ridge,policy)},X)
    write_timing(root/'comparison.json',result);rebuilt=read_timing(root/'comparison.json')
    _atomic_json(root/'comparison-rebuilt.json',rebuilt['summary'])
    assert not result['summary']['precision_reached']  # four complete blocks cannot assess p95
    duplicates=result['raw_blocks']+result['raw_blocks']*10
    assert rebuild_comparison(duplicates,result['policies'])==result['summary']
    block=result['summary']['complete_block_ids'][0]
    one=[r for r in result['raw_blocks'] if r['block']==block]
    assert not rebuild_comparison(one*20,result['policies'])['precision_reached']
    record=dict(schema_version=1,scope='bounded engineering method blocks and failed-anchor cost lineage, not WP05 calibration',
        source_sha256=source_hash(),implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        core_lock_sha256=lock_hash(),backend_producer_sha256=implementation_hash(),runtime_hash=runtime_hash(),
        applicable_cpu_receipt={'path':str(receipt),'sha256':file_hash(receipt),'threads':{'cpu':1,'interop':1}},
        first_attempt_ns=first.attempt_charged_ns,resumed_attempt_ns=second.attempt_charged_ns,cumulative_ns=second.cumulative_charged_ns,
        prior_anchor_status=second.prior_trials[0]['status'],prior_anchor_error=second.prior_trials[0]['error'],
        raw_sampler_rejection=raw_rejection,verified_attempts=len(attempts),historical_models_promoted=False,
        timing_summary=rebuilt['summary'],timing_precision_status=rebuilt['precision_status'],
        within_block_duplication_invariant=True,single_block_rejected_for_precision=True,raw_summary_rebuilt=True,
        final_WP05_policy=None)
    _atomic_json(root/'fix1.json',record);_atomic_json(root/'resources-after.json',_resources(torch))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
