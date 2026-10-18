"""Retain artificial nested timers and paired savings arithmetic, without claims."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from treeneuron.analysis.cost_v2 import attribute_costs,paired_timing_savings
from treeneuron.research.ledger import write_new_json
from treeneuron.research.seeds import seed64
from treeneuron.train import source_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    timers=[dict(timer_id='cohort',parent_id=None,start_ns=0,end_ns=110,requires=[]),
        dict(timer_id='parent',parent_id='cohort',start_ns=10,end_ns=40,requires=[]),
        dict(timer_id='selected',parent_id='cohort',start_ns=40,end_ns=95,requires=['parent']),
        dict(timer_id='score',parent_id='selected',start_ns=40,end_ns=45,requires=[]),
        dict(timer_id='random',parent_id='cohort',start_ns=95,end_ns=110,requires=['parent'])]
    procedures=dict(selected=['selected'],random=['random']);seed=2026101902
    rng=np.random.default_rng(seed64(seed,'paired-deployment-timing','order'))
    blocks=[dict(block_id=i,order='selected_first' if order==0 else 'baseline_first',selected_ns=50,baseline_ns=100)
        for i,order in enumerate(rng.integers(0,2,size=30))]
    write_new_json(args.output/'inputs.json',dict(timers=timers,procedures=procedures,blocks=blocks))
    paired=paired_timing_savings(blocks,seed=seed,quality_risk_ratio_upper=1.01)
    savings=paired.pop('bootstrap_savings_fractions');times=paired.pop('bootstrap_paired_time_savings_ns')
    np.savez_compressed(args.output/'draws.npz',savings_fraction=savings,paired_time_saving_ns=times)
    result=dict(schema='research/v2-cost-analysis-proof',
        attribution=attribute_costs(timers,procedures,cohort_timer='cohort'),paired=paired,
        scientific_deployment_blocks=0,models_timed=0,T_accessed=False,cost_claim_authorized=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256=source_hash(),
        argv=sys.argv,cwd=str(Path.cwd()),draws_sha256=hashlib.sha256((args.output/'draws.npz').read_bytes()).hexdigest(),
        evidence_level='Artificial timer/block values; no measured performance or coverage validation')
    write_new_json(args.output/'result.json',result)
    print(json.dumps(dict(physical_cohort_wall_ns=result['attribution']['physical_cohort_wall_ns'],
        standalone_attributed_sum_ns=result['attribution']['standalone_attributed_sum_ns'],
        artificial_savings_interval=paired['savings_fraction_interval'],scientific_deployment_blocks=0,
        cost_claim_authorized=False,source_sha256=result['source_sha256']),sort_keys=True))


if __name__=='__main__':main()
