"""Retain actual pinned TPE replay on a wholly artificial objective."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from treeneuron.research.ledger import write_new_json
from treeneuron.research.search_v2 import ReplayTPEV2
from treeneuron.train import source_hash


def objective(params):
    return float((params['x']-.11)**2+(params['width']-96)**2/10000)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    identity=dict(study='artificial-TPE-replay',source_sha256=source_hash())
    support=dict(x=dict(kind='float',low=.001,high=1.,log=True),
        width=dict(kind='integer',low=64,high=128,step=16,log=False))
    control=ReplayTPEV2(seed=871,support=support,identity=identity)
    for index in range(18):
        params=control.ask();control.tell(None if index in (2,9) else objective(params))
    pending=control.ask();ref=control.save(args.output/'pending-state.json')
    resumed=ReplayTPEV2.load(args.output/'pending-state.json',expected_sha256=ref['sha256'],identity=identity)
    assert resumed.pending_suggestion==pending
    control.tell(objective(pending));resumed.tell(objective(pending))
    expected=control.ask();actual=resumed.ask();assert expected==actual
    control.tell(objective(expected));resumed.tell(objective(actual));assert control.history==resumed.history
    final=resumed.save(args.output/'resumed-state.json')
    result=dict(schema='research/v2-tpe-replay-proof',argv=sys.argv,cwd=str(Path.cwd()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256=source_hash(),
        identity=identity,support=support,initial_state=ref,final_state=final,
        paid_suggestion_count=20,failed_suggestions=2,pending_suggestion_reproduced=pending,
        uninterrupted_next_suggestion=expected,resumed_next_suggestion=actual,
        complete_histories_identical=True,model_fits=0,labels_read=[],T_accessed=False,
        execution_authorized=False,evidence_level='Actual pinned sampler replay on an artificial scalar objective')
    write_new_json(args.output/'result.json',result)
    print(json.dumps({k:result[k] for k in ('source_sha256','paid_suggestion_count','failed_suggestions',
        'complete_histories_identical','model_fits','T_accessed')},sort_keys=True))


if __name__=='__main__':main()
