"""Retain an artificial outer search with an interrupted actual-TPE suggestion."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from treeneuron.config import canonical_hash
from treeneuron.research.ledger import write_new_json
from treeneuron.research.search_controller_v2 import make_search_plan,run_search
from treeneuron.research.search_v2 import ReplayTPEV2
from treeneuron.train import source_hash


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter_ns();clock=[0];requests=[]
    runtime=dict(python=sys.version,platform=platform.platform())
    support=dict(lr=dict(kind='float',low=.001,high=.1,log=True))
    source=source_hash()
    identity=dict(job_id='artificial/P03',family_id='P03',search_repeat=0,source_sha256=source,
        runtime_sha256=canonical_hash(runtime),support_sha256=canonical_hash(support),split_sha256=canonical_hash(dict(no_data=True)))
    plan=make_search_plan(identity=identity,fit_seed=487,hpo_seed32=19,anchors=[dict(width=17)],
        mode='tpe',support=support,candidates=[],max_attempts=18,deadline_ns=1000,prefixes_ns=[120,160])
    def coordinator(request):
        requests.append(request);clock[0]+=10
        interrupted=request['candidate_id']=='tpe-12' and request['retry_index']==0
        failed=request['candidate_id']=='tpe-2'
        status='interrupted_recoverable' if interrupted else 'numerical_failure' if failed else 'completed_eligible'
        score=None if interrupted or failed else .5 if request['candidate_origin']=='anchor' else (request['config']['lr']-.03)**2
        path=Path(request['attempt_directory'])/'artificial-result.json'
        write_new_json(path,dict(artificial_only=True,model_fitted=False,config=request['config'],fit_seed=request['fit_seed']))
        ref=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return dict(status=status,V_score=score,files=[ref],checkpoint_files=[ref] if interrupted else [],
            reason=None if status=='completed_eligible' else status)
    first=run_search(output/'search',plan=plan,evaluate_trial=coordinator,clock_ns=lambda:clock[0],invocation_preparation_ns=7)
    write_new_json(output/'first-invocation.json',first)
    clock[0]+=1000000
    second=run_search(output/'search',plan=plan,evaluate_trial=coordinator,clock_ns=lambda:clock[0],
        resume=True,expected_journal_hash=first['journal_hash'],invocation_preparation_ns=11)
    write_new_json(output/'second-invocation.json',second)
    reference=ReplayTPEV2(seed=19,support=support,identity=identity,max_suggestions=17)
    for request in requests:
        if request['candidate_origin']!='tpe' or request['retry_index']!=0:continue
        assert reference.ask()==request['config']
        reference.tell(None if request['candidate_id']=='tpe-2' else (request['config']['lr']-.03)**2)
    assert reference.history==second['sampler_history']
    assert second['attempts_paid']==18 and second['total_charged_ns']==198
    assert first['stop_reason']=='interrupted_recoverable' and second['stop_reason']=='attempt_limit'
    assert requests[13]['config']==requests[14]['config'] and requests[14]['checkpoint_files']
    record=dict(schema='research/v2-artificial-search-controller-proof',source_sha256=source,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runtime_metadata=runtime,
        argv=sys.argv,cwd=str(Path.cwd()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        real_models_fitted=0,scientific_teachers=0,real_V_targets_read=0,protected_access=False,
        attempts_paid=second['attempts_paid'],artificial_charged_ns=second['total_charged_ns'],
        paid_failure_count=sum(row['status']!='completed_eligible' for row in second['records']),
        adaptive_observations=len(second['sampler_history']),exact_reference_history=True,
        same_seed_and_pending_candidate_recovery=True,off_worker_pause_excluded=True,
        observed_proof_process_body_ns=time.perf_counter_ns()-started,
        qualification_scope='Actual pinned TPE with artificial objective/files/clock; no native worker or full-process budget qualification')
    write_new_json(output/'proof.json',record)
    print(json.dumps(record,sort_keys=True))


if __name__=='__main__':main()
