"""Retain artificial R2 teacher risks and both prospective uncertainty outputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from treeneuron.analysis.design_v2 import DESIGN_ALTERNATIVE
from treeneuron.analysis.primary_v2 import analyze_teacher_risks,holm_family
from treeneuron.research.ledger import write_new_json
from treeneuron.train import source_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    teachers=[f'artificial-{i}' for i in range(32)];rows=[]
    for index,name in enumerate(teachers):
        effect=DESIGN_ALTERNATIVE+.02*math.sin(index)
        for repeat,risk in enumerate((1.,3.)):
            for role,value in [('selected',risk),('random',risk*math.exp(effect))]:
                rows.append(dict(teacher_id=name,repetition=repeat,role=role,risk=value,status='completed_eligible'))
    write_new_json(args.output/'risks.json',dict(planned_teacher_ids=teachers,records=rows))
    result=analyze_teacher_risks(teachers,rows,seed=2026101901)
    draws=result.pop('bootstrap_mean_log_benefits')
    np.savez_compressed(args.output/'draws.npz',bootstrap_mean_log_benefits=draws)
    result.update(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=source_hash(),argv=sys.argv,cwd=str(Path.cwd()),
        scientific_teacher_count=0,models_fit=0,T_accessed=False,
        evidence_level='Artificial arithmetic fixture; no measured efficacy or nominal-coverage validation',
        bootstrap_draws_file_sha256=hashlib.sha256((args.output/'draws.npz').read_bytes()).hexdigest(),
        holm_manual_fixture=holm_family(dict(a=.001,b=.01,c=.9),planned_hypotheses=['a','b','c']))
    write_new_json(args.output/'result.json',result)
    print(json.dumps({key:result[key] for key in ('status','teacher_count','mean_log_benefit',
        'student_t_interval','bootstrap_interval','useful_margin_decision','scientific_teacher_count',
        'source_sha256','claim_authorized')},sort_keys=True))


if __name__=='__main__':main()
