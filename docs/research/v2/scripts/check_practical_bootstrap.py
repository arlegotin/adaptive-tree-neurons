"""Retain an artificial paired-bootstrap proof; no fitted or protected data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from treeneuron.analysis.practical_v2 import paired_group_bootstrap

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=Path(__file__).resolve().parents[4];args.output.mkdir(parents=True,exist_ok=False)
packets={};inputs={}
for name in ('airfoil','concrete','ccpp','parkinsons','bike','energy'):
    count=10 if name=='bike' else 3
    baseline=np.repeat(np.arange(1,count+1,dtype=float)[:,None],2,axis=1)
    packet=dict(group_ids=[str(i) for i in range(count)],selected=baseline/2,baseline=baseline)
    if name=='bike':packet['day_ordinals']=np.arange(100,110)
    packets[name]=packet
    inputs[name]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in packet.items()}
result=paired_group_bootstrap(packets,seed=2026101802)
arrays={'aggregate':result.pop('aggregate_bootstrap_log_benefits')}
arrays.update({name:values for name,values in result.pop('dataset_bootstrap_log_benefits').items()})
with (args.output/'draws.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
assert np.max(np.abs(arrays['aggregate']-math.log(2)))<1e-14
assert result['iterations']==20000 and result['discarded_resamples']==0
assert result['population_superiority_confirmed'] is False
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(schema='research/v2-practical-bootstrap-artificial-proof',inputs=inputs,result=result,
    numpy=np.__version__,draws_sha256=sha(args.output/'draws.npz'),
    module_sha256=sha(root/'src/treeneuron/analysis/practical_v2.py'),
    script_sha256=sha(Path(__file__)),argv=sys.argv,executable=sys.executable,cwd=str(Path.cwd()),
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    independent_teachers=0,scientific_datasets_fit=0,T_accessed=False,
    proof_scope='Artificial constant risk ratio validates paired arithmetic/resampling retention, not coverage or predictive quality')
with (args.output/'result.json').open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
print(json.dumps(dict(iterations=result['iterations'],interval=result['aggregate_interval'],T_accessed=False)))
