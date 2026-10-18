"""Check fixed-count model-family relabeling; no fitting or efficacy experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from treeneuron.train import SelectiveRegressor,source_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter_ns();records=[]
    for fixture in range(32):
        torch.manual_seed(2026101805+fixture)
        d,h,m=16,64,16
        source=SelectiveRegressor(d,h).double()
        ids=torch.arange(m,dtype=torch.int64)
        source.layer.grow(ids,torch.randn(m,d+1,dtype=torch.float64))
        with torch.no_grad():
            source.layer.gate.weight.normal_();source.layer.gate.bias.normal_()
        permutation=torch.randperm(h);destination_ids=torch.nonzero(permutation<m).flatten()
        source_rows=permutation[destination_ids]
        destination=SelectiveRegressor(d,h).double()
        contrasts=torch.cat((source.layer.contrast.weight,source.layer.contrast.bias[:,None]),dim=1).detach()
        destination.layer.grow(destination_ids,contrasts[source_rows])
        with torch.no_grad():
            destination.layer.base.weight.copy_(source.layer.base.weight[permutation])
            destination.layer.base.bias.copy_(source.layer.base.bias[permutation])
            destination.head.weight.copy_(source.head.weight[:,permutation]);destination.head.bias.copy_(source.head.bias)
            destination.layer.gate.weight.copy_(source.layer.gate.weight[source_rows])
            destination.layer.gate.bias.copy_(source.layer.gate.bias[source_rows])
        X=torch.randn(257,d,dtype=torch.float64);y=torch.randn(257,dtype=torch.float64)
        a,b=source(X),destination(X)
        discrepancy=float((a-b).abs().max().detach())
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        ((a-y)**2).mean().backward();((b-y)**2).mean().backward()
        pairs=[(source.layer.base.weight.grad[permutation],destination.layer.base.weight.grad),
               (source.layer.base.bias.grad[permutation],destination.layer.base.bias.grad),
               (source.head.weight.grad[:,permutation],destination.head.weight.grad),
               (source.head.bias.grad,destination.head.bias.grad)]
        for name in ('gate','contrast'):
            old,new=getattr(source.layer,name),getattr(destination.layer,name)
            pairs.extend(((old.weight.grad[source_rows],new.weight.grad),(old.bias.grad[source_rows],new.bias.grad)))
        for a,b in pairs:torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)
        records.append(dict(fixture=fixture,seed=2026101805+fixture,source_ids=ids.tolist(),
            destination_ids=destination_ids.tolist(),permutation=permutation.tolist(),
            parameters=sum(p.numel() for p in source.parameters()),
            destination_parameters=sum(p.numel() for p in destination.parameters()),
            maximum_prediction_error=discrepancy,
            maximum_gradient_error=max(float((a-b).abs().max()) for a,b in pairs)))
    result=dict(schema='research/v2-fixed-count-model-relabeling',source_sha256=source_hash(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),argv=sys.argv,cwd=str(Path.cwd()),
        torch_version=torch.__version__,dtype='float64',threads=1,fixtures=records,
        elapsed_ns=time.perf_counter_ns()-started,training_updates=0,independent_scientific_teachers=0,
        inference='Fixed-count unrestricted model families are related by hidden-unit relabeling',
        limitations=['This permutes the entire parent, head and grown blocks; it does not equate different allocations on one fixed parent',
            'No optimizer trajectory, finite-training ranking or predictive advantage is tested'],T_accessed=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(fixtures=len(records),prediction_error=max(r['maximum_prediction_error'] for r in records),
        gradient_error=max(r['maximum_gradient_error'] for r in records),parameters=records[0]['parameters'])))


if __name__=='__main__':main()
