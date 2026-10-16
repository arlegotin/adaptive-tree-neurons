"""Executable B7 independent vendor parity and full-size bounded cost gate."""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types
import numpy as np
import torch
from treeneuron.baselines.node import NodePredictor, node_identity

TOLERANCE=dict(rtol=1e-6,atol=1e-7)


def vendor_reference():
    root=Path(__file__).resolve().parents[3]
    vendor=root/'configs/vendor/node'
    for record in json.loads((vendor/'sources.json').read_text()):
        path=vendor/'upstream'/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'Archived vendor integrity mismatch: {path}')
    # Bypass broad legacy lib.__init__; load the unchanged numerical files.
    # These site packages exist only for utils.py import-time requests/tqdm.
    sys.path.insert(0,str(root/'environment/node-reference-site'))
    package=types.ModuleType('node_gate_independent_vendor')
    package.__path__=[str(vendor/'upstream/lib')];sys.modules[package.__name__]=package
    try:
        nn=importlib.import_module(package.__name__+'.nn_utils')
        arch=importlib.import_module(package.__name__+'.arch')
    finally:sys.path.pop(0)
    return nn,arch


def verify_node(directory: Path):
    directory.mkdir(parents=True,exist_ok=True)
    nn,arch=vendor_reference()
    fixture=NodePredictor(dict(recipe='engineering_fixture',trees=5,depth=2,batch_size=8,report_frequency=2))
    x=torch.tensor(np.random.RandomState(39).normal(size=(1000,4)),dtype=torch.float32)
    np.random.seed(1337);torch.manual_seed(1337)
    model=fixture.build_model(4)
    with torch.no_grad():actual=model(x)
    np.random.seed(1337);torch.manual_seed(1337)
    reference=arch.DenseBlock(4,5,1,tree_dim=3,depth=2,flatten_output=False,
                              choice_function=nn.entmax15,bin_function=nn.entmoid15)
    with torch.no_grad():expected=reference(x)[...,0].mean(-1)
    torch.testing.assert_close(actual,expected,**TOLERANCE)
    model(x).square().mean().backward();reference(x)[...,0].mean(-1).square().mean().backward()
    for a,b in zip(model[0].parameters(),reference.parameters()):
        torch.testing.assert_close(a,b,**TOLERANCE)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,**TOLERANCE)
    for f,g,values,kwargs in (
        (model[0][0].choice_function,nn.entmax15,torch.arange(60.).reshape(3,5,4).permute(2,1,0),{'dim':0}),
        (model[0][0].bin_function,nn.entmoid15,torch.tensor([-3.,-2.,-1.999,0.,1.999,2.,3.]),{})):
        a=values.clone().requires_grad_();b=values.clone().requires_grad_()
        first=f(a,**kwargs);second=g(b,**kwargs)
        torch.testing.assert_close(first,second,**TOLERANCE)
        weights=torch.sin(torch.arange(first.numel()).reshape(first.shape))
        (first*weights).sum().backward();(second*weights).sum().backward()
        torch.testing.assert_close(a.grad,b.grad,**TOLERANCE)
    assert not torch.allclose(nn.entmoid15(torch.tensor([-3.,3.])),torch.sigmoid(torch.tensor([-3.,3.])),**TOLERANCE)
    output=directory/'default-cost.json'
    command=[sys.executable,'-m','treeneuron.baselines.node_worker','--output',str(output),'--directory',str(directory/'anchor')]
    t=time.perf_counter();run=subprocess.run(command,capture_output=True,text=True)
    (directory/'worker.stdout.txt').write_text(run.stdout);(directory/'worker.stderr.txt').write_text(run.stderr)
    if run.returncode:raise RuntimeError(f'NODE default-cost worker exit {run.returncode}: {run.stderr[-2000:]}')
    anchor=json.loads(output.read_text())
    if anchor['identity']!=node_identity():raise ValueError('Worker identity mismatch')
    if anchor['status']!='passed':raise RuntimeError(str(anchor))
    return dict(status='passed',identity=node_identity(),parity_tolerance=TOLERANCE,
        independent_reference='immutable archived upstream modules, separately imported; numerical modules unchanged',
        independent_reference_max_abs=float((actual-expected).abs().max()),
        sigmoid_counterexample_rejected=True,anchor=anchor['resources'],cost_worker=anchor,
        startup_inclusive_worker_seconds=time.perf_counter()-t,
        scope='synthetic correctness and bounded full-size cost; no YEAR score or default plateau claim')
