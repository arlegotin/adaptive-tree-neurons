"""Actual CPU correctness receipts for each causal topology, never throughput claims.

The reference evaluates separate affine leaves with explicit per-unit reductions.
A second optimizer consumes its gradients. Save/load checks include optimizer
moments and a subsequent update. This diagnostic is separate from study timing.
"""
from __future__ import annotations
import copy
import io
import math
import torch
from treeneuron.config import canonical_hash


def _reference(model, x):
    p = dict(model.named_parameters())
    def affine(name):
        # Per-row reductions provide an independent path from the batched GEMM.
        return torch.stack([(x*w).sum(-1)+b for w,b in zip(p[name+'.weight'],p[name+'.bias'])],dim=1)
    layer = model.layer
    if model.export_config()['kind'] == 'widened_relu_scalar':
        old = torch.relu(affine('layer.base'))
        result = (old*p['head.base.weight']).sum(-1)+p['head.base.bias'][0]
        if layer.added is not None:
            result = result+(torch.relu(affine('layer.added'))*p['head.added.weight']).sum(-1)
        return result
    if model.export_config()['kind'] == 'selective_relu_scalar':
        z = affine('layer.base')
        if layer.grown_ids.numel():
            gate = torch.sigmoid(affine('layer.gate')/layer.tau)
            a = z[:,layer.grown_ids]; q = affine('layer.contrast')
            z = z.index_copy(1,layer.grown_ids,gate*(a+q)+(1-gate)*(a-q))
    else:
        parts=[]
        if layer.count:
            gate=torch.sigmoid(affine('layer.gate')/layer.tau)
            if layer.coordinates=='centered':
                a,q=affine('layer.center'),affine('layer.contrast')
                parts.append(gate*(a+q)+(1-gate)*(a-q))
            else:
                parts.append(gate*affine('layer.left')+(1-gate)*affine('layer.right'))
        if layer.ordinary is not None: parts.append(affine('layer.ordinary'))
        z=torch.cat(parts,dim=1)
    return (torch.relu(z)*p['head.weight']).sum(-1)+p['head.bias'][0]


def probe_causal_cpu(model, *, batch_size=128):
    from treeneuron.train import capture_rng, restore_rng, source_hash
    if (type(batch_size) is not int or not 0 < batch_size <= 128
            or any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in model.parameters())):
        raise ValueError('Causal probe requires CPU float32 and batch <=128')
    before=capture_rng(); threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        # Probe independent copies, preserving the actual caller tensors and RNG.
        a,b=copy.deepcopy(model),copy.deepcopy(model)
        g=torch.Generator().manual_seed(1602041)
        x=torch.randn(batch_size,model.layer.d,generator=g,dtype=torch.float32)
        y=torch.randn(batch_size,generator=g,dtype=torch.float32)
        opts=dict(lr=.001,weight_decay=.01,foreach=False)
        oa,ob=torch.optim.AdamW(a.parameters(),**opts),torch.optim.AdamW(b.parameters(),**opts)
        out,ref=a(x),_reference(b,x)
        torch.testing.assert_close(out,ref,rtol=2e-5,atol=2e-6)
        ((out-y)**2).mean().backward(); ((ref-y)**2).mean().backward()
        grad_error=max(float((p.grad-q.grad).abs().max()) for p,q in zip(a.parameters(),b.parameters()))
        for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)
        oa.step(); ob.step()
        optimizer_error=max(float((p-q).abs().max().detach()) for p,q in zip(a.parameters(),b.parameters()))
        for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-6)
        payload=io.BytesIO(); torch.save({'model':a.state_dict(),'optimizer':oa.state_dict()},payload)
        payload.seek(0); saved=torch.load(payload,weights_only=True)
        c=type(a).from_config(a.export_config()).to(dtype=torch.float32); c.load_state_dict(saved['model'],strict=True)
        oc=torch.optim.AdamW(c.parameters(),**opts); oc.load_state_dict(saved['optimizer'])
        for candidate,optimizer in ((a,oa),(c,oc)):
            optimizer.zero_grad(); ((candidate(x)-y)**2).mean().backward(); optimizer.step()
        for p,q in zip(a.parameters(),c.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            for key in ('step','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(oa.state[p][key],oc.state[q][key],rtol=0,atol=0)
        return dict(schema_version=1,status='passed',source_hash=source_hash(),topology=model.export_config(),
            parameter_shapes={n:list(p.shape) for n,p in model.named_parameters()},
            total_parameters=sum(p.numel() for p in model.parameters()),batch_size=batch_size,
            device='cpu',dtype='float32',execution='eager',seed=1602041,
            forward_max_abs=float((out-ref).abs().max().detach()),gradient_max_abs=grad_error,
            optimizer_max_abs=optimizer_error,reload_next_update_max_abs=0.,scope='forward/backward/AdamW/serialization correctness only; no throughput measurement')
    finally:
        torch.set_num_threads(threads); restore_rng(before)


def attach_causal_evidence(backend, model, batch_size):
    result=copy.deepcopy(backend)
    if result['device']!='cpu': raise ValueError('Causal controls are verified only on selected CPU')
    result['causal_cpu_evidence']=probe_causal_cpu(model,batch_size=batch_size)
    result['performance_calibration_matches']=False
    result['performance_scope']='Actual causal topology CPU correctness; performance uncalibrated'
    result['workload']['h']=model.layer.h
    result['workload']['coordinate_family']=model.export_config()['kind'] if not hasattr(model.layer,'coordinates') else model.layer.coordinates
    result['workload']['actual_growth_count']=len(model.layer.grown_ids) if hasattr(model.layer,'grown_ids') else getattr(model.layer,'count',model.layer.h-getattr(model.layer,'parent_width',model.layer.h))
    return result


def validate_causal_evidence(backend, topology, config):
    from treeneuron.train import source_hash
    e=backend.get('causal_cpu_evidence',{})
    if not isinstance(e,dict): raise ValueError('Malformed causal CPU evidence')
    if (e.get('status')!='passed' or e.get('source_hash')!=source_hash() or e.get('topology')!=topology
            or e.get('batch_size')!=config.training.batch_size or e.get('device')!='cpu'
            or e.get('dtype')!='float32' or e.get('execution')!='eager' or backend.get('device')!='cpu'
            or backend.get('performance_calibration_matches') is not False
            or not 0 < config.model.d <=16 or not 0 < config.model.h <=95 or config.data.n_fit>2306):
        raise ValueError('Missing or mismatched actual causal CPU evidence')

    required={'schema_version','status','source_hash','topology','parameter_shapes','total_parameters',
              'batch_size','device','dtype','execution','seed','forward_max_abs','gradient_max_abs',
              'optimizer_max_abs','reload_next_update_max_abs','scope'}
    if set(e)!=required or e['schema_version']!=1 or e['seed']!=1602041:
        raise ValueError('Malformed causal CPU evidence schema')
    for key in ('forward_max_abs','gradient_max_abs','optimizer_max_abs','reload_next_update_max_abs'):
        if type(e[key]) not in (float,int) or not math.isfinite(e[key]) or not 0<=e[key]<=1e-4:
            raise ValueError('Failed causal CPU numerical evidence')
    if e['reload_next_update_max_abs']!=0:
        raise ValueError('CPU causal replay must be exact')
    d,h=config.model.d,config.model.h
    shapes={}
    def linear(name,rows,columns,bias=True):
        shapes[name+'.weight']=[rows,columns]
        if bias: shapes[name+'.bias']=[rows]
    kind=topology.get('kind')
    if kind=='widened_relu_scalar':
        old=config.model.parent_width
        linear('layer.base',old,d);linear('head.base',1,old)
        if h>old: linear('layer.added',h-old,d);linear('head.added',1,h-old,False)
    elif kind=='static_gated_relu_scalar':
        m=config.model.static_count
        if m<h: linear('layer.ordinary',h-m,d)
        if m:
            for name in ('gate','center','contrast') if config.model.coordinates=='centered' else ('gate','left','right'):
                linear('layer.'+name,m,d)
        linear('head',1,h)
    elif kind=='selective_relu_scalar':
        linear('layer.base',h,d);linear('head',1,h)
        m=len(topology['layer']['grown_ids'])
        if m: linear('layer.gate',m,d);linear('layer.contrast',m,d)
    else: raise ValueError('Unknown causal coordinate topology')
    if e['parameter_shapes']!=shapes or e['total_parameters']!=sum(math.prod(s) for s in shapes.values()):
        raise ValueError('Causal evidence parameter shapes/count mismatch')
