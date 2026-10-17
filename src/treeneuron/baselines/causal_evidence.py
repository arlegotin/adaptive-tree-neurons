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


def _reference(model, x, *, mask=None, preactivations=None):
    p = dict(model.named_parameters())
    offset=0
    def relu(z):
        nonlocal offset
        if preactivations is not None: preactivations.append(z.detach())
        selected=None if mask is None else mask[:,offset:offset+z.shape[1]]
        offset+=z.shape[1]
        # Reference-only VJP: preserve the independently evaluated forward value.
        return torch.relu(z) if selected is None else torch.relu(z).detach()+(z-z.detach())*selected
    def affine(name):
        # Per-row reductions provide an independent path from the batched GEMM.
        return torch.stack([(x*w).sum(-1)+b for w,b in zip(p[name+'.weight'],p[name+'.bias'])],dim=1)
    layer = model.layer
    if model.export_config()['kind'] == 'widened_relu_scalar':
        old = relu(affine('layer.base'))
        result = (old*p['head.base.weight']).sum(-1)+p['head.base.bias'][0]
        if layer.added is not None:
            result = result+(relu(affine('layer.added'))*p['head.added.weight']).sum(-1)
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
    return (relu(z)*p['head.weight']).sum(-1)+p['head.bias'][0]



def validate_relu_branches(model, x, actual, reference):
    """Certify differing ReLU masks only for independently bounded affine rows.

    For d products, their reduction and bias addition, k=2*d+2 elementary
    binary32 roundings conservatively cover fused or unfused implementations.
    |fl(a)-a| <= gamma_k * (sum |x_i*w_i| + |b|) plus k*tiny covers
    gradual underflow. The binary64 center gets its own gamma_k allowance.
    Both computed preactivations must lie in that interval and it must contain
    zero. Gated/contrast rows have no affine-only certificate: fail closed there.
    """
    different=(actual>0)!=(reference>0)
    result=dict(semantics='actual float32 ReLU branch; affine-rounding-certified reference only',
        ambiguous_count=int(different.sum()),mismatches=[],
        interval='gamma_(2d+2) absolute affine sum + binary64 center error + (2d+2)*tiny32; gated rows unsupported')
    if not different.any(): return result
    layer=model.layer
    kind=model.export_config()['kind']
    affine={}
    if kind=='widened_relu_scalar':
        affine.update({j:(layer.base,j) for j in range(layer.parent_width)})
        if layer.added is not None:
            affine.update({layer.parent_width+j:(layer.added,j) for j in range(layer.h-layer.parent_width)})
    elif kind=='selective_relu_scalar':
        grown=set(layer.grown_ids.tolist())
        affine.update({j:(layer.base,j) for j in range(layer.h) if j not in grown})
    elif layer.ordinary is not None:
        affine.update({layer.count+j:(layer.ordinary,j) for j in range(layer.h-layer.count)})
    k=2*layer.d+2
    u32,u64=torch.finfo(torch.float32).eps/2,torch.finfo(torch.float64).eps/2
    gamma=k*u32/(1-k*u32)+k*u64/(1-k*u64)
    for row,column in different.nonzero().tolist():
        if column not in affine:
            raise AssertionError('ReLU branch disagreement has no supported composed rounding interval')
        linear,j=affine[column]
        operands=x[row].detach().double()*linear.weight[j].detach().double()
        bias=linear.bias[j].detach().double()
        center=float(operands.sum()+bias)
        radius=gamma*float(operands.abs().sum()+bias.abs())+k*torch.finfo(torch.float32).tiny
        a,b=float(actual[row,column]),float(reference[row,column])
        if not (abs(center)<=radius and abs(a-center)<=radius and abs(b-center)<=radius):
            raise AssertionError('ReLU mask lies outside independently derived rounding interval')
        result['mismatches'].append(dict(index=[row,column],actual=a,reference=b,center=center,radius=radius))
    return result


def _reference_gradients(a,b,x,y,*,atol,rtol):
    """Independent forward and VJP oracle; raw kink differences stay explicit."""
    actual_x=x.detach().clone().requires_grad_()
    reference_x=x.detach().to(dtype=next(b.parameters()).dtype).requires_grad_()
    out=a(actual_x)
    parts=[]
    raw=_reference(b,reference_x,preactivations=parts)
    reference_z=torch.cat(parts,dim=1)
    actual_z=a.layer(actual_x).detach()
    branch=validate_relu_branches(a,actual_x,actual_z,reference_z)
    ref=raw if branch['ambiguous_count']==0 else _reference(b,reference_x,mask=(actual_z>0))
    target=y.to(dtype=reference_x.dtype)
    raw_grad=torch.autograd.grad((raw-target).square().mean(),(reference_x,*b.parameters()),retain_graph=True)
    actual_loss=(out-y).square().mean();reference_loss=(ref-target).square().mean()
    actual_loss.backward();reference_loss.backward()
    actual_grad=(actual_x.grad,*[p.grad for p in a.parameters()])
    conditioned_grad=(reference_x.grad,*[p.grad for p in b.parameters()])
    def normalized(left,right):
        return float(((left.detach().double()-right.detach().double()).abs()/(atol+rtol*right.detach().double().abs())).max())
    raw_error=max(normalized(p,q) for p,q in zip(actual_grad,raw_grad))
    conditioned_error=max(normalized(p,q) for p,q in zip(actual_grad,conditioned_grad))
    forward_error=normalized(out,ref)
    loss_error=normalized(actual_loss,reference_loss)
    torch.testing.assert_close(actual_loss.double(),reference_loss.double(),rtol=rtol,atol=atol)
    torch.testing.assert_close(out.double(),ref.double(),rtol=rtol,atol=atol)
    for p,q in zip(actual_grad,conditioned_grad):
        torch.testing.assert_close(p.double(),q.double(),rtol=rtol,atol=atol)
    return out,ref,dict(reference_semantics='relu-affine-branch-conditioned-v2',branch=branch,
        raw_gradient_max_normalized_error=raw_error,conditioned_gradient_max_normalized_error=conditioned_error,
        forward_max_normalized_error=forward_error,loss_max_normalized_error=loss_error,
        raw_gradient_contract='diagnostic only at certified ReLU ambiguity; never relabelled as unconditional parity')


def compare_reference(model,x,y,*,reference_dtype):
    """Check CPU32 execution against an independent branch-aware reference."""
    from treeneuron.backend import TOLERANCE
    a,b=copy.deepcopy(model),copy.deepcopy(model).to(dtype=reference_dtype)
    tolerance=TOLERANCE if reference_dtype==torch.float64 else dict(atol=2e-6,rtol=2e-4)
    _,_,result=_reference_gradients(a,b,x,y,**tolerance)
    return result


def _first_adam_rounding(initial, gradient, *, lr, weight_decay, eps):
    """Independent bias-corrected first AdamW step and float32 rounding bound.

    With zero moments, bias correction gives m_hat=g, v_hat=g**2.
    Thus p1=p0*(1-lr*decay)-lr*g/(abs(g)+eps). Evaluation here is
    float64 from the exact float32 operands. gamma_20 bounds the at-most
    twenty elementary roundings (moments, bias correction, sqrt, division,
    decay and addition); summing both term magnitudes covers cancellation.
    The tiny term covers underflow. This is an error bound, not a flat atol.
    """
    p,g=initial.double(),gradient.double()
    decayed=p*(1-lr*weight_decay); step=lr*g/(g.abs()+eps)
    unit=torch.finfo(torch.float32).eps/2
    gamma=20*unit/(1-20*unit)
    rounding=gamma*(decayed.abs()+step.abs())+20*torch.finfo(torch.float32).tiny
    return decayed-step,rounding


def _check_first_adam(a, b, initial, gradients_a, gradients_b, opts):
    """Separate own-gradient correctness from cross-reference conditioning."""
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        ga,gb=gradients_a[name],gradients_b[name]
        expected_a,round_a=_first_adam_rounding(initial[name],ga,**opts)
        expected_b,round_b=_first_adam_rounding(initial[name],gb,**opts)
        if (torch.any((p.double()-expected_a).abs()>round_a)
                or torch.any((q.double()-expected_b).abs()>round_b)):
            raise AssertionError(f'Incorrect first AdamW formula: {name}')
        ga,gb=ga.double(),gb.double(); eps=opts['eps']
        # Same sign: |f(g)-f(h)| = eps*|g-h|/((|g|+eps)(|h|+eps)).
        # Opposite signs: |f(g)-f(h)| = |g|/(|g|+eps)+|h|/(|h|+eps).
        # Forward/gradient equivalence was checked independently before step().
        conditioning=torch.where(ga*gb>=0,
            eps*(ga-gb).abs()/((ga.abs()+eps)*(gb.abs()+eps)),
            ga.abs()/(ga.abs()+eps)+gb.abs()/(gb.abs()+eps))
        allowance=opts['lr']*conditioning+round_a+round_b
        if torch.any((p.double()-q.double()).abs()>allowance):
            raise AssertionError(f'First AdamW conditioning bound exceeded: {name}')


def probe_causal_cpu(model, *, batch_size=128, n_rows=None):
    from treeneuron.train import capture_rng, restore_rng, source_hash
    if (type(batch_size) is not int or not 0 < batch_size <= 16384
            or any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in model.parameters())):
        raise ValueError('Causal probe requires CPU float32 and batch <=16384')
    if n_rows is not None and (type(n_rows) is not int or n_rows<=0):
        raise ValueError('Require positive actual F row bound')
    actual_batch=min(batch_size,n_rows) if n_rows is not None else batch_size
    before=capture_rng(); threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        # Probe independent copies, preserving the actual caller tensors and RNG.
        a,b=copy.deepcopy(model),copy.deepcopy(model)
        g=torch.Generator().manual_seed(1602041)
        x=torch.randn(actual_batch,model.layer.d,generator=g,dtype=torch.float32)
        y=torch.randn(actual_batch,generator=g,dtype=torch.float32)
        opts=dict(lr=.001,weight_decay=.01,eps=1e-8)
        oa,ob=torch.optim.AdamW(a.parameters(),foreach=False,**opts),torch.optim.AdamW(b.parameters(),foreach=False,**opts)
        out,ref,reference_check=_reference_gradients(a,b,x,y,atol=2e-6,rtol=2e-4)
        torch.testing.assert_close(out,ref,rtol=2e-5,atol=2e-6)
        grad_error=max(float((p.grad-q.grad).abs().max()) for p,q in zip(a.parameters(),b.parameters()))
        for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)
        initial={name:p.detach().clone() for name,p in a.named_parameters()}
        gradients_a={name:p.grad.clone() for name,p in a.named_parameters()}
        gradients_b={name:p.grad.clone() for name,p in b.named_parameters()}
        oa.step(); ob.step()
        optimizer_error=max(float((p-q).abs().max().detach()) for p,q in zip(a.parameters(),b.parameters()))
        _check_first_adam(a,b,initial,gradients_a,gradients_b,opts)
        payload=io.BytesIO(); torch.save({'model':a.state_dict(),'optimizer':oa.state_dict()},payload)
        payload.seek(0); saved=torch.load(payload,weights_only=True)
        c=type(a).from_config(a.export_config()).to(dtype=torch.float32); c.load_state_dict(saved['model'],strict=True)
        oc=torch.optim.AdamW(c.parameters(),foreach=False,**opts); oc.load_state_dict(saved['optimizer'])
        for candidate,optimizer in ((a,oa),(c,oc)):
            optimizer.zero_grad(); ((candidate(x)-y)**2).mean().backward(); optimizer.step()
        for p,q in zip(a.parameters(),c.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            for key in ('step','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(oa.state[p][key],oc.state[q][key],rtol=0,atol=0)
        return dict(schema_version=2,status='passed',source_hash=source_hash(),topology=model.export_config(),
            reference_check=reference_check,
            parameter_shapes={n:list(p.shape) for n,p in model.named_parameters()},
            total_parameters=sum(p.numel() for p in model.parameters()),batch_size=batch_size,actual_batch_size=actual_batch,
            device='cpu',dtype='float32',execution='eager',seed=1602041,
            forward_max_abs=float((out-ref).abs().max().detach()),gradient_max_abs=grad_error,
            optimizer_max_abs=optimizer_error,reload_next_update_max_abs=0.,scope='float32 forward and certified branch-conditioned gradient equivalence (raw discrepancies retained); own-gradient first AdamW formula and componentwise conditioning; exact serialization replay; no throughput measurement')
    finally:
        torch.set_num_threads(threads); restore_rng(before)


def attach_causal_evidence(backend, model, batch_size):
    result=copy.deepcopy(backend)
    if result['device']!='cpu': raise ValueError('Causal controls are verified only on selected CPU')
    result['causal_cpu_evidence']=probe_causal_cpu(model,batch_size=batch_size,n_rows=backend['workload']['n_fit'])
    result['performance_calibration_matches']=False
    result['performance_scope']='Actual causal topology CPU correctness; performance uncalibrated'
    result['workload']['h']=model.layer.h
    result['workload']['coordinate_family']=model.export_config()['kind'] if not hasattr(model.layer,'coordinates') else model.layer.coordinates
    result['workload']['actual_growth_count']=len(model.layer.grown_ids) if hasattr(model.layer,'grown_ids') else getattr(model.layer,'count',model.layer.h-getattr(model.layer,'parent_width',model.layer.h))
    return result


def validate_causal_evidence(backend, topology, config):
    from treeneuron.train import source_hash
    from treeneuron.cpu_envelope import requires_envelope, validate_envelope_reference
    if requires_envelope(config):
        validate_envelope_reference(backend.get('cpu_envelope',{}),config,topology)
    e=backend.get('causal_cpu_evidence',{})
    if not isinstance(e,dict): raise ValueError('Malformed causal CPU evidence')
    if (e.get('status')!='passed' or e.get('source_hash')!=source_hash() or e.get('topology')!=topology
            or e.get('batch_size')!=config.training.batch_size
            or e.get('actual_batch_size')!=min(config.training.batch_size,config.data.n_fit) or e.get('device')!='cpu'
            or e.get('dtype')!='float32' or e.get('execution')!='eager' or backend.get('device')!='cpu'
            or backend.get('performance_calibration_matches') is not False
            or not 0 < config.model.d <=16 or not 0 < config.model.h <=95 or (config.data.n_fit>2306 and not requires_envelope(config))):
        raise ValueError('Missing or mismatched actual causal CPU evidence')

    required={'reference_check','schema_version','status','source_hash','topology','parameter_shapes','total_parameters',
              'batch_size','actual_batch_size','device','dtype','execution','seed','forward_max_abs','gradient_max_abs',
              'optimizer_max_abs','reload_next_update_max_abs','scope'}
    if set(e)!=required or e['schema_version']!=2 or e['seed']!=1602041:
        raise ValueError('Malformed causal CPU evidence schema')
    for key in ('forward_max_abs','gradient_max_abs','optimizer_max_abs','reload_next_update_max_abs'):
        if type(e[key]) not in (float,int) or not math.isfinite(e[key]) or not 0<=e[key]<=1e-4:
            raise ValueError('Failed causal CPU numerical evidence')
    validate_reference_check(e['reference_check'])
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


def validate_reference_check(check):
    if not isinstance(check,dict): raise ValueError('Malformed reference check')
    if check.get('reference_semantics')!='relu-affine-branch-conditioned-v2':
        raise ValueError('Unsupported causal reference semantics')
    for key in ('raw_gradient_max_normalized_error','conditioned_gradient_max_normalized_error','forward_max_normalized_error','loss_max_normalized_error'):
        value=check.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid reference diagnostic')
    if max(check[k] for k in ('conditioned_gradient_max_normalized_error','forward_max_normalized_error','loss_max_normalized_error'))>1:
        raise ValueError('Reference forward/conditioned gradient failed')
    branch=check['branch']
    if (type(branch['ambiguous_count']) is not int or branch['ambiguous_count']!=len(branch['mismatches'])
            or branch['semantics']!='actual float32 ReLU branch; affine-rounding-certified reference only'):
        raise ValueError('Invalid ReLU branch certificate')
    if not branch['ambiguous_count'] and check['raw_gradient_max_normalized_error']>1:
        raise ValueError('Unconditional gradient failed without certified ambiguity')
    for row in branch['mismatches']:
        a,b,center,radius=(row[k] for k in ('actual','reference','center','radius'))
        if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (a,b,center,radius))
                or radius<0 or (a>0)==(b>0) or abs(center)>radius or abs(a-center)>radius or abs(b-center)>radius):
            raise ValueError('Invalid ReLU rounding interval')
