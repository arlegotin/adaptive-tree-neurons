import torch,copy,json
from treeneuron.models.widened import WidenedRegressor
from treeneuron.baselines.causal_evidence import _reference
torch.set_num_threads(1);torch.set_num_interop_threads(1)
s=torch.load('.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/whole-h1-fix-evidence/failed-widened-operation.pt',weights_only=True)
a=WidenedRegressor.from_config(s['topology']);a.load_state_dict(s['model']);b=copy.deepcopy(a)
x,y=s['x'][:9224],s['y'][:9224]
pred=a(x);ref=_reference(b,x)
(pred-y).square().mean().backward();(ref-y).square().mean().backward()
errors={n:float(((p.grad-q.grad).abs()/(2e-6+2e-4*q.grad.abs())).max()) for (n,p),(_,q) in zip(a.named_parameters(),b.named_parameters())}
print(json.dumps(dict(reference='existing independent float32 per-row reductions',actual_rows=9224,forward_max_abs=float((pred-ref).abs().max().detach()),gradient_normalized_errors=errors)))
