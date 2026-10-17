import torch,copy,json
from treeneuron.models.widened import WidenedRegressor
from treeneuron.backend import TOLERANCE
torch.set_num_threads(1);torch.set_num_interop_threads(1)
s=torch.load('.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/whole-h1-fix-evidence/failed-widened-operation.pt',weights_only=True)
model=WidenedRegressor.from_config(s['topology']);model.load_state_dict(s['model'])
for n in (9224,8192,4096,2048,1024,512,256,128):
 outs=[]
 for dtype in (torch.float64,torch.float32):
  net=copy.deepcopy(model).to(dtype=dtype);x=s['x'][:n].to(dtype=dtype);y=s['y'][:n].to(dtype=dtype)
  z=net.layer(x);pred=net(x);loss=(pred-y).square().mean();loss.backward()
  outs.append(dict(z=z.detach().double(),prediction=pred.detach().double(),**{k:p.grad.double() for k,p in net.named_parameters()}))
 errors={k:float(((outs[1][k]-v).abs()/(TOLERANCE['atol']+TOLERANCE['rtol']*v.abs())).max()) for k,v in outs[0].items()}
 print(json.dumps(dict(actual_rows=n,errors=errors,changed_activation_count=int(((outs[0]['z']>0)!=(outs[1]['z']>0)).sum()))))
