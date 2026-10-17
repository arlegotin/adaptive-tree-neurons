import copy,json,torch
from treeneuron.cpu_envelope import _topologies,LOGICAL_BATCHES
from treeneuron.train import SelectiveRegressor
from treeneuron.models.static import StaticRegressor
from treeneuron.models.widened import WidenedRegressor
from treeneuron.backend import TOLERANCE
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(1602042)
classes={'selective_relu_scalar':SelectiveRegressor,'static_gated_relu_scalar':StaticRegressor,'widened_relu_scalar':WidenedRegressor}
models={p:classes[s['kind']].from_config(s) for p,s in _topologies().items()}
with torch.no_grad():
 models['grown_epoch'].layer.contrast.weight.normal_(0,.05);models['grown_epoch'].layer.contrast.bias.fill_(.01)
 models['grown_epoch'].layer.gate.weight.normal_(0,.04);models['grown_epoch'].layer.gate.bias.fill_(.1)
for phase,model in models.items():
 for batch in LOGICAL_BATCHES:
  x=torch.randn(batch,16);y=torch.randn(batch)
  outcomes=[]
  for dtype in (torch.float64,torch.float32):
   net=copy.deepcopy(model).to(dtype=dtype);v=x.to(dtype=dtype).requires_grad_();target=y.to(dtype=dtype)
   z=net.layer(v);pred=net(v);loss=(pred-target).square().mean();loss.backward()
   outcomes.append(dict(preactivation=z.detach().double(),prediction=pred.detach().double(),loss=loss.detach().double(),input_gradient=v.grad.double(),**{name:p.grad.double() for name,p in net.named_parameters()}))
  errors={name:float(((outcomes[1][name]-value).abs()/(TOLERANCE['atol']+TOLERANCE['rtol']*value.abs())).max()) for name,value in outcomes[0].items()}
  if max(errors.values())>1:
   z64,z32=outcomes[0]['preactivation'],outcomes[1]['preactivation']
   changed=(z64>0)!=(z32>0)
   print(json.dumps(dict(phase=phase,batch=batch,errors=errors,changed_activation_count=int(changed.sum()),changed_indices=changed.nonzero().tolist(),z64=z64[changed].tolist(),z32=z32[changed].tolist())))
   torch.save(dict(model=model.state_dict(),topology=model.export_config(),x=x,y=y),'.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/whole-h1-fix-evidence/failed-widened-operation.pt')
