import os,time,json,copy,resource,statistics
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron.train import SelectiveRegressor,source_hash
from treeneuron.backend import lock_hash,runtime_hash,_resources,_guard
before=_resources(torch);_guard(before)
torch.manual_seed(1602042)
x=torch.randn(9224,16);y=torch.randn(9224)
base=SelectiveRegressor(16,64);grown=copy.deepcopy(base)
grown.layer.grow(torch.arange(16),torch.randn(16,17)*.05)
rows=[]
for name,model in [('base',base),('grown',grown)]:
 for batch in (4096,8192,16384,32768):
  samples=[]
  for repeat in range(4):
   net=copy.deepcopy(model);opt=torch.optim.AdamW(net.parameters(),lr=.001,foreach=False)
   start=time.perf_counter_ns()
   for i in range(0,len(x),batch):
    opt.zero_grad();(net(x[i:i+batch])-y[i:i+batch]).square().mean().backward();opt.step()
   elapsed=time.perf_counter_ns()-start
   if repeat:samples.append(elapsed)
  rows.append(dict(phase=name,n_fit=len(x),d=16,h=64,m=0 if name=='base' else 16,logical_batch=batch,physical_batch=batch,actual_max_rows=min(batch,len(x)),updates=(len(x)+batch-1)//batch,samples_ns=samples,median_ns=statistics.median(samples)))
report=dict(scope='bounded engineering exploration, epoch training only, initialization excluded; ambient CPU interference; no plateau claim',source_sha256=source_hash(),lock_sha256=lock_hash(),runtime_sha256=runtime_hash(),resources_before=before,resources_after=_resources(torch),rows=rows)
print(json.dumps(report,sort_keys=True))
