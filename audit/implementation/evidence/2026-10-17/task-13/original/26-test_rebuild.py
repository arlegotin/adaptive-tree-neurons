import sys
sys.path.insert(0,'/Users/artemlegotin/Documents/git_alter/adaptive-tree-neurons/tests')
import copy,json,traceback
from pathlib import Path
import torch
from test_protocol import repository,prepared
from test_train import cpu_training_evidence
OUT=Path('/tmp/task13-seed42-diagnostic')

def test_exact_failed_unit(prepared,cpu_training_evidence,monkeypatch):
    from treeneuron.protocol import confirmation_seeds
    from treeneuron.evaluation.confirm import repetition_config
    from treeneuron.data.synthetic import make_from_config
    from treeneuron.config import resolve_experiment,canonical_hash
    from treeneuron.pilot.calibrate import fit_repetition
    from treeneuron.baselines import causal_evidence as ce
    root,base,sources=prepared
    seeds=confirmation_seeds('toy','negative-fixture',42,6,2)
    tc=repetition_config(base,seeds,0)
    teacher,data=make_from_config(tc)
    cfg=repetition_config(base,seeds,1).to_dict()
    cfg['execution']['backend_report']=str(cpu_training_evidence)
    cfg['backend']['interop_threads']=torch.get_num_interop_threads()
    config=resolve_experiment(cfg)
    (OUT/'reconstruction.json').write_text(json.dumps(dict(seeds=seeds,config=cfg,manifest=data.manifest.to_dict()),indent=2))
    original=ce.probe_causal_cpu
    def capture(model,**kwargs):
        try: return original(model,**kwargs)
        except AssertionError:
            (OUT/'traceback.txt').write_text(traceback.format_exc())
            torch.save(dict(topology=model.export_config(),state=model.state_dict()),OUT/'failed-probe-model.pt')
            diagnose(model,kwargs['batch_size'],ce)
            raise
    monkeypatch.setattr(ce,'probe_causal_cpu',capture)
    fit_repetition(config,data,OUT/'reconstructed-attempt',canonical_hash(config.to_dict()))

def diagnose(model,batch,ce):
    torch.set_num_threads(1)
    g=torch.Generator().manual_seed(1602041)
    x=torch.randn(batch,model.layer.d,generator=g,dtype=torch.float32)
    y=torch.randn(batch,generator=g,dtype=torch.float32)
    outcomes={}
    for dtype in (torch.float32,torch.float64):
        a,b=copy.deepcopy(model).to(dtype=dtype),copy.deepcopy(model).to(dtype=dtype)
        opta=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.01,foreach=False)
        optb=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=.01,foreach=False)
        xx,yy=x.to(dtype),y.to(dtype)
        out,ref=a(xx),ce._reference(b,xx)
        ((out-yy)**2).mean().backward(); ((ref-yy)**2).mean().backward()
        initial={n:p.detach().clone() for n,p in a.named_parameters()}
        ga={n:p.grad.clone() for n,p in a.named_parameters()}
        gb={n:p.grad.clone() for n,p in b.named_parameters()}
        opta.step(); optb.step()
        pa,pb=dict(a.named_parameters()),dict(b.named_parameters())
        details=[]
        for n,p in pa.items():
            q=pb[n]; diff=(p-q).abs(); mask=diff>2e-6+2e-5*q.abs()
            indices=mask.nonzero().tolist()
            if indices:
                for idx in indices:
                    ix=tuple(idx)
                    details.append(dict(parameter=n,index=idx,initial=initial[n][ix].item(),gradient_actual=ga[n][ix].item(),gradient_reference=gb[n][ix].item(),updated_actual=p[ix].item(),updated_reference=q[ix].item(),absolute_difference=diff[ix].item()))
        outcomes[str(dtype)]=dict(forward_max_abs=(out-ref).abs().max().item(),gradient_max_abs=max((ga[n]-gb[n]).abs().max().item() for n in ga),update_max_abs=max((pa[n]-pb[n]).abs().max().item() for n in pa),failed_elements=details)
        torch.save(dict(initial=initial,grad_actual=ga,grad_reference=gb,updated_actual={n:p.detach() for n,p in pa.items()},updated_reference={n:p.detach() for n,p in pb.items()},x=xx,y=yy),OUT/f'operands-{dtype}.pt')
    (OUT/'diagnostic.json').write_text(json.dumps(dict(topology=model.export_config(),outcomes=outcomes),indent=2))
