"""Exact seed42/unit6/rep1 B6 probe regression; no scientific observations."""
from pathlib import Path
import copy
import pytest
import torch


def failed_probe():
    from treeneuron.train import SelectiveRegressor
    saved=torch.load(Path(__file__).parent/'fixtures/causal_seed42_probe.pt',weights_only=True)
    model=SelectiveRegressor.from_config(saved['topology'])
    model.load_state_dict(saved['state'],strict=True)
    return model


def test_seed42_causal_probe_accepts_conditioned_first_adam_step():
    from treeneuron.baselines.causal_evidence import probe_causal_cpu
    model=failed_probe(); original=copy.deepcopy(model.state_dict())
    report=probe_causal_cpu(model,batch_size=8)
    assert report['optimizer_max_abs']==pytest.approx(2.442335244268179e-6,abs=1e-10)
    assert report['reload_next_update_max_abs']==0.
    assert 'conditioning' in report['scope']
    for key,value in original.items(): torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)


def test_causal_probe_rejects_incorrect_first_adam_step(monkeypatch):
    from treeneuron.baselines.causal_evidence import probe_causal_cpu
    step=torch.optim.AdamW.step
    def incorrect(optimizer,*args,**kwargs):
        result=step(optimizer,*args,**kwargs)
        with torch.no_grad(): optimizer.param_groups[0]['params'][0].add_(1e-4)
        return result
    monkeypatch.setattr(torch.optim.AdamW,'step',incorrect)
    with pytest.raises(AssertionError,match='first AdamW formula'):
        probe_causal_cpu(failed_probe(),batch_size=8)


def widened_rounding_case():
    from treeneuron.models.widened import WidenedRegressor
    saved=torch.load(Path(__file__).parent/'fixtures/h1_widened_relu_rounding.pt',weights_only=True)
    model=WidenedRegressor.from_config(saved['topology']);model.load_state_dict(saved['model'])
    return model,saved['x'][:9224],saved['y'][:9224]


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_retained_widened_relu_case_checks_actual_branch_without_hiding_raw_error(dtype):
    from treeneuron.baselines.causal_evidence import compare_reference
    model,x,y=widened_rounding_case()
    result=compare_reference(model,x,y,reference_dtype=dtype)
    assert result['raw_gradient_max_normalized_error']>18
    assert result['conditioned_gradient_max_normalized_error']<1
    assert result['branch']['ambiguous_count']==1
    assert result['branch']['mismatches'][0]['index']==[1417,66]
    assert result['branch']['semantics']=='actual float32 ReLU branch; affine-rounding-certified reference only'


def test_wrong_relu_mask_outside_rounding_interval_is_rejected():
    from treeneuron.baselines.causal_evidence import validate_relu_branches
    model,x,_=widened_rounding_case()
    actual=model.layer(x).detach();reference=copy.deepcopy(model).double().layer(x.double()).detach()
    actual[0,0]=-reference[0,0].sign().float()
    with pytest.raises(AssertionError,match='rounding interval'):
        validate_relu_branches(model,x,actual,reference)


def test_branch_reference_still_rejects_incorrect_backward(monkeypatch):
    from treeneuron.baselines.causal_evidence import compare_reference
    from treeneuron.models.widened import WidenedRegressor
    model,x,y=widened_rounding_case()
    original=WidenedRegressor.forward
    def wrong(self,x):
        return original(self,x)+(self.layer.added.weight-self.layer.added.weight.detach()).sum()*.01
    monkeypatch.setattr(WidenedRegressor,'forward',wrong)
    with pytest.raises(AssertionError): compare_reference(model,x,y,reference_dtype=torch.float64)


def test_gated_relu_ambiguity_without_composed_interval_fails_closed():
    from treeneuron.train import SelectiveRegressor
    from treeneuron.baselines.causal_evidence import validate_relu_branches
    model=SelectiveRegressor(2,2)
    model.layer.grow(torch.tensor([0]),torch.ones(1,3))
    with pytest.raises(AssertionError,match='composed rounding interval'):
        validate_relu_branches(model,torch.ones(1,2),torch.tensor([[1e-9,1.]]),torch.tensor([[-1e-9,1.]]))
