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
