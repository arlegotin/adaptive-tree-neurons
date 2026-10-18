"""A full disk must not replace a control failure or leak its backend receipt."""
import pytest

from test_train import training_case,cpu_training_evidence


def test_control_failure_publication_keeps_original_error_and_restores_receipt(training_case,tmp_path,monkeypatch):
    from treeneuron import train
    from treeneuron.research import control_runner
    from treeneuron.research.roles import fitting_view
    data,config,_=training_case
    original=train.BACKEND_REPORT_PATH
    failure=FloatingPointError('original numerical failure')
    def fail(*args,**kwargs):raise failure
    def full(*args,**kwargs):raise OSError('disk full during control report')
    monkeypatch.setattr(control_runner,'initialize_control_state',fail)
    monkeypatch.setattr(control_runner,'write_new_json',full)
    with pytest.raises(FloatingPointError,match='original numerical failure') as caught:
        control_runner.fit_control_pair(config,fitting_view(data),tmp_path/'failed',('keep',),control=object())
    assert train.BACKEND_REPORT_PATH==original
    assert any('disk full' in note for note in caught.value.__notes__)
