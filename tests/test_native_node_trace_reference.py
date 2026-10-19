"""Independent audit arithmetic for native NODE minibatches and stopping."""
import copy
import importlib.util
from pathlib import Path

import pytest


def reference():
    path=Path(__file__).resolve().parents[1]/'docs/research/v2/scripts/native_node_trace_reference.py'
    assert path.is_file(),'Missing independent native NODE trace reference'
    spec=importlib.util.spec_from_file_location('native_node_trace_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.check_native_trace


def example():
    epochs=[[0,1,2,3,4],[4,3,2,1,0],[1,4,0,3,2],[2,0,4,3,1]]
    trace=[]
    for epoch,rows in enumerate(epochs):
        for part in (rows[:3],rows[3:]):
            trace.append(dict(step=len(trace)+1,epoch=epoch,rows=part,loss=1.,applied=True,optimizer_entered=True))
    reports=[dict(step=2,g_mse=1.,snapshot_steps=[2]),dict(step=4,g_mse=.9,snapshot_steps=[2,4]),
        dict(step=6,g_mse=.9,snapshot_steps=[2,4,6]),dict(step=8,g_mse=1.1,snapshot_steps=[4,6,8])]
    return dict(trace=trace,reports=reports,n_fit=5,batch_size=3,report_frequency=2,
        patience=3,latest_checkpoints=3,selected_step=4,executed_updates=8)


def test_multibatch_epochs_keep_the_actual_tail_and_stop_after_earliest_tied_G_best():
    out=reference()(**example())
    assert out['rows_updated']==20 and out['complete_epochs']==4
    assert out['native_G_evaluations']==4 and out['selected_step']==4 and out['executed_updates']==8
    assert out['native_stopping_independently_checked'] is True and out['training_reexecuted'] is False


@pytest.mark.parametrize('change',['short_batch','duplicate','epoch','not_applied','nonfinite_G',
    'bad_window','tied_selected','too_early','too_late','missing_report','duplicate_report','negative_loss'])
def test_inconsistent_native_evidence_cannot_pass_completion(change):
    fn=reference();case=copy.deepcopy(example())
    if change=='short_batch':case['trace'][0]['rows'].pop()
    elif change=='duplicate':case['trace'][1]['rows'][0]=0
    elif change=='epoch':case['trace'][1]['epoch']=1
    elif change=='not_applied':case['trace'][0]['applied']=False
    elif change=='nonfinite_G':case['reports'][0]['g_mse']=float('nan')
    elif change=='bad_window':case['reports'][-1]['snapshot_steps']=[2,4,6,8]
    elif change=='tied_selected':case['selected_step']=6
    elif change=='too_early':case['trace'].pop();case['reports'].pop();case['executed_updates']=7
    elif change=='too_late':case['trace'].append(dict(step=9,epoch=4,rows=[0,1,2],loss=1.,applied=True,optimizer_entered=True));case['executed_updates']=9
    elif change=='missing_report':case['reports'].pop(1)
    elif change=='duplicate_report':case['reports'][1]['step']=2
    else:case['trace'][0]['loss']=-1.
    with pytest.raises(ValueError):fn(**case)


def test_small_F_really_does_cover_all_rows_per_native_update():
    case=example();case.update(n_fit=2,batch_size=512)
    for step,row in enumerate(case['trace']):row.update(epoch=step,rows=[1,0])
    out=reference()(**case)
    assert out['rows_updated']==16 and out['complete_epochs']==8
