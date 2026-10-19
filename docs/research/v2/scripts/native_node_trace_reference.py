"""Independent row/epoch and stopping reference; no model imports or training."""
import math


def _finite_nonnegative(value):
    try:return type(value) in (int,float) and math.isfinite(value) and value>=0
    except OverflowError:return False


def check_native_trace(*,trace,reports,n_fit,batch_size,report_frequency,patience,
                       latest_checkpoints,selected_step,executed_updates):
    if (any(type(v) is not int or v<=0 for v in
            (n_fit,batch_size,report_frequency,patience,latest_checkpoints,selected_step,executed_updates))
            or n_fit<2 or type(trace) is not list or len(trace)!=executed_updates or type(reports) is not list):
        raise ValueError('Explicit positive native trace geometry and complete lists required')
    epoch=0;cursor=0;seen=set();complete=0;rows_updated=0
    for step,row in enumerate(trace,1):
        if cursor==n_fit:epoch+=1;cursor=0;seen=set()
        if (type(row) is not dict or type(row.get('step')) is not int or row['step']!=step
                or type(row.get('epoch')) is not int or row['epoch']!=epoch
                or type(row.get('rows')) is not list or len(row['rows'])!=min(batch_size,n_fit-cursor)
                or any(type(i) is not int or not 0<=i<n_fit for i in row['rows'])
                or len(set(row['rows']))!=len(row['rows']) or seen.intersection(row['rows'])
                or row.get('applied') is not True or row.get('optimizer_entered') is not True
                or not _finite_nonnegative(row.get('loss'))):
            raise ValueError('Applied native minibatch/epoch coverage differs')
        seen.update(row['rows']);cursor+=len(row['rows']);rows_updated+=len(row['rows'])
        if cursor==n_fit:
            if seen!=set(range(n_fit)):raise ValueError('Completed native epoch lacks a training row')
            complete+=1
    steps=list(range(report_frequency,executed_updates+1,report_frequency))
    if len(reports)!=len(steps) or not steps:raise ValueError('Native G selection grid is incomplete')
    by_step={}
    for step,report in zip(steps,reports):
        window=list(range(max(report_frequency,step-(latest_checkpoints-1)*report_frequency),step+1,report_frequency))
        if (type(report) is not dict or type(report.get('step')) is not int or report['step']!=step
                or not _finite_nonnegative(report.get('g_mse'))
                or type(report.get('snapshot_steps')) is not list
                or any(type(v) is not int for v in report['snapshot_steps']) or report['snapshot_steps']!=window):
            raise ValueError('Native G report/rolling checkpoint window differs')
        by_step[step]=report
    best=math.inf;best_step=0;first_stop=None
    for step in range(1,executed_updates+1):
        if step in by_step and by_step[step]['g_mse']<best:
            best=by_step[step]['g_mse'];best_step=step
        if step>best_step+patience:first_stop=step;break
    if first_stop!=executed_updates or best_step!=selected_step:
        raise ValueError('Endpoint is not the first native patience stop with strict earliest G best')
    return dict(training_rows=n_fit,rows_updated=rows_updated,complete_epochs=complete,
        final_epoch_index=epoch,final_epoch_rows=cursor,executed_updates=executed_updates,
        selected_step=best_step,native_G_evaluations=len(reports),native_stopping_independently_checked=True,
        training_reexecuted=False,
        scope='Independent trace geometry, strict G selection, rolling window and first native patience endpoint only; no RNG or weight-update reexecution')
