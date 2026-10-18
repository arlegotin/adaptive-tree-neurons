"""Reconstruct M1 scalar results from retained predictions without model loading."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from treeneuron.config import canonical_hash
from .control_analysis import positive_sensitivity
from .ledger import AttemptLedger


@contextmanager
def _inactive_study(root):
    import fcntl
    # A shared, read-only lease prevents auditing a partially published group.
    with (root/'worker-lease.json').open('r') as stream:
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PermissionError('Cannot audit an active study worker') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def _file(path, root, commitments):
    path = Path(path)
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError('Retained audit input escaped the study or is unavailable')
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    commitments[str(resolved.relative_to(root))] = digest
    return resolved,digest


def _vector(value, name):
    if value.ndim != 1 or not len(value) or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
        raise ValueError(f'Invalid retained {name} vector')
    return value.astype(np.float64)


def _row_ids(value, count):
    if (value.ndim != 1 or value.shape != (count,) or value.dtype.kind not in 'Uiu'
            or len(np.unique(value)) != count
            or (value.dtype.kind == 'U' and np.any(np.char.str_len(value) == 0))):
        raise ValueError('Retained row IDs must be unique one-dimensional length-matched strings or integers')
    return value


def _metrics(target, prediction, variance):
    errors = [abs(float(p)-float(y)) for y,p in zip(target,prediction)]
    squared = [e*e for e in errors]
    if not all(math.isfinite(e) for e in squared):
        raise ValueError('Retained squared prediction error is nonfinite')
    mse = math.fsum(squared)/len(errors)
    return dict(n=len(errors),mse=mse,normalized_mse=mse/variance if variance else None,
        rmse=math.sqrt(mse),mae=math.fsum(errors)/len(errors),
        absolute_error_p95=float(np.quantile(errors,.95)),max_absolute_error=max(errors))


def audit_control_study(directory, *, expected_plan_hash):
    """An external plan commitment is required; current source need not refit it.

    This is raw prediction arithmetic conditional on retained inputs. It neither
    regenerates teachers nor authenticates prediction values by rerunning models.
    """
    root = Path(directory).resolve()
    with _inactive_study(root):
        plan = json.loads((root/'plan.json').read_text())
        if (plan.get('plan_hash') != expected_plan_hash
                or plan['plan_hash'] != canonical_hash({k:v for k,v in plan.items() if k != 'plan_hash'})
                or plan['schema'] != 'research/v2-plan' or plan['config']['kind'] != 'm1_controls'):
            raise ValueError('Control audit plan commitment/schema differs')
        ledger = AttemptLedger.resume(root,plan)
        latest = {e['attempt_id']:e for e in ledger.events}
        running = {e['attempt_id']:e for e in ledger.events if e['status'] == 'running'}
        commitments = {}
        _file(root/'plan.json',root,commitments)
        _file(root/'events.jsonl',root,commitments)
        rows = []
        comparisons = 0
        maximum_difference = 0.
        for case in plan['attempts']:
            event = latest.get(case['attempt_id'])
            record = dict(case,status='never_started' if event is None else event['status'])
            if record['status'] != 'completed_eligible':
                rows.append(record)
                continue
            start = running.get(case['attempt_id'])
            if start is None or type(start.get('recovery_attempt')) is not int or start['recovery_attempt'] not in (0,1):
                raise ValueError('Eligible control lacks its declared native transaction')
            repeat,attempt = case['repetition'],start['recovery_attempt']
            parent = root/case['cell']/f"teacher-{case['teacher_index']}"
            pair = parent/f'repeat-{repeat}-attempt-{attempt}'
            pointer,digest = _file(event['retained_model']['path'],root,commitments)
            if not pointer.is_relative_to(pair) or digest != event['retained_model']['sha256']:
                raise ValueError('Retained model pointer commitment/transaction differs')
            observed,_ = _file(parent/f'data-repeat-{repeat}-attempt-{attempt}/observations.npz',root,commitments)
            predicted,_ = _file(pair/'predictions.npz',root,commitments)
            with np.load(observed,allow_pickle=False) as data,np.load(predicted,allow_pickle=False) as values:
                fit = _vector(data['F_y'],'F target')
                variance = float(np.var(fit,ddof=0))
                metrics = {}
                for role in ('V','E'):
                    target = _vector(data[role+'_y'],role+' target')
                    prediction = _vector(values[case['role']+'_'+role],role+' prediction')
                    original_ids = _row_ids(data[role+'_row_id'],len(target))
                    prediction_ids = _row_ids(values[role+'_row_id'],len(prediction))
                    if (target.shape != prediction.shape
                            or original_ids.dtype != prediction_ids.dtype
                            or not np.array_equal(original_ids,prediction_ids)
                            or data[role+'_y'].dtype != values[role+'_y'].dtype
                            or data[role+'_y'].tobytes() != values[role+'_y'].tobytes()):
                        raise ValueError('Retained prediction row/target alignment differs')
                    metrics[role] = _metrics(target,prediction,variance)
                    for name,value in metrics[role].items():
                        saved = event['metrics'][role].get(name)
                        if value is None:
                            agrees = saved is None
                        elif name == 'n':
                            agrees = type(saved) is int and saved == value
                        else:
                            agrees = type(saved) in (int,float) and math.isfinite(saved) and math.isclose(value,saved,rel_tol=1e-12,abs_tol=1e-14)
                            if agrees:maximum_difference = max(maximum_difference,abs(value-saved))
                        if not agrees:
                            raise ValueError(f"Retained {case['attempt_id']} {role} metric {name} differs")
                        comparisons += 1
            record.update(metrics=metrics,V_NMSE=metrics['V']['normalized_mse'],fit_target_variance=variance)
            rows.append(record)
        complete = bool(rows) and all(r['status'] == 'completed_eligible' for r in rows)
        return dict(schema='research/v2-control-audit',status='completed' if complete else 'incomplete_or_ineligible',
            plan_hash=plan['plan_hash'],source_sha256=plan['source_sha256'],
            evidence_level='retained_raw_prediction_arithmetic',model_predictions_reexecuted=False,
            teacher_data_regenerated=False,rows=rows,positive_sensitivity=positive_sensitivity(rows),
            audited_predictions=sum('metrics' in r for r in rows)*2,metric_comparisons=comparisons,
            maximum_absolute_metric_difference=maximum_difference,file_commitments=commitments,
            ledger_event_hash=ledger.events[-1]['event_hash'] if ledger.events else '0'*64,
            charged_standalone_ns=ledger.charged_ns,physical_cost_inferred_from_sum=False,
            T_accessed=False,efficacy_claim=False)
