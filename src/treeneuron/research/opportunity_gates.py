"""Conditional work is admitted from committed outcomes, never failed artifacts."""
import json
from pathlib import Path

from treeneuron.config import canonical_hash


def _unavailable(parent, requested):
    from .parent_artifacts import _committed
    report = parent['report']
    request = report['request']
    missing = report['unavailable_snapshots']
    executed = report['executed_updates']
    if (request != dict(snapshot_updates=requested, fixed_budget=False)
            or type(executed) is not int or not 0 <= executed <= 8000
            or not isinstance(missing, list)
            or any(type(u) is not int for u in missing)
            or missing != sorted(set(missing))
            or not set(missing) <= set(requested)):
        raise ValueError('Original parent snapshot disposition differs from its fixed request')
    capture = json.loads(_committed(Path(parent['artifact_path']), report['capture']).read_text())
    expected_captured = [u for u in requested if u <= executed]
    if (set(capture) != {'requested','captured','unavailable','records'}
            or capture['requested'] != requested or capture['captured'] != expected_captured
            or capture['unavailable'] != missing
            or missing != [u for u in requested if u > executed]
            or not isinstance(capture['records'], list)):
        raise ValueError('Original capture and executed snapshot counts disagree')
    current = [r['update'] for r in capture['records'] if r['label'] == 'current']
    if (any(type(u) is not int for u in current) or sorted(current) != expected_captured):
        raise ValueError('Original capture records and snapshot partition disagree')
    return missing


def fallback_gate(job, dependencies):
    """The queue authenticates each complete prerequisite before this call."""
    p = job['payload']
    record = dependencies[p['original_parent']]
    parent = record['value']
    evidence = dict(original_parent_job_id=p['original_parent'],
                    original_result_hash=canonical_hash(record))
    if parent['status'] == 'completed_ineligible':
        return dict(status='completed_ineligible', reason='Original conditional parent was not admitted', **evidence)
    if parent['status'] != 'completed_eligible':
        return dict(status='irrecoverable_missing', reason='Original parent failed; fallback cannot replace it', **evidence)
    missing = _unavailable(parent, p['snapshot_updates'])
    return dict(status='completed_eligible' if missing else 'completed_ineligible',
                unavailable_snapshots=missing,
                reason='Only recorded unreached actual updates admit fixed-parent training', **evidence)


def snapshot_gate(job, dependencies):
    p = job['payload']
    parent = dependencies[p['original_parent']]['value']
    score = dependencies[p['original_score']]['value']
    admission = dependencies[p['fallback_gate']]['value']
    if any(v['status'] not in ('completed_eligible', 'completed_ineligible')
           for v in (parent, score, admission)):
        return dict(status='irrecoverable_missing', reason='A required original snapshot prerequisite failed')
    if parent['status'] == 'completed_ineligible':
        return dict(status='completed_ineligible', reason='Original extension was not admitted')
    if p['snapshot'] not in ('actual_250', 'actual_1000'):
        raise ValueError('Fallback snapshot is outside the fixed actual-update inventory')
    update = int(p['snapshot'][7:])
    missing = _unavailable(parent, [250, 1000])
    if ((update in missing) != (score['status'] == 'completed_ineligible')
            or admission.get('unavailable_snapshots') != missing):
        raise ValueError('Original score and parent snapshot disposition disagree')
    if update not in missing:
        return dict(status='completed_ineligible', reason='Use the available original actual snapshot')
    if admission['status'] != 'completed_eligible':
        raise ValueError('Missing original snapshot lacks its parent fallback admission')
    return dict(status='completed_eligible', snapshot=p['snapshot'],
                reason='Original actual update was recorded unreached',
                original_parent_job_id=p['original_parent'])
