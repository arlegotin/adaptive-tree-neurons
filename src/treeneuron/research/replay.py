"""Check observed unchanged continuation against a parent's unsuccessful tail.

This is an execution diagnostic. Even a complete exact replay does not measure
headroom under a different minibatch stream, schedule or optimizer treatment.
"""
import json
import math
from pathlib import Path

from treeneuron.config import canonical_hash


def _stage(events, interval):
    if (not isinstance(events, list) or not events or events[0].get('type') != 'stage_started'
            or events[-1].get('type') != 'selection'
            or sum(e.get('type') == 'stage_started' for e in events) != 1
            or sum(e.get('type') == 'selection' for e in events) != 1):
        raise ValueError('Require one ordered complete native stage, from start through terminal selection')
    start = current = events[0]['update']
    if type(start) is not int or start < 0:
        raise ValueError('Stage start must be a nonnegative integer')
    updates, evaluations = {}, {}
    for event in events[1:-1]:
        kind, update = event['type'], event['update']
        if type(update) is not int:
            raise ValueError('Native event update must be an integer')
        if kind == 'update':
            if update != current+1 or event['stage_update'] != update-start:
                raise ValueError('Native updates must be contiguous, unique and ordered')
            current = update
            if (type(event['rows']) is not int or event['rows'] <= 0
                    or not isinstance(event['row_ids'], list) or len(event['row_ids']) != event['rows']
                    or any(type(row) is not int for row in event['row_ids'])
                    or not math.isfinite(event['loss_standardized_mse'])
                    or not math.isfinite(event['learning_rate'])):
                raise ValueError('Incomplete or nonfinite native update evidence')
            updates[update] = event
        elif kind == 'evaluation':
            if (event['split'] != 'G' or update != current or update in evaluations
                    or event['stage_update'] != update-start or not math.isfinite(event['metrics']['mse'])
                    or event['metrics']['mse'] < 0):
                raise ValueError('Native G evaluations must be unique, ordered and finite')
            evaluations[update] = event
        elif kind == 'checkpoint':
            if update != current:
                raise ValueError('Checkpoint event differs from the current complete update')
        else:
            raise ValueError('Unexpected event type in native stage evidence')
    selection = events[-1]
    if (selection['last_update'] != current or selection['executed_updates'] != current-start
            or type(selection['last_update']) is not int or type(selection['executed_updates']) is not int):
        raise ValueError('Terminal selection bounds/count contradict ordered native updates')
    grid = list(range(start, current+1, interval))
    if grid[-1] != current:
        grid.append(current)
    if list(evaluations) != grid:
        raise ValueError('Missing or inconsistent native G evaluation cadence')
    earliest_best = min(evaluations, key=lambda update: (evaluations[update]['metrics']['mse'], update))
    if type(selection['selected_update']) is not int or selection['selected_update'] != earliest_best:
        raise ValueError('Selected update differs from earliest strict G-best')
    return start, selection, updates, evaluations


def replay_evidence(parent_events, keep_events, *, validation_interval, patience,
                    parent_max_updates, unchanged_model=True, full_state_restored=True):
    for value in (validation_interval, patience, parent_max_updates):
        if type(value) is not int or value <= 0:
            raise ValueError('Replay policy requires positive integer interval, patience and cap')
    result = dict(schema='research/v2-replay', classification='incomplete_evidence',
        matching_updates=0, matching_G_evaluations=0, mismatches=[],
        training_headroom_exhausted=False, unchanged_model=unchanged_model,
        full_state_restored=full_state_restored)
    try:
        result['event_hashes'] = dict(parent=canonical_hash({'events': parent_events}),
                                     keep=canonical_hash({'events': keep_events}))
        parent_start, selection, parent_updates, parent_G = _stage(parent_events, validation_interval)
        keep_start, kept, keep_updates, keep_G = _stage(keep_events, validation_interval)
        if parent_start != 0 or keep_start != selection['selected_update']:
            raise ValueError('Unchanged continuation must start at this parent fit\'s selected update')
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        result['reason'] = f'Incomplete or contradictory raw stage evidence: {error}'
        return result
    selected, last = selection['selected_update'], selection['last_update']
    result.update(parent_selected_update=selected, parent_last_update=last,
        keep_selected_stage_update=kept['selected_update']-selected)
    bad = [e for u, e in parent_G.items() if u > selected]
    stopped_by_patience = (len(bad) >= patience and last-selected == patience*validation_interval
        and selected in parent_G
        and all(e['metrics']['mse'] >= parent_G[selected]['metrics']['mse'] for e in bad))
    result['parent_stop'] = ('patience_and_cap' if stopped_by_patience and last == parent_max_updates
        else 'patience' if stopped_by_patience else 'cap' if last == parent_max_updates else 'unresolved')
    if not (stopped_by_patience and unchanged_model and full_state_restored):
        result.update(classification='not_replay_guaranteed',
            reason='Patience tail plus unchanged complete-state treatment premises do not all hold')
        return result
    expected_updates = range(selected+1, last+1)
    expected_G = range(selected, last+1, validation_interval)
    for expected, left, right, fields in (
        (expected_updates, parent_updates, keep_updates,
         ('loss_standardized_mse', 'learning_rate', 'rows', 'row_ids')),
        (expected_G, parent_G, keep_G, ('metrics',))):
        for update in expected:
            if update not in left or update not in right or any(
                    key not in left[update] or key not in right[update] for key in fields):
                result['reason'] = f'Missing complete raw update/evaluation evidence at {update}'
                return result
            matches = True
            for key in fields:
                if left[update][key] != right[update][key]:
                    result['mismatches'].append(dict(update=update, field=key,
                        event_type='update' if 'row_ids' in fields else 'evaluation'))
                    matches = False
            if matches:
                result['matching_updates' if 'row_ids' in fields else 'matching_G_evaluations'] += 1
    if result['mismatches']:
        result['classification'] = 'replay_mismatch'
    elif kept['last_update'] == last and kept['selected_update'] == selected:
        result['classification'] = 'verified_patience_tail_replay'
    else:
        result.update(classification='matching_suffix_only',
            reason='Compared suffix matches but continuation has a different terminal selection or budget')
    return result


def reference_replay(root, report, training):
    """Read original per-stage streams, without loading or modifying models."""
    root = Path(root)
    parent = root/'runs'/report['parent_run_id']/'events.jsonl'
    keep = root/report['methods']['keep']['stages'][0]['path']/'events.jsonl'
    for path in (parent, keep):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Replay stream escaped its retained native pair')
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines()]
    unchanged = report['methods']['keep']['initial_pairing'] == report['parent_pairing']
    return replay_evidence(read(parent), read(keep), validation_interval=training.validation_interval,
        patience=training.patience, parent_max_updates=training.base_updates,
        unchanged_model=True, full_state_restored=unchanged)
