"""Stored, target-free group balancing and expanding-window transfer roles."""
import numpy as np

from treeneuron.config import canonical_hash
from .seeds import seed64

TRANSFER_ROOT = 2026101803
BIKE_BOUNDARIES = ('2012-01-01', '2012-03-14', '2012-05-26', '2012-08-07', '2012-10-19', '2013-01-01')


def balance_groups(counts, capacities, minima, *, seed):
    """Lexicographic deterministic rule from the v2 contract; labels absent."""
    if (not counts or set(capacities) != set(minima) or any(v <= 0 for v in counts.values())
            or any(type(v) is not int or v < 0 for v in minima.values())
            or len(counts) < sum(minima.values()) or any(v <= 0 for v in capacities.values())):
        raise ValueError('Group balancing requires feasible positive counts/capacities and group minima')
    names = tuple(capacities)
    row_counts, group_counts = dict.fromkeys(names, 0), dict.fromkeys(names, 0)
    order_keys = {g: seed64(seed, 'group-order', str(g)) for g in counts}
    priorities = {role: seed64(seed, 'role-priority', role) for role in names}
    order = sorted(counts, key=lambda g: (-counts[g], order_keys[g], str(g)))
    assignments = {}
    for index, group in enumerate(order):
        remaining = len(order)-index-1

        def score(role):
            after_groups = {r: group_counts[r]+int(r == role) for r in names}
            violation = max(0, sum(max(0, minima[r]-after_groups[r]) for r in names)-remaining)
            deviation = sum((row_counts[r]+(counts[group] if r == role else 0)-capacities[r])**2 for r in names)
            return violation, deviation, group_counts[role], priorities[role]

        role = min(names, key=score)
        assignments[group] = role
        row_counts[role] += counts[group]
        group_counts[role] += 1
    moves = []
    while any(group_counts[r] < minima[r] for r in names):
        choices = []
        before = sum((row_counts[r]-capacities[r])**2 for r in names)
        for destination in names:
            if group_counts[destination] >= minima[destination]:
                continue
            for group, origin in assignments.items():
                if group_counts[origin] <= minima[origin]:
                    continue
                deviation = sum((row_counts[r]+counts[group]*(int(r == destination)-int(r == origin))-capacities[r])**2 for r in names)
                choices.append((counts[group], deviation-before, priorities[destination], order_keys[group], str(group), group, origin, destination))
        if not choices:
            raise ValueError('Deterministic repair cannot satisfy group minima')
        _, _, _, _, _, group, origin, destination = min(choices)
        assignments[group] = destination
        row_counts[origin] -= counts[group]
        row_counts[destination] += counts[group]
        group_counts[origin] -= 1
        group_counts[destination] += 1
        moves.append(dict(group=group, origin=origin, destination=destination))
    return assignments, dict(seed=seed, targets=capacities, actual_rows=row_counts,
        actual_groups=group_counts, minima=minima, sorted_groups=order,
        group_tie_keys={str(g): order_keys[g] for g in counts}, role_priorities=priorities, moves=moves)


def grouped_splits(dataset, group_id):
    groups = np.asarray(group_id)
    if groups.ndim != 1 or groups.dtype != np.int64:
        raise ValueError('Group IDs must be an int64 vector')
    ids, sizes = np.unique(groups, return_counts=True)
    counts = {int(g): int(n) for g, n in zip(ids, sizes)}
    outer, outer_trace = balance_groups(counts,
        {str(i): len(groups)/5 for i in range(5)}, {str(i): 1 for i in range(5)},
        seed=seed64(TRANSFER_ROOT, 'transfer', dataset, 'outer-v1'))
    splits = []
    for fold in range(5):
        development = {g: n for g, n in counts.items() if outer[g] != str(fold)}
        n_development = sum(development.values())
        inner, trace = balance_groups(development,
            dict(F=.7*n_development, G=.15*n_development, V=.15*n_development),
            dict(F=2, G=2, V=2), seed=seed64(TRANSFER_ROOT, 'transfer', dataset, 'inner-v1', str(fold)))
        roles = ['T' if outer[int(g)] == str(fold) else inner[int(g)] for g in groups]
        record = dict(schema='research/v2-transfer-split', dataset=dataset,
            outer_fold=fold, primary=fold == 0, roles=roles,
            actual_rows={role: roles.count(role) for role in 'FGVT'},
            outer_trace=outer_trace, inner_trace=trace, split_unit='group',
            unused_future_rows=0, embargo_rows=0, role_boundaries={})
        record['split_hash'] = canonical_hash(record)
        splits.append(record)
    return splits


def bike_splits(timestamps):
    hours = np.asarray(timestamps)
    if (hours.ndim != 1 or not np.issubdtype(hours.dtype, np.datetime64) or not len(hours)
            or np.isnat(hours).any() or np.any(np.diff(hours) <= np.timedelta64(0, 'h'))):
        raise ValueError('Bike timestamps must be nonempty unique increasing datetimes')
    hours = hours.astype('datetime64[h]')
    splits = []
    for fold in range(5):
        start, end = (np.datetime64(t, 'h') for t in BIKE_BOUNDARIES[fold:fold+2])
        days = np.unique(hours[hours < start].astype('datetime64[D]'))
        first, second = int(.7*len(days)), int(.85*len(days))
        if min(first, second-first, len(days)-second) < 2:
            raise ValueError('Bike development needs at least two dates in F/G/V')
        boundaries = dict(G=days[first].astype('datetime64[h]'),
                          V=days[second].astype('datetime64[h]'), T=start)
        roles = np.full(len(hours), 'unused_future', dtype='<U13')
        roles[hours < boundaries['G']] = 'F'
        roles[(hours >= boundaries['G']) & (hours < boundaries['V'])] = 'G'
        roles[(hours >= boundaries['V']) & (hours < start)] = 'V'
        roles[(hours >= start) & (hours < end)] = 'T'
        omissions = {}
        for role, boundary in boundaries.items():
            omit = (roles == role) & (hours < boundary+np.timedelta64(24, 'h'))
            omissions[role] = np.flatnonzero(omit).tolist()
            roles[omit] = 'embargo'
        record = dict(schema='research/v2-transfer-split', dataset='bike', outer_fold=fold,
            primary=fold == 4, roles=roles.tolist(), split_unit='day',
            actual_rows={role: int(np.sum(roles == role)) for role in ('F', 'G', 'V', 'T', 'embargo', 'unused_future')},
            role_boundaries={role: str(value) for role, value in boundaries.items()},
            test_end_exclusive=str(end), embargo_rows=int(np.sum(roles == 'embargo')),
            unused_future_rows=int(np.sum(roles == 'unused_future')), embargo_omitted_row_indices=omissions,
            embargo_hours=24, development_date_counts=dict(F=first, G=second-first, V=len(days)-second))
        record['split_hash'] = canonical_hash(record)
        splits.append(record)
    return splits
