"""Portable F-score artifacts and exact finite M2 candidate bindings."""
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from treeneuron.growth.allocate import allocate
from treeneuron.growth.score import ScoreTable
from .diagnostics import ScoredParent,_parent
from .ledger import write_new_json


def save_scores(root,scored):
    if (not isinstance(scored,ScoredParent) or scored.table.table_hash!=scored.table_sha256
            or type(scored.elapsed_ns) is not int or scored.elapsed_ns<0):
        raise ValueError('Invalid bound score table')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    arrays,metadata={},{}
    for field in fields(ScoreTable):
        value=getattr(scored.table,field.name)
        if isinstance(value,torch.Tensor):
            arrays[field.name]=value.detach().cpu().numpy()
        else:
            metadata[field.name]=value
    path=root/'arrays.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
    write_new_json(root/'scores.json',dict(schema='research/v2-F-scores',
        parent_sha256=scored.parent_sha256,table_sha256=scored.table_sha256,
        elapsed_ns=scored.elapsed_ns,metadata=metadata,array_keys=sorted(arrays),
        arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def load_scores(root,parent):
    root=Path(root);record=json.loads((root/'scores.json').read_text())
    required={'schema','parent_sha256','table_sha256','elapsed_ns','metadata','array_keys','arrays_sha256'}
    if (set(record)!=required or record['schema']!='research/v2-F-scores'
            or record['parent_sha256']!=_parent(parent)[1]):
        raise ValueError('F-score binding belongs to a changed parent/schema')
    path=root/'arrays.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['arrays_sha256']:
        raise ValueError('F-score array checksum mismatch')
    with np.load(path,allow_pickle=False) as values:
        if sorted(values.files)!=record['array_keys'] or set(values.files)&set(record['metadata']):
            raise ValueError('F-score array inventory mismatch')
        kwargs={**record['metadata'],**{k:torch.from_numpy(values[k].copy()) for k in values.files}}
    kwargs['status']=tuple(kwargs['status'])
    table=ScoreTable(**kwargs)
    if table.table_hash!=record['table_sha256']:
        raise ValueError('F-score table checksum mismatch')
    if type(record['elapsed_ns']) is not int or record['elapsed_ns']<0:
        raise ValueError('Invalid measured scoring time')
    return ScoredParent(table,record['parent_sha256'],record['table_sha256'],record['elapsed_ns'])


def horizon_slots(table,seed):
    """Bind all nine nominal slots using F-only scores and a local RNG."""
    eligible=set(allocate(table,len(table.ids),policy='adam',seed=seed).ids.tolist())
    rows={int(unit):i for i,unit in enumerate(table.ids.tolist())}
    ranked=sorted(eligible,key=lambda unit:(-float(table.adam_scores[rows[unit]]),unit))
    top=ranked[:4]
    remaining=sorted(eligible-set(top))
    order=np.random.default_rng(seed).permutation(len(remaining))[:4]
    sampled=[remaining[int(i)] for i in order]
    slots=[dict(slot='keep',role='keep',unit_ids=[],status='bound')]
    for label,chosen in (('top',top),('sampled',sampled)):
        for i in range(4):
            available=i<len(chosen)
            slots.append(dict(slot=f'{label}-{i}',role='subset',unit_ids=[chosen[i]] if available else [],
                status='bound' if available else 'completed_ineligible',
                reason=None if available else 'common eligible pool has no candidate for this nominal slot'))
    return slots


def pair_oracle(records):
    """G-selected pair only; all nominal failed/ineligible pairs stay counted."""
    pairs=[r for r in records if len(r['unit_ids'])==2]
    eligible=[r for r in pairs if r['status']=='completed_eligible']
    for row in eligible:
        score=row['selected_g_mse']
        if (type(score) not in (int,float) or not math.isfinite(score) or score<0
                or row['unit_ids']!=sorted(set(row['unit_ids']))):
            raise ValueError('Pair oracle requires finite risks and canonical unit-ID pairs')
    if not eligible:
        return dict(status='completed_ineligible',unit_ids=None,attempted_pair_count=len(pairs),
            reason='No eligible fitted pair; no teacher/subset replacement')
    best=min(r['selected_g_mse'] for r in eligible)
    winner=min((r for r in eligible if r['selected_g_mse']<=best+1e-12*max(1.,abs(best))),
        key=lambda r:tuple(r['unit_ids']))
    return dict(status='completed_eligible',unit_ids=list(winner['unit_ids']),
        selected_g_mse=winner['selected_g_mse'],attempted_pair_count=len(pairs),
        selection_role='G',selection_metric='original_unit_mse')
