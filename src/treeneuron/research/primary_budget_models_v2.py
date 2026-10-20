"""Byte-bind models retained at each fixed search view; no native authority."""
import copy
from pathlib import Path


_PREFIXES=(600_000_000_000,1800_000_000_000,7200_000_000_000)


def inspect_budget_model_views(*,search_result,candidate_models):
    """Validate supplied scores/bytes; actual native/HPO provenance is separate."""
    from treeneuron.analysis.practical_v2 import select_completed_trial
    from treeneuron.config import canonical_json
    from .qualified_primary_models_v2 import inspect_selected_snapshot_binding
    from .primary_search_worker_v2 import _digest
    if type(search_result) is not dict or type(candidate_models) is not dict:
        raise ValueError('Complete supplied search result and candidate model map required')
    records=search_result.get('records');total=search_result.get('total_charged_ns')
    if (type(records) is not list or type(total) is not int or total<0
            or records and total<records[-1]['completion_ns']):
        raise ValueError('Actual complete search endpoint and paid records required')
    native=select_completed_trial(records,deadline_ns=total)
    prefixes={str(t):select_completed_trial(records,deadline_ns=t) for t in _PREFIXES}
    if (canonical_json(dict(winner=search_result.get('winner')))!=canonical_json(dict(winner=native['winner']))
            or canonical_json(dict(prefixes=search_result.get('prefixes')))!=canonical_json(dict(prefixes=prefixes))):
        raise ValueError('Native or prefix selections contradict completion times/V ties')
    eligible={row['attempt_index'] for row in records if row['status']=='completed_eligible'}
    if set(candidate_models)!=eligible or any(type(i) is not int for i in candidate_models):
        raise ValueError('Every eligible paid model needs evidence, with no extra or boolean index')
    models={};fields={'selected_metadata','snapshot_root','snapshot_inventory_sha256','generation_sha256'}
    for index,model in candidate_models.items():
        if type(model) is not dict or set(model)!=fields or not _digest(model['generation_sha256']):
            raise ValueError('Exact candidate snapshot and generation metadata required')
        inspect_selected_snapshot_binding(selected_model=model['selected_metadata'],expected_metadata=model['selected_metadata'],
            snapshot_root=Path(model['snapshot_root']),expected_inventory_sha256=model['snapshot_inventory_sha256'])
        models[index]=dict(attempt_index=index,**copy.deepcopy(model))
    views={}
    for name,selection in [('native_quality',native),*prefixes.items()]:
        winner=selection['winner'];deadline=None if name=='native_quality' else int(name)
        views[name]=dict(status='retained_model_bytes' if winner is not None else
            'unavailable_no_eligible_model' if deadline is None else 'unavailable_at_prefix',
            deadline_ns=deadline,V_score=None if winner is None else winner['V_score'],
            model_binding=None if winner is None else models[winner['attempt_index']])
    return dict(schema='research/v2-primary-budget-model-bindings',views=views,
        native_model_generation_authenticated=False,protected_access_authorized=False,
        scope='Supplied completion/V arithmetic and exact snapshot bytes only. Actual native generation, HPO provenance, protocol and protected authority remain separate.')
