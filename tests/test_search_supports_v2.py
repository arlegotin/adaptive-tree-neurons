"""Frozen semantic ranges compile to native schemas without fitting or labels."""
import copy
import importlib.util

import pytest


SCHEMA=dict(columns=['x','kind'],categorical=['kind'])
FAMILIES=[f'P{i:02}' for i in range(14)]


def api():
    assert importlib.util.find_spec('treeneuron.research.search_supports_v2') is not None, 'Missing fixed native family search supports'
    from treeneuron.research import search_supports_v2
    return search_supports_v2


def parse(family,value):
    if family in ('P00','P01','P02','P05'):
        from treeneuron.research.gated_config import GatedConfig
        return GatedConfig.parse(family,value).to_state()
    if family in ('P03','P04'):
        from treeneuron.research.ordinary_neural import OrdinaryNeuralConfig
        return OrdinaryNeuralConfig.parse(family,value).to_state()
    if family=='P06':
        from treeneuron.research.realmlp_native import RealMLPConfig
        return RealMLPConfig.parse(value).to_state()
    if family=='P07':
        from treeneuron.research.tabm_native import TabMConfig
        return TabMConfig.parse(value).to_state()
    if family in ('P08','P09','P10','P11'):
        from treeneuron.research.native_trees import NativeTreeConfig
        return NativeTreeConfig.parse(family,value).to_state()
    if family=='P12':
        from treeneuron.research.node_v2 import NodeV2Config
        return NodeV2Config.parse(value).to_state()
    from treeneuron.research.p13_simple import _config
    return _config(value)


def draw(support,high=False):
    return {name:spec['choices'][-1 if high else 0] if spec['kind']=='categorical' else spec['high' if high else 'low']
        for name,spec in support.items()}


@pytest.mark.parametrize('family',FAMILIES)
def test_every_family_retains_native_anchors_and_declared_budget_without_qualification(family):
    module=api();spec=module.family_search_spec(family,SCHEMA)
    assert spec['family_id']==family and spec['max_attempts']==32 and spec['deadline_ns']==7200000000000
    assert spec['descriptive_prefixes_ns']==[600000000000,1800000000000]
    assert spec['execution_authorized'] is spec['native_workload_qualified'] is spec['protected_access_authorized'] is False
    assert spec['anchors'] and all(parse(family,a['config'])==a['config'] for a in spec['anchors'])
    if family=='P06':
        assert spec['search_kind']=='native_official_schedule' and spec['support'] is None
        assert spec['native_sampler']=='realmlp_search.realmlp_schedule'
    elif family=='P13':
        assert spec['search_kind']=='finite' and spec['finite_unique_cap']==29
        assert len(spec['finite_configurations'])==29 and spec['finite_configurations'][0]==spec['anchors'][0]['config']
        assert all(parse(family,c)==c for c in spec['finite_configurations'])
    else:
        assert spec['search_kind']=='tpe'
        from treeneuron.research.search_v2 import _support
        assert _support(spec['support'])==spec['support']
        for high in (False,True):
            cfg=module.compile_search_draw(family,SCHEMA,draw(spec['support'],high))
            assert cfg['mode']=='search' and parse(family,cfg)==cfg


def test_fixed_anchor_variants_and_historical_coordinate_are_explicit_without_changing_native_recipes():
    module=api();expected={'P00':1,'P01':1,'P02':2,'P03':2,'P04':1,'P05':4,'P06':1,'P07':2,
        'P08':1,'P09':1,'P10':1,'P11':1,'P12':1,'P13':1}
    assert {f:len(module.family_search_spec(f,SCHEMA)['anchors']) for f in FAMILIES}==expected
    mlp=module.family_search_spec('P03',SCHEMA)['anchors']
    assert mlp[0]['config']['mode']=='native_anchor' and mlp[0]['config']['parameters']=={}
    assert mlp[1]['config']['parameters']['hidden_dims']==[64] and mlp[1]['config']['parameters']['lr']==.005
    assert 'coordinate' in mlp[1]['anchor_id']
    p04=module.family_search_spec('P04',SCHEMA);assert .25 not in p04['support']['hidden_dropout']['choices']
    assert {a['config']['variant'] for a in module.family_search_spec('P07',SCHEMA)['anchors']}=={'plain','piecewise'}


def test_selected_and_random_receive_the_exact_same_support_directions_count_and_treatment():
    module=api();a=module.family_search_spec('P00',SCHEMA);b=module.family_search_spec('P01',SCHEMA)
    assert a['support']==b['support']
    for high in (False,True):
        value=draw(a['support'],high);left=module.compile_search_draw('P00',SCHEMA,value);right=module.compile_search_draw('P01',SCHEMA,value)
        left.pop('family');right.pop('family');assert left==right


def test_compiler_translates_width_depth_zero_regularization_and_embedding_conditions():
    module=api();spec=module.family_search_spec('P03',SCHEMA);value=draw(spec['support'])
    value.update(depth=3,width=128,weight_decay_mode='positive',weight_decay_value=.002)
    cfg=module.compile_search_draw('P03',SCHEMA,value)
    assert cfg['parameters']['hidden_dims']==[128]*3 and cfg['parameters']['weight_decay']==.002
    value['weight_decay_mode']='zero';assert module.compile_search_draw('P03',SCHEMA,value)['parameters']['weight_decay']==0.
    spec=module.family_search_spec('P07',SCHEMA);value=draw(spec['support']);value['variant']='plain'
    plain=module.compile_search_draw('P07',SCHEMA,value);assert not {'n_bins','d_embedding'}&set(plain['parameters'])
    value['variant']='piecewise';embedded=module.compile_search_draw('P07',SCHEMA,value)
    assert {'n_bins','d_embedding'}<=set(embedded['parameters'])


@pytest.mark.parametrize('change',['unknown','missing','nonfinite','bool_integer','off_step','unadmitted_direction'])
def test_invalid_or_outside_support_draws_fail_instead_of_being_replaced(change):
    module=api();family='P07' if change=='off_step' else 'P00';spec=module.family_search_spec(family,SCHEMA);value=draw(spec['support'])
    if change=='unknown':value['extra']=1
    elif change=='missing':value.pop('lr')
    elif change=='nonfinite':value['lr']=float('nan')
    elif change=='bool_integer':value['width']=True
    elif change=='off_step':value['d_block']=65
    else:value['direction']='new_direction'
    with pytest.raises(ValueError):module.compile_search_draw(family,SCHEMA,value)


@pytest.mark.parametrize('family',['P06','P13'])
def test_native_sampler_and_exhaustive_family_cannot_be_silently_replaced_by_generic_TPE(family):
    module=api()
    with pytest.raises(ValueError):module.compile_search_draw(family,SCHEMA,{})


def test_input_or_returned_dictionary_mutation_cannot_change_later_support_or_its_commitment():
    module=api();schema=copy.deepcopy(SCHEMA);first=module.family_search_spec('P00',schema);digest=first['support_hash']
    schema['columns'].append('later');first['support']['width']['choices'].append(999)
    second=module.family_search_spec('P00',SCHEMA)
    assert second['support_hash']==digest and second['support']['width']['choices']==[64,128]
