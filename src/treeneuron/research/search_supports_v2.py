"""Fixed semantic HPO supports and native-schema compilation; no fit/admission."""
import copy
import hashlib
import importlib
import math
from pathlib import Path

from treeneuron.config import canonical_hash


_ROOT=Path(__file__).resolve().parents[3]
_PARSERS={**{f:('gated_config','GatedConfig',True) for f in ('P00','P01','P02','P05')},
    **{f:('ordinary_neural','OrdinaryNeuralConfig',True) for f in ('P03','P04')},
    'P06':('realmlp_native','RealMLPConfig',False),'P07':('tabm_native','TabMConfig',False),
    **{f:('native_trees','NativeTreeConfig',True) for f in ('P08','P09','P10','P11')},
    'P12':('node_v2','NodeV2Config',False)}


def _cat(*values):return dict(kind='categorical',choices=list(values))
def _float(low,high,log=False):return dict(kind='float',low=low,high=high,log=log)
def _integer(low,high,step=1):return dict(kind='integer',low=low,high=high,step=step,log=False)


def _optional(name,low,high,*,log=True):
    return {name+'_mode':_cat('zero','positive'),name+'_value':_float(low,high,log)}


def _schema(value):
    from .native_trees import _schema as check
    result=check(value)
    if len(result['columns'])>128:raise ValueError('Raw semantic schema exceeds the bounded v2 worker support')
    return result


def _parse(family,config):
    if family=='P13':
        from .p13_simple import _config
        return _config(config)
    module,name,with_family=_PARSERS[family]
    parser=getattr(importlib.import_module('treeneuron.research.'+module),name)
    return (parser.parse(family,config) if with_family else parser.parse(config)).to_state()


def _config(family,schema,parameters,*,mode='native_anchor',variant=None):
    name=('gated-native' if family in ('P00','P01','P02','P05') else
        'ordinary-neural' if family in ('P03','P04') else 'realmlp' if family=='P06' else
        'tabm' if family=='P07' else 'node' if family=='P12' else 'native-tree')
    result=dict(schema='research/v2-'+name+'-config',family=family,mode=mode,
        feature_schema=copy.deepcopy(schema),parameters=parameters)
    if family=='P12':result['execution']='bounded_memory_v1'
    if variant is not None:result['variant']=variant
    return _parse(family,result)


def _support(family):
    if family in ('P06','P13'):return None
    if family in ('P00','P01','P02','P05','P03','P04'):
        result=dict(lr=_float(1e-4,1e-2,True),batch_size=_cat(128,256),**_optional('weight_decay',1e-7,1e-2))
        if family in ('P00','P01','P02','P05'):
            result.update(width=_cat(32,64,128) if family=='P05' else _cat(64,128),scheduler=_cat('constant','cosine'))
            if family=='P05':result.update(variant=_cat('centered_partial','centered_full','direct_partial','direct_full'),dropout=_cat(0.,.1,.2))
            else:
                result['growth_fraction']=_cat(1/8,1/4,1/2)
                if family=='P02':result['variant']=_cat('lower','upper')
        elif family=='P03':result.update(depth=_cat(1,2,3),width=_cat(64,128,256,512),activation=_cat('ReLU','GELU'),dropout=_cat(0.,.1,.2))
        else:result.update(blocks=_cat(2,3,4),width=_cat(128,256,512),hidden_dropout=_cat(0.,.1,.2),residual_dropout=_cat(0.,.1,.2))
        return result
    if family=='P07':return dict(variant=_cat('plain','piecewise'),n_blocks=_cat(2,3,4),d_block=_integer(64,1024,16),
        lr=_float(1e-4,5e-3,True),**_optional('weight_decay',1e-4,.1),**_optional('dropout',0.,.5,log=False),
        n_bins=_integer(2,128),d_embedding=_integer(8,32,4))
    if family=='P08':return dict(depth=_cat(4,6,8,10),learning_rate=_float(.01,.3,True),l2_leaf_reg=_float(1.,100.,True))
    if family=='P09':return dict(max_depth=_cat(3,5,7,9),eta=_float(.01,.3,True),min_child_weight=_float(1.,100.,True),
        subsample=_float(.6,1.),colsample_bytree=_float(.6,1.),reg_lambda=_float(1e-3,100.,True),**_optional('reg_alpha',1e-4,10.))
    if family=='P10':return dict(num_leaves=_cat(15,31,63,127),learning_rate=_float(.01,.2,True),
        min_data_in_leaf=_cat(5,10,20,50,100),feature_fraction=_float(.6,1.),bagging_fraction=_float(.6,1.),**_optional('lambda_l2',1e-3,100.))
    if family=='P11':return dict(num_leaves=_cat(7,15,31),learning_rate=_float(.01,.2,True),
        min_data_in_leaf=_cat(20,50,100),**_optional('lambda_l2',1e-3,100.),**_optional('linear_lambda',1e-3,100.))
    return dict(trees=_cat(512,1024,2048),depth=_cat(4,6),lr_multiplier=_cat(.5,1,2))


def family_search_spec(family,feature_schema):
    """Return a copied, hash-bound declaration; it cannot admit search execution."""
    if type(family) is not str or family not in (*_PARSERS,'P13'):raise ValueError('Mandatory family P00–P13 required')
    schema=_schema(feature_schema);anchors=[];finite=[]
    def anchor(name,config):anchors.append(dict(anchor_id=name,config=config))
    if family=='P13':
        from .p13_simple import p13_configurations
        finite=p13_configurations(schema);anchor('mean',finite[0]);kind='finite'
    else:
        variants=('lower','upper') if family=='P02' else ('centered_partial','centered_full','direct_partial','direct_full') if family=='P05' else ('plain','piecewise') if family=='P07' else (None,)
        for variant in variants:anchor('native_'+(variant or 'anchor'),_config(family,schema,{},variant=variant))
        if family=='P03':anchor('historical_width_lr_coordinate_under_native_preprocessing',_config(family,schema,
            dict(hidden_dims=[64],activation='ReLU',dropout=0.,lr=.005,weight_decay=0.,batch_size=128),mode='search'))
        kind='native_official_schedule' if family=='P06' else 'tpe'
    module='p13_simple' if family=='P13' else _PARSERS[family][0]
    sources={str(p.relative_to(_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__),_ROOT/'src/treeneuron/research'/f'{module}.py')}
    if family=='P06':
        for relative in ('src/treeneuron/research/realmlp_search.py','locks/research-v2/realmlp/source-lock.json'):
            sources[relative]=hashlib.sha256((_ROOT/relative).read_bytes()).hexdigest()
    result=dict(schema='research/v2-family-search-support',family_id=family,feature_schema=schema,
        search_kind=kind,anchors=anchors,support=_support(family),finite_configurations=finite,
        finite_schedule_includes_anchor=family=='P13',finite_unique_cap=29 if family=='P13' else 32,
        native_sampler='realmlp_search.realmlp_schedule' if family=='P06' else None,
        max_attempts=32,deadline_ns=7200000000000,descriptive_prefixes_ns=[600000000000,1800000000000],
        all_anchors_failures_and_duplicate_draws_paid=True,anchors_in_TPE_surrogate=False,
        conditional_coordinates='Inactive zero/value and plain-embedding coordinates are retained in the raw paid draw',
        source_bindings=sources,execution_authorized=False,native_workload_qualified=False,protected_access_authorized=False)
    result['support_hash']=canonical_hash(result)
    return copy.deepcopy(result)


def _draw(support,value):
    if type(value) is not dict or set(value)!=set(support):raise ValueError('Complete exact raw search draw required')
    for name,spec in support.items():
        item=value[name];kind=spec['kind']
        if kind=='categorical':valid=any(type(item) is type(choice) and item==choice for choice in spec['choices'])
        elif kind=='integer':valid=type(item) is int and spec['low']<=item<=spec['high'] and (item-spec['low'])%spec['step']==0
        else:
            try:valid=type(item) in (int,float) and spec['low']<=item<=spec['high'] and math.isfinite(item)
            except OverflowError:valid=False
        if not valid:raise ValueError(f'Search coordinate outside the fixed support: {name}')
    return copy.deepcopy(value)


def compile_search_draw(family,feature_schema,draw):
    """Compile one paid atomic TPE draw; never resample or fit an adapter."""
    spec=family_search_spec(family,feature_schema)
    if spec['search_kind']!='tpe':raise ValueError('This family requires its native official schedule or finite exhaustive support')
    parameters=_draw(spec['support'],draw);variant=parameters.pop('variant',None)
    for name in ('weight_decay','dropout','reg_alpha','lambda_l2','linear_lambda'):
        if name+'_mode' in parameters:
            mode=parameters.pop(name+'_mode');value=parameters.pop(name+'_value')
            parameters[name]=0. if mode=='zero' else value
    if family=='P03':parameters['hidden_dims']=[parameters.pop('width')]*parameters.pop('depth')
    if family=='P07' and variant=='plain':
        parameters.pop('n_bins');parameters.pop('d_embedding')
    return _config(family,spec['feature_schema'],parameters,mode='search',variant=variant)
