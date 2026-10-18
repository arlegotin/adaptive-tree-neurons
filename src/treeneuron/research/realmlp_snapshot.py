"""Closed, inference-only serialization of explicitly pinned native RealMLP layers.

No class/function is imported from a manifest string. Fitting/controller objects
are omitted; native tensor-dictionary evaluation does not read them. This format
cannot resume training. Source/dependency/external-digest checks live in the owner.
"""
from __future__ import annotations

import copy
import functools
import math
import numpy as np


def _plain(value):
    if isinstance(value,np.generic):value=value.item()
    if value is None or type(value) in (str,bool,int):return value
    if type(value) is float and math.isfinite(value):return value
    if isinstance(value,(list,tuple)):return [_plain(x) for x in value]
    if type(value) is dict and all(type(k) is str for k in value):return {k:_plain(v) for k,v in value.items()}
    raise ValueError('Unsupported/nonfinite closed native graph scalar')


def _registry():
    from pytabkit.models.nn_models import base,nn,activations,categorical
    classes=[base.SequentialLayer,base.IdentityLayer,base.ConcatParallelLayer,base.FilterTensorsLayer,
        base.RenameTensorLayer,base.FunctionLayer,base.BiasLayer,base.ScaleLayer,base.WeightLayer,
        activations.ParametricActivationLayer,nn.DropoutLayer,nn.NormalizeOutputLayer,nn.ClampLayer,
        nn.PLREmbeddingsLayer,nn.PLREmbeddingsLayerCosBias,categorical.EncodingLayer,
        categorical.SingleOneHotLayer,categorical.SingleEmbeddingLayer]
    return {c.__name__:c for c in classes}


_ATTRIBUTES={
    'SequentialLayer':(), 'IdentityLayer':(), 'ConcatParallelLayer':(),
    'FilterTensorsLayer':('include_keys','exclude_keys'),
    'RenameTensorLayer':('old_name','new_name'), 'FunctionLayer':('function',),
    'BiasLayer':('factor',), 'ScaleLayer':(), 'WeightLayer':('factor',),
    'ParametricActivationLayer':('function',), 'DropoutLayer':('evaluation_probability',),
    'NormalizeOutputLayer':(), 'ClampLayer':(),
    'PLREmbeddingsLayer':('plr_act_name','plr_use_densenet'),
    'PLREmbeddingsLayerCosBias':('plr_act_name','plr_use_densenet'),
    'EncodingLayer':('enc_output_name',),
    'SingleOneHotLayer':('onoff','cat_size','use_missing_zero','use_1d_binary_onehot'),
    'SingleEmbeddingLayer':()}
_CHILDREN={'SequentialLayer':'tfms','ConcatParallelLayer':'layers','EncodingLayer':'emb_layers'}
_PARAMETERS={'BiasLayer':('bias',),'ScaleLayer':('scale',),'WeightLayer':('weight',),
    'ParametricActivationLayer':('weight',),'NormalizeOutputLayer':('mean','std'),
    'ClampLayer':('low','high'),'PLREmbeddingsLayer':('weight_1','weight_2','bias_2'),
    'PLREmbeddingsLayerCosBias':('weight_1','bias_1','weight_2','bias_2'),
    'SingleEmbeddingLayer':('emb',)}


def _function(value):
    import torch
    from pytabkit.models.nn_models.activations import mish
    from pytabkit.models.nn_models.models import smooth_clip_func
    for name,function in (('relu',torch.relu),('selu',torch.selu),('mish',mish)):
        if value is function:return dict(name=name)
    if (type(value) is functools.partial and value.func is smooth_clip_func and not value.args
            and value.keywords=={'max_abs_value':3.0}):
        return dict(name='smooth_clip_func',max_abs_value=3.0)
    raise ValueError('Unqualified native graph function')


def _read_function(value):
    import torch
    from pytabkit.models.nn_models.activations import mish
    from pytabkit.models.nn_models.models import smooth_clip_func
    if value=={'name':'smooth_clip_func','max_abs_value':3.0}:
        return functools.partial(smooth_clip_func,max_abs_value=3.0)
    if value=={'name':'relu'}:return torch.relu
    if value=={'name':'selu'}:return torch.selu
    if value=={'name':'mish'}:return mish
    raise ValueError('Unknown/unqualified closed graph function')


def encode_graph(model,prefix,arrays):
    from pytabkit.models.nn_models.base import Variable
    registry=_registry();name=type(model).__name__
    if registry.get(name) is not type(model):raise ValueError('Unknown native layer in closed graph')
    attrs={}
    for key in _ATTRIBUTES[name]:
        if key=='function':attrs[key]=_function(model.f)
        elif key=='evaluation_probability':attrs[key]=float(model.hyper_getter())
        else:attrs[key]=_plain(getattr(model,key))
    parameters={}
    actual=dict(model.named_parameters(recurse=False))|dict(model.named_buffers(recurse=False))
    if set(actual)!=set(_PARAMETERS.get(name,())):
        raise ValueError('Unqualified native layer tensor fields')
    for key,value in actual.items():
        if type(value) is not Variable:raise ValueError('Native parameter lost its Variable contract')
        array=value.detach().cpu().numpy().copy()
        if array.dtype!=np.float32 or not np.isfinite(array).all():raise ValueError('Invalid native tensor dtype/finiteness')
        tensor_key=prefix+'/'+key
        arrays[tensor_key]=array
        parameters[key]=dict(tensor=tensor_key,trainable=bool(value.trainable),requires_grad=bool(value.requires_grad),
            hyper_factors=_plain(value.hyper_factors),scope=list(value.context.scope._names))
    children=[]
    if name in _CHILDREN:
        for i,child in enumerate(getattr(model,_CHILDREN[name])):
            children.append(encode_graph(child,prefix+'/'+str(i),arrays))
    return dict(type=name,attributes=attrs,parameters=parameters,children=children,
        scope=list(model.context.scope._names),remove_keys=_plain(model.remove_keys),
        tensor_infos={k:v.to_dict() for k,v in model.new_tensor_infos.items()},
        omitted_training_fitter=type(model.fitter).__module__+'.'+type(model.fitter).__qualname__ if model.fitter else None)


def decode_graph(node,arrays,used,depth=0):
    import torch
    from pytabkit.models.nn_models.base import Variable,Scope,set_scope_context,set_hp_context
    from pytabkit.models.training.coord import HyperparamManager
    from pytabkit.models.data.data import TensorInfo
    registry=_registry()
    fields={'type','attributes','parameters','children','scope','remove_keys','tensor_infos','omitted_training_fitter'}
    if type(node) is not dict or set(node)!=fields or node.get('type') not in registry or depth>32:
        raise ValueError('Invalid/unknown closed native graph node')
    name=node['type'];attrs=node['attributes'];params=node['parameters'];scope=node['scope']
    if (type(attrs) is not dict or set(attrs)!=set(_ATTRIBUTES[name]) or type(params) is not dict
            or set(params)!=set(_PARAMETERS.get(name,())) or type(node['children']) is not list
            or len(node['children'])>256 or name not in _CHILDREN and node['children']
            or type(scope) is not list or any(type(x) is not str for x in scope)):
        raise ValueError('Closed native graph fields differ from qualified constructor')
    _plain(attrs)
    if name=='FilterTensorsLayer' and attrs not in (
            dict(include_keys=['x_cont'],exclude_keys=None),dict(include_keys=None,exclude_keys=['x_cont'])):
        raise ValueError('Unqualified native tensor filter')
    if name=='RenameTensorLayer' and attrs!=dict(old_name='x_one_hot',new_name='x_cont'):
        raise ValueError('Unqualified native tensor rename')
    if name=='EncodingLayer' and attrs['enc_output_name'] not in ('x_cont','x_one_hot'):
        raise ValueError('Unqualified native encoding output')
    if name=='SingleOneHotLayer' and (type(attrs['cat_size']) is not int or not 1<=attrs['cat_size']<=9
            or attrs['onoff']!=[1.,0.] or attrs['use_missing_zero'] is not True
            or attrs['use_1d_binary_onehot'] is not True):
        raise ValueError('Unqualified native one-hot representation')
    if name in ('BiasLayer','WeightLayer') and (type(attrs['factor']) not in (float,int)
            or not math.isfinite(attrs['factor']) or attrs['factor']<=0):
        raise ValueError('Invalid native affine factor')
    expected_remove='x_cat' if name=='SingleEmbeddingLayer' else None
    if node['remove_keys']!=expected_remove:raise ValueError('Unqualified native tensor removal')
    variables={}

    for key,value in params.items():
        if (type(value) is not dict or set(value)!={'tensor','trainable','requires_grad','hyper_factors','scope'}
                or type(value['trainable']) is not bool or type(value['requires_grad']) is not bool
                or type(value['hyper_factors']) is not dict or type(value['scope']) is not list
                or any(type(x) is not str for x in value['scope'])):
            raise ValueError('Invalid native Variable declaration')
        tensor=value['tensor']
        if tensor in used or tensor not in arrays:raise ValueError('Missing/duplicated native tensor reference')
        array=arrays[tensor]
        if array.dtype!=np.float32 or not 1<=array.ndim<=4 or not np.isfinite(array).all():
            raise ValueError('Native tensor dtype/shape/finiteness differs')
        if any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in value['hyper_factors'].values()):
            raise ValueError('Invalid native Variable hyper factor')
        with set_scope_context(Scope(value['scope'])):
            variables[key]=Variable(torch.from_numpy(array.copy()),trainable=value['trainable'],
                requires_grad=value['requires_grad'],hyper_factors=value['hyper_factors'])
        used.add(tensor)
    children=[decode_graph(child,arrays,used,depth+1) for child in node['children']]
    probability=attrs.get('evaluation_probability',0.)
    if type(probability) not in (float,int) or not 0<=probability<=1:raise ValueError('Invalid inference dropout probability')
    with set_scope_context(Scope(scope)),set_hp_context(HyperparamManager(p_drop=probability)):
        constructor=registry[name]
        if name=='SequentialLayer':model=constructor(children)
        elif name=='IdentityLayer':model=constructor()
        elif name=='ConcatParallelLayer':model=constructor(children,fitter=None)
        elif name=='FilterTensorsLayer':model=constructor(attrs['include_keys'],attrs['exclude_keys'],fitter=None)
        elif name=='RenameTensorLayer':model=constructor(attrs['old_name'],attrs['new_name'],fitter=None)
        elif name=='FunctionLayer':model=constructor(_read_function(attrs['function']))
        elif name=='ParametricActivationLayer':model=constructor(_read_function(attrs['function']),variables['weight'])
        elif name=='DropoutLayer':model=constructor()
        elif name=='EncodingLayer':model=constructor(children,attrs['enc_output_name'],fitter=None)
        elif name=='SingleOneHotLayer':model=constructor(fitter=None,**attrs)
        elif name in ('PLREmbeddingsLayer','PLREmbeddingsLayerCosBias'):model=constructor(fitter=None,**variables,**attrs)
        else:model=constructor(**variables,**attrs)
    if type(node['tensor_infos']) is not dict or not set(node['tensor_infos'])<={'x_cont','x_cat','y'}:
        raise ValueError('Invalid native tensor metadata')
    for info in node['tensor_infos'].values():
        if type(info) is not dict or set(info)!={'feat_shape','cat_sizes'}:raise ValueError('Invalid native tensor-info fields')
    model.new_tensor_infos={k:TensorInfo.from_dict(v) for k,v in node['tensor_infos'].items()}
    model.remove_keys=node['remove_keys']
    model.eval()
    return model


def converter_state(converter,schema):
    import pandas as pd
    cats=[];missing=[]
    if schema['categorical']:
        encoder=converter.cat_tf.named_transformers_['categorical']
        for array in encoder.categories_:
            cats.append([x for x in array.tolist() if not pd.isna(x)])
            missing.append(any(pd.isna(x) for x in array))
    return dict(schema=copy.deepcopy(schema),categories=cats,F_missing=missing,
        cardinalities=converter.tensor_infos['x_cat'].get_cat_sizes().tolist())


def restore_converter(value):
    import pandas as pd
    from pytabkit.models.data.conversion import ToDictDatasetConverter
    from treeneuron.research.native_trees import _schema
    if type(value) is not dict or set(value)!={'schema','categories','cardinalities','F_missing'}:
        raise ValueError('Invalid F-only converter snapshot')
    schema=_schema(value['schema']);categories=value['categories'];cards=value['cardinalities'];missing=value['F_missing']
    names=[x for x in schema['columns'] if x in schema['categorical']]
    if (type(categories) is not list or len(categories)!=len(names) or type(cards) is not list
            or len(cards)!=len(names) or any(type(k) is not int or not 1<=k<=4096 for k in cards)
            or sum(cards)>4096 or type(missing) is not list or len(missing)!=len(names)
            or any(type(flag) is not bool for flag in missing)):
        raise ValueError('Invalid F vocabulary cardinality/missing declaration')
    for array,card,flag in zip(categories,cards,missing,strict=True):
        if (type(array) is not list or len(array)!=card-1 or any(type(x) is not str for x in array)
                or array!=sorted(set(array)) or not array and not flag):
            raise ValueError('Invalid F category order/value/missingness')
    n=max([len(x)+int(flag) for x,flag in zip(categories,missing,strict=True)],default=1)
    columns={name:np.zeros(n,np.float32) for name in schema['columns']}
    expected_codes=[]
    for name,array,flag in zip(names,categories,missing,strict=True):
        period=len(array)+int(flag)
        indexes=[i%period for i in range(n)]
        columns[name]=pd.Series([array[i] if i<len(array) else np.nan for i in indexes],dtype=object)
        expected_codes.append([i+1 if i<len(array) else 0 for i in indexes])
    carrier=pd.DataFrame(columns)
    converter=ToDictDatasetConverter(cat_features=[x in names for x in schema['columns']],verbosity=0)
    converted=converter.fit_transform(carrier)
    if converter_state(converter,schema)!=value:raise ValueError('Reconstructed F vocabulary changed category order/cardinality/missingness')
    for i,expected in enumerate(expected_codes):
        if converted.tensors['x_cat'][:,i].tolist()!=expected:raise ValueError('Reconstructed native category codes differ')
    return converter


def walk(node):
    yield node
    for child in node['children']:yield from walk(child)


def _operation(node):
    name=node['type']
    if name=='ConcatParallelLayer':
        return dict(type=name,attributes=node['attributes'],branches=[_operations(child) for child in node['children']])
    return dict(type=name,attributes=node['attributes'],children=[_operation(child) for child in node['children']])


def _operations(node):
    if node['type']=='SequentialLayer':
        return [operation for child in node['children'] for operation in _operations(child)]
    return [_operation(node)]


def _preprocessing_recipe(parameters,vocabulary):
    """Pinned factory/split-off-dynamic semantics, without fitting observations."""
    def leaf(name,**attrs):return dict(type=name,attributes=attrs,children=[])
    def encoding(children,output):return dict(type='EncodingLayer',attributes=dict(enc_output_name=output),children=children)
    def onehot(card):return leaf('SingleOneHotLayer',cat_size=card,onoff=[1.,0.],use_missing_zero=True,use_1d_binary_onehot=True)
    def stats(width):
        return ([leaf('BiasLayer',factor=1.),leaf('ScaleLayer')] if width else [])+[
            leaf('FunctionLayer',function=dict(name='smooth_clip_func',max_abs_value=3.0))]
    cards=vocabulary['cardinalities'];large=[card for card in cards if card>9]
    initial=[encoding([onehot(card) if card<=9 else leaf('IdentityLayer') for card in cards],'x_one_hot')] if cards else []
    remaining=[encoding([leaf('IdentityLayer') for card in large],'x_one_hot')] if large else []
    learned=[encoding([leaf('SingleEmbeddingLayer') for card in large],'x_cont')] if large else []
    rename=leaf('RenameTensorLayer',old_name='x_one_hot',new_name='x_cont')
    numeric=len(vocabulary['schema']['columns'])-len(vocabulary['schema']['categorical'])
    categorical=sum(1 if card==3 else card-1 for card in cards if card<=9)
    kind=parameters['num_emb_type']
    if kind=='none':
        fixed=initial+remaining+[rename]+stats(numeric+categorical)
        return fixed,learned
    numerical=[leaf('FilterTensorsLayer',include_keys=['x_cont'],exclude_keys=None),rename,*stats(numeric),
        leaf('PLREmbeddingsLayerCosBias' if kind=='pbld' else 'PLREmbeddingsLayer',
            plr_act_name='relu' if kind=='plr' else 'linear',plr_use_densenet=kind=='pbld')]
    categories=[leaf('FilterTensorsLayer',include_keys=None,exclude_keys=['x_cont']),*remaining,rename,*stats(categorical),*learned]
    parallel=dict(type='ConcatParallelLayer',attributes={},branches=[numerical,categories])
    return initial,[parallel]


def validate_recipe(graph,static,arrays,parameters,vocabulary):
    """Check admitted graph layout against the actual resolved factory config."""
    children=graph['children']
    dynamic=parameters['num_emb_type']!='none' or any(card>9 for card in vocabulary['cardinalities'])
    if (graph['type']!='SequentialLayer' or len(children)!=(4 if dynamic else 3)
            or static['type']!='SequentialLayer'):
        raise ValueError('Unqualified native top-level/static preprocessing partition')
    net,normal,clamp=children[-3:]
    prep=children[0] if dynamic else static
    expected_static,expected_dynamic=_preprocessing_recipe(parameters,vocabulary)
    actual_dynamic=_operations(prep) if dynamic else []
    if _operations(static)!=expected_static or actual_dynamic!=expected_dynamic:
        raise ValueError('Native preprocessing function/order/static partition differs from resolved recipe')
    if net['type']!='SequentialLayer' or normal['type']!='NormalizeOutputLayer' or clamp['type']!='ClampLayer':
        raise ValueError('Native output/net layout differs from resolved config')
    def tensor(node,key):return arrays[node['parameters'][key]['tensor']]
    if np.any(tensor(normal,'std')<0) or np.any(tensor(clamp,'low')>tensor(clamp,'high')):
        raise ValueError('Invalid native output scale/clipping bounds')
    if any(tensor(node,key).shape!=(1,1,1) for node,keys in ((normal,('mean','std')),(clamp,('low','high'))) for key in keys):
        raise ValueError('Native output statistic shape differs')
    dims=len(vocabulary['schema']['columns'])-len(vocabulary['schema']['categorical'])
    kind=parameters['num_emb_type'];embeddings=[n for n in walk(prep) if n['type'].startswith('PLREmbeddingsLayer')]
    if kind=='none':
        if embeddings:raise ValueError('Unexpected native numerical embedding')
        width=dims
    else:
        if len(embeddings)!=1:raise ValueError('Unqualified numerical embedding count')
        emb=embeddings[0];expected_type='PLREmbeddingsLayerCosBias' if kind=='pbld' else 'PLREmbeddingsLayer'
        if (emb['type']!=expected_type or emb['attributes']!=dict(plr_act_name='relu' if kind=='plr' else 'linear',plr_use_densenet=kind=='pbld')):
            raise ValueError('Numerical embedding differs from resolved native variant')
        hidden=parameters['plr_hidden_1'];out=parameters['plr_hidden_2']-(1 if kind=='pbld' else 0)
        expected={'weight_1':(1,dims,1,hidden if kind=='pbld' else hidden//2),
            'weight_2':(1,dims,hidden,out),'bias_2':(1,dims,1,out)}
        if kind=='pbld':expected['bias_1']=(1,dims,1,hidden)
        if any(tensor(emb,k).shape!=shape for k,shape in expected.items()):raise ValueError('Native numerical embedding shape differs')
        width=dims*parameters['plr_hidden_2']
    width+=sum(1 if card==3 else card-1 if card<=parameters['max_one_hot_cat_size'] else parameters['embedding_size'] for card in vocabulary['cardinalities'])
    blocks=net['children'];hidden=parameters['hidden_sizes']
    if len(blocks)!=len(hidden)+1:raise ValueError('Native hidden layout differs from resolved config')
    for i,(block,out) in enumerate(zip(blocks,[*hidden,1],strict=True)):
        expected=['WeightLayer','BiasLayer']
        if i<len(hidden):expected+=['ParametricActivationLayer','DropoutLayer']
        if i==0 and parameters['add_front_scale']:expected=['ScaleLayer',*expected]
        if block['type']!='SequentialLayer' or [n['type'] for n in block['children']]!=expected:
            raise ValueError('Native block layout differs from resolved config')
        parts=block['children'];offset=int(i==0 and parameters['add_front_scale'])
        weight,bias=parts[offset:offset+2]
        if weight['attributes']['factor']!=float(np.sqrt(1.0/width)) or bias['attributes']['factor']!=1.0:
            raise ValueError('Native NTP affine factor differs from resolved config')
        if tensor(weight,'weight').shape!=(1,width,out) or tensor(bias,'bias').shape!=(1,1,out):
            raise ValueError('Native weight shape differs from resolved layout')
        if offset and tensor(parts[0],'scale').shape!=(1,1,width):raise ValueError('Native front scale shape differs')
        if i<len(hidden):
            activation=parts[offset+2]
            if activation['attributes']['function']!={'name':parameters['act']} or tensor(activation,'weight').shape!=(1,1,out):
                raise ValueError('Native activation differs from resolved config')
        width=out

    onehots=[node for node in walk(static) if node['type']=='SingleOneHotLayer']
    if [node['attributes']['cat_size'] for node in onehots]!=[card for card in vocabulary['cardinalities'] if card<=9]:
        raise ValueError('Native static one-hot graph differs from F category cardinalities')
    learned=[node for node in walk(prep) if node['type']=='SingleEmbeddingLayer']
    large=[card for card in vocabulary['cardinalities'] if card>9]
    if len(learned)!=len(large) or any(tensor(node,'emb').shape!=(1,card,parameters['embedding_size']) for node,card in zip(learned,large,strict=True)):
        raise ValueError('Native learned categorical embedding differs from F cardinalities')
