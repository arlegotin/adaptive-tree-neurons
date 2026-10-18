"""Independent immutable official NODE numerical path; never adapter math.

Exact source bodies: Dataset class/fetch definitions/registry and Trainer's
average_checkpoints. No uninstalled data or UI dependency is replaced numerically.
Forbidden dependency callables raise; they must never execute. Real Torch
checkpoint I/O is used only by this trusted reference, never portable adapter state.
"""
import ast
from collections import deque
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import types

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'configs/vendor/node/upstream'


def forbidden(*args,**kwargs):raise AssertionError('Forbidden broad-import dependency reached the reference numerical/data path')


def modules():
    for row in json.loads((ROOT/'configs/vendor/node/sources.json').read_text()):
        assert hashlib.sha256((ARCHIVE/row['path']).read_bytes()).hexdigest()==row['sha256']
    overlay=Path(json.loads((ROOT/'locks/research-v2/core.json').read_text())['executable']).parents[2]/'environment/node-reference-site'
    paths=[str(overlay),str(ARCHIVE/'qhoptim-wheel')];old=sys.path[:];sys.path[:0]=paths
    package=types.ModuleType('node_v2_independent_official');package.__path__=[str(ARCHIVE/'lib')]
    sys.modules[package.__name__]=package
    try:
        nn=importlib.import_module(package.__name__+'.nn_utils')
        arch=importlib.import_module(package.__name__+'.arch')
        utils=importlib.import_module(package.__name__+'.utils')
        qh=importlib.import_module('qhoptim.pyt')
        assert Path(qh.__file__).resolve()==(ARCHIVE/'qhoptim-wheel/qhoptim/pyt/__init__.py').resolve()
    finally:sys.path[:]=old
    namespace=dict(np=np,pd=pd,torch=torch,random=random,QuantileTransformer=QuantileTransformer,
        download=forbidden,LeaveOneOutEncoder=forbidden,load_svmlight_file=forbidden,train_test_split=forbidden)
    tree=ast.parse((ARCHIVE/'lib/data.py').read_text())
    bodies=[v for v in tree.body if isinstance(v,(ast.FunctionDef,ast.ClassDef)) or isinstance(v,ast.Assign)]
    exec(compile(ast.Module(body=bodies,type_ignores=[]),str(ARCHIVE/'lib/data.py'),'exec'),namespace)
    tree=ast.parse((ARCHIVE/'lib/trainer.py').read_text())
    average=next(m for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Trainer'
        for m in c.body if isinstance(m,ast.FunctionDef) and m.name=='average_checkpoints')
    average_namespace={'torch':torch,'deepcopy':deepcopy}
    exec(compile(ast.Module(body=[average],type_ignores=[]),str(ARCHIVE/'lib/trainer.py'),'exec'),average_namespace)
    return nn,arch,utils,qh.QHAdam,namespace['Dataset'],average_namespace['average_checkpoints']


def reference(req,resolved,directory):
    nn,arch,utils,QHAdam,Dataset,average=modules();directory.mkdir()
    # Only caller-provided synthetic F/G and one unused synthetic feature row.
    data=Dataset('explicit_inline_fixture',req.seed,quantile_transform=True,quantile_noise=.001,
        X_train=np.asarray(req.X_fit,dtype=np.float32),y_train=np.asarray(req.y_fit,dtype=np.float64),
        X_valid=np.asarray(req.X_select,dtype=np.float32),y_valid=np.asarray(req.y_select,dtype=np.float64),
        X_test=np.zeros((1,2),dtype=np.float32),y_test=np.zeros(1))
    mean=float(data.y_train.mean());scale=float(data.y_train.std());scale=scale or 1.
    y=((data.y_train-mean)/scale).astype(np.float32);g=((data.y_valid-mean)/scale).astype(np.float32)
    model=torch.nn.Sequential(arch.DenseBlock(data.X_train.shape[1],resolved['trees'],num_layers=1,tree_dim=3,
        depth=resolved['depth'],flatten_output=False,choice_function=nn.entmax15,bin_function=nn.entmoid15),
        nn.Lambda(lambda x:x[...,0].mean(-1))).cpu().float()
    with torch.no_grad():model(torch.as_tensor(data.X_train[:1000]))
    initial=deepcopy(model.state_dict());opt=QHAdam(model.parameters(),**resolved['optimizer'])
    paths=deque(maxlen=5);reports=[];best=None;best_opt=None;best_step=0;best_risk=float('inf');raw=None
    trace=[];selected_window=[]
    for step,(batch,target,indices) in enumerate(utils.iterate_minibatches(data.X_train,y,np.arange(len(y)),
            batch_size=resolved['batch_size'],shuffle=True,epochs=float('inf')),1):
        model.train();opt.zero_grad();loss=(model(torch.as_tensor(batch))-torch.as_tensor(target)).square().mean()
        loss.backward();opt.step();trace.append(indices.tolist())
        if step%resolved['report_frequency']==0:
            raw=deepcopy(model.state_dict());path=directory/f'raw-{step}.pth'
            torch.save({'model':raw,'opt':deepcopy(opt.state_dict()),'step':step},path);paths.append(path)
            output=directory/'average.pth';average(None,paths=list(reversed(paths)),out_path=str(output))
            averaged=torch.load(output,weights_only=True);model.load_state_dict(averaged['model']);model.eval()
            with torch.no_grad():pred=utils.process_in_chunks(model,torch.as_tensor(data.X_valid),batch_size=1024).numpy()
            risk=float(((g-pred)**2).mean());reports.append(dict(step=step,g_mse=risk,snapshot_steps=[int(p.stem.split('-')[1]) for p in paths]))
            if risk<best_risk:
                best_risk=risk;best_step=step;best=deepcopy(model.state_dict());best_opt=deepcopy(averaged['opt']);selected_window=list(paths)
            model.load_state_dict(raw)
        if step>best_step+resolved['patience'] or step>=req.budget['max_updates']:break
    raw=deepcopy(model.state_dict());raw_opt=deepcopy(opt.state_dict())
    if best is not None:model.load_state_dict(best)
    model.eval()
    with torch.no_grad():prediction=utils.process_in_chunks(model,torch.as_tensor(data.X_valid),batch_size=1024).numpy().astype(np.float64)*scale+mean
    return dict(initial=initial,model=model,raw=raw,raw_optimizer=raw_opt,selected_optimizer=best_opt,
        prediction=prediction,reports=reports,selected_step=best_step if best is not None else None,trace=trace,
        transformed_F=data.X_train,selected_window=selected_window)
