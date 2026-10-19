"""CPU v2 complete-state stage engine; no persistent resume/quality admission.

Paid update events are sent once to an external sink. G-best restoration only
rewinds model-coupled state, never that outer event/cost history. No V/T interface
exists. Legacy trainers, checkpoint schemas and finite guards are unchanged.
"""
import copy
import math
import random

import numpy as np
import torch

from treeneuron.train import UpdateSchedule, capture_rng, restore_rng


def _integer(x,low,high):return type(x) is int and low<=x<=high


def _finite_tensor(x,*,shape,dtype):
    return (isinstance(x,torch.Tensor) and x.device.type=='cpu' and x.dtype==dtype
        and x.shape==shape and bool(torch.isfinite(x).all()))


def _check_rng(value):
    def finite_cached(item):
        if isinstance(item,dict):
            for v in item.values():finite_cached(v)
        elif isinstance(item,(tuple,list)):
            for v in item:finite_cached(v)
        elif isinstance(item,(float,np.floating)) and not math.isfinite(item):
            raise ValueError('Nonfinite cached RNG scalar')
    try:
        finite_cached(value)
        if set(value)!={'python','numpy','torch_cpu','torch_mps'}:raise ValueError('RNG fields differ')
        random.Random().setstate(value['python'])
        n=value['numpy']
        if set(n)!={'name','keys','position','has_gauss','cached_gaussian'}:raise ValueError('NumPy RNG fields differ')
        if not _finite_tensor(n['keys'],shape=(624,),dtype=torch.int64):raise ValueError('Invalid NumPy RNG keys')
        if bool((n['keys']<0).any()) or bool((n['keys']>2**32-1).any()):raise ValueError('Invalid NumPy RNG key range')
        np.random.RandomState().set_state((n['name'],n['keys'].numpy().astype(np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
        torch.Generator().set_state(value['torch_cpu'])
        m=value['torch_mps']
        if set(m)!={'state','reason'}:raise ValueError('Invalid auxiliary RNG')
        if m['state'] is not None:
            if not _finite_tensor(m['state'],shape=capture_rng()['torch_mps']['state'].shape,dtype=torch.uint8):
                raise ValueError('Invalid auxiliary RNG tensor')
        elif type(m['reason']) is not str:raise ValueError('Missing auxiliary RNG disposition')
    except (KeyError,TypeError,RuntimeError,AttributeError,OverflowError) as error:
        raise ValueError('Invalid complete RNG state') from error


class TrainingSession:
    @classmethod
    def create(cls,model,*,n_rows,sampler_seed,lr,weight_decay,scheduler,horizon,batch_size):
        if (not _integer(n_rows,2,20000) or not _integer(sampler_seed,0,2**63-1)
                or not _integer(batch_size,1,256) or not _integer(horizon,1,16000)
                or type(lr) not in (int,float) or not 0<lr<=.01
                or type(weight_decay) not in (int,float) or not 0<=weight_decay<=.01):
            raise ValueError('Invalid finite v2 CPU training envelope')
        parameters=dict(model.named_parameters())
        if not parameters or any(not _finite_tensor(p,shape=p.shape,dtype=torch.float32) or not p.requires_grad for p in parameters.values()):
            raise ValueError('Trainable parameters must be finite CPU float32')
        width=next((m.in_features for m in model.modules() if isinstance(m,torch.nn.Linear)),None)
        if not _integer(width,1,128):raise ValueError('Invalid separate v2 input geometry')
        result=cls();result.model=model;result.n_rows=n_rows;result.batch_size=batch_size;result.input_width=width
        result.optimizer=torch.optim.AdamW([dict(params=list(parameters.values()),param_names=list(parameters))],
            lr=float(lr),weight_decay=float(weight_decay),betas=(.9,.999),eps=1e-8,foreach=False)
        result.scheduler=UpdateSchedule(result.optimizer,kind=scheduler,horizon=horizon,min_lr=0.,base_lrs=[float(lr)])
        result.sampler_generator=torch.Generator().manual_seed(sampler_seed)
        result.order=torch.randperm(n_rows,generator=result.sampler_generator)
        result.cursor=0;result.epoch=0;result.update=0
        result.births={name:0 for name in parameters};result.required_state=set()
        result._parameter_ids={id(p):name for name,p in parameters.items()}
        result.rng=capture_rng();result.stage={}
        result.snapshot()
        return result

    def snapshot(self):
        result=dict(schema='research/v2-gated-training-session',model=copy.deepcopy(self.model.state_dict()),
            optimizer=copy.deepcopy(self.optimizer.state_dict()),scheduler=self.scheduler.state_dict(),
            order=self.order.clone(),sampler_generator=self.sampler_generator.get_state().clone(),
            cursor=self.cursor,epoch=self.epoch,update=self.update,births=copy.deepcopy(self.births),
            required_state=sorted(self.required_state),rng=copy.deepcopy(self.rng),training=self.model.training,
            stage=copy.deepcopy(self.stage),n_rows=self.n_rows,batch_size=self.batch_size)
        self._validate(result)
        return result

    def _validate(self,value):
        fields={'schema','model','optimizer','scheduler','order','sampler_generator','cursor','epoch','update',
            'births','required_state','rng','training','stage','n_rows','batch_size'}
        if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-gated-training-session'
                or type(value['training']) is not bool or value['n_rows']!=self.n_rows or value['batch_size']!=self.batch_size
                or not _integer(value['update'],0,self.scheduler.horizon) or not _integer(value['cursor'],0,self.n_rows)
                or not _integer(value['epoch'],0,16000) or type(value['stage']) is not dict):
            raise ValueError('Invalid complete session identity or counters')
        per_epoch=math.ceil(self.n_rows/self.batch_size);update=value['update']
        epoch=(update-1)//per_epoch if update else 0
        cursor=min(self.n_rows,((update-1)%per_epoch+1)*self.batch_size) if update else 0
        if value['epoch']!=epoch or value['cursor']!=cursor:
            raise ValueError('Sampler phase differs from the continued absolute update stream')
        stage=value['stage']
        if stage and (set(stage)!={'start_update','updates_completed','bad_evaluations','best_mse','complete'}
                or not _integer(stage['start_update'],0,update)
                or not _integer(stage['updates_completed'],0,8000)
                or stage['updates_completed']!=update-stage['start_update']
                or not _integer(stage['bad_evaluations'],0,8000) or type(stage['complete']) is not bool
                or type(stage['best_mse']) not in (int,float) or not 0<=stage['best_mse']<float('inf')):
            raise ValueError('Invalid selected stage progress or risk')
        expected=self.model.state_dict();saved=value['model']
        if (not isinstance(saved,dict) or set(saved)!=set(expected) or any(
                not _finite_tensor(saved[k],shape=p.shape,dtype=p.dtype) for k,p in expected.items())):
            raise ValueError('Model snapshot shape/dtype/finiteness mismatch; casting is forbidden')
        if (not _finite_tensor(value['order'],shape=(self.n_rows,),dtype=torch.int64)
                or not torch.equal(value['order'].sort().values,torch.arange(self.n_rows))):
            raise ValueError('Sampler order must be a full F permutation')
        try:torch.Generator().set_state(value['sampler_generator'])
        except (RuntimeError,TypeError) as error:raise ValueError('Invalid sampler RNG') from error
        _check_rng(value['rng'])
        names=dict(self.model.named_parameters());births=value['births']
        if (type(births) is not dict or set(births)!=set(names)
                or any(not _integer(v,0,value['update']) for v in births.values())
                or births!=self.births or value['required_state']!=sorted(self.required_state)):
            raise ValueError('Birth metadata differs from the admitted model lineage')
        schedule=value['scheduler'];live=self.scheduler.state_dict()
        if (type(schedule) is not dict or set(schedule)!=set(live)
                or any(schedule[k]!=live[k] for k in ('kind','horizon','min_lr','base_lrs'))
                or schedule['position']!=value['update']):raise ValueError('Absolute scheduler state mismatch')
        state=value['optimizer'];current=self.optimizer.state_dict()
        if (type(state) is not dict or set(state)!={'state','param_groups'} or type(state['state']) is not dict
                or type(state['param_groups']) is not list or len(state['param_groups'])!=len(current['param_groups'])):
            raise ValueError('Invalid optimizer schema or group topology')
        identifiers=set()
        factor=1. if schedule['kind']=='constant' else (1+math.cos(math.pi*value['update']/schedule['horizon']))/2
        for index,(group,reference) in enumerate(zip(state['param_groups'],current['param_groups'])):
            if (type(group) is not dict or set(group)!=set(reference)
                    or any(group[k]!=reference[k] for k in group if k!='lr')
                    or group['lr']!=schedule['base_lrs'][index]*factor):
                raise ValueError('Optimizer ownership/options or scheduled LR mismatch')
            for identifier,name in zip(group['params'],group['param_names']):
                if identifier in identifiers:raise ValueError('Duplicate optimizer parameter')
                identifiers.add(identifier);p=names[name];record=state['state'].get(identifier,{})
                age=value['update']-births[name]
                if not record:
                    if age or name in self.required_state:raise ValueError('Missing Adam history')
                    continue
                if type(record) is not dict or set(record)!={'step','exp_avg','exp_avg_sq'}:
                    raise ValueError('Incomplete Adam history')
                step=record['step']
                if not _finite_tensor(step,shape=(),dtype=torch.float32) or float(step)!=age:
                    raise ValueError('Adam applied-update age differs from birth/global state')
                for key in ('exp_avg','exp_avg_sq'):
                    if not _finite_tensor(record[key],shape=p.shape,dtype=p.dtype) or key=='exp_avg_sq' and bool((record[key]<0).any()):
                        raise ValueError('Adam shape/dtype/finiteness or second-moment sign invalid')
        if not set(state['state'])<=identifiers:raise ValueError('Orphan Adam state')

    def restore(self,value):
        self._validate(value)
        saved=copy.deepcopy(value)
        self.model.load_state_dict(saved['model'],strict=True)
        self.optimizer.load_state_dict(saved['optimizer'])
        self.scheduler.position=saved['scheduler']['position'];self.scheduler._apply()
        self.order=saved['order'];self.sampler_generator.set_state(saved['sampler_generator'])
        self.cursor=saved['cursor'];self.epoch=saved['epoch'];self.update=saved['update']
        self.rng=saved['rng'];self.stage=saved['stage'];self.model.train(saved['training'])
        restore_rng(self.rng)

    def adopt_growth(self,model):
        names=dict(model.named_parameters());old_ids=set(self._parameter_ids)
        if not old_ids<={id(p) for p in names.values()}:raise ValueError('Growth dropped an incumbent parameter')
        births={};required=set()
        for name,p in names.items():
            old_name=self._parameter_ids.get(id(p))
            births[name]=self.births[old_name] if old_name is not None else self.update
            if old_name is None or old_name in self.required_state:required.add(name)
        self.model=model;self.births=births;self.required_state=required
        self._parameter_ids={id(p):name for name,p in names.items()}
        count=len(self.optimizer.param_groups)-len(self.scheduler.base_lrs)
        if count<0:raise ValueError('Growth dropped an optimizer group')
        self.scheduler.base_lrs.extend([self.scheduler.base_lrs[0]]*count);self.scheduler._apply()
        self.stage={};self.snapshot()

    def _data(self,X,y,*,fit):
        expected=self.n_rows if fit else len(X)
        if (not isinstance(X,torch.Tensor) or X.ndim!=2 or not 1<=X.shape[1]<=128
                or not _finite_tensor(X,shape=(expected,X.shape[1]),dtype=torch.float32)
                or X.shape[1]!=self.input_width or not 1<=len(X)<=20000
                or not isinstance(y,torch.Tensor) or y.dtype not in ((torch.float32,) if fit else (torch.float32,torch.float64))
                or not _finite_tensor(y,shape=(expected,),dtype=y.dtype)):
            raise ValueError('Require finite nonbroadcast CPU float32 F/G arrays in the v2 envelope')
        return X,y

    def step(self,X,y):
        self._data(X,y,fit=True)
        if self.update>=self.scheduler.horizon:raise ValueError('Absolute schedule exhausted')
        if self.cursor==self.n_rows:
            self.epoch+=1;self.order=torch.randperm(self.n_rows,generator=self.sampler_generator);self.cursor=0
        indices=self.order[self.cursor:self.cursor+self.batch_size]
        restore_rng(self.rng);self.model.train();self.optimizer.zero_grad(set_to_none=True)
        prediction=self.model(X.index_select(0,indices));target=y.index_select(0,indices)
        if prediction.shape!=target.shape:raise ValueError('Prediction broadcasting is forbidden')
        loss=(prediction-target).square().mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite F loss')
        loss.backward();group_norms=[]
        for group in self.optimizer.param_groups:
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in group['params']):
                raise FloatingPointError('Missing or nonfinite trainable gradient')
            group_norms.append(math.sqrt(sum(float(p.grad.double().square().sum()) for p in group['params'])))
            for p in group['params']:
                if not self.optimizer.state.get(p):
                    self.optimizer.state[p]=dict(step=torch.tensor(0.,dtype=torch.float32),
                        exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        rates=[g['lr'] for g in self.optimizer.param_groups]
        self.optimizer.step();self.scheduler.step();self.update+=1;self.cursor+=len(indices)
        self.rng=capture_rng()
        if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            raise FloatingPointError('Nonfinite updated parameter')
        return dict(type='update',update=self.update,row_indices=indices.tolist(),
            F_online_mse=float(loss.detach()),learning_rates=rates,gradient_norms=group_norms)

    def fit_stage(self,F,yF,G,yG,*,max_updates,validation_interval,patience,event_sink,target_mean=0.,target_scale=1.):
        self._data(F,yF,fit=True);self._data(G,yG,fit=False)
        if F.shape[1]!=G.shape[1]:raise ValueError('F/G feature widths differ')
        if (type(target_mean) not in (int,float) or not -float('inf')<target_mean<float('inf')
                or type(target_scale) not in (int,float) or not 0<target_scale<float('inf')):
            raise ValueError('G selection requires finite F-fitted target location/positive scale')
        if (not _integer(max_updates,1,self.scheduler.horizon-self.update)
                or not _integer(validation_interval,1,8000) or not _integer(patience,1,8000)
                or not callable(event_sink)):
            raise ValueError('Invalid stage budget/cadence or missing external event sink')
        start=self.update;best=None
        self.stage=dict(start_update=start,updates_completed=0,bad_evaluations=0,best_mse=None,complete=False)
        def evaluate():
            nonlocal best
            self.model.eval();total=0.
            with torch.no_grad():
                for begin in range(0,len(G),self.batch_size):
                    p=self.model(G[begin:begin+self.batch_size])
                    target=yG[begin:begin+self.batch_size]
                    if p.shape!=target.shape:raise ValueError('G prediction broadcasting is forbidden')
                    total+=float((p.double()*target_scale+target_mean-target.double()).square().sum())
            risk=total/len(G)
            if not math.isfinite(risk):raise FloatingPointError('Nonfinite G risk')
            improved=self.stage['best_mse'] is None or risk<self.stage['best_mse']
            if improved:self.stage['best_mse']=risk;self.stage['bad_evaluations']=0
            else:self.stage['bad_evaluations']+=1
            event_sink(dict(type='evaluation',update=self.update,stage_update=self.update-start,G_mse=risk,improved=improved))
            if improved:best=self.snapshot()
        evaluate();executed=0
        for offset in range(1,max_updates+1):
            event=self.step(F,yF);executed=offset;self.stage['updates_completed']=offset
            event_sink(dict(event,stage_update=offset))
            if offset%validation_interval==0 or offset==max_updates:evaluate()
            if self.stage['bad_evaluations']>=patience:break
        selected=best['update']-start;stop='patience' if self.stage['bad_evaluations']>=patience else 'max_updates'
        event_sink(dict(type='selection',selected_absolute_update=best['update'],selected_stage_update=selected,
            executed_stage_updates=executed,stop=stop))
        self.restore(best);self.stage['complete']=True
        return dict(selected_absolute_update=self.update,selected_stage_update=selected,
            executed_stage_updates=executed,executed_absolute_update=start+executed,stop=stop,G_mse=self.stage['best_mse'])
