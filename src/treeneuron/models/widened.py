"""Blockwise ordinary widening: old Adam ages survive, added units start fresh."""
import math
import torch
from torch import nn


class WidenedLayer(nn.Module):
    def __init__(self,d,h,parent_width,tau):
        super().__init__()
        self.d,self.h,self.parent_width,self.tau=d,h,parent_width,float(tau)
        self.base=nn.Linear(d,parent_width,dtype=torch.float32)
        self.added=nn.Linear(d,h-parent_width,dtype=torch.float32) if h>parent_width else None

    def forward(self,x):
        old=self.base(x)
        return old if self.added is None else torch.cat((old,self.added(x)),dim=1)


class WidenedHead(nn.Module):
    def __init__(self,h,parent_width):
        super().__init__()
        self.parent_width=parent_width
        self.base=nn.Linear(parent_width,1,dtype=torch.float32)
        self.added=nn.Linear(h-parent_width,1,bias=False,dtype=torch.float32) if h>parent_width else None

    def forward(self,x):
        old=self.base(x[:,:self.parent_width])
        return old if self.added is None else old+self.added(x[:,self.parent_width:])


class WidenedRegressor(nn.Module):
    def __init__(self,d,h,parent_width,tau=1.):
        super().__init__()
        if (type(d) is not int or not 0<d<=16 or type(h) is not int or not 0<h<=95
                or type(parent_width) is not int or not 0<parent_width<=min(h,64)
                or type(tau) not in (int,float) or not math.isfinite(tau) or tau<=0):
            raise ValueError('Invalid bounded widened topology')
        self.layer=WidenedLayer(d,h,parent_width,tau)
        self.head=WidenedHead(h,parent_width)

    def forward(self,x):
        return self.head(torch.relu(self.layer(x))).squeeze(-1)

    def export_config(self):
        l=self.layer
        return dict(schema_version=1,kind='widened_relu_scalar',
            layer=dict(d=l.d,h=l.h,parent_width=l.parent_width,tau=l.tau))

    @classmethod
    def from_config(cls,config):
        if (set(config)!={'schema_version','kind','layer'} or type(config['schema_version']) is not int
                or config['schema_version']!=1 or config['kind']!='widened_relu_scalar'
                or set(config['layer'])!={'d','h','parent_width','tau'}):
            raise ValueError('Invalid widened topology')
        return cls(**config['layer'])
