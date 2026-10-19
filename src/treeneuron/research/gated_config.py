"""Strict own-method practical recipes; no strategy or workload admission."""
from dataclasses import dataclass
import json
import math

from .native_trees import _schema


def _number(value,lower,upper):
    return type(value) in (int,float) and lower<=value<=upper and math.isfinite(value)


@dataclass(frozen=True)
class GatedConfig:
    _encoded:str

    @classmethod
    def parse(cls,family,value):
        fields={'schema','family','mode','feature_schema','parameters'}
        if (family not in ('P00','P01','P02','P05') or type(value) is not dict
                or value.get('schema')!='research/v2-gated-native-config'
                or value.get('family')!=family
                or value.get('mode') not in ('native_anchor','search','engineering_fixture')):
            raise ValueError('Invalid own-method v2 configuration or family')
        if family in ('P02','P05'):fields.add('variant')
        fixture=value['mode']=='engineering_fixture'
        if fixture:fields.update({'base_updates','continuation_updates','validation_interval','patience'})
        if set(value)!=fields:raise ValueError('Unknown setting or implicit native budget override')
        if family=='P02' and value['variant'] not in ('lower','upper'):
            raise ValueError('Ordinary warm capacity requires its declared bracket')
        if family=='P05' and value['variant'] not in ('centered_partial','centered_full','direct_partial','direct_full'):
            raise ValueError('Unknown static coordinate/capacity variant')
        _schema(value['feature_schema'])
        parameters=value['parameters']
        if type(parameters) is not dict or value['mode']=='native_anchor' and parameters:
            raise ValueError('Native anchors have no supplied parameter overrides')
        allowed={'width','lr','weight_decay','batch_size','scheduler'}
        allowed.add('dropout' if family=='P05' else 'growth_fraction')
        for key,item in parameters.items():
            if key not in allowed:raise ValueError('Unknown or unadmitted method setting')
            if key=='width':
                choices=(4,8) if fixture else (32,64,128) if family=='P05' else (64,128)
                valid=type(item) is int and item in choices
            elif key=='lr':valid=_number(item,1e-4,1e-2)
            elif key=='weight_decay':valid=_number(item,0.,1e-2) and (item==0 or item>=1e-7)
            elif key=='batch_size':valid=type(item) is int and item in (128,256)
            elif key=='scheduler':valid=type(item) is str and item in ('constant','cosine')
            elif key=='growth_fraction':valid=_number(item,1/8,1/2) and item in (1/8,1/4,1/2)
            else:valid=_number(item,0.,.2) and item in (0.,.1,.2)
            if not valid:raise ValueError('Own-method parameter outside the frozen support')
        if fixture and any(type(value[k]) is not int or not 1<=value[k]<=25
                for k in ('base_updates','continuation_updates','validation_interval','patience')):
            raise ValueError('Engineering stage/selection limits must be explicit integers1–25')
        return cls(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False))

    def to_state(self):return json.loads(self._encoded)

    @property
    def feature_schema(self):return self.to_state()['feature_schema']

    def resolved(self):
        raw=self.to_state();family=raw['family'];fixture=raw['mode']=='engineering_fixture'
        width=8 if fixture else 64
        result=dict(family=family,placement={'P00':'legacy_adam','P01':'random'}.get(family),
            variant=raw.get('variant'),width=width,growth_fraction=1/4,lr=.005,
            weight_decay=0.,batch_size=128,scheduler='constant',dropout=0.,
            epsilon=.05,tau=1.,optimizer='AdamW',betas=[.9,.999],adam_epsilon=1e-8,
            dtype='float32',device='cpu',checkpoint_selection='strict_earliest_complete_G_best',
            evaluate_update_zero=True,direction='current_signed_moment',
            parent_snapshot='late_G_best',stream='continued',newborn_lr_multiplier=1.,parent_freeze_updates=0)
        result.update(raw['parameters'])
        for name in ('base_updates','continuation_updates','validation_interval','patience'):
            result[name]=raw[name] if fixture else {'base_updates':8000,'continuation_updates':8000,
                'validation_interval':25,'patience':20}[name]
        result['schedule_horizon']=result['base_updates']+result['continuation_updates']
        result['requested_count']=(result['width'] if family=='P05' and raw['variant'].endswith('_full')
            else result['width']//4 if family=='P05' else int(result['width']*result['growth_fraction']))
        return result
