"""Retain exact M2 arrays; fitting loads F/G and validation features only."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from treeneuron import train
from treeneuron.config import canonical_hash,resolve_experiment
from treeneuron.data.synthetic import make_from_config,sample_teacher
from treeneuron.data.types import DevelopmentData,ObservationSet,SplitManifest,array_hash,_readonly
from .controls import BASE_ROWS,ControlParameters
from .ledger import write_new_json
from .roles import FittingData
from .seeds import seed64

ROW_KEYS=('X','y','row_id','group_id')
REQUEST_KEYS={'population','teacher_index','teacher_seed','data_seed','reference_seed',
              'evaluation_seed','evaluation_rows'}


def _declared(spec):
    factors=np.random.default_rng(spec['teacher_seed']).uniform(.95,1.05,8)
    factors[1]=factors[0]
    head=np.asarray((.5,.5,1.,.2,-.2,.1,-.1,.1))*factors
    return dict(base_rows=np.asarray(BASE_ROWS,dtype=np.float64).tolist(),head=head.tolist(),
        gamma=2.,sigma=.1,generating_ids=[0,2],contrasts=[[0.,.5,0.],[.5,0.,0.]],
        gates=[[2.,0.,0.],[0.,2.,0.]])


def _rows_header(rows):
    keys=ROW_KEYS+(('noise_free_target',) if rows.noise_free_target is not None else ())
    return dict(stream_id=rows.stream_id,feature_names=list(rows.feature_names),
        arrays={k:dict(dtype=str(getattr(rows,k).dtype),shape=list(getattr(rows,k).shape),
            sha256=array_hash(getattr(rows,k))) for k in keys})


def _save(path,arrays):
    with path.open('xb') as stream:np.savez_compressed(stream,**arrays)


def _request(base,spec):
    if (not isinstance(spec,dict) or set(spec)!=REQUEST_KEYS
            or spec['population'] not in ('primary','positive_small')
            or type(spec['teacher_index']) is not int or spec['teacher_index']<0
            or type(spec['evaluation_rows']) is not int or spec['evaluation_rows']<=0):
        raise ValueError('Unknown or invalid development data request; protected roles are forbidden')
    for k in ('teacher_seed','data_seed','reference_seed','evaluation_seed'):seed64(spec[k],'validate')
    if spec['population']=='positive_small' and spec['evaluation_rows']!=16384:
        raise ValueError('Planted control requires its declared16384 E observations')
    raw=json.loads(json.dumps(base));raw['data']['population']='primary'
    if raw['data'].get('n_test') is not None:raise ValueError('Protected data request is forbidden')
    raw['seeds'].update({k:spec[k+'_seed'] for k in ('teacher','data','reference')})
    if spec['population']=='positive_small':
        raw['model'].update(d=2,h=8);raw['growth']['m_requested']=2
        raw['data'].update(n_fit=512,n_growth=256,n_validation=256,noise_sigma=.1)
    return resolve_experiment(raw)


def prepare_data_artifact(base,spec,output):
    config=_request(base,spec)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    write_new_json(root/'request.json',dict(spec=spec,resolved_data_config=config.to_dict()))
    if spec['population']=='primary':
        teacher,data=make_from_config(config)
        E=sample_teacher(teacher,spec['evaluation_rows'],spec['evaluation_seed'],config.data.noise_sigma)
        write_new_json(root/'teacher.json',teacher.to_dict())
    else:
        from .controls import make_positive_control
        declared=_declared(spec)
        write_new_json(root/'parent-coefficients.json',declared)
        control=make_positive_control(instance_seed=spec['teacher_seed'],data_seed=spec['data_seed'],gamma=2.,sigma=.1)
        if (not np.array_equal(control.base_rows,declared['base_rows'])
                or not np.array_equal(control.head,declared['head'])):
            raise ValueError('Planted coefficients differ from their pre-observation declaration')
        data,E=control.development,control.evaluation
        write_new_json(root/'teacher.json',dict(version='research/v2-planted',**declared,
            instance_seed=spec['teacher_seed'],data_seed=spec['data_seed'],teacher_id=data.teacher_id))
    _save(root/'fitting.npz',{f'{role}_{key}':getattr(getattr(data,role),key)
        for role in 'FG' for key in ROW_KEYS})
    _save(root/'validation-features.npz',{f'V_{key}':getattr(data.V,key) for key in ('X','row_id','group_id')})
    evaluator={'V_y':data.V.y,**{'E_'+key:getattr(E,key) for key in ROW_KEYS}}
    for role,rows in (('F',data.F),('G',data.G),('V',data.V),('E',E)):
        if rows.noise_free_target is not None:evaluator[role+'_noise_free_target']=rows.noise_free_target
    _save(root/'evaluator.npz',evaluator)
    write_new_json(root/'split.json',data.manifest.to_dict())
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    record=dict(schema='research/v2-data-artifact',status='completed_eligible',source_sha256=train.source_hash(),
        request=spec,teacher_id=data.teacher_id,manifest_sha256=canonical_hash(data.manifest.to_dict()),
        headers={r:_rows_header(rows) for r,rows in (('F',data.F),('G',data.G),('V',data.V),('E',E))},
        files=files,T_created=False,labels_policy='F/G worker; V/E and all noiseless targets evaluator only')
    write_new_json(root/'result.json',record)
    return record


def _metadata(root):
    root=Path(root);value=json.loads((root/'result.json').read_text())
    expected={'request.json','teacher.json','fitting.npz','validation-features.npz','evaluator.npz','split.json'}
    if value.get('request',{}).get('population')=='positive_small':expected.add('parent-coefficients.json')
    if (value.get('schema')!='research/v2-data-artifact' or value.get('status')!='completed_eligible'
            or value.get('source_sha256')!=train.source_hash() or value.get('T_created') is not False
            or set(value['headers'])!=set('FGVE') or set(value['files'])!=expected):
        raise ValueError('Data artifact source/role inventory differs')
    saved=json.loads(_file(root,value,'request.json').read_text())
    if saved['spec']!=value['request']:raise ValueError('Retained data request differs')
    config=_request(saved['resolved_data_config'],saved['spec'])
    if config.to_dict()!=saved['resolved_data_config']:
        raise ValueError('Retained data config differs from its request/seeds')
    expected_counts=dict(F=config.data.n_fit,G=config.data.n_growth,V=config.data.n_validation,
        E=value['request']['evaluation_rows'])
    for role,n in expected_counts.items():
        header=value['headers'][role];keys=set(ROW_KEYS)|{'noise_free_target'}
        if (set(header)!= {'stream_id','feature_names','arrays'} or set(header['arrays'])!=keys
                or len(header['feature_names'])!=config.model.d
                or header['arrays']['X']['shape']!=[n,config.model.d]
                or any(header['arrays'][k]['shape']!=[n] for k in keys-{'X'})):
            raise ValueError('Retained role shapes/schema differ from the declared data config')
    raw=json.loads(_file(root,value,'split.json').read_text())
    if canonical_hash(raw)!=value['manifest_sha256']:raise ValueError('Data split commitment differs')
    manifest=SplitManifest(**{k:v for k,v in raw.items() if k not in ('assignment_hash','feature_schema_hash')})
    if (canonical_hash(manifest.to_dict())!=canonical_hash(raw) or manifest.teacher_id!=value['teacher_id']
            or manifest.phase!='exploratory' or set(manifest.assignments)!=set('FGV')):
        raise ValueError('Data manifest roles/identity differ')
    if [k for k,_ in manifest.hashes]!=[r+'.'+k for r in 'FGV' for k in ('X','y')]:
        raise ValueError('Native development hash inventory differs')
    expected_id=canonical_hash(dict(teacher_id=manifest.teacher_id,hashes=manifest.hashes,
        row_id=manifest.row_id,feature_names=manifest.feature_names,feature_schema_hash=manifest.feature_schema_hash))
    if manifest.dataset_id!=expected_id:
        raise ValueError('Native raw dataset identity differs from committed arrays and teacher')
    return root,value,manifest


def _file(root,value,name):
    path=root/name
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=value['files'][name]:
        raise ValueError('Data file commitment differs')
    return path


def _arrays(root,value,name,keys):
    with _file(root,value,name).open('rb') as stream,np.load(stream,allow_pickle=False) as archive:
        if set(archive.files)!=set(keys):raise ValueError('Data array inventory differs')
        arrays={key:archive[key].copy() for key in keys}
    for name,array in arrays.items():
        role,key=name.split('_',1);header=value['headers'][role]['arrays'][key]
        dtype='int64' if key in ('row_id','group_id') else 'float64'
        if (str(array.dtype)!=dtype or header['dtype']!=dtype or list(array.shape)!=header['shape']
                or array_hash(array)!=header['sha256'] or not np.isfinite(array).all()):
            raise ValueError('Data array dtype/shape/bytes differ')
    return arrays


def _observation(value,role,arrays):
    h=value['headers'][role]
    return ObservationSet(**{k:arrays[role+'_'+k] for k in ROW_KEYS},stream_id=h['stream_id'],
        feature_names=tuple(h['feature_names']),noise_free_target=arrays.get(role+'_noise_free_target'))


def load_fitting_data(root):
    root,value,manifest=_metadata(root)
    fit=_arrays(root,value,'fitting.npz',[r+'_'+k for r in 'FG' for k in ROW_KEYS])
    features=_arrays(root,value,'validation-features.npz',['V_'+k for k in ('X','row_id','group_id')])
    rows={r:_observation(value,r,fit) for r in 'FG'}
    hashes=dict(manifest.hashes)
    for r in 'FG':
        if any(array_hash(getattr(rows[r],k))!=hashes[r+'.'+k] for k in ('X','y')):
            raise ValueError('F/G arrays differ from their split manifest')
    if array_hash(features['V_X'])!=hashes['V.X']:raise ValueError('V features differ from split manifest')
    ids=tuple(int(i) for r in 'FGV' for i in (getattr(rows[r],'row_id') if r in rows else features['V_row_id']))
    groups=tuple(int(i) for r in 'FGV' for i in (getattr(rows[r],'group_id') if r in rows else features['V_group_id']))
    assignments=tuple(r for r in 'FGV' for _ in range(len(rows[r].y) if r in rows else len(features['V_X'])))
    if (ids!=manifest.row_id or groups!=manifest.group_id or assignments!=manifest.assignments
            or any(tuple(value['headers'][r]['feature_names'])!=manifest.feature_names for r in 'FGV')):
        raise ValueError('Worker rows/features differ from the declared split')
    return FittingData(SimpleNamespace(**rows,V=SimpleNamespace(X=_readonly(features['V_X'],np.dtype('float64'))),
        manifest=manifest,teacher_id=value['teacher_id']))


def load_evaluation_data(root):
    root,value,manifest=_metadata(root);fit=load_fitting_data(root)
    features=_arrays(root,value,'validation-features.npz',['V_'+k for k in ('X','row_id','group_id')])
    keys=['V_y',*('E_'+k for k in ROW_KEYS),*(r+'_noise_free_target' for r in 'FGVE'
        if 'noise_free_target' in value['headers'][r]['arrays'])]
    arrays=_arrays(root,value,'evaluator.npz',keys)
    for r in 'FG':
        arrays.update({r+'_'+k:getattr(getattr(fit,r),k) for k in ROW_KEYS})
    arrays.update(features)
    data=DevelopmentData(*(_observation(value,r,arrays) for r in 'FGV'),value['teacher_id'],manifest)
    return data,_observation(value,'E',arrays)


def load_control_parameters(root):
    root,value,_=_metadata(root)
    if value['request']['population']!='positive_small':raise ValueError('Require a planted parent declaration')
    raw=json.loads(_file(root,value,'parent-coefficients.json').read_text())
    if raw!=_declared(value['request']):raise ValueError('Planted parent differs from predeclared coefficients')
    return ControlParameters(_readonly(raw['base_rows'],np.dtype('float64')),
        _readonly(raw['head'],np.dtype('float64')),raw['gamma'])
