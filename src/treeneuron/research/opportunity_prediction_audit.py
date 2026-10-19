"""Independent M2 retained-prediction arithmetic, without fitting/model imports.

Queue byte authentication is shared infrastructure. Scalar loss/variance/quantile
arithmetic below does not call the producer metrics or its data/model loaders.
This does not regenerate teachers, reexecute predictors or authenticate tensor
semantics. It accepts a frozen producer identity distinct from the auditor.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from treeneuron.config import canonical_hash
from .workqueue import authenticated_queue


def _finite(value):
    try:return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:return False


def _vector(value):
    value=np.asarray(value)
    if value.dtype!=np.dtype('float64') or value.ndim!=1 or not len(value) or not np.isfinite(value).all():
        raise ValueError('Retained prediction/target must be a finite nonempty float64 vector')
    return value


def _mean(values):
    # Divide before summing: the mean may be finite even if the total is not.
    result=math.fsum(value/len(values) for value in values)
    if not math.isfinite(result):raise ValueError('Independent mean overflowed')
    return result


def prediction_metrics(target,prediction,variance,noisefree):
    target=_vector(target);prediction=_vector(prediction);noisefree=_vector(noisefree)
    if (target.shape!=prediction.shape or target.shape!=noisefree.shape
            or not _finite(variance) or variance<0):raise ValueError('Prediction rows or F variance differ')
    errors=[abs(float(p)-float(y)) for y,p in zip(target,prediction)]
    squared=[e*e for e in errors]
    noiseless=[(float(p)-float(y))**2 for p,y in zip(prediction,noisefree)]
    if not all(math.isfinite(e) for e in squared+noiseless):raise ValueError('Independent loss overflowed')
    mse=_mean(squared);nmse=mse/variance if variance else None
    if nmse is not None and not math.isfinite(nmse):raise ValueError('Independent normalized loss overflowed')
    ordered=sorted(errors);position=.95*(len(errors)-1);lower=math.floor(position);upper=math.ceil(position)
    quantile=ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)
    return dict(n=len(errors),mse=mse,normalized_mse=nmse,rmse=math.sqrt(mse),mae=_mean(errors),
        absolute_error_p95=quantile,max_absolute_error=ordered[-1],noise_free_mse=_mean(noiseless),
        normalized_mse_reason=None if variance else 'zero F population target variance; degenerate task',
        group_macro_mse=None,worst_group_mse=None,group_metrics=None)


def _path(path,root,commitments):
    path=Path(path);resolved=path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(root) or not path.is_file():
        raise ValueError('Retained audit file escapes its authenticated artifact or is absent')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    commitments[str(resolved)]=digest
    return resolved,digest


def _array_digest(array):
    digest=hashlib.sha256();digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes());return digest.hexdigest()


def _data(root,job,result,source,commitments):
    artifact=Path(result['value']['artifact_path']).resolve()
    result_directory=Path(result['_audit_result_path']).parent.resolve()
    if not artifact.is_relative_to(result_directory):raise ValueError('Data artifact belongs to another queue job')
    meta_path,_=_path(artifact/'result.json',root,commitments);metadata=json.loads(meta_path.read_text())
    required={'request.json','teacher.json','split.json','fitting.npz','validation-features.npz','evaluator.npz'}
    if job['payload'].get('population')=='positive_small':required.add('parent-coefficients.json')
    if (metadata.get('schema')!='research/v2-data-artifact' or metadata.get('status')!='completed_eligible'
            or metadata.get('source_sha256')!=source or metadata.get('T_created') is not False
            or metadata.get('request')!=job['payload'] or metadata.get('teacher_id')!=result['value'].get('teacher_id')
            or set(metadata.get('headers',{}))!=set('FGVE') or set(metadata.get('files',{}))!=required):
        raise ValueError('Data role/source/request inventory differs from the producer plan')
    for name,digest in metadata['files'].items():
        _,actual=_path(artifact/name,artifact,commitments)
        if actual!=digest:raise ValueError('Data file commitment differs')
    request=json.loads((artifact/'request.json').read_text())
    if request.get('spec')!=job['payload']:raise ValueError('Data request differs')
    arrays={}
    inventories={'fitting.npz':[r+'_'+k for r in 'FG' for k in ('X','y','row_id','group_id')],
        'validation-features.npz':['V_X','V_row_id','V_group_id'],
        'evaluator.npz':['V_y',*['E_'+k for k in ('X','y','row_id','group_id')],*[r+'_noise_free_target' for r in 'FGVE']]}
    for name,keys in inventories.items():
        with np.load(artifact/name,allow_pickle=False) as archive:
            if len(archive.files)!=len(keys) or set(archive.files)!=set(keys):raise ValueError('Unexpected data array/role inventory')
            arrays.update({k:archive[k] for k in keys})
    for role in 'FGVE':
        header=metadata['headers'][role];n=len(arrays[role+'_y']);d=len(header.get('feature_names',[]))
        if set(header.get('arrays',{}))!={'X','y','row_id','group_id','noise_free_target'} or not n or not d:
            raise ValueError('Data array header inventory differs')
        for key in ('X','y','row_id','group_id','noise_free_target'):
            value=arrays[role+'_'+key];dtype='int64' if key in ('row_id','group_id') else 'float64'
            declared=header['arrays'][key];shape=[n,d] if key=='X' else [n]
            if (str(value.dtype)!=dtype or list(value.shape)!=shape or not np.isfinite(value).all()
                    or declared!=dict(dtype=dtype,shape=shape,sha256=_array_digest(value))):
                raise ValueError('Data array dtype/shape/bytes differ from its header')
        if len(np.unique(arrays[role+'_row_id']))!=n:raise ValueError('Data row identifiers are not unique')
    split=json.loads((artifact/'split.json').read_text())
    if (canonical_hash(split)!=metadata.get('manifest_sha256') or split.get('phase')!='exploratory'
            or split.get('teacher_id')!=metadata['teacher_id']
            or split.get('row_id')!=[int(i) for r in 'FGV' for i in arrays[r+'_row_id']]
            or split.get('group_id')!=[int(i) for r in 'FGV' for i in arrays[r+'_group_id']]
            or split.get('assignments')!=[r for r in 'FGV' for _ in arrays[r+'_y']]
            or split.get('hashes')!=[[r+'.'+k,_array_digest(arrays[r+'_'+k])] for r in 'FGV' for k in ('X','y')]):
        raise ValueError('Data array rows/targets differ from their split commitment')
    values=[float(v) for v in arrays['F_y']];mean=_mean(values)
    variance=_mean([(v-mean)**2 for v in values])
    return dict(variance=variance,arrays={key:value for key,value in arrays.items()
        if key.startswith(('V_','E_')) and not key.endswith('_X')})


def _compare(actual,saved,where):
    if not isinstance(saved,dict) or set(saved)!=set(actual):raise ValueError(f'{where}: scalar metric inventory differs')
    maximum=0.
    for key,value in actual.items():
        other=saved[key]
        if value is None or type(value) is str:agrees=other==value
        elif key=='n':agrees=type(other) is int and other==value
        else:
            agrees=_finite(other) and math.isclose(value,other,rel_tol=1e-12,abs_tol=1e-14)
            if agrees:maximum=max(maximum,abs(value-other))
        if not agrees:raise ValueError(f'{where}: independently reconstructed metric {key} differs')
    return maximum


def audit_opportunity_predictions(directory,*,expected_plan_hash,expected_ledger_hash):
    """Authenticate an inactive producer queue and independently rebuild its risks.

    Every planned evaluator is retained, including missing/ineligible outcomes.
    All paid attempt files are authenticated by the shared queue reader. Only
    completed eligible evaluators have predictions reconstructed here.
    """
    return _audit_predictions(directory,expected_plan_hash=expected_plan_hash,expected_ledger_hash=expected_ledger_hash,
        kind='m2_opportunity',schema='research/v2-opportunity-prediction-audit',gate='Gate_A_decided')


def audit_joint_predictions(directory,*,expected_plan_hash,expected_ledger_hash):
    """Reconstruct M3 development predictions; source/design review stays separate."""
    return _audit_predictions(directory,expected_plan_hash=expected_plan_hash,expected_ledger_hash=expected_ledger_hash,
        kind='m3_joint',schema='research/v2-joint-prediction-audit',gate='Gate_B_decided')


def _audit_predictions(directory,*,expected_plan_hash,expected_ledger_hash,kind,schema,gate):
    root=Path(directory).resolve();commitments={};rows=[];cache={};vectors=0;comparisons=0;maximum=0.
    with authenticated_queue(directory,expected_plan_hash=expected_plan_hash) as (plan,ledger,results):
        endpoint=ledger.events[-1]['event_hash'] if ledger.events else '0'*64
        if endpoint!=expected_ledger_hash:raise ValueError('External ledger endpoint differs')
        if plan['identity'].get('study_config',{}).get('kind')!=kind:raise ValueError('Exploratory queue kind differs from this audit')
        source=plan['identity']['source_sha256'];jobs={j['job_id']:j for j in plan['attempts']}
        statuses={j:'never_started' for j in jobs};statuses.update({e['attempt_id']:e['status'] for e in ledger.events})
        references={e['attempt_id']:e['result'] for e in ledger.events if 'result' in e}
        for name,record in results.items():record['_audit_result_path']=references[name]['path']
        for job in plan['attempts']:
            if job['kind']!='evaluate':continue
            name=job['job_id'];row=dict(job_id=name,payload=job['payload'],status=statuses[name]);rows.append(row)
            if statuses[name]!='completed_eligible':continue
            value=results[name]['value']
            if value.get('T_accessed') is not False or value.get('E_exploratory') is not True:
                raise ValueError('Evaluator protected/exploratory scope differs')
            data=[j for j in job['depends_on'] if jobs[j]['kind']=='data']
            producer=[j for j in job['depends_on'] if jobs[j]['kind']!='data']
            if len(data)!=1 or len(producer)!=1 or any(statuses[j]!='completed_eligible' for j in data+producer):
                raise ValueError('Eligible evaluator lacks its declared eligible data/model producer')
            key=data[0]
            if key not in cache:cache[key]=_data(root,jobs[key],results[key],source,commitments)
            retained=cache[key];variance=retained['variance'];original=retained['arrays']
            declared_variance=value.get('fit_target_variance')
            if not _finite(declared_variance) or not math.isclose(variance,declared_variance,rel_tol=1e-12,abs_tol=1e-14):
                raise ValueError('F population target variance differs from retained F labels')
            report=results[producer[0]]['value']['report'];models=set(report['models'])
            prefixes=report.get('prefixes',{}).get('records',[])
            names=['prefix_'+str(r['prefix']) for r in prefixes]
            if len(names)!=len(set(names)) or models.intersection(names):raise ValueError('Retained model names/prefixes collide')
            models.update(names)
            if 'selected' not in models or set(value['metrics'])!=models:raise ValueError('Retained model/metric inventory differs')
            path,_=_path(value['predictions_path'],Path(results[name]['_audit_result_path']).parent.resolve(),commitments)
            expected={r+'_'+k for r in ('V','E') for k in ('y','row_id','group_id')}|{m+'_'+r for m in models for r in ('V','E')}
            actual={}
            with np.load(path,allow_pickle=False) as archive:
                if len(archive.files)!=len(expected) or set(archive.files)!=expected:raise ValueError('Prediction vector inventory differs')
                for role in ('V','E'):
                    for kind in ('y','row_id','group_id'):
                        key=role+'_'+kind;observed=archive[key];wanted=original[key]
                        if observed.dtype!=wanted.dtype or observed.shape!=wanted.shape or observed.tobytes()!=wanted.tobytes():
                            raise ValueError('Prediction row/group/target alignment differs')
                for model in sorted(models):
                    if set(value['metrics'][model])!={'V','E'}:raise ValueError('Unexpected evaluated target roles')
                    actual[model]={}
                    for role in ('V','E'):
                        metric=prediction_metrics(original[role+'_y'],archive[model+'_'+role],variance,original[role+'_noise_free_target'])
                        maximum=max(maximum,_compare(metric,value['metrics'][model][role],f'{name}/{model}/{role}'))
                        actual[model][role]=metric;vectors+=1;comparisons+=len(metric)
            row.update(metrics=actual,fit_target_variance=variance)
        complete=all(s in ('completed_eligible','completed_ineligible') for s in statuses.values())
        return dict(schema=schema,status='completed_arithmetic' if complete else 'incomplete_or_failed',
            queue_plan_hash=expected_plan_hash,ledger_event_hash=endpoint,producer_source_sha256=source,
            planned_evaluation_jobs=len(rows),audited_evaluation_jobs=sum('metrics' in r for r in rows),
            audited_prediction_vectors=vectors,scalar_metric_comparisons=comparisons,
            maximum_absolute_metric_difference=maximum,rows=rows,statuses=statuses,file_commitments=commitments,
            evidence_level='authenticated queue bytes plus independently reconstructed retained-prediction arithmetic',
            model_predictions_reexecuted=False,checkpoint_tensor_chain_authenticated=False,teacher_data_regenerated=False,
            feature_values_revalidated=True,**{gate:False},T_accessed=False,efficacy_claim=False)
