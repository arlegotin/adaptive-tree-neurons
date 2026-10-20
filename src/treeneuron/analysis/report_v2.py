"""Canonical formatting of an auditor-supplied evidence record.

This renderer does not open data, authenticate an audit, run inference, decide
novelty or grant protected access. The sealed artifact auditor owns those gates.
"""
import hashlib
import math
import os
from pathlib import Path
import re

from treeneuron.research.ledger import TERMINAL


_FIELDS={'schema','study_id','status','evidence_scope','source','provenance','protocol_sha256',
    'independent_units','planned_prediction_ids','predictions','attempts','comparisons','costs','limitations','sources'}


def _keys(value,fields):
    if type(value) is not dict or set(value)!=set(fields):raise ValueError('Canonical report record fields differ')


def _text(value):
    if type(value) is not str or not value:raise ValueError('Nonempty report text required')
    return value.replace('\\','\\\\').replace('|','\\|').replace('\r',' ').replace('\n',' ')


def _digest(value,n=64):
    if type(value) is not str or re.fullmatch('[0-9a-f]{'+str(n)+'}',value) is None:
        raise ValueError('Committed source/protocol/reference digest required')
    return value


def _names(values):
    if type(values) is not list or any(type(v) is not str or not v for v in values) or len(values)!=len(set(values)):
        raise ValueError('Unique declared report inventory required')
    return values


def _number(value):
    if value is None:return 'unavailable'
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite comparison number or explicit unknown required')
    return format(value,'.10g')


def _validated(value):
    if type(value) is not dict:raise ValueError('Canonical audit record required')
    _keys(value,_FIELDS|({'secondary_families'} if 'secondary_families' in value else set()))
    _secondary_families(value.get('secondary_families',[]))
    if (value['schema']!='research/v2-canonical-audit'
            or value['status'] not in ('completed','estimation','retired','blocked')
            or value['evidence_scope'] not in ('software','arithmetic','development','confirmation')):
        raise ValueError('Explicit supported study status and evidence scope required')
    _text(value['study_id']);_text(value['independent_units'])
    _keys(value['source'],('commit','package_sha256'))
    _digest(value['source']['commit'],40);_digest(value['source']['package_sha256'])
    _keys(value['provenance'],('source_authenticated','raw_predictions_authenticated','protected_protocol_authenticated'))
    if any(type(v) is not bool for v in value['provenance'].values()):raise ValueError('Explicit audit attestation flags required')
    if value['evidence_scope']=='confirmation':
        if not all(value['provenance'].values()):raise ValueError('Supplied confirmation audit lacks mandatory authentication')
        _digest(value['protocol_sha256'])
    elif value['protocol_sha256'] is not None or value['provenance']['protected_protocol_authenticated']:
        raise ValueError('Unqualified scope cannot masquerade as protected confirmation')
    planned=_names(value['planned_prediction_ids']);predictions={}
    if type(value['predictions']) is not list:raise ValueError('Final prediction inventory required')
    for p in value['predictions']:
        _keys(p,('prediction_id','method_id','task_id','repetition','status','normalized_risk',
            'normalization_status','primary_MSE','row_RMSE','row_MAE'))
        identifier=p['prediction_id']
        if (type(identifier) is not str or identifier not in planned or identifier in predictions
                or type(p['repetition']) is not int or p['repetition'] not in (0,1)
                or p['status'] not in TERMINAL|{'absent','running'}):raise ValueError('Invalid or unplanned final prediction')
        _text(p['method_id']);_text(p['task_id']);risk=p['normalized_risk']
        if p['status']=='completed_eligible':
            for key in ('primary_MSE','row_RMSE','row_MAE'):
                _number(p[key])
                if p[key] is None or p[key]<0:raise ValueError('Eligible prediction requires finite original-unit metrics')
            if p['normalization_status']=='defined':
                if type(risk) not in (int,float) or not math.isfinite(risk) or risk<0:
                    raise ValueError('Defined normalization requires finite nonnegative risk')
            elif p['normalization_status']!='undefined_F_variance' or risk is not None:
                raise ValueError('Explicit undefined F variance must retain an unavailable normalized risk')
        elif (any(p[k] is not None for k in ('normalized_risk','primary_MSE','row_RMSE','row_MAE'))
                or p['normalization_status']!='unavailable_prediction'):
            raise ValueError('Missing/failed prediction cannot have imputed metrics')
        predictions[identifier]=p
    if set(predictions)!=set(planned):raise ValueError('Every planned prediction must have an explicit status record')
    if type(value['attempts']) is not list:raise ValueError('Paid attempt inventory required')
    attempts=set()
    for row in value['attempts']:
        _keys(row,('attempt_id','status'));name=row['attempt_id'];_text(name)
        if name in attempts or row['status'] not in TERMINAL|{'absent','running'}:raise ValueError('Duplicate or invalid attempt')
        attempts.add(name)
        if value['status']=='completed' and row['status'] in ('absent','running','interrupted_recoverable'):
            raise ValueError('A completed study cannot retain an unfinished planned attempt')
    missing=any(p['status']!='completed_eligible' or p['normalized_risk'] is None for p in predictions.values())
    if type(value['comparisons']) is not list:raise ValueError('Explicit comparison decisions required')
    comparison_ids=set()
    for row in value['comparisons']:
        _keys(row,('comparison_id','decision','explanation','estimate','lower','upper','margin','independent_unit','n_units'))
        name=row['comparison_id'];_text(name);_text(row['explanation']);_text(row['independent_unit'])
        for key in ('estimate','lower','upper','margin'):_number(row[key])
        if ((row['lower'] is None)!=(row['upper'] is None)
                or row['lower'] is not None and row['lower']>row['upper']
                or row['n_units'] is not None and (type(row['n_units']) is not int or row['n_units']<1)):
            raise ValueError('Ordered supplied confidence interval and valid unit count required')
        if (name in comparison_ids or row['decision'] not in ('passed','failed','inconclusive','not_tested','blocked','development_screen_passed')):
            raise ValueError('Invalid comparison decision')
        comparison_ids.add(name)
        if row['decision']=='passed' and (not planned or not attempts or missing
                or value['evidence_scope']!='confirmation' or value['status']!='completed'):
            raise ValueError('A positive confirmation cannot omit planned predictions or lack qualified complete scope')
        if row['decision']=='passed' and (any(row[k] is None for k in ('estimate','lower','upper','margin'))
                or row['lower']<=row['margin']):
            raise ValueError('Positive comparison requires its supplied lower bound strictly above the declared margin')
    _keys(value['costs'],('physical_elapsed_ns','standalone_attributed_ns','scope'));_text(value['costs']['scope'])
    for key in ('physical_elapsed_ns','standalone_attributed_ns'):
        amount=value['costs'][key]
        if amount is not None and (type(amount) is not int or amount<0):raise ValueError('Nonnegative declared cost or unknown required')
    if type(value['limitations']) is not list:raise ValueError('Explicit limitations inventory required')
    for item in value['limitations']:_text(item)
    if type(value['sources']) is not list or not value['sources']:raise ValueError('Evidence references required')
    seen=set()
    for ref in value['sources']:
        _keys(ref,('path','sha256'));_text(ref['path']);_digest(ref['sha256'])
        if ref['path'] in seen:raise ValueError('Duplicate evidence reference')
        seen.add(ref['path'])
    return predictions,missing



def _secondary_families(values):
    if type(values) is not list:raise ValueError('Explicit secondary family list required')
    seen=set();result=[]
    for f in values:
        practical=type(f) is dict and f.get('construction')=='paired_group_bootstrap_guilbaud2008_choice2_strict'
        fields={'family_id','construction','planned_hypotheses','independent_unit','n_units','alpha','margin','status','unavailable_reason','rows'}
        _keys(f,fields|({'bike_block_days','iterations','dataset_assumptions'} if practical else set()))
        _text(f['family_id']);planned=_names(f['planned_hypotheses']);_number(f['margin']);_number(f['alpha'])
        if practical:_practical_family_units(f)
        elif (f['construction']!='teacher_t_guilbaud2008_choice2_strict' or f['independent_unit']!='teacher'
                or type(f['n_units']) is not int or f['n_units']<2):
            raise ValueError('Invalid teacher family construction or independent units')
        if (f['family_id'] in seen or not planned
                or f['alpha'] is None or not 0<f['alpha']<1 or f['margin'] is None
                or f['status'] not in ('completed','unavailable_family') or type(f['rows']) is not list):
            raise ValueError('Invalid secondary family, construction or independent units')
        seen.add(f['family_id']);rows={}
        for row in f['rows']:
            _keys(row,('hypothesis_id','raw_p','adjusted_p','lower','lower_endpoint_excluded','rejected'))
            name=row['hypothesis_id']
            if type(name) is not str or name not in planned or name in rows:raise ValueError('Secondary family inventory differs')
            rows[name]=row
            if f['status']=='unavailable_family':
                if any(row[k] is not None for k in ('raw_p','adjusted_p','lower','lower_endpoint_excluded','rejected')):
                    raise ValueError('Unavailable family cannot carry adjusted inference')
            else:
                for key in ('raw_p','adjusted_p','lower'):
                    _number(row[key])
                    if row[key] is None:raise ValueError('Completed secondary inference requires finite numbers')
                if (not 0<=row['raw_p']<=row['adjusted_p']<=1 or type(row['rejected']) is not bool
                        or row['rejected']!=(row['adjusted_p']<f['alpha']) or type(row['lower_endpoint_excluded']) is not bool):
                    raise ValueError('Secondary adjusted probability/rejection differs')
        if set(rows)!=set(planned):raise ValueError('Every secondary hypothesis must remain visible')
        if f['status']=='unavailable_family':_text(f['unavailable_reason'])
        else:
            if f['unavailable_reason'] is not None:raise ValueError('Completed secondary family has an unavailable reason')
            all_rejected=all(row['rejected'] for row in rows.values())
            if all_rejected:
                endpoints={row['lower'] for row in rows.values()}
                if len(endpoints)!=1:raise ValueError('All-rejected construction requires its common lower bound')
            for row in rows.values():
                endpoint=row['lower'];excluded=row['lower_endpoint_excluded']
                if all_rejected:
                    valid=endpoint>=f['margin'] and excluded==(endpoint==f['margin'])
                elif row['rejected']:valid=endpoint==f['margin'] and excluded
                else:valid=endpoint<=f['margin'] and not excluded
                if not valid:raise ValueError('Secondary endpoint contradicts its strict Holm region')
        result.append((f,[rows[name] for name in planned]))
    return result


def _practical_family_units(value):
    from .practical_v2 import OBSERVATIONAL,FAMILIES
    if (value['independent_unit']!='held_out_groups_or_days_within_fixed_tasks'
            or type(value['n_units']) is not dict or set(value['n_units'])!=set(OBSERVATIONAL)
            or any(type(n) is not int or n<2 for n in value['n_units'].values())
            or value['planned_hypotheses']!=list(FAMILIES[1:])
            or type(value['iterations']) is not int or value['iterations']<2
            or type(value['bike_block_days']) is not int or value['bike_block_days'] not in (3,7,14)
            or value['n_units']['bike']<value['bike_block_days']
            or type(value['dataset_assumptions']) is not dict or set(value['dataset_assumptions'])!=set(OBSERVATIONAL)):
        raise ValueError('Exact practical family and per-task group/day sampling units required')
    for name in OBSERVATIONAL:
        expected=['fixed_fitted_models','local_dependence_covered_by_declared_calendar_block'] if name=='bike' else ['fixed_fitted_models','held_out_groups_exchangeable']
        if name=='ccpp':expected.append('unverified_exchangeability')
        if name=='parkinsons':expected.append('few_subject_clusters; resample_count_does_not_increase_units')
        assumptions=_names(value['dataset_assumptions'][name])
        if assumptions!=expected:raise ValueError('Practical conditional/dependence assumptions changed')


def teacher_family_report_record(family_id,result):
    """Project supplied teacher arithmetic; authenticate no data or protocol."""
    _text(family_id)
    if (type(result) is not dict or result.get('schema')!='research/v2-holm-compatible-teacher-bounds'
            or result.get('status') not in ('completed','unavailable_family')):
        raise ValueError('Explicit compatible teacher-region result required')
    planned=_names(result['planned_hypotheses']);teachers=_names(result['teacher_ids'])
    if result['teacher_count']!=len(teachers) or result['family_size']!=len(planned):
        raise ValueError('Teacher/family inventory differs from its supplied inference')
    available=result['status']=='completed';rows=[]
    for name in planned:
        bound=result['bounds'][name] if available else None
        rows.append(dict(hypothesis_id=name,raw_p=result['holm']['raw_p'][name] if available else None,
            adjusted_p=result['holm']['adjusted_p'][name] if available else None,
            lower=bound['lower'] if available else None,
            lower_endpoint_excluded=bound['lower_endpoint_excluded'] if available else None,
            rejected=bound['rejects_margin_null'] if available else None))
    reason=None if available else ('Unavailable complete family; missing '+str(result['missing_hypotheses'])+
        '; degenerate '+str(result['degenerate_hypotheses'])+'.')
    value=dict(family_id=family_id,construction='teacher_t_guilbaud2008_choice2_strict',
        planned_hypotheses=list(planned),independent_unit='teacher',n_units=len(teachers),
        alpha=result['alpha'],margin=result['margin'],status=result['status'],unavailable_reason=reason,rows=rows)
    _secondary_families([value])
    return value


def practical_family_report_record(family_id,result):
    """Project supplied paired bootstrap arithmetic, granting no authentication."""
    from .practical_v2 import FAMILIES
    _text(family_id)
    if type(result) is not dict or result.get('schema')!='research/v2-complete-practical-comparison-arithmetic':
        raise ValueError('Complete practical comparison arithmetic required')
    family=result['multiplicity'];champion=result['champion'];comparisons=result['family_comparisons']
    planned=list(FAMILIES[1:]);statuses=('completed','undefined_family','insufficient_resampling_resolution')
    if (type(family) is not dict or family.get('schema')!='research/v2-practical-bootstrap-margin-family'
            or family.get('planned_hypotheses')!=planned or family.get('family_size')!=13
            or family.get('status') not in statuses or type(comparisons) is not dict or set(comparisons)!=set(planned)):
        raise ValueError('All thirteen unchanged practical family hypotheses required')
    keys=('iterations','bike_block_days','pairing_index_sha256','resampling_units','dataset_assumptions')
    for name in planned:
        if any(comparisons[name].get(key)!=champion.get(key) for key in keys):
            raise ValueError('Practical family resampling pairing or units differ')
    available=family['status']=='completed';rows=[]
    if available and (type(family.get('holm')) is not dict or type(family.get('bounds')) is not dict
            or set(family['bounds'])!=set(planned) or family['holm'].get('planned_hypotheses')!=planned
            or family['holm'].get('family_size')!=13):
        raise ValueError('Complete Holm and compatible-region records required')
    for name in planned:
        bound=family['bounds'][name] if available else None
        rows.append(dict(hypothesis_id=name,raw_p=family['holm']['raw_p'][name] if available else None,
            adjusted_p=family['holm']['adjusted_p'][name] if available else None,
            lower=bound['lower'] if available else None,
            lower_endpoint_excluded=bound['lower_endpoint_excluded'] if available else None,
            rejected=bound['rejects_margin_null'] if available else None))
    value=dict(family_id=family_id,construction='paired_group_bootstrap_guilbaud2008_choice2_strict',
        planned_hypotheses=planned,independent_unit='held_out_groups_or_days_within_fixed_tasks',
        n_units=dict(champion['resampling_units']),alpha=family['alpha'],margin=family['margin'],
        status='completed' if available else 'unavailable_family',
        unavailable_reason=None if available else 'Unavailable complete family: '+family['status']+'; '+str(family['unavailable_hypotheses']),
        rows=rows,bike_block_days=champion['bike_block_days'],iterations=champion['iterations'],
        dataset_assumptions={k:list(v) for k,v in champion['dataset_assumptions'].items()})
    _secondary_families([value])
    return value

def render_report(audit):
    """Format supplied audit facts deterministically; never infer study success."""
    predictions,missing=_validated(audit)
    lines=[f'# {_text(audit["study_id"])}', '',f'Study status: {audit["status"]}. Evidence scope: {audit["evidence_scope"]}.','']
    if audit['evidence_scope']!='confirmation':
        lines+=['No protected confirmation is authorized by this report.','']
    else:lines+=['Protected source, protocol and raw-prediction authentication are attested by the supplied audit; the renderer does not perform that authentication.','']
    for key,label in [('source_authenticated','Source authentication'),('raw_predictions_authenticated','Raw prediction authentication'),
            ('protected_protocol_authenticated','Protected protocol authentication')]:
        lines.append(label+': '+str(audit['provenance'][key]).lower()+'.')
    lines.append('')
    lines+=[_text(audit['independent_units']),'',f'Source commit: {audit["source"]["commit"]}.',
        f'Package SHA256: {audit["source"]["package_sha256"]}.','']
    if audit['protocol_sha256'] is not None:lines += [f'Protocol SHA256: {audit["protocol_sha256"]}.','']
    lines+=['## Final prediction inventory','',
        '| Prediction | Method | Task | Repeat | Status | Normalization | Normalized risk | Primary MSE | Row RMSE | Row MAE |',
        '| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |']
    for identifier in audit['planned_prediction_ids']:
        row=predictions[identifier];risk='unavailable' if row['normalized_risk'] is None else format(row['normalized_risk'],'.10g')
        lines.append('| '+' | '.join([_text(identifier),_text(row['method_id']),_text(row['task_id']),str(row['repetition']),
            row['status'],row['normalization_status'],risk,*[_number(row[k]) for k in ('primary_MSE','row_RMSE','row_MAE')]])+' |')
    if any(row['status']!='completed_eligible' for row in predictions.values()):
        lines+=['','Incomplete planned predictions block a complete positive primary claim.']
    if any(row['normalization_status']=='undefined_F_variance' for row in predictions.values()):
        lines+=['','Undefined normalization retains original-unit metrics and blocks a complete positive primary claim.']
    lines+=['','## Comparison decisions','',
        '| Comparison | Decision | Estimate | Lower | Upper | Margin | Unit | Count | Explanation |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |']
    for row in audit['comparisons']:
        cells=[_text(row['comparison_id']),row['decision'],*[_number(row[k]) for k in ('estimate','lower','upper','margin')],
            _text(row['independent_unit']),'unavailable' if row['n_units'] is None else str(row['n_units']),_text(row['explanation'])]
        lines.append('| '+' | '.join(cells)+' |')
    for family,rows in _secondary_families(audit.get('secondary_families',[])):
        practical=family['construction']=='paired_group_bootstrap_guilbaud2008_choice2_strict'
        if practical:
            from .practical_v2 import OBSERVATIONAL
            unit_text='; '.join(name+'='+str(family['n_units'][name]) for name in OBSERVATIONAL)
            explanation=['Construction: '+family['construction']+'. Held-out groups/days by fixed task: '+unit_text+'.',
                str(family['iterations'])+' bootstrap draws; Bike calendar-day block: '+str(family['bike_block_days'])+'.',
                'Approximate conditional sampling for fixed fitted models; not retraining or dataset-population uncertainty. Valid marginal bootstrap and dependence assumptions are required. Supplied calculations do not authenticate data or protocol.']
            explanation+=['Assumptions for '+name+': '+', '.join(_text(s) for s in family['dataset_assumptions'][name])+'.' for name in OBSERVATIONAL]
        else:explanation=['Construction: '+family['construction']+'. Independent units: teacher; count '+str(family['n_units'])+'.',
            'Conditional on valid marginal teacher t inference. Supplied calculations do not authenticate data or protocol.']
        lines+=['','## Secondary family: '+_text(family['family_id']),'',
            explanation[0],
            'Family alpha: '+_number(family['alpha'])+'. Margin: '+_number(family['margin'])+'.',
            *explanation[1:],
            'No numeric gap beyond the margin is established by an open endpoint equal to that margin.','',
            '| Hypothesis | Raw p | Holm adjusted p | Lower assertion | Margin null test |',
            '| --- | ---: | ---: | --- | --- |']
        if family['status']=='unavailable_family':lines+=['',_text(family['unavailable_reason']),'']
        for row in rows:
            assertion=('unavailable' if row['lower'] is None else
                'effect '+('> ' if row['lower_endpoint_excluded'] else '>= ')+_number(row['lower']))
            test='unavailable' if row['rejected'] is None else 'rejected (arithmetic)' if row['rejected'] else 'not rejected'
            lines.append('| '+' | '.join([_text(row['hypothesis_id']),_number(row['raw_p']),_number(row['adjusted_p']),assertion,test])+' |')
    lines+=['','## Attempt inventory','','| Attempt | Status |','| --- | --- |']
    for row in audit['attempts']:lines.append(f'| {_text(row["attempt_id"])} | {row["status"]} |')
    costs=audit['costs'];lines+=['','## Resources','']
    for key,label in [('physical_elapsed_ns','Containing physical seconds'),('standalone_attributed_ns','Standalone attributed seconds')]:
        amount=costs[key];lines.append(label+': '+('unavailable' if amount is None else f'{amount/1e9:.6f}')+'.')
    lines+=['',_text(costs['scope']),'','## Limitations','']
    lines.extend('- '+_text(item) for item in audit['limitations'])
    lines+=['','## Evidence references','','| Path | SHA256 |','| --- | --- |']
    for ref in sorted(audit['sources'],key=lambda row:row['path']):lines.append(f'| {_text(ref["path"])} | {ref["sha256"]} |')
    return '\n'.join(lines)+'\n'


def write_report(audit,output):
    """Publish a fresh report only after complete renderer validation."""
    content=render_report(audit).encode('utf-8');path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(content);stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path),bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
