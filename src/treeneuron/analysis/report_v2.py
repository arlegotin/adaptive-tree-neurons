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
    _keys(value,_FIELDS)
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
