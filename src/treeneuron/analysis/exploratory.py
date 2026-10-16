"""Rebuild descriptive G/V metrics and paired differences from saved artifacts."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np

from treeneuron.evaluation.metrics import regression_metrics
from treeneuron.data.types import array_hash
from treeneuron.experiments.paired import audit_pairing, file_hash


def summarize_exploratory(study_dir: Path) -> dict[str, Any]:
    root = Path(study_dir)
    study = json.loads((root/'study.json').read_text())
    for name, expected in study['artifacts'].items():
        path = root/name
        if not path.is_file() or file_hash(path)!=expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    if study['status'] != 'completed':
        return study  # failures remain first-class retained results
    audit = audit_pairing(study)
    table = json.loads((root/'score-table.json').read_text())
    for name in ('top','random','sgd'):
        allocation = study['methods'][name]['allocation']
        if allocation['table_hash'] != table['table_hash']: raise ValueError('Score table identity mismatch')
        for unit, contrast in zip(allocation['ids'],allocation['contrasts_float64']):
            if contrast != table['contrasts'][unit]: raise ValueError('Per-ID contrast mismatch')
    with np.load(root/'development.npz',allow_pickle=False) as raw:
        variance=float(np.var(raw['F_y']))
        validation_x=raw['V_X'].copy()
    predictions={}
    methods={}
    for name,record in study['methods'].items():
        for stage in record['stages']:
            events=[json.loads(line) for line in (root/stage['path']/'events.jsonl').read_text().splitlines()]
            rows=[array_hash(np.array(e['row_ids'],dtype=np.int64)) for e in events if e['type']=='update']
            if rows!=stage['row_hashes'] or len(rows)!=stage['executed_updates']:
                raise ValueError('pairing row stream/count disagrees with executed events')
        values={}
        metrics={}
        for split,link in record['predictions'].items():
            artifact=json.loads((root/link['path']).read_text())
            if file_hash(root/link['path']) != link['sha256']: raise ValueError('Prediction hash mismatch')
            values[split]=artifact
            metrics[split]=regression_metrics(np.array(artifact['y']),np.array(artifact['prediction']),
                fit_target_variance=variance,groups=np.array(artifact['group_id']),
                noisefree=None if artifact['noise_free_target'] is None else np.array(artifact['noise_free_target']))
        if metrics != record['metrics']: raise ValueError('Saved metrics disagree with predictions')
        if study['config']['data']['population']=='conditional_slope_2d':
            # Descriptive slopes on each side of the actual conditional boundary.
            predicted=np.array(values['V']['prediction'])
            slopes={}
            for side,mask in [('left',validation_x[:,0]<0),('right',validation_x[:,0]>=0)]:
                design=np.column_stack((np.ones(int(mask.sum())),validation_x[mask,1]))
                coefficients=np.linalg.lstsq(design,predicted[mask],rcond=None)[0]
                slopes[side]=dict(intercept=float(coefficients[0]),slope=float(coefficients[1]),rows=int(mask.sum()))
            record['conditional_slopes_V']=slopes
        predictions[name]=values
        methods[name]={k:v for k,v in record.items() if k not in ('executed_row_ids','stages','growth_metadata')}
        methods[name]['stages']=[{k:v for k,v in s.items() if k not in ('training_loss','row_hashes')}
                                 for s in record['stages']]
    differences={}
    for left,right in [('top','random'),('top','keep'),('random','keep'),('gaussian','top'),('sham','keep')]:
        pair={}
        for split in ('G','V'):
            a,b=predictions[left][split],predictions[right][split]
            if any(a[key]!=b[key] for key in ('row_id','group_id','y')):
                raise ValueError('Prediction pairing rows/targets mismatch')
            pa,pb=np.array(a['prediction']),np.array(b['prediction'])
            pair[split]=dict(mse=methods[left]['metrics'][split]['mse']-methods[right]['metrics'][split]['mse'],
                max_prediction_abs=float(np.abs(pa-pb).max()),
                mean_prediction_difference=float((pa-pb).mean()))
        differences[f'{left}_minus_{right}']=pair
    if any(differences['sham_minus_keep'][s]['max_prediction_abs']!=0 for s in ('G','V')):
        raise ValueError('Identical sham continuation differs')
    return dict(schema_version=1,status=study['status'],study_id=study['study_id'],
        study_sha256=file_hash(root/'study.json'),phase='exploratory',interpretation=study['interpretation'],
        config=study['config'],config_hash=study['config_hash'],provenance=study['provenance'],data=study['data'],
        methods=methods,pairing_audit=audit,paired_differences=differences,
        tiny_gate_probe=study['tiny_gate_probe'],costs=study['costs'],resources=study['resources'],
        failures=study['failures'],test_labels_accessed=False,
        ceilings=[{'method':name,'stage':i,'improving_at_ceiling':stage['improving_at_ceiling']}
            for name,record in study['methods'].items() for i,stage in enumerate(record['stages']) if stage['hit_ceiling']],
        pending=['B0 spline','B6','B7','practical data scope','calibrated allowances and budget','independent pilot/freeze/confirmation/audit'])
