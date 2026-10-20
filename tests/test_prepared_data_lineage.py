"""Recorded data/source byte consistency must not decode targets or fit models."""
from collections import Counter
import copy
from dataclasses import asdict,replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from treeneuron.config import canonical_hash


def api():
    assert importlib.util.find_spec('treeneuron.research.prepared_lineage_v2') is not None,'Missing prepared data byte/source audit'
    from treeneuron.research import prepared_lineage_v2
    return prepared_lineage_v2


def publish(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(module,tmp_path,monkeypatch):
    from treeneuron.data.tabular import DATASETS
    root=Path(__file__).resolve().parents[1]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    source=module.archived_scientific_source(commit)['source_sha256']
    prepared=tmp_path/'prepared';prepared.mkdir();acquisition=tmp_path/'acquisition.json';entries=[];datasets={}
    for name,original in list(DATASETS.items()):
        raw=tmp_path/'raw'/str(original.uci_id)/original.raw_member;raw.parent.mkdir(parents=True,exist_ok=True)
        raw.write_bytes(('opaque fixture raw '+name).encode());schema=replace(original,raw_sha256=sha(raw))
        monkeypatch.setitem(DATASETS,name,schema)
        entries.append(dict(dataset_id=schema.uci_id,status='raw_hash_matched',matched_members=[dict(
            path=raw.relative_to(tmp_path).as_posix(),bytes=raw.stat().st_size,sha256=sha(raw))]))
        manifest=dict(schema='research/v2-sealed-transfer',dataset=name,raw_sha256=schema.raw_sha256,
            rows=schema.rows,groups=schema.groups,semantic_schema=asdict(schema),feature_names=list(schema.features),
            categorical_features=list(schema.categorical),raw_identity_verified=True,protected_labels_read_for_modeling=False,
            files={},splits=[])
        for fold in range(5):
            roles=['F']*(schema.rows-6)+['G']*2+['V']*2+['T']*2
            split=dict(schema='research/v2-transfer-split',dataset=name,outer_fold=fold,
                primary=fold==(4 if name=='bike' else 0),roles=roles,actual_rows=dict(Counter(roles)))
            split['split_hash']=canonical_hash(split);manifest['splits'].append(split)
            names=[f'outer-{fold}/{r}.npz' for r in 'FG']+[f'outer-{fold}/{r}-features.npz' for r in 'VT']
            names += [f'coordinator/outer-{fold}/V-labels.npz',f'protected/outer-{fold}/T-labels.npz']
            for relative in names:
                path=prepared/name/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(('opaque sealed fixture '+name+'/'+relative).encode())
                manifest['files'][relative]=dict(bytes=path.stat().st_size,sha256=sha(path))
        manifest['manifest_hash']=canonical_hash(manifest);publish(prepared/name/'manifest.json',manifest)
        datasets[name]={k:manifest[k] for k in ('manifest_hash','rows','groups','raw_sha256')}
    publish(acquisition,dict(datasets=entries))
    suite=dict(schema='research/v2-prepared-suite',status='completed',datasets=datasets,models_fitted=False,
        protected_labels_read_for_modeling=False,source_sha256=source,dependencies={'fixture_only':True},
        argv=['fixture'],cwd=str(tmp_path),environment_overrides={},config=dict(schema='research/v2-data-preparation',
            datasets=list(DATASETS),acquisition_manifest='historical-locator-not-read',acquisition_manifest_sha256=sha(acquisition)))
    suite['suite_hash']=canonical_hash(suite);publish(prepared/'suite.json',suite)
    return dict(prepared_root=prepared,expected_suite_sha256=sha(prepared/'suite.json'),
        acquisition_manifest=acquisition,expected_acquisition_sha256=sha(acquisition),preparation_commit=commit)


def test_byte_audit_binds_all_six_datasets_without_numpy_pandas_or_target_decoding(tmp_path,monkeypatch):
    module=api();args=fixture(module,tmp_path,monkeypatch)
    import numpy as np
    import pandas as pd
    def forbidden(*a,**kw):raise AssertionError('Byte/source audit cannot decode tables or arrays')
    monkeypatch.setattr(np,'load',forbidden);monkeypatch.setattr(pd,'read_csv',forbidden);monkeypatch.setattr(pd,'read_excel',forbidden)
    result=module.audit_prepared_transfer_suite(**args)
    assert result['recorded_preparation_source_matches_commit'] is True
    assert result['raw_assets_hashed']==6 and result['role_archives_hashed']==180 and result['protected_label_archives_hashed']==30
    assert set(result['datasets'])=={'airfoil','concrete','ccpp','parkinsons','bike','energy'}
    assert result['arrays_or_labels_decoded'] is result['data_content_rederived'] is result['native_workload_qualified'] is result['protected_access_authorized'] is False
    assert result['models_fit']==0 and result['protected_metrics_computed']==0


@pytest.mark.parametrize('change',['suite_reference','raw_bytes','protected_bytes','source','traversal','duplicate'])
def test_corrupt_or_self_rehashed_data_records_cannot_override_the_pinned_sources(tmp_path,monkeypatch,change):
    module=api();args=fixture(module,tmp_path,monkeypatch);suite_path=args['prepared_root']/'suite.json'
    suite=json.loads(suite_path.read_text());acq=json.loads(args['acquisition_manifest'].read_text())
    if change=='suite_reference':args['expected_suite_sha256']='0'*64
    elif change=='raw_bytes':(tmp_path/acq['datasets'][0]['matched_members'][0]['path']).write_bytes(b'changed raw bytes')
    elif change=='protected_bytes':(args['prepared_root']/'airfoil/protected/outer-0/T-labels.npz').write_bytes(b'opaque corrupt bytes')
    else:
        if change=='source':suite['source_sha256']='0'*64
        else:
            if change=='traversal':acq['datasets'][0]['matched_members'][0]['path']='../escape'
            else:acq['datasets'].append(acq['datasets'][0])
            publish(args['acquisition_manifest'],acq);args['expected_acquisition_sha256']=sha(args['acquisition_manifest'])
            suite['config']['acquisition_manifest_sha256']=args['expected_acquisition_sha256']
        suite['suite_hash']=canonical_hash({k:v for k,v in suite.items() if k!='suite_hash'})
        publish(suite_path,suite);args['expected_suite_sha256']=sha(suite_path)
    with pytest.raises(ValueError):module.audit_prepared_transfer_suite(**args)


def test_metadata_replacement_after_decoding_cannot_bind_old_values_to_new_file_hash(tmp_path,monkeypatch):
    module=api();args=fixture(module,tmp_path,monkeypatch);original=module._json;changed=[]
    def replace_after_decode(content):
        value=original(content)
        if value.get('schema')=='research/v2-sealed-transfer' and value.get('dataset')=='airfoil' and not changed:
            changed.append(True);different=copy.deepcopy(value)
            different['feature_names']=list(reversed(different['feature_names']))
            different['manifest_hash']=canonical_hash({k:v for k,v in different.items() if k!='manifest_hash'})
            publish(args['prepared_root']/'airfoil/manifest.json',different)
        return value
    monkeypatch.setattr(module,'_json',replace_after_decode)
    with pytest.raises(ValueError):module.audit_prepared_transfer_suite(**args)
    assert changed
