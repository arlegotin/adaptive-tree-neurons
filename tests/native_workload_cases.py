"""Actual native software fixtures, never official-data qualification."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


def actual_native_family(tmp_path,family,*,variant=None):
    from treeneuron.config import canonical_hash
    from treeneuron.data.tabular import DATASETS,table_from_frame
    from treeneuron.research.deployment_worker_v2 import _ROOT,_source
    from treeneuron.research.native_fit_v2 import run_native_fit
    from treeneuron.research.search_supports_v2 import family_search_spec
    from treeneuron.research.transfer import TransferStore,seal_table
    rows_per_group=2 if family=='P12' else 65
    n=12*rows_per_group;schema=DATASETS['energy'];index=np.arange(n);group=index//rows_per_group
    frame=pd.DataFrame({name:index.astype(float) for name in schema.raw_columns})
    for name in ('X1','X2','X3','X4','X5'):frame[name]=group.astype(float)
    frame['X6']=index%4;frame['X8']=index%6;frame['X7']=(index%5)/10
    frame['Y1']=group+.2*frame.X6+.1*frame.X7-.1*frame.X8
    # Synthetic content with a schema locator. The full registry must reject
    # treating this as the actual 768-row acquisition/real workload.
    table=table_from_frame('energy',frame,raw_sha256=schema.raw_sha256)
    sealed=tmp_path/'sealed';seal_table(table,sealed);store=TransferStore(sealed)
    manifest=store.manifest;feature_schema=dict(columns=manifest['feature_names'],categorical=manifest['categorical_features'])
    anchors=family_search_spec(family,feature_schema)['anchors']
    if variant is not None:anchor=next(a for a in anchors if a['config'].get('variant')==variant)
    else:anchor=next(a for a in anchors if a['config'].get('variant')=='upper') if family=='P02' else anchors[0]
    config=copy.deepcopy(anchor['config'])
    if family=='P12':
        # A full native stopping procedure at a declared search geometry, not
        # a shortened update budget or a substitute for the 2048-tree anchor.
        config['mode']='search';config['parameters'].update(trees=512,depth=4)
    attempt=tmp_path/'generation';inputs=attempt/'input';inputs.mkdir(parents=True)
    store._access(0,['F_labels','G_labels','V_features']);files={}
    for role in ('F','G','V_features'):
        relative='outer-0/'+('V-features.npz' if role=='V_features' else role+'.npz')
        target=inputs/(role+'.npz');shutil.copyfile(sealed/relative,target)
        files[role]=dict(path=target.name,sha256=manifest['files'][relative]['sha256'])
    request=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',
        job_id='workload-fixture/'+family,family_id=family,fit_seed=17,scientific_source_sha256=_source(),
        config=config,budget=dict(native_stopping=True),files=files,
        observation_identity=dict(dataset='energy',split_kind='grouped',split_sha256=manifest['splits'][0]['split_hash']))
    path=inputs/'request.json';path.write_text(json.dumps(request,sort_keys=True)+'\n')
    digest=hashlib.sha256(path.read_bytes()).hexdigest();names={
        'P06':'realmlp','P07':'tabm','P08':'tree','P09':'tree','P10':'tree','P11':'tree'}
    overlays=(_ROOT/('environment/research-v2-'+names[family]+'-site'),) if family in names else ()
    native=run_native_fit(path,expected_request_sha256=digest,output=attempt/'native',
        interpreter=sys.executable,extra_pythonpath=overlays,rss_limit_bytes=4*1024**3)
    assert native['status']=='completed_ineligible',native
    return dict(request_path=path,expected_request_sha256=digest,native_reference=native['recovery_reference'],
        prepared_root=sealed,expected_manifest_sha256=hashlib.sha256((sealed/'manifest.json').read_bytes()).hexdigest(),
        outer_fold=0,expected_fit_seed=17,expected_configuration=config),attempt
