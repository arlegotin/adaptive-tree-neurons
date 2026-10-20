"""Search inspections need their own closed reader, independent of anchor IDs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.native_inspection_reader_v2') is not None,'Missing generic closed native inspection reader'
    from treeneuron.research import native_inspection_reader_v2
    return native_inspection_reader_v2


def reference(root):
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    return dict(root=str(root),inventory_sha256=canonical_hash(_inventory(root)))


@pytest.fixture(scope='module')
def candidates(tmp_path_factory):
    cached=[]
    def get():
        if not cached:
            from test_native_trial_bridge import setup,bridge,run
            from treeneuron.research.native_trial_bridge_v2 import InspectedNativeTrialBridge
            root=tmp_path_factory.mktemp('closed-search-candidates')
            with pytest.MonkeyPatch.context() as patch:args,old,spec,plan=setup(root,patch)
            search=root/'search';result=run(search,plan,bridge(InspectedNativeTrialBridge,args,plan,search))
            assert result['attempts_paid']==2 and result['winner'] is None
            rows=[]
            for i in range(2):
                attempt=search/f'attempt-{i:03}';generation=attempt/'generation'
                request=generation/'input/request.json';value=json.loads(request.read_text())
                native=json.loads((generation/'native-fit-result.json').read_text())
                rows.append(dict(reference=reference(attempt/'inspection'),immutable_generation_root=generation,
                    generation_arguments=dict(args,request_path=request,expected_request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),
                        native_reference=native['recovery_reference'],expected_configuration=value['config'])))
            cached.extend(rows)
        return cached
    return get


@pytest.mark.parametrize('index',[0,1])
def test_real_anchor_and_nonanchor_native_search_inspections_reopen_without_new_fit_or_prediction(candidates,monkeypatch,index):
    module=api();row=candidates()[index]
    from treeneuron.research.p13_simple import P13Predictor
    def forbidden(*a,**kw):raise AssertionError('Closed inspection reader cannot fit or predict')
    monkeypatch.setattr(P13Predictor,'fit',forbidden);monkeypatch.setattr(P13Predictor,'predict',forbidden)
    result=module.load_primary_native_inspection(**row)
    assert result['inspection_authenticated'] is result['prediction_reload_exact'] is True
    assert result['generation']['configuration_sha256']
    assert result['validation']['score']>=0 and result['V_labels_read_by_coordinator'] is True
    assert result['new_fit_or_inference'] is result['native_workload_qualified'] is result['quality_eligible'] is result['T_accessed'] is False


def test_other_valid_native_candidate_cannot_supply_the_inspection_generation(candidates):
    module=api();a,b=candidates();row=dict(a,generation_arguments=b['generation_arguments'],immutable_generation_root=b['immutable_generation_root'])
    with pytest.raises(ValueError,match='generation|container|original'):
        module.load_primary_native_inspection(**row)


@pytest.mark.parametrize('fault',['schema','authority','V_score','replay_status','rss_count','snapshot_sidecar',
    'replay_query_dtype','replay_prediction_dtype'])
def test_consistently_recommitted_bad_inspections_cannot_supply_production_evidence(candidates,fault):
    module=api();row=candidates()[0];root=Path(row['reference']['root'])
    paths=[root/'result.json',root/'process/process.json',root/'replay/terminal.json']
    before={p:p.read_bytes() for p in paths};sidecar=root/'input/snapshot/unrecorded.txt'
    try:
        saved=json.loads(paths[0].read_text())
        if fault=='schema':saved['schema']='research/v99-unrecognized'
        elif fault=='authority':saved['quality_eligible']=True
        elif fault=='V_score':saved['validation']['score']+=.5
        elif fault=='replay_status':
            replay=json.loads(paths[2].read_text());replay['prediction_reload_exact']=False
            paths[2].write_text(json.dumps(replay));saved['replay']=replay
        elif fault=='rss_count':
            process=json.loads(paths[1].read_text());process['rss_sample_count']+=1
            paths[1].write_text(json.dumps(process));saved['process']=process
        elif fault in ('replay_query_dtype','replay_prediction_dtype'):
            replay=json.loads(paths[2].read_text());replay['query_dtype' if fault=='replay_query_dtype' else 'prediction_dtype']='float32'
            paths[2].write_text(json.dumps(replay));saved['replay']=replay
        else:sidecar.write_text('extra replay state')
        from treeneuron.config import canonical_hash
        from treeneuron.research.workqueue import _inventory
        saved['replay_artifact_inventory_sha256']=canonical_hash(_inventory(root/'replay'))
        saved['replay_process_inventory_sha256']=canonical_hash(_inventory(root/'process'))
        paths[0].write_text(json.dumps(saved));changed=dict(row,reference=reference(root))
        with pytest.raises(ValueError):module.load_primary_native_inspection(**changed)
    finally:
        if sidecar.exists():sidecar.unlink()
        for p,raw in before.items():p.write_bytes(raw)
        assert reference(root)==row['reference']


def test_current_source_drift_rejects_before_new_V_access(candidates,monkeypatch):
    module=api();row=candidates()[0];access=Path(row['generation_arguments']['prepared_root'])/'access.jsonl';before=access.read_bytes()
    monkeypatch.setattr(module,'_source',lambda:'0'*64)
    with pytest.raises(ValueError,match='source|identity'):module.load_primary_native_inspection(**row)
    assert access.read_bytes()==before


def test_late_primary_label_change_after_real_arithmetic_rejects(candidates,monkeypatch):
    module=api();row=candidates()[0];path=Path(row['generation_arguments']['prepared_root'])/'coordinator/outer-0/V-labels.npz'
    original=module.score_primary_validation_prediction;before=path.read_bytes()
    def change_after_score(**kw):
        result=original(**kw);temporary=path.with_name('late-reader-change.tmp')
        temporary.write_bytes(before+b' ');temporary.replace(path);return result
    monkeypatch.setattr(module,'score_primary_validation_prediction',change_after_score)
    try:
        with pytest.raises(ValueError,match='validation|bytes|arithmetic'):
            module.load_primary_native_inspection(**row)
    finally:path.write_bytes(before)


def test_actual_temporal_primary_fold_uses_day_arithmetic_and_keeps_T_closed(tmp_path):
    module=api()
    import sys,shutil
    import numpy as np
    import pandas as pd
    from treeneuron.data.tabular import DATASETS,table_from_frame
    from treeneuron.research.transfer import seal_table
    from treeneuron.research.search_supports_v2 import family_search_spec
    from treeneuron.research.method_worker import scientific_source_sha256
    from treeneuron.research.native_fit_v2 import run_native_fit
    from treeneuron.research.native_candidate_v2 import inspect_primary_native_candidate
    days=np.arange(np.datetime64('2011-01-01'),np.datetime64('2013-01-01'));n=len(days);schema=DATASETS['bike']
    frame=pd.DataFrame({name:np.arange(n,dtype=float) for name in schema.raw_columns})
    frame['dteday']=days.astype(str);frame['hr']=12.;frame['cnt']=20.+np.sin(np.arange(n)/10.)
    # Synthetic chronology fixture; a raw-digest tag is not official acquisition.
    sealed=tmp_path/'sealed';manifest=seal_table(table_from_frame('bike',frame,raw_sha256=schema.raw_sha256),sealed)
    generation=tmp_path/'generation';inputs=generation/'input';inputs.mkdir(parents=True);files={}
    for role,name in (('F','F.npz'),('G','G.npz'),('V_features','V-features.npz')):
        target=inputs/(role+'.npz');shutil.copyfile(sealed/'outer-4'/name,target)
        files[role]=dict(path=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    config=family_search_spec('P13',dict(columns=list(schema.features),categorical=list(schema.categorical)))['anchors'][0]['config']
    request=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',job_id='fixture/P13/temporal',family_id='P13',
        fit_seed=17,scientific_source_sha256=scientific_source_sha256(),config=config,budget=dict(native_stopping=True),
        observation_identity=dict(dataset='bike',split_kind='temporal',split_sha256=manifest['splits'][4]['split_hash']),files=files)
    path=inputs/'request.json';path.write_text(json.dumps(request));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    native=run_native_fit(path,expected_request_sha256=digest,output=generation/'native',interpreter=sys.executable)
    assert native['status']=='completed_ineligible'
    args=dict(request_path=path,expected_request_sha256=digest,native_reference=native['recovery_reference'],prepared_root=sealed,
        expected_manifest_sha256=hashlib.sha256((sealed/'manifest.json').read_bytes()).hexdigest(),outer_fold=4,expected_fit_seed=17,expected_configuration=config)
    inspected=tmp_path/'inspection';result=inspect_primary_native_candidate(generation_arguments=args,immutable_generation_root=generation,
        output=inspected,interpreter=sys.executable)
    assert result['status']=='completed_candidate_inspection'
    proof=module.load_primary_native_inspection(reference(inspected),generation_arguments=args,immutable_generation_root=generation)
    assert proof['generation']['outer_fold']==4 and proof['generation']['dataset']=='bike'
    assert proof['validation']['score']==result['validation']['score']
    assert proof['T_accessed'] is proof['quality_eligible'] is False
    assert all(not any(role.startswith('T') for role in json.loads(r)['roles'])
        for r in (sealed/'access.jsonl').read_text().splitlines())
