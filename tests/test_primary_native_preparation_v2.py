"""Primary trial templates are byte-staged without exposing validation targets."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.primary_native_preparation_v2') is not None,'Missing exact primary native callback preparation'
    from treeneuron.research import primary_native_preparation_v2
    return primary_native_preparation_v2


def setup(tmp_path):
    from test_native_qualification_uncapped import prepared_config
    from treeneuron.research.method_worker import _ROOT,_sha,scientific_source_sha256
    from treeneuron.research.search_supports_v2 import family_search_spec
    from treeneuron.research.search_controller_v2 import make_search_plan
    _,value=prepared_config(tmp_path/'input');row=value['datasets']['energy'];prepared=Path(row['root'])
    manifest=json.loads((prepared/'manifest.json').read_text());spec=family_search_spec('P13',
        dict(columns=manifest['feature_names'],categorical=manifest['categorical_features']))
    plan=make_search_plan(identity=dict(job_id='energy/outer-0/repeat-0/P13',family_id='P13',search_repeat=0,
        source_sha256=scientific_source_sha256(),runtime_sha256=_sha(_ROOT/'locks/research-v2/core.json'),
        support_sha256=spec['support_hash'],split_sha256=manifest['splits'][0]['split_hash']),
        fit_seed=17,hpo_seed32=91,anchors=[a['config'] for a in spec['anchors']],mode='finite',support=None,
        candidates=spec['finite_configurations'][1:],max_attempts=32,deadline_ns=None,prefixes_ns=[600000000000])
    return dict(plan=plan,prepared_root=prepared,expected_manifest_sha256=row['manifest_sha256'],outer_fold=0,output=tmp_path/'template')


def test_actual_primary_role_bytes_stage_without_label_or_array_decoding(tmp_path,monkeypatch):
    module=api();args=setup(tmp_path)
    def forbidden(*a,**kw):raise AssertionError('Template staging must not decode role arrays or labels')
    monkeypatch.setattr(np,'load',forbidden)
    result=module.stage_primary_native_template(**args)
    assert result['staged_roles']==['F','G','V_features']
    assert result['native_workload_qualified'] is result['quality_eligible'] is result['T_accessed'] is False
    assert result['arrays_or_labels_decoded'] is False
    path=Path(result['request_path']);request=json.loads(path.read_text())
    assert request['budget']=={'native_stopping':True} and request['fit_seed']==17
    assert set(request['files'])=={'F','G','V_features'}
    from treeneuron.research.method_worker import _sha
    for role,ref in request['files'].items():
        name='V-features.npz' if role=='V_features' else role+'.npz'
        assert _sha(path.parent/ref['path'])==ref['sha256']==_sha(args['prepared_root']/('outer-0/'+name))
    assert not (path.parent/'V.npz').exists() and not (path.parent/'T.npz').exists()


@pytest.mark.parametrize('fault',['secondary','manifest','role_bytes','source','runtime','support','split','anchor','deadline','self_output'])
def test_bad_primary_template_contract_rejects_before_staging(tmp_path,fault):
    module=api();args=setup(tmp_path)
    if fault=='secondary':args['outer_fold']=1
    elif fault=='manifest':args['expected_manifest_sha256']='0'*64
    elif fault=='role_bytes':(args['prepared_root']/'outer-0/F.npz').write_bytes(b'changed source role')
    elif fault=='self_output':args['output']=args['prepared_root']/'fitting-input'
    else:
        from treeneuron.research.search_controller_v2 import make_search_plan
        value=args['plan'].to_dict();keys=('identity','fit_seed','hpo_seed32','anchors','mode','support','candidates','max_attempts','deadline_ns','prefixes_ns')
        fields={k:copy.deepcopy(value[k]) for k in keys}
        if fault in ('source','runtime','support','split'):fields['identity'][fault+'_sha256']='0'*64
        elif fault=='anchor':fields['anchors'][0]['parameters']={'undeclared':True}
        else:fields['deadline_ns']=7200000000000
        args['plan']=make_search_plan(**fields)
    with pytest.raises((ValueError,PermissionError)):module.stage_primary_native_template(**args)
    assert not args['output'].exists()


def test_validation_label_sentinel_inside_self_rehashed_query_archive_rejects(tmp_path):
    module=api();args=setup(tmp_path);path=args['prepared_root']/'outer-0/V-features.npz'
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    np.savez(path,**arrays,y=np.zeros(len(arrays['X'])))
    from treeneuron.config import canonical_hash
    from treeneuron.research.method_worker import _sha
    manifest=args['prepared_root']/'manifest.json';value=json.loads(manifest.read_text())
    value['files']['outer-0/V-features.npz'].update(sha256=_sha(path),bytes=path.stat().st_size)
    value['manifest_hash']=canonical_hash({k:v for k,v in value.items() if k!='manifest_hash'})
    manifest.write_text(json.dumps(value));args['expected_manifest_sha256']=_sha(manifest)
    with pytest.raises(ValueError,match='role|query|label|archive'):module.stage_primary_native_template(**args)
    assert not args['output'].exists()


def test_false_registry_cannot_reach_template_staging_or_callback_construction(tmp_path,monkeypatch):
    module=api();root=tmp_path/'false-registry';root.mkdir();(root/'manifest.json').write_text('{"native_workload_qualified":true}')
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    def forbidden(**kw):raise AssertionError('False qualification reached staging')
    monkeypatch.setattr(module,'stage_primary_native_template',forbidden)
    with pytest.raises(ValueError):module.prepare_primary_native_callback(practical_plan=None,expected_practical_plan_hash='0'*64,
        job_id='energy/outer-0/repeat-0/P13',support_reference=None,
        qualification_reference=dict(root=str(root),inventory_sha256=canonical_hash(_inventory(root))),
        output=tmp_path/'not-created',search_directory=tmp_path/'search')
    assert not (tmp_path/'not-created').exists() and not (tmp_path/'search').exists()


@pytest.mark.parametrize('fault',[None,'other_working_output','working_search','proof_output','proof_search',
    'source_search','overlap','guard_bool','guard_above_track'])
def test_pure_factory_locations_preserve_every_working_and_proof_root(tmp_path,fault):
    module=api();assert hasattr(module,'_check_preparation_locations'),'Missing all-input factory path boundary'
    working=[tmp_path/'energy',tmp_path/'airfoil'];proofs=[tmp_path/'immutable-proof']
    output=tmp_path/'output';search=tmp_path/'search';guard=4*1024**3
    if fault=='other_working_output':output=working[1]/'new-template'
    elif fault=='working_search':search=working[0]/'search'
    elif fault=='proof_output':output=proofs[0]/'new-template'
    elif fault=='proof_search':search=proofs[0]/'search'
    elif fault=='source_search':
        from treeneuron.research.method_worker import _ROOT
        search=_ROOT/'src/should-never-be-created'
    elif fault=='overlap':search=output/'search'
    elif fault=='guard_bool':guard=True
    elif fault=='guard_above_track':guard=17*1024**3
    kwargs=dict(output=output,search=search,working_roots=working,proof_roots=proofs,rss_limit_bytes=guard)
    if fault is None:module._check_preparation_locations(**kwargs)
    else:
        with pytest.raises(ValueError):module._check_preparation_locations(**kwargs)
    assert not output.exists() and not search.exists()
