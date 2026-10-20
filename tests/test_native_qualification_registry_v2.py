"""Complete coverage never substitutes for actual current-source native proofs."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from test_native_anchor_execution import evidence as native_anchor_evidence
from test_preparation_execution_v2 import actual_preparation as official_preparation


def api():
    assert importlib.util.find_spec('treeneuron.research.native_qualification_registry_v2') is not None, 'Missing complete native qualification registry'
    from treeneuron.research import native_qualification_registry_v2
    return native_qualification_registry_v2


def sentinel_bundle(tmp_path):
    """Untrusted coverage sentinels; NEVER a fake qualified happy-path bundle."""
    from treeneuron.data.tabular import DATASETS
    from treeneuron.research.native_verification_v2 import verification_profiles
    from treeneuron.research.search_supports_v2 import family_search_spec
    from treeneuron.research.support_probes_v2 import support_probe_cases
    def ref(label):return dict(root=str((tmp_path/label).resolve()),inventory_sha256='0'*64)
    anchors=[];supports=[]
    for dataset,s in DATASETS.items():
        schema=dict(columns=list(s.features),categorical=list(s.categorical))
        for i in range(14):
            family=f'P{i:02}'
            for a in family_search_spec(family,schema)['anchors']:
                anchors.append(dict(dataset=dataset,family_id=family,anchor_id=a['anchor_id'],reference=ref(dataset+family+a['anchor_id'])))
            for c in support_probe_cases(family,schema):
                supports.append(dict(dataset=dataset,family_id=family,case_id=c['case_id'],reference=ref(dataset+c['case_id'])))
    return dict(preparation_reference=ref('preparation'),profile_references={k:ref(k) for k in verification_profiles()},
        anchor_references=anchors,support_references=supports,recovery_reference=ref('recovery'),
        prepared_roots={name:str((tmp_path/'working-data'/name).resolve()) for name in DATASETS})


def test_fixed_catalog_covers_every_native_anchor_and_large_geometry_without_labels():
    catalog=api()._catalog()
    assert set(catalog)=={'profiles','anchors','supports'}
    assert len(catalog['profiles'])==7 and len(catalog['anchors'])==120 and len(catalog['supports'])==258
    for dataset in ('airfoil','concrete','ccpp','parkinsons','bike','energy'):
        anchors=[r for r in catalog['anchors'] if r['dataset']==dataset]
        assert len(anchors)==20 and {r['family_id'] for r in anchors}=={f'P{i:02}' for i in range(14)}
        assert len([r for r in catalog['supports'] if r['dataset']==dataset])==43
    bike=next(r for r in catalog['anchors'] if r['dataset']=='bike' and r['family_id']=='P13')
    assert bike['configuration']['feature_schema']['categorical']==['season','mnth','hr','weekday','weathersit','holiday','workingday']
    assert {r['primary_fold'] for r in catalog['anchors'] if r['dataset']=='bike'}=={4}
    assert {r['primary_fold'] for r in catalog['anchors'] if r['dataset']!='bike'}=={0}


@pytest.mark.parametrize('fault',['missing_profile','unknown_profile','missing_anchor','duplicate_anchor',
    'unknown_anchor','missing_support','duplicate_support','unknown_support','extra_field','bad_reference'])
def test_incomplete_or_ambiguous_coverage_rejects_before_any_component_reader(tmp_path,monkeypatch,fault):
    module=api();bundle=sentinel_bundle(tmp_path)
    if fault=='missing_profile':bundle['profile_references'].pop('node')
    elif fault=='unknown_profile':bundle['profile_references']['foundation']=bundle['profile_references'].pop('node')
    elif fault=='missing_anchor':bundle['anchor_references'].pop()
    elif fault=='duplicate_anchor':bundle['anchor_references'][-1]=copy.deepcopy(bundle['anchor_references'][0])
    elif fault=='unknown_anchor':bundle['anchor_references'][-1]['dataset']='unlisted'
    elif fault=='missing_support':bundle['support_references'].pop()
    elif fault=='duplicate_support':bundle['support_references'][-1]=copy.deepcopy(bundle['support_references'][0])
    elif fault=='unknown_support':bundle['support_references'][-1]['case_id']='invented'
    elif fault=='extra_field':bundle['anchor_references'][0]['quality_eligible']=True
    else:bundle['recovery_reference']['inventory_sha256']='not-a-digest'
    def forbidden(*a,**kw):raise AssertionError('Invalid coverage reached an evidence reader')
    monkeypatch.setattr(module,'load_preparation_execution',forbidden)
    output=tmp_path/'registry'
    with pytest.raises(ValueError,match='coverage|coordinate|profile|reference|schema'):
        module.seal_native_qualification_registry(**bundle,output=output)
    assert not output.exists()


def test_syntactically_complete_but_handfilled_proofs_never_publish_a_registry(tmp_path):
    module=api();bundle=sentinel_bundle(tmp_path);output=tmp_path/'registry'
    with pytest.raises((ValueError,FileNotFoundError)):
        module.seal_native_qualification_registry(**bundle,output=output)
    assert not output.exists()


def test_handfilled_authority_and_legacy_manifest_cannot_be_loaded(tmp_path):
    module=api();root=tmp_path/'handfilled';root.mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(schema='research/v2-native-qualification-registry',
        native_workload_qualified=True,protected_access_authorized=True,anchor_count=120)))
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    with pytest.raises(ValueError,match='schema|registry|proof|authority'):
        module.load_native_qualification_registry(dict(root=str(root.resolve()),inventory_sha256=canonical_hash(_inventory(root))))


def test_actual_official_preparation_alone_cannot_promote_missing_native_proofs(tmp_path,monkeypatch):
    from test_preparation_execution_v2 import copied_acquisition
    from treeneuron.research.preparation_execution_v2 import run_managed_preparation
    import sys
    module=api();path,digest=copied_acquisition(tmp_path)
    actual=run_managed_preparation(path,expected_config_sha256=digest,output=tmp_path/'actual-data-proof',
        interpreter=sys.executable,data_overlay=Path(__file__).resolve().parents[1]/'environment/research-v2-data-site')
    assert actual['status']=='completed_preparation'
    bundle=sentinel_bundle(tmp_path);bundle['preparation_reference']=actual['reference'];seen=[]
    original=module.load_preparation_execution
    def observing(ref):
        proof=original(ref);seen.append(proof);return proof
    monkeypatch.setattr(module,'load_preparation_execution',observing)
    output=tmp_path/'not-qualified'
    with pytest.raises((ValueError,FileNotFoundError)):
        module.seal_native_qualification_registry(**bundle,output=output)
    assert len(seen)==1 and seen[0]['role_archives_hashed']==180 and seen[0]['models_fit']==0
    assert seen[0]['arrays_or_labels_decoded'] is False and not output.exists()


@pytest.fixture(scope='module')
def partial_anchor(tmp_path_factory,native_anchor_evidence):
    from test_native_anchor_execution import qualify
    from treeneuron.research import native_anchor_v2
    from treeneuron.research.native_verification_v2 import load_native_implementation_verification
    from treeneuron.research.workqueue import _inventory
    from treeneuron.config import canonical_hash
    args,old,software=native_anchor_evidence();root=tmp_path_factory.mktemp('registry-actual-anchor')/'anchor'
    qualify(native_anchor_v2,args,old,software,root)
    return args,dict(root=str(root),inventory_sha256=canonical_hash(_inventory(root))),load_native_implementation_verification(software,expected_profile='simple')


def inspect_partial_anchor(module,partial_anchor):
    args,ref,software=partial_anchor
    coordinate=next(r for r in module._catalog()['anchors'] if r['dataset']=='energy' and r['family_id']=='P13')
    assert hasattr(module,'_inspect_anchor'),'Missing checked anchor/workload join'
    return module._inspect_anchor(coordinate,ref,args['prepared_root'],args['expected_manifest_sha256'],software,
        software['identity']['scientific_source_sha256'])


def test_one_actual_anchor_joins_its_own_native_workload_without_qualifying_the_suite(partial_anchor):
    result=inspect_partial_anchor(api(),partial_anchor)
    assert result['dataset']=='energy' and result['family_id']=='P13' and result['anchor_id']=='mean'
    assert result['training']['estimator']=='mean'
    assert 'native_workload_qualified' not in result and 'quality_eligible' not in result


def test_a_different_valid_native_generation_cannot_supply_an_anchors_workload(tmp_path,monkeypatch,partial_anchor):
    from test_native_generation_chain import actual
    from treeneuron.research.native_workload_v2 import inspect_native_workload
    module=api()
    with pytest.MonkeyPatch.context() as patch:args,other=actual(tmp_path,patch)
    different=inspect_native_workload(**args)
    # A real second valid numerical fixture; never a mocked complete registry.
    monkeypatch.setattr(module,'inspect_native_workload',lambda **kw:different)
    with pytest.raises(ValueError,match='different valid generation'):
        inspect_partial_anchor(module,partial_anchor)


def working_data(tmp_path,official_preparation):
    import shutil
    from treeneuron.research.preparation_execution_v2 import load_preparation_execution
    from treeneuron.data.tabular import DATASETS
    actual,_,_,_=official_preparation;proof=load_preparation_execution(actual['reference'])
    roots={}
    for name in DATASETS:
        source=Path(actual['reference']['root'])/'worker/data'/name;target=tmp_path/'working'/name
        target.mkdir(parents=True);manifest=json.loads((source/'manifest.json').read_text())
        for relative in ('manifest.json',*manifest['files']):
            dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,dest)
        roots[name]=str(target.resolve())
    bundle=sentinel_bundle(tmp_path);bundle.update(preparation_reference=actual['reference'],prepared_roots=roots)
    return bundle,proof


def test_working_access_logs_do_not_mutate_the_immutable_official_preparation(tmp_path,official_preparation):
    from treeneuron.research.preparation_execution_v2 import load_preparation_execution
    from treeneuron.research.transfer import TransferStore
    module=api();bundle,proof=working_data(tmp_path,official_preparation)
    assert hasattr(module,'_prepared_datasets'),'Missing separate immutable preparation/working-data join'
    before=module._prepared_datasets(bundle,proof)
    TransferStore(bundle['prepared_roots']['energy'])._access(0,['F_labels','G_labels','V_features'])
    assert module._prepared_datasets(bundle,proof)==before
    assert load_preparation_execution(bundle['preparation_reference'])==proof


@pytest.mark.parametrize('fault',['protected_bytes','manifest','root_alias'])
def test_separate_working_data_needs_identical_manifest_and_every_opaque_role(tmp_path,official_preparation,fault):
    module=api();bundle,proof=working_data(tmp_path,official_preparation)
    assert hasattr(module,'_prepared_datasets'),'Missing separate immutable preparation/working-data join'
    root=Path(bundle['prepared_roots']['energy'])
    if fault=='protected_bytes':(root/'protected/outer-0/T-labels.npz').write_bytes(b'changed opaque protected bytes')
    elif fault=='manifest':(root/'manifest.json').write_text('{}')
    else:bundle['prepared_roots']['energy']=str(Path(bundle['preparation_reference']['root'])/'worker/data/energy')
    with pytest.raises(ValueError,match='working|role|manifest|immutable|bytes'):
        module._prepared_datasets(bundle,proof)
