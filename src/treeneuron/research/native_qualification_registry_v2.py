"""Join actual native prerequisites; no score selection or protected authority.

Coverage is fixed before evidence is opened. A syntactically complete inventory
still needs every current-source component proof, including official preparation.
"""
import copy
from pathlib import Path
import sys

from treeneuron.config import canonical_hash
from treeneuron.data.tabular import DATASETS
from .ledger import write_new_json
from .method_worker import _ROOT, _digest, _read_json, _sha
from .native_anchor_v2 import _closed_reference, _bound_json, load_native_anchor_execution
from .native_recovery_proof_v2 import load_fullsize_node_recovery
from .native_verification_v2 import (verification_identity, verification_profiles,
    load_native_implementation_verification)
from .native_workload_v2 import inspect_native_workload, _bound_reader, _process_observation
from .preparation_execution_v2 import load_preparation_execution
from .search_supports_v2 import family_search_spec
from .support_probes_v2 import support_probe_cases, load_support_probe
from .workqueue import _inventory
from .validation_invocation_v2 import _within_validation, _load_once


_INPUTS={'preparation_reference','profile_references','anchor_references',
    'support_references','recovery_reference','prepared_roots'}
_SCHEMA='research/v2-native-qualification-registry'
_SCOPE=('Complete fixed native software/official-data/anchor/workload/short-support/'
    'recovery prerequisite join. Short supports exercise declared geometry, not '
    'every Cartesian hyperparameter combination. Each later candidate requires '
    'its own full native workload and inference/validation admission. No HPO '
    'journal, selected model, practical quality, global freeze or protected authority.')
_FLAGS=dict(native_workload_qualified=True,quality_eligible=False,
    HPO_journal_authenticated=False,protected_access_authorized=False,T_accessed=False)
_FAILURE_POLICY=dict(missing_mandatory='block_full_suite_without_predictive_imputation',
    P12='one_paid_identical_source_config_seed_checkpoint_resume',
    other_families='one_paid_identical_source_config_seed_restart',
    optional_foundation_tier='unqualified_separate_admission_required')


def _catalog():
    profiles=verification_profiles();anchors=[];supports=[]
    for dataset,schema in sorted(DATASETS.items()):
        features=dict(columns=list(schema.features),categorical=list(schema.categorical))
        fold=4 if dataset=='bike' else 0
        for index in range(14):
            family=f'P{index:02}';spec=family_search_spec(family,features)
            profile=next(p for p,v in profiles.items() if family in v['families'])
            common=dict(dataset=dataset,family_id=family,primary_fold=fold,
                support_sha256=spec['support_hash'],profile=profile)
            anchors.extend(dict(common,anchor_id=a['anchor_id'],configuration=a['config']) for a in spec['anchors'])
            supports.extend(dict(common,case_id=c['case_id'],configuration=c['native_config'])
                for c in support_probe_cases(family,features))
    return dict(profiles=sorted(profiles),anchors=anchors,supports=supports)


def _reference(value):
    if (type(value) is not dict or set(value)!={'root','inventory_sha256'}
            or type(value['root']) is not str or not _digest(value['inventory_sha256'])):
        raise ValueError('Exact closed prerequisite reference schema required')
    root=Path(value['root'])
    if not root.is_absolute() or root.is_symlink() or root.resolve()!=root:
        raise ValueError('Absolute ordinary prerequisite reference required')


def _checked_inputs(value):
    if type(value) is not dict or set(value)!=_INPUTS:
        raise ValueError('Exact native registry input schema required')
    value=copy.deepcopy(value);catalog=_catalog()
    _reference(value['preparation_reference']);_reference(value['recovery_reference'])
    roots=value['prepared_roots']
    if (type(roots) is not dict or set(roots)!=set(DATASETS)
            or any(type(p) is not str or not Path(p).is_absolute() or Path(p).resolve()!=Path(p)
                or Path(p).is_symlink() for p in roots.values())):
        raise ValueError('Exact six canonical working-data root references required')
    profiles=value['profile_references']
    if type(profiles) is not dict or set(profiles)!=set(catalog['profiles']):
        raise ValueError('Exactly seven native software profiles required')
    for ref in profiles.values():_reference(ref)
    for kind,id_field in (('anchor','anchor_id'),('support','case_id')):
        rows=value[kind+'_references'];keys=('dataset','family_id',id_field)
        expected={tuple(r[k] for k in keys) for r in catalog[kind+'s']};seen=set()
        if type(rows) is not list:raise ValueError('Exact native coordinate coverage list required')
        for row in rows:
            if type(row) is not dict or set(row)!=set(keys)|{'reference'}:
                raise ValueError('Exact native coordinate reference schema required')
            if any(type(row[k]) is not str for k in keys):raise ValueError('Native coordinate strings required')
            coordinate=tuple(row[k] for k in keys)
            if coordinate not in expected or coordinate in seen:
                raise ValueError('Duplicate or unknown native coordinate coverage')
            seen.add(coordinate);_reference(row['reference'])
        if seen!=expected:raise ValueError('Incomplete mandatory native coordinate coverage')
        rows.sort(key=lambda r:tuple(r[k] for k in keys))
    return value,catalog


def _all_references(inputs):
    return [inputs['preparation_reference'],inputs['recovery_reference'],
        *inputs['profile_references'].values(),
        *(row['reference'] for kind in ('anchor','support') for row in inputs[kind+'_references'])]


def native_registry_storage_roots(*,inputs,registry_reference):
    """Metadata boundaries, not qualification; include the registry container."""
    checked,_=_checked_inputs(inputs);_reference(registry_reference)
    return dict(working_roots=sorted(set(checked['prepared_roots'].values())),
        proof_roots=sorted({registry_reference['root'],*(r['root'] for r in _all_references(checked))}),
        native_qualification_authenticated=False)


def _resources(reference):
    root,inventory=_closed_reference(reference);read,refs=_bound_reader(root,inventory)
    return _process_observation(root,read,refs)


def _generation_arguments(anchor,coordinate,prepared,manifest_sha):
    generation=anchor['generation'];native=generation['native_inventory']
    process=_read_json(Path(native['root'])/'process/process.json');argv=process.get('argv',[])
    if argv.count('--request')!=1 or argv.index('--request')+1>=len(argv):
        raise ValueError('Authenticated native generation request is missing/ambiguous')
    return dict(request_path=Path(argv[argv.index('--request')+1]),
        expected_request_sha256=generation['request_sha256'],native_reference=native,
        prepared_root=prepared,expected_manifest_sha256=manifest_sha,
        outer_fold=coordinate['primary_fold'],expected_fit_seed=17,
        expected_configuration=coordinate['configuration'])


def _prepared_datasets(inputs,preparation):
    """Hash all role bytes; access logs live outside the immutable prep proof."""
    result={}
    for dataset,schema in DATASETS.items():
        root=Path(inputs['prepared_roots'][dataset])
        if (not root.is_absolute() or root.resolve()!=root or root.is_symlink()
                or any(root.is_relative_to(Path(ref['root'])) for ref in _all_references(inputs))
                or any(root.is_relative_to((_ROOT/p).resolve()) for p in ('src','tests','configs','locks','protocol'))):
            raise ValueError('Working data must stay outside immutable proof/executable roots')
        path=root/'manifest.json';expected=preparation['datasets'][dataset]['manifest_file_sha256']
        if path.is_symlink() or _sha(path)!=expected:raise ValueError('Working manifest bytes differ from official preparation')
        manifest=_read_json(path)
        if (manifest['feature_names']!=list(schema.features)
                or manifest['categorical_features']!=list(schema.categorical)):
            raise ValueError('Working official semantic schema differs')
        for relative,ref in manifest['files'].items():
            archive=root/relative
            if (archive.is_symlink() or archive.resolve()!=archive or not archive.is_file()
                    or archive.stat().st_size!=ref['bytes'] or _sha(archive)!=ref['sha256']):
                raise ValueError('Working opaque role bytes differ from official preparation')
        result[dataset]=(root,expected)
    return result


def _inspect_anchor(coordinate,reference,prepared,manifest_sha,software,source):
    proof=load_native_anchor_execution(reference,prepared_root=prepared,expected_manifest_sha256=manifest_sha,
        expected_dataset=coordinate['dataset'],expected_family=coordinate['family_id'],expected_anchor_id=coordinate['anchor_id'])
    if proof['source_sha256']!=source or proof['software']!=software:
        raise ValueError('Anchor and registry software proofs differ')
    workload=inspect_native_workload(**_generation_arguments(proof,coordinate,prepared,manifest_sha))
    if workload['generation']!=proof['generation'] or workload['source_sha256']!=source:
        raise ValueError('Recorded workload belongs to a different valid generation')
    return dict(coordinate,reference=reference,proof_sha256=canonical_hash(proof),
        generation_sha256=canonical_hash(proof['generation']),workload_sha256=canonical_hash(workload),
        validation_sha256=canonical_hash(proof['validation']),input_rows=workload['input_rows'],
        training=workload['training'],resources=workload['process_observation'])


def _inspect(inputs,catalog):
    identity=verification_identity();source=identity['scientific_source_sha256']
    preparation=load_preparation_execution(inputs['preparation_reference'])
    if (preparation['source_sha256']!=source or set(preparation['datasets'])!=set(DATASETS)
            or preparation['raw_to_sealed_generation_chain_authenticated'] is not True):
        raise ValueError('Actual current-source official preparation proof required')
    manifests=_prepared_datasets(inputs,preparation)
    software={}
    for profile in catalog['profiles']:
        proof=load_native_implementation_verification(inputs['profile_references'][profile],expected_profile=profile)
        if proof['identity']!=identity:raise ValueError('Native profile source/test/lock identities differ')
        software[profile]=proof
    recovery=load_fullsize_node_recovery(inputs['recovery_reference'])
    if recovery['source_sha256']!=source or recovery['recovery_qualified'] is not True:
        raise ValueError('Current-source full-size recovery capability required')
    anchors=[];supports=[]
    coordinates={(r['dataset'],r['family_id'],r['anchor_id']):r for r in catalog['anchors']}
    for row in inputs['anchor_references']:
        coordinate=coordinates[(row['dataset'],row['family_id'],row['anchor_id'])]
        prepared,sha=manifests[row['dataset']]
        anchors.append(_inspect_anchor(coordinate,row['reference'],prepared,sha,software[coordinate['profile']],source))
    coordinates={(r['dataset'],r['family_id'],r['case_id']):r for r in catalog['supports']}
    for row in inputs['support_references']:
        coordinate=coordinates[(row['dataset'],row['family_id'],row['case_id'])]
        prepared,sha=manifests[row['dataset']]
        proof=load_support_probe(row['reference'],prepared_root=prepared,expected_manifest_sha256=sha,
            expected_family=row['family_id'],expected_case_id=row['case_id'])
        if (proof['source_sha256']!=source or proof['native_configuration']!=coordinate['configuration']
                or proof['outer_fold']!=coordinate['primary_fold']):
            raise ValueError('Short support source/configuration/primary role differs')
        supports.append(dict(coordinate,reference=row['reference'],proof_sha256=canonical_hash(proof),
            input_rows=proof['complete_input_rows'],resources=_resources(row['reference']),full_native_training=False))
    software_records={name:dict(proof_sha256=canonical_hash(proof),test_cases=proof['test_cases'],
        resources=_resources(inputs['profile_references'][name])) for name,proof in software.items()}
    if (verification_identity()!=identity or load_preparation_execution(inputs['preparation_reference'])!=preparation
            or _prepared_datasets(inputs,preparation)!=manifests):
        raise ValueError('Registry source or official data changed during inspection')
    for ref in _all_references(inputs):_closed_reference(ref)
    return dict(schema=_SCHEMA,status='qualified_native_prerequisites',inputs=inputs,identity=identity,
        catalog_sha256=canonical_hash(catalog),counts=dict(datasets=6,profiles=7,anchors=120,supports=258),
        preparation=dict(proof_sha256=canonical_hash(preparation),datasets=preparation['datasets'],
            resources=preparation['process_observation']),
        software=software_records,
        recovery=dict(proof_sha256=canonical_hash(recovery),resources=recovery['process_observation'],
            total_applied_updates=recovery['total_applied_updates']),anchors=anchors,supports=supports,
        failure_policy=copy.deepcopy(_FAILURE_POLICY),scope=_SCOPE,**_FLAGS)


def seal_native_qualification_registry(*,preparation_reference,profile_references,anchor_references,
        support_references,recovery_reference,prepared_roots,output):
    """Publish only after all actual prerequisites pass; never fit a model here."""
    inputs,catalog=_checked_inputs(dict(preparation_reference=preparation_reference,profile_references=profile_references,
        anchor_references=anchor_references,support_references=support_references,recovery_reference=recovery_reference,
        prepared_roots=prepared_roots))
    root=Path(output)
    protected=[Path(sys.prefix).resolve(),*(Path(p) for p in inputs['prepared_roots'].values()),
        *(Path(ref['root']) for ref in _all_references(inputs)),
        *((_ROOT/p).resolve() for p in ('src','tests','configs','locks','protocol','environment','.git','.agents','.codex','.aws'))]
    if (root.exists() or root.is_symlink()
            or any(root.resolve().is_relative_to(path) for path in protected)):
        raise ValueError('Fresh registry output must preserve working data, interpreter and executable/proof inputs')
    body=_inspect(inputs,catalog)
    root=root.resolve();root.mkdir(parents=True,exist_ok=False);write_new_json(root/'manifest.json',body)
    return dict(body,reference=dict(root=str(root),inventory_sha256=canonical_hash(_inventory(root))))


@_within_validation
def load_native_qualification_registry(reference):
    """Load real prerequisites once within this checked public invocation."""
    reference = copy.deepcopy(reference)
    def guard(value):
        from .qualified_native_trial_v2 import QualifiedNativeTrialBridge
        QualifiedNativeTrialBridge._unchanged(dict(registry=value, qualification_reference=reference))
    return _load_once('native_registry', reference,
        lambda: _load_native_qualification_registry(reference), guard=guard,
        check=lambda value: _closed_reference(reference))


def _load_native_qualification_registry(reference):
    """Reopen every prerequisite; no hand-filled authority or source migration."""
    reference=copy.deepcopy(reference);root,inventory=_closed_reference(reference)
    if {r['path'] for r in inventory['files']}!={'manifest.json'}:raise ValueError('Exact registry manifest inventory required')
    saved=_bound_json(root,inventory,'manifest.json')
    fields={'schema','status','inputs','identity','catalog_sha256','counts','preparation','software',
        'recovery','anchors','supports','failure_policy','scope',*_FLAGS}
    if (type(saved) is not dict or set(saved)!=fields or saved.get('schema')!=_SCHEMA
            or saved.get('status')!='qualified_native_prerequisites'
            or any(type(saved.get(k)) is not bool or saved[k]!=v for k,v in _FLAGS.items())
            or saved['identity']!=verification_identity()):
        raise ValueError('Registry schema/source/authority proof differs')
    inputs,catalog=_checked_inputs(saved['inputs']);actual=_inspect(inputs,catalog)
    if actual!=saved or _inventory(root)!=inventory:raise ValueError('Native registry proof changed or differs')
    return dict(actual,reference=reference)
