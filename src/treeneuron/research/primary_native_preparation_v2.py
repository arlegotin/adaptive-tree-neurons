"""Prepare an exact primary callback; whole-job/protocol admission is separate."""
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
import zipfile

from .immutable_inputs import stage_immutable_file
from .ledger import write_new_json
from .method_worker import _ROOT, _sha, scientific_source_sha256
from .native_anchor_v2 import load_native_anchor_execution
from .native_qualification_registry_v2 import load_native_qualification_registry,native_registry_storage_roots
from .primary_search_declaration_v2 import compile_primary_search_declaration
from .qualified_native_trial_v2 import QualifiedNativeTrialBridge, _bind_procedure
from .search_controller_v2 import SearchPlan, _checked_plan
from .search_supports_v2 import family_search_spec
from .transfer import TransferStore
from .workload import _checked_manifest


def stage_primary_native_template(*,plan,prepared_root,expected_manifest_sha256,outer_fold,output):
    """Stage three immutable role archives without decoding arrays or labels.

    The caller's hashes are checked, not authenticated as official acquisition.
    Actual registry qualification and worker content validation remain required.
    """
    started=time.perf_counter_ns();value=_checked_plan(plan);prepared=Path(prepared_root);output=Path(output)
    if (not prepared.is_absolute() or prepared.resolve()!=prepared or prepared.is_symlink()
            or not output.is_absolute() or output.resolve()!=output or output.exists() or output.is_symlink()
            or output.is_relative_to(prepared) or prepared.is_relative_to(output)
            or any(output.is_relative_to((_ROOT/p).resolve()) for p in ('src','tests','configs','locks','protocol'))):
        raise ValueError('Fresh canonical template outside prepared/executable roots required')
    manifest=prepared/'manifest.json'
    if manifest.is_symlink() or _sha(manifest)!=expected_manifest_sha256:raise ValueError('Prepared manifest commitment differs')
    store=TransferStore(prepared);_checked_manifest(store.manifest['dataset'],store.manifest)
    if type(outer_fold) is not int or outer_fold!=(4 if store.manifest['dataset']=='bike' else 0):
        raise ValueError('Primary template cannot read secondary development roles before freeze')
    split=store._split(outer_fold);identity=value['identity'];source=scientific_source_sha256()
    core=_ROOT/'locks/research-v2/core.json';spec=family_search_spec(identity['family_id'],
        dict(columns=store.manifest['feature_names'],categorical=store.manifest['categorical_features']))
    if (value['deadline_ns'] is not None or identity['source_sha256']!=source
            or identity['runtime_sha256']!=_sha(core) or identity['split_sha256']!=split['split_hash']
            or identity['support_sha256']!=spec['support_hash'] or value['anchors']!=[a['config'] for a in spec['anchors']]):
        raise ValueError('Primary template source/runtime/support/split/native anchor differs')
    originals={};files={}
    for role in ('F','G','V_features'):
        relative=f'outer-{outer_fold}/'+('V-features.npz' if role=='V_features' else role+'.npz')
        path=prepared/relative;ref=store.manifest['files'][relative]
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=ref['bytes'] or _sha(path)!=ref['sha256']:
            raise ValueError('Primary role archive bytes differ')
        # Inspect names only: a query label must never be copied to a fit worker.
        try:
            with zipfile.ZipFile(path) as archive:names=archive.namelist()
        except zipfile.BadZipFile as error:raise ValueError('Invalid primary role archive') from error
        required={'X.npy','row_id.npy','group_id.npy'}|({'y.npy'} if role!='V_features' else set())
        allowed=required|({'timestamps.npy'} if store.manifest['dataset']=='bike' else set())
        if len(names)!=len(set(names)) or set(names)!=allowed:
            raise ValueError('Primary role/query archive has missing fields or forbidden labels')
        originals[role]=(path,ref['sha256']);files[role]=dict(path=role+'.npz',sha256=ref['sha256'])
    output.mkdir(parents=True,exist_ok=False);material=[]
    for role,(path,digest) in originals.items():
        material.append(dict(stage_immutable_file(path,output/(role+'.npz'),expected_sha256=digest),role=role))
    request=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',
        job_id=identity['job_id'],family_id=identity['family_id'],fit_seed=value['fit_seed'],scientific_source_sha256=source,
        config=copy.deepcopy(spec['anchors'][0]['config']),budget={'native_stopping':True},files=files,
        observation_identity=dict(dataset=store.manifest['dataset'],split_kind='temporal' if store.manifest['dataset']=='bike' else 'grouped',
            split_sha256=split['split_hash']))
    path=output/'request.json';write_new_json(path,request)
    write_new_json(output/'materialization.json',dict(files=material,arrays_or_labels_decoded=False,
        native_workload_qualified=False,quality_eligible=False,T_accessed=False))
    if (any(_sha(original)!=digest or _sha(output/(role+'.npz'))!=digest for role,(original,digest) in originals.items())
            or _sha(manifest)!=expected_manifest_sha256 or scientific_source_sha256()!=source
            or _sha(core)!=identity['runtime_sha256']):
        raise ValueError('Primary template source/data/core changed during staging; partial output retained')
    return dict(schema='research/v2-staged-primary-native-template',request_path=str(path),request_sha256=_sha(path),
        source_sha256=source,plan_hash=value['plan_hash'],staged_roles=['F','G','V_features'],
        arrays_or_labels_decoded=False,raw_acquisition_authenticated=False,native_workload_qualified=False,
        quality_eligible=False,T_accessed=False,elapsed_before_return_ns=time.perf_counter_ns()-started,
        scope='Byte-bound primary metadata/archive-name/staging component only. Official acquisition, array contents/role disjointness, native worker validation, registry and protocol admission remain separate.')


@dataclass(frozen=True)
class PreparedPrimaryNativeCallback:
    plan:SearchPlan
    callback:QualifiedNativeTrialBridge
    manifest_path:Path
    manifest_sha256:str
    preparation_before_return_ns:int
    native_sampler_minimum_prerequisite_ns:int


def _check_preparation_locations(*,output,search,working_roots,proof_roots,rss_limit_bytes):
    protected=[*(Path(p) for p in working_roots),*(Path(p) for p in proof_roots),
        Path(sys.prefix).resolve(),
        *((_ROOT/p).resolve() for p in ('src','tests','configs','locks','protocol','environment','.git','.agents','.codex','.aws'))]
    if (not output.is_absolute() or output.resolve()!=output or output.exists() or output.is_symlink()
            or not search.is_absolute() or search.resolve()!=search or search.is_symlink()
            or output.is_relative_to(search) or search.is_relative_to(output)
            or any(output.is_relative_to(p) or search.is_relative_to(p) for p in protected)
            or type(rss_limit_bytes) is not int or not 0<rss_limit_bytes<=16*1024**3):
        raise ValueError('Preparation/search must preserve all working/proof/executable roots and the declared16GiB worker ceiling')


def prepare_primary_native_callback(*,practical_plan,expected_practical_plan_hash,job_id,support_reference,
        qualification_reference,output,search_directory,official_schedule_reference=None,rss_limit_bytes=4*1024**3):
    """Require actual qualification before staging any fit inputs.

    The actual native anchor supplies method-specific interpreter/import paths.
    This prepares a callback; it does not execute a search, qualify whole external
    cost, freeze the global procedure or authorize protected/secondary access.
    """
    started=time.perf_counter_ns();registry=load_native_qualification_registry(qualification_reference)
    declaration=compile_primary_search_declaration(practical_plan=practical_plan,
        expected_practical_plan_hash=expected_practical_plan_hash,job_id=job_id,support_reference=support_reference,
        official_schedule_reference=official_schedule_reference)
    from treeneuron.config import canonical_json
    plan=SearchPlan(canonical_json(declaration['search_plan']));job=declaration['job'];dataset=job['dataset']
    prepared=Path(registry['inputs']['prepared_roots'][dataset]);manifest=registry['preparation']['datasets'][dataset]['manifest_file_sha256']
    _bind_procedure(registry,dict(plan=plan,prepared_root=prepared,expected_manifest_sha256=manifest,outer_fold=job['outer_fold']))
    output=Path(output);search=Path(search_directory)
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=qualification_reference)
    _check_preparation_locations(output=output,search=search,working_roots=roots['working_roots'],
        proof_roots=roots['proof_roots'],rss_limit_bytes=rss_limit_bytes)
    spec=family_search_spec(job['family_id'],dict(columns=TransferStore(prepared).manifest['feature_names'],
        categorical=TransferStore(prepared).manifest['categorical_features']))
    anchor=next(row for row in registry['anchors'] if row['dataset']==dataset and row['family_id']==job['family_id']
        and row['anchor_id']==spec['anchors'][0]['anchor_id'])
    proof=load_native_anchor_execution(anchor['reference'],prepared_root=prepared,expected_manifest_sha256=manifest,
        expected_family=job['family_id'],expected_anchor_id=anchor['anchor_id'],expected_dataset=dataset)
    import json
    process=json.loads((Path(proof['generation']['native_inventory']['root'])/'process/process.json').read_text())
    interpreter=process['argv'][0];paths=process['environment_overrides']['PYTHONPATH'].split(os.pathsep)
    if (Path(interpreter).resolve()!=Path(sys.executable).resolve() or not paths or paths[0]!=str(_ROOT/'src')
            or any(not Path(p).is_absolute() or not Path(p).is_dir()
                or not Path(p).resolve().is_relative_to((_ROOT/'environment').resolve()) for p in paths[1:])):
        raise ValueError('Actual qualified native interpreter/import overlays differ')
    staged=stage_primary_native_template(plan=plan,prepared_root=prepared,expected_manifest_sha256=manifest,
        outer_fold=job['outer_fold'],output=output/'template')
    callback=QualifiedNativeTrialBridge(qualification_reference=qualification_reference,plan=plan,
        template_path=Path(staged['request_path']),expected_template_sha256=staged['request_sha256'],
        prepared_root=prepared,expected_manifest_sha256=manifest,outer_fold=job['outer_fold'],search_directory=search,
        interpreter=interpreter,extra_pythonpath=paths[1:],rss_limit_bytes=rss_limit_bytes,
        official_schedule_reference=official_schedule_reference,native_hpo_seed=declaration['native_hpo_seed'])
    again=compile_primary_search_declaration(practical_plan=practical_plan,expected_practical_plan_hash=expected_practical_plan_hash,
        job_id=job_id,support_reference=support_reference,official_schedule_reference=official_schedule_reference)
    if again!=declaration:raise ValueError('Primary declaration changed during callback preparation')
    record=dict(schema='research/v2-prepared-primary-native-callback',declaration=declaration,
        qualification_reference=copy.deepcopy(qualification_reference),template=staged,search_directory=str(search),
        interpreter=interpreter,interpreter_sha256=_sha(Path(interpreter)),extra_pythonpath=paths[1:],
        rss_limit_bytes=rss_limit_bytes,actual_anchor_reference=anchor['reference'],
        native_sampler_minimum_prerequisite_ns=callback.native_sampler_minimum_prerequisite_ns,
        elapsed_before_publication_ns=time.perf_counter_ns()-started,search_executed=False,
        whole_process_cost_authenticated=False,production_execution_authorized=False,protected_access_authorized=False,T_accessed=False,
        clock_scope='Factory entry through final preparation/publication; caller startup and any native sampler run before entry need separately measured paid preparation before first prefix. Nested callback preparation is included, never added twice.')
    target=output/'factory.json';write_new_json(target,record)
    return PreparedPrimaryNativeCallback(plan,callback,target,_sha(target),time.perf_counter_ns()-started,
        callback.native_sampler_minimum_prerequisite_ns)
