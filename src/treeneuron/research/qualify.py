"""Finite primary native-anchor engineering, without V scores or eligibility.

This orchestrator records native executions. Workload/range/recovery review and
the final common-source qualification remain separate Gate D prerequisites.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

from treeneuron.config import canonical_hash
from .ledger import write_new_json
from .method_worker import _ROOT,_read_json,scientific_source_sha256
from .search_supports_v2 import family_search_spec,_schema
from .workload import _checked_manifest
from .workqueue import make_queue_plan,run_queue


_DATASETS=('airfoil','concrete','ccpp','parkinsons','bike','energy')
_FAMILIES=tuple(f'P{i:02}' for i in range(14))
_FIELDS={'schema','study_id','source_sha256','locks_sha256','interpreter','datasets','families','overlays'}
_DEADLINE=7_200_000_000_000
_RSS=4*1024**3


def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def qualification_source_sha256():return scientific_source_sha256()


def qualification_locks_sha256():
    paths=[_ROOT/'uv.lock',*sorted((_ROOT/'locks/research-v2').rglob('*'))]
    files={}
    for path in paths:
        if path.is_symlink():raise ValueError('Qualification locks cannot contain symbolic links')
        if path.is_file():files[path.relative_to(_ROOT).as_posix()]=_sha(path)
    return canonical_hash(dict(files=files))


def load_qualification_config(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Ordinary qualification configuration required')
    value=_read_json(path)
    if (type(value) is not dict or set(value)!=_FIELDS or value['schema']!='research/v2-native-qualification'
            or type(value['study_id']) is not str or re.fullmatch('[A-Za-z0-9_-]{1,100}',value['study_id']) is None
            or value['source_sha256']!=qualification_source_sha256()
            or value['locks_sha256']!=qualification_locks_sha256()):
        raise ValueError('Qualification schema/source/locks differ')
    families=value['families'];datasets=value['datasets'];overlays=value['overlays']
    if (type(families) is not list or not families or any(type(f) is not str or f not in _FAMILIES for f in families)
            or families!=[f for f in _FAMILIES if f in families]
            or type(datasets) is not dict or not datasets or not set(datasets)<=set(_DATASETS)
            or type(overlays) is not dict or not set(overlays)<=set(families)):
        raise ValueError('Finite ordered families and fixed primary datasets required')
    if type(value['interpreter']) is not str:raise ValueError('Explicit qualified interpreter path required')
    interpreter=Path(value['interpreter'])
    if not interpreter.is_absolute() or not interpreter.is_file():raise ValueError('Explicit qualified interpreter required')
    for family,entries in overlays.items():
        if type(entries) is not list or any(type(entry) is not str for entry in entries) or len(entries)!=len(set(entries)):
            raise ValueError('Unique explicit adapter overlays required')
        for entry in entries:
            if type(entry) is not str:raise ValueError('Relative environment overlay path required')
            relative=Path(entry);directory=_ROOT/relative
            if (relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0]!='environment'
                    or not directory.is_dir() or not directory.resolve().is_relative_to((_ROOT/'environment').resolve())):
                raise ValueError('Overlay escaped the pinned project environment or is missing')
    for dataset,reference in datasets.items():
        if type(reference) is not dict or set(reference)!={'root','manifest_sha256'} or type(reference['root']) is not str:
            raise ValueError('Explicit prepared manifest commitment required')
        directory=Path(reference['root']);manifest=directory/'manifest.json'
        if (not directory.is_absolute() or directory.is_symlink() or manifest.is_symlink()
                or not manifest.is_file() or _sha(manifest)!=reference['manifest_sha256']):
            raise ValueError('Prepared manifest bytes/path differ')
        metadata=_read_json(manifest);_checked_manifest(dataset,metadata)
        _schema(dict(columns=metadata.get('feature_names'),categorical=metadata.get('categorical_features')))
    return value


def qualification_plan(config_path):
    config_path=Path(config_path).resolve();c=load_qualification_config(config_path)
    jobs=[]
    for family in c['families']:
        for dataset in _DATASETS:
            if dataset not in c['datasets']:continue
            reference=c['datasets'][dataset];manifest=_read_json(Path(reference['root'])/'manifest.json')
            schema=dict(columns=manifest['feature_names'],categorical=manifest['categorical_features'])
            spec=family_search_spec(family,schema)
            for anchor in spec['anchors']:
                jobs.append(dict(job_id=f'{family}/{anchor["anchor_id"]}/{dataset}/primary',
                    kind='native_anchor',depends_on=[],payload=dict(family=family,dataset=dataset,
                        outer_fold=4 if dataset=='bike' else 0,anchor_id=anchor['anchor_id'],
                        config=anchor['config'],support_sha256=spec['support_hash'])))
    identity=dict(schema='research/v2-native-qualification-identity',study_id=c['study_id'],
        config_path=str(config_path),config_sha256=_sha(config_path),config=c,
        source_sha256=c['source_sha256'],locks_sha256=c['locks_sha256'],interpreter_sha256=_sha(c['interpreter']),
        full_native_anchor_inventory_planned=set(c['families'])==set(_FAMILIES) and set(c['datasets'])==set(_DATASETS),
        fit_seed=17,per_anchor_deadline_ns=_DEADLINE,sampled_root_rss_limit_bytes=_RSS,
        resource_floor=dict(memory_proxy_bytes=11*1024**3//2,free_disk_bytes=1536*1024**2),
        quality_eligible=False,Gate_D=False,V_labels_read=False,T_accessed=False,
        purpose='Primary native-anchor execution engineering only; no HPO or predictive comparison')
    plan=make_queue_plan(identity,jobs)
    if load_qualification_config(config_path)!=c:raise ValueError('Qualification inputs changed during planning')
    return plan


def _resources(output):
    from .opportunity_execution import live_resources
    return live_resources(output)


def _execute_anchor(job,directory,identity):
    from .transfer import TransferStore
    from .trial_bridge_v2 import EngineeringTrialBridge
    from .search_controller_v2 import make_search_plan,run_search
    started=time.perf_counter_ns();case=job['payload'];c=identity['config']
    family=case['family'];dataset=case['dataset'];fold=case['outer_fold']
    store=TransferStore(c['datasets'][dataset]['root']);view=store.development(fold)
    input_root=directory/'input';input_root.mkdir();files={};material=[]
    for role in ('F','G','V_features'):
        relative=f'outer-{fold}/'+('V-features.npz' if role=='V_features' else role+'.npz')
        original=store.root/relative;expected=store.manifest['files'][relative]['sha256']
        if original.is_symlink() or _sha(original)!=expected:raise ValueError('Qualification role archive changed')
        target=input_root/(role+'.npz')
        try:os.link(original,target);mode='hard_link'
        except OSError as error:
            import errno
            if error.errno!=errno.EXDEV:raise
            shutil.copyfile(original,target);mode='cross_device_copy'
        if _sha(target)!=expected:raise ValueError('Qualification staged role changed')
        files[role]=dict(path=target.name,sha256=expected);material.append(dict(role=role,mode=mode,sha256=expected))
    write_new_json(input_root/'materialization.json',dict(files=material))
    request=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',
        job_id='qualification/'+job['job_id'],family_id=family,fit_seed=17,scientific_source_sha256=c['source_sha256'],
        config=case['config'],budget=dict(max_seconds=7200.),files=files,
        observation_identity=dict(dataset=dataset,split_kind='temporal' if dataset=='bike' else 'grouped',split_sha256=view.split_hash))
    path=input_root/'request.json';write_new_json(path,request)
    trial_identity=dict(job_id=request['job_id'],family_id=family,search_repeat=0,source_sha256=c['source_sha256'],
        runtime_sha256=canonical_hash(dict(interpreter=identity['interpreter_sha256'],locks=c['locks_sha256'])),
        support_sha256=case['support_sha256'],split_sha256=view.split_hash)
    plan=make_search_plan(identity=trial_identity,fit_seed=17,hpo_seed32=73,anchors=[case['config']],mode='finite',
        support=None,candidates=[],max_attempts=1,deadline_ns=_DEADLINE,prefixes_ns=[])
    bridge=EngineeringTrialBridge(plan=plan,template_path=path,expected_template_sha256=_sha(path),
        search_directory=directory/'search',interpreter=c['interpreter'],
        extra_pythonpath=[_ROOT/entry for entry in c['overlays'].get(family,[])],rss_limit_bytes=_RSS)
    result=run_search(directory/'search',plan=plan,evaluate_trial=bridge,invocation_preparation_ns=time.perf_counter_ns()-started)
    write_new_json(directory/'search-result.json',result)
    status=result['records'][0]['status'] if result['records'] else 'timeout'
    return dict(status=status,family=family,dataset=dataset,outer_fold=fold,anchor_id=case['anchor_id'],
        config=case['config'],source_sha256=c['source_sha256'],manifest_sha256=c['datasets'][dataset]['manifest_sha256'],
        paid_search_ns=result['total_charged_ns'],quality_eligible=False,Gate_D=False,V_labels_read=False,T_accessed=False)


def qualify(config_path,output,*,resume=False):
    started=time.perf_counter_ns();config_path=Path(config_path).resolve();output=Path(output).resolve()
    plan=qualification_plan(config_path);identity=plan['identity'];c=identity['config']
    if type(resume) is not bool:raise ValueError('Explicit boolean qualification resume required')
    if any(output.is_relative_to(Path(ref['root']).resolve()) for ref in c['datasets'].values()):
        raise ValueError('Qualification output must stay outside sealed datasets')
    from treeneuron.protocol import _execution
    if output.is_relative_to(_ROOT) and _execution(output.relative_to(_ROOT).as_posix()):
        raise ValueError('Qualification output cannot change executable scientific inputs')
    if resume:
        if output.is_symlink() or not (output/'queue/plan.json').is_file():raise ValueError('Existing exact qualification queue required')
    else:output.mkdir(parents=True,exist_ok=False)
    index=len(list(output.glob('invocation-*.json')))
    write_new_json(output/f'invocation-{index}.json',dict(argv=sys.argv,cwd=str(Path.cwd()),
        config_path=str(config_path),config_sha256=identity['config_sha256'],plan_hash=plan['plan_hash'],
        source_sha256=c['source_sha256'],started_utc_ns=time.time_ns(),resume=resume,T_accessed=False))
    def unchanged(value):
        if (value!=identity or _sha(config_path)!=identity['config_sha256']
                or load_qualification_config(config_path)!=c or _sha(c['interpreter'])!=identity['interpreter_sha256']):
            raise ValueError('Qualification identity changed')
    observations=0
    def before_start(job,ledger):
        nonlocal observations
        observed=_resources(output)
        memory=observed['available_memory_bytes'];disk=observed['free_disk_bytes']
        decision=dict(admitted=memory is not None and memory>=11*1024**3//2 and disk>=1536*1024**2,
            observed=observed,required_memory_proxy_bytes=11*1024**3//2,required_free_disk_bytes=1536*1024**2,
            sampled_root_rss_limit_bytes=_RSS,job_id=job['job_id'],utc_ns=time.time_ns())
        write_new_json(output/f'admission-{index}-{observations}.json',decision);observations+=1
        return decision
    try:
        result=run_queue(plan,output/'queue',lambda job,directory,dependencies,recovery_attempt:
            _execute_anchor(job,directory,identity),check_identity=unchanged,resume=resume,before_start=before_start)
        result|=dict(quality_eligible=False,Gate_D=False,V_labels_read=False,T_accessed=False,
            native_anchor_executions_complete=result['status']=='completed',
            full_native_anchor_inventory_planned=identity['full_native_anchor_inventory_planned'],
            source_sha256=c['source_sha256'],operator_body_elapsed_ns=time.perf_counter_ns()-started,
            clock_scope='Operator body includes queue; do not add nested queue/search/process clocks; fresh CLI startup needs outer receipt')
        write_new_json(output/f'terminal-{index}.json',result)
        return result
    except BaseException as error:
        write_new_json(output/f'error-{index}.json',dict(type=type(error).__name__,message=str(error),
            elapsed_ns=time.perf_counter_ns()-started,T_accessed=False))
        raise
