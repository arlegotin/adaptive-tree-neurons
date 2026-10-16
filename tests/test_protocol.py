"""Simulated, temporary protocol fixtures; never scientific evidence."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from treeneuron.config import canonical_hash, resolve_experiment


def test_scientific_purpose_requires_readiness_without_changing_legacy_identity():
    legacy = resolve_experiment({'study_id': 'old'}).to_dict()
    assert 'pilot' not in legacy
    with pytest.raises(ValueError, match='readiness'):
        resolve_experiment({'study_id': 's', 'pilot': {'purpose': 'scientific'}})
    config = resolve_experiment({'study_id': 's', 'pilot': {
        'purpose': 'scientific', 'readiness_path': 'readiness.json'}})
    assert config.pilot.purpose == 'scientific'
    assert 'readiness_path' not in resolve_experiment({'study_id': 'e', 'pilot': {}}).to_dict()['pilot']


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def write(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value))
    return target


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / 'repo'; root.mkdir()
    git(root, 'init', '-q'); git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'user.name', 'Simulated fixture')
    for name in ('src/model.py', 'configs/scientific.yaml', 'analysis/infer.py', 'requirements.lock'):
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# fixture\n')
    git(root, 'add', '.'); git(root, 'commit', '-qm', 'simulated execution')
    return root


def test_execution_identity_allows_docs_commits_but_rejects_scientific_changes(repository):
    from treeneuron.protocol import execution_snapshot
    root = repository; pinned = git(root, 'rev-parse', 'HEAD')
    initial = execution_snapshot(root, pinned)
    write(root, 'docs/note.json', {'text': 'documentation only'})
    git(root, 'add', '.'); git(root, 'commit', '-qm', 'docs only')
    current = execution_snapshot(root, pinned)
    assert current['execution_sha256'] == initial['execution_sha256']
    assert current['checkout_commit'] != current['pinned_commit']
    (root / 'src/model.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='[Dd]irty|drift'):
        execution_snapshot(root, pinned)


@pytest.mark.parametrize('name', ['src/override.py', 'configs/new.yaml', 'analysis/new.py', 'plugin.py'])
def test_untracked_executable_or_scientific_input_is_rejected(repository, name):
    from treeneuron.protocol import execution_snapshot
    path = repository / name; path.parent.mkdir(exist_ok=True, parents=True); path.write_text('x=1\n')
    with pytest.raises(ValueError, match='[Uu]ntracked'):
        execution_snapshot(repository, git(repository, 'rev-parse', 'HEAD'))


def test_historical_markdown_has_no_implicit_numeric_authority():
    from treeneuron.protocol import load_design
    with pytest.raises(ValueError, match='companion'):
        load_design(Path('pilot/design_decision.md'))


def ref(root, path):
    import hashlib
    return {'path': str(Path(path).relative_to(root)), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


@pytest.fixture
def prepared(repository, monkeypatch):
    """Simulated committed reviews and sealed ledgers, never real WP05 evidence."""
    from treeneuron import protocol
    from treeneuron import train
    from treeneuron.data.types import array_hash
    import numpy as np
    from test_design import constraints
    root = repository; pinned = git(root, 'rev-parse', 'HEAD')
    snapshot = protocol.execution_snapshot(root, pinned)
    reviews = {}
    for task in ('9', '10', '13'):
        artifact=write(root,f'audit/findings-{task}.json',{'review':'simulated fixture findings'})
        review = write(root, f'audit/review-{task}.json', dict(task=task, verdict='accepted',
            implementation_commit=pinned, execution_sha256=snapshot['execution_sha256'],files=snapshot['files'],artifacts=[ref(root,artifact)]))
        reviews[task] = ref(root, review)
    git(root, 'add', '.'); git(root, 'commit', '-qm', 'simulated accepted reviews')
    config = resolve_experiment(dict(study_id='toy', data={'n_fit':2,'n_growth':1,'n_validation':1}, execution={'backend_report':'evidence/runtime.json'}, backend={'cpu_threads': 1, 'interop_threads': 1},
        training=dict(base_updates=10, continuation_updates=10, patience=5, validation_interval=1, batch_size=8),
        pilot=dict(purpose='scientific', readiness_path='evidence/readiness.json',
            initial_teachers=8, max_teachers=32, evaluation_rows=5, max_seconds=10000., bootstrap_iterations=2000, endpoint_tolerance=.1,
            constraints_path='evidence/constraints.json')))
    c = constraints(); c['inference_supported'] = False
    write(root, 'evidence/constraints.json', c)
    receipt = write(root, 'evidence/runtime.json', {'simulated': True})
    # Only numerical hardware receipt verification is replaced: no numerical work here.
    monkeypatch.setattr(protocol, '_validate_runtime', lambda config, value: None)
    sample=write(root,'evidence/resource-sample.json',dict(status='completed',teacher_count=1,
        enclosing_seconds=1.,retained_bytes=1000,peak_memory_bytes=1000,recipe_sha256=protocol.recipe_hash(config),
        runtime_sha256=ref(root,receipt)['sha256'],source_sha256=snapshot['execution_sha256']))
    measurement = write(root, 'evidence/resources.json', dict(samples=[ref(root,sample)],
        measured_seconds_per_teacher=1., measured_bytes_per_teacher=1000,
        peak_memory_bytes=1000, machine_memory_bytes=100000, reserved_memory_bytes=10000,
        free_disk_bytes=10000000, max_seconds=10000., max_storage_bytes=1000000,
        retries=1, failure_policy='retain_and_block', source_sha256=snapshot['execution_sha256']))
    readiness = dict(schema_version=1, pinned_commit=pinned,
        execution_sha256=snapshot['execution_sha256'], reviews=reviews,
        config_sha256=canonical_hash(config.to_dict()), runtime=ref(root, receipt),
        measurement=ref(root, measurement), constraints=ref(root, root/'evidence/constraints.json'),
        seed_plan={'namespace': 'exploratory/pilot', 'seeds': config.to_dict()['seeds'],
            'max_teachers': 32, 'repetitions': 2})
    ready_path = write(root, 'evidence/readiness.json', readiness)
    # Uniformly spread teacher effects give a stable SD bootstrap; mean zero -> estimation.
    records=[]; attempts=[]
    for i in range(32):
        teacher_dir=root/f'evidence/pilot/teacher-{i}'; teacher_dir.mkdir(parents=True)
        E_X=np.zeros((5,config.model.d)); E_y=np.zeros(5); E_rows=np.arange(5,dtype=np.int64)
        np.savez(teacher_dir/'observations.npz',F_y=np.array([0.,2.]),E_X=E_X,E_y=E_y,E_rows=E_rows)
        evaluation_hash=canonical_hash(dict(X=array_hash(E_X),y=array_hash(E_y),rows=array_hash(E_rows)))
        for j in range(2):
            row=dict(teacher_id=f't{i}', repetition=j, data_hash=f'd{i}', evaluation_hash=evaluation_hash,
                recipe_hash=canonical_hash(config.to_dict()), h1_rows=[(i-15.5)*.02+j*.001+k*.001 for k in range(5)],
                static_rows=[0.]*5, controls=['B1','B2','B3','B4','B5','B6'], status='completed',
                source_sha256=train.source_hash(), enclosing_ns=100000, retained_bytes=100,
                V_accessed=False, T_accessed=False, child_runs=[],
                methods={k:{'method_id':v,'status':'completed'} for k,v in dict(keep='B1',top='B5',random='B4',
                    greedy='B6',widen_upper='B2',ordinary_lower='B2',ordinary_upper='B2',static_partial='B3').items()})
            prediction_path=teacher_dir/f'repetition-{j}/E-predictions.npz'
            prediction_path.parent.mkdir(parents=True)
            h=np.array(row['h1_rows']); predictions={name:np.zeros(5) for name in protocol.EVALUATION_METHODS}
            predictions.update(top=np.sqrt(np.maximum(-h,0)),random=np.sqrt(np.maximum(h,0)))
            np.savez(prediction_path,**predictions,y=E_y,row_id=E_rows)
            row['h1_rows']=(predictions['random']**2-predictions['top']**2).tolist()
            row['E_predictions_sha256']=ref(root,prediction_path)['sha256']
            simulated_pair_graph(prediction_path.parent,row,config)
            write(root, f'evidence/pilot/teacher-{i}/repetition-{j}/pair.json', row)
            records.append(row); attempts.append(dict(teacher_index=i,repetition=j,status='completed',record=row))
    raw=dict(schema_version=1,purpose='scientific',status='completed',config=config.to_dict(),
        provenance={'source_sha256':train.source_hash()},readiness=ref(root,ready_path),
        constraints=c,accepted_records=records,attempts=attempts,failed_pairs=0,
        batch_runtimes=[1.,1.,1.],batch_completed_counts=[8,16,32],enclosing_ns=6400000,retained_bytes=6400)
    raw_path=write(root,'evidence/pilot/raw.json',raw)
    seal_path=write(root,'evidence/pilot/raw-seal.json',protocol.pilot_seal(raw_path.parent))
    scientific=dict(hypothesis='H1-A',contrast=['B4','B5'],population='primary',coordinate='selective',
        independent_unit='teacher',exclusions=[],metric='squared_error_over_F_variance',
        grouping='row',preprocessing='F_only',refit=False,evaluation_methods=list(protocol.EVALUATION_METHODS),analysis=ref(root,root/'analysis/infer.py'))
    sources=dict(root=str(root),pinned_commit=pinned,readiness=ref(root,ready_path),
        raw=ref(root,raw_path),raw_seal=ref(root,seal_path),scientific=scientific,cohort=None)
    return root, config, sources


def test_pre_pilot_readiness_does_not_require_future_pilot_or_K(prepared):
    from treeneuron.protocol import validate_pilot_readiness
    root, config, sources=prepared
    import shutil
    shutil.rmtree(root/'evidence/pilot')
    ready=validate_pilot_readiness(config,root/sources['readiness']['path'],root=root)
    assert ready['forecast']['teachers']==32
    assert ready['forecast']['seconds']==64
    assert 'K' not in ready['forecast']


@pytest.mark.parametrize('change',['review','runtime','config','resource','seed','unknown'])
def test_pre_pilot_rejects_missing_or_drifting_evidence(prepared,change):
    from treeneuron.protocol import validate_pilot_readiness
    root,config,sources=prepared; path=root/sources['readiness']['path']
    value=json.loads(path.read_text())
    if change=='review': value['reviews'].pop('13')
    elif change=='runtime': (root/value['runtime']['path']).write_text('{}')
    elif change=='config': value['config_sha256']='0'*64
    elif change=='resource': (root/value['measurement']['path']).write_text('{}')
    elif change=='seed': value['seed_plan']['namespace']='confirmatory/pilot'
    else: value['approved']=True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): validate_pilot_readiness(config,path,root=root)


def test_post_pilot_recomputes_design_and_rejects_hand_filled_or_corrupt_ledgers(prepared):
    from treeneuron.protocol import eligible_design
    root,config,sources=prepared
    decision=eligible_design(sources)
    assert decision.branch=='estimation' and decision.K==32 and decision.n_T>=1
    pair=root/'evidence/pilot/teacher-0/repetition-0/pair.json'
    pair.write_text('{}')
    with pytest.raises(ValueError,match='pair|Input hash'): eligible_design(sources)


def test_canonical_companion_binds_exact_markdown_and_current_sources(prepared):
    from treeneuron.protocol import eligible_design,publish_design,load_design
    root,config,sources=prepared
    decision=eligible_design(sources)
    md=root/'published/design_decision.md'
    publish_design(decision,sources,md)
    got,_=load_design(md)
    assert got.to_dict()==decision.to_dict()
    md.write_text(md.read_text()+'hand edited\n')
    with pytest.raises(ValueError,match='Markdown'): load_design(md)


@pytest.fixture
def candidate(prepared):
    from dataclasses import replace
    from treeneuron.protocol import eligible_design, confirmation_seeds
    from treeneuron.evaluation.protected import test_manifest as make_manifest
    from treeneuron.data.synthetic import make_teacher
    from treeneuron.data.types import SplitManifest,array_hash
    import numpy as np
    root,config,sources=prepared
    decision=eligible_design(sources)
    base=make_teacher(999,config.model.d)
    units=[]; manifests=[]
    for i in range(decision.K):
        seeds=confirmation_seeds('toy','cohort-a',123,i,decision.repetitions)
        # Simulated teacher specifications: these are not a generated scientific cohort.
        teacher=replace(base,seed=seeds['teacher'],reference_seed=seeds['reference'])
        teacher_path=write(root,f'evidence/confirmation/teacher-{i}.json',teacher.to_dict())
        fit=np.array([0.,2.])
        fit_path=root/f'evidence/confirmation/F-{i}.npy'; np.save(fit_path,fit)
        dev=SplitManifest(f'dev-{i}',(1,2,3,4),(1,2,3,4),('F','F','G','V'),
            teacher_id=teacher.teacher_id,hashes=(('F.y',array_hash(fit)),))
        manifest=make_manifest(dev,namespace='confirmatory/toy/cohort-a/T',seed=seeds['test'],n_test=decision.n_T)
        units.append(dict(unit_id=f'unit-{i}',seeds=seeds,teacher=ref(root,teacher_path),fit_targets=ref(root,fit_path),
            manifest=manifest.to_dict(),run_ids=[f'unit-{i}-rep-{j}' for j in range(decision.repetitions)]))
        manifests.append(manifest)
    sources['cohort']=dict(study_id='toy',cohort_id='cohort-a',root_seed=123,units=units)
    return root,decision,sources,manifests


def test_freeze_preserves_identity_across_document_and_protocol_commit(candidate):
    from treeneuron.protocol import freeze,validate_frozen
    root,decision,sources,manifests=candidate
    path=root/'protocol/frozen.yaml'
    frozen=freeze(decision,sources,path)
    git(root,'add','protocol'); git(root,'commit','-qm','frozen artifact only')
    checked=validate_frozen(path,{'root':str(root)})
    assert checked.protocol_hash==frozen.protocol_hash
    assert checked.checkout_commit!=checked.pinned_commit
    with pytest.raises((ValueError,FileExistsError)): freeze(decision,sources,root/'protocol/second.yaml')


@pytest.mark.parametrize('change',['draft','number','source','lock','config','analysis','split','transfer','namespace'])
def test_freeze_rejections_leave_no_protocol_or_access(candidate,change):
    from dataclasses import replace
    from treeneuron.protocol import freeze
    root,decision,sources,manifests=candidate
    if change=='draft': decision=replace(decision,branch='blocked')
    elif change=='number': decision=replace(decision,n_T=decision.n_T+1)
    elif change in ('source','lock','config','analysis'):
        path={'source':'src/model.py','lock':'requirements.lock','config':'configs/scientific.yaml','analysis':'analysis/infer.py'}[change]
        (root/path).write_text('changed')
    elif change=='split': sources['cohort']['units'][0]['manifest']['assignment_hash']='0'*64
    elif change=='transfer': sources['scientific']['hypothesis']='H3'
    else: sources['cohort']['cohort_id']='exploratory'
    with pytest.raises(ValueError): freeze(decision,sources,root/'protocol/frozen.yaml')
    assert not (root/'protocol/frozen.yaml').exists()
    assert not list(root.rglob('predictions.json'))
    assert not list(root.rglob('access.jsonl'))


def test_cli_freeze_requires_verified_companion_and_rejects_legacy_report(tmp_path):
    from treeneuron.cli import main
    with pytest.raises(ValueError,match='companion'):
        main(['freeze','--design','pilot/design_decision.md','--output',str(tmp_path/'frozen.yaml')])
    assert not (tmp_path/'frozen.yaml').exists()


def test_scientific_calibrate_checks_readiness_before_runtime_or_attempt(tmp_path,monkeypatch):
    from unittest.mock import Mock
    from treeneuron.pilot.calibrate import calibrate
    import importlib
    module=importlib.import_module('treeneuron.pilot.calibrate')
    trap=Mock(side_effect=AssertionError('runtime before readiness'))
    monkeypatch.setattr(module,'_runtime_preflight',trap)
    config=resolve_experiment({'study_id':'blocked','execution':{'artifacts_root':str(tmp_path/'runs')},
        'pilot':{'purpose':'scientific','readiness_path':str(tmp_path/'absent.json')}})
    with pytest.raises(ValueError,match='evidence|readiness'):
        calibrate(config)
    trap.assert_not_called()
    assert not (tmp_path/'runs').exists()


def test_review_commit_can_precede_later_unrelated_reviewed_implementation(prepared):
    from treeneuron.protocol import validate_pilot_readiness,execution_snapshot
    root,config,sources=prepared
    # Each task's review commits exact affected files; later task source is allowed
    # only via a newly bound readiness identity, never by changing reviewed files.
    path=root/'evidence/readiness.json'; ready=json.loads(path.read_text())
    for review_ref in ready['reviews'].values():
        rp=root/review_ref['path']; review=json.loads(rp.read_text())
        review['files']={'src/model.py':execution_snapshot(root,review['implementation_commit'])['files']['src/model.py']}
        rp.write_text(json.dumps(review)); review_ref['sha256']=ref(root,rp)['sha256']
    write(root,'src/new_task.py',{'implemented':'later'})
    git(root,'add','.'); git(root,'commit','-qm','later independently reviewed implementation')
    pinned=git(root,'rev-parse','HEAD'); snapshot=execution_snapshot(root,pinned)
    ready['pinned_commit']=pinned; ready['execution_sha256']=snapshot['execution_sha256']
    measurement=root/ready['measurement']['path']; measured=json.loads(measurement.read_text())
    measured['source_sha256']=snapshot['execution_sha256']
    sample=root/measured['samples'][0]['path']; sample_value=json.loads(sample.read_text())
    sample_value['source_sha256']=snapshot['execution_sha256']; sample.write_text(json.dumps(sample_value))
    measured['samples']=[ref(root,sample)]; measurement.write_text(json.dumps(measured))
    ready['measurement']=ref(root,measurement); path.write_text(json.dumps(ready))
    assert validate_pilot_readiness(config,path,root=root)['snapshot']['pinned_commit']==pinned


def test_post_pilot_control_claims_must_match_completed_method_ledgers(prepared):
    from treeneuron.protocol import eligible_design
    root,config,sources=prepared
    raw_path=root/sources['raw']['path']; raw=json.loads(raw_path.read_text())
    # A label on controls is insufficient when B6's method has no completed run.
    record=raw['accepted_records'][0]; record['methods']={}
    raw['attempts'][0]['record']=record
    write(root,'evidence/pilot/teacher-0/repetition-0/pair.json',record)
    raw_path.write_text(json.dumps(raw)); sources['raw']=ref(root,raw_path)
    from treeneuron.protocol import pilot_seal
    seal=root/'evidence/pilot/raw-seal.json'; seal.write_text(json.dumps(pilot_seal(raw_path.parent)))
    sources['raw_seal']=ref(root,seal)
    with pytest.raises(ValueError,match='control|method'): eligible_design(sources)


def test_pre_pilot_requires_raw_measured_resource_support(prepared):
    from treeneuron.protocol import validate_pilot_readiness
    root,config,sources=prepared
    path=root/sources['readiness']['path']; ready=json.loads(path.read_text())
    measured_path=root/ready['measurement']['path']; measured=json.loads(measured_path.read_text())
    measured.pop('samples',None)
    measured_path.write_text(json.dumps(measured)); ready['measurement']=ref(root,measured_path)
    path.write_text(json.dumps(ready))
    with pytest.raises(ValueError,match='sample|measured'): validate_pilot_readiness(config,path,root=root)


def test_post_pilot_rejects_incomplete_batch_identity(prepared):
    from treeneuron.protocol import eligible_design
    root,config,sources=prepared
    raw_path=root/sources['raw']['path']; raw=json.loads(raw_path.read_text())
    raw['batch_completed_counts']=[8,16]  # reports 32 teachers but only 16 completed
    raw_path.write_text(json.dumps(raw)); sources['raw']=ref(root,raw_path)
    from treeneuron.protocol import pilot_seal
    seal=root/'evidence/pilot/raw-seal.json'; seal.write_text(json.dumps(pilot_seal(raw_path.parent)))
    sources['raw_seal']=ref(root,seal)
    with pytest.raises(ValueError,match='batch'): eligible_design(sources)


def test_corrupted_raw_E_predictions_reject_before_freeze(prepared):
    from treeneuron.protocol import eligible_design
    root,config,sources=prepared
    path=root/'evidence/pilot/teacher-0/repetition-0/E-predictions.npz'
    path.write_bytes(b'corrupt retained predictions')
    with pytest.raises(ValueError,match='prediction|Input hash'): eligible_design(sources)


def test_hand_filled_effect_rows_cannot_replace_saved_predictions(prepared):
    from treeneuron.protocol import eligible_design,pilot_seal
    root,config,sources=prepared
    raw_path=root/sources['raw']['path']; raw=json.loads(raw_path.read_text())
    record=raw['accepted_records'][0]; record['h1_rows']=[x+.01 for x in record['h1_rows']]
    raw['attempts'][0]['record']=record
    write(root,'evidence/pilot/teacher-0/repetition-0/pair.json',record)
    raw_path.write_text(json.dumps(raw)); sources['raw']=ref(root,raw_path)
    seal=root/'evidence/pilot/raw-seal.json'; seal.write_text(json.dumps(pilot_seal(raw_path.parent)))
    sources['raw_seal']=ref(root,seal)
    with pytest.raises(ValueError,match='prediction|row'): eligible_design(sources)


def test_review_cannot_name_missing_file_with_null_hash(prepared):
    from treeneuron.protocol import validate_pilot_readiness
    root,config,sources=prepared
    ready_path=root/sources['readiness']['path']; ready=json.loads(ready_path.read_text())
    review_path=root/ready['reviews']['9']['path']; review=json.loads(review_path.read_text())
    review['files']={'src/absent.py':None}; review_path.write_text(json.dumps(review))
    git(root,'add','audit'); git(root,'commit','-qm','simulated malformed acceptance record')
    ready['reviews']['9']=ref(root,review_path); ready_path.write_text(json.dumps(ready))
    with pytest.raises(ValueError,match='Reviewed'): validate_pilot_readiness(config,ready_path,root=root)


@pytest.mark.parametrize('field,value', [
    ('repetitions',3),('evaluation_rows',6),('probe_updates',2),('bootstrap_iterations',3000)])
def test_resource_recipe_binds_inner_workload(prepared,field,value):
    from treeneuron.protocol import validate_pilot_readiness
    root,config,sources=prepared
    changed=config.to_dict(); changed['pilot'][field]=value
    config=resolve_experiment(changed)
    path=root/sources['readiness']['path']; ready=json.loads(path.read_text())
    ready['config_sha256']=canonical_hash(config.to_dict())
    ready['seed_plan']['repetitions']=config.pilot.repetitions
    path.write_text(json.dumps(ready))
    # A newly issued readiness config cannot reuse a cheaper measured workload.
    with pytest.raises(ValueError,match='recipe'):
        validate_pilot_readiness(config,path,root=root)


@pytest.mark.parametrize('name', ['src/model.py', 'configs/scientific.yaml', 'analysis/infer.py', 'requirements.lock'])
def test_staged_execution_change_with_restored_worktree_is_rejected(repository, name):
    from treeneuron.protocol import execution_snapshot
    pinned=git(repository,'rev-parse','HEAD'); path=repository/name
    original=path.read_bytes(); path.write_text('# staged drift\n')
    git(repository,'add',name); path.write_bytes(original)
    assert git(repository,'diff','--name-only','HEAD')==''
    with pytest.raises(ValueError,match='Dirty'):
        execution_snapshot(repository,pinned)


def test_staged_document_only_change_remains_allowed(repository):
    from treeneuron.protocol import execution_snapshot
    pinned=git(repository,'rev-parse','HEAD')
    expected=execution_snapshot(repository,pinned)
    write(repository,'docs/note.json',{'simulated':'staged documentation'})
    git(repository,'add','docs')
    assert execution_snapshot(repository,pinned)==expected


def reseal_pair(root,sources,raw,record):
    from treeneuron.protocol import pilot_seal
    raw['accepted_records'][0]=record; raw['attempts'][0]['record']=record
    write(root,'evidence/pilot/teacher-0/repetition-0/pair.json',record)
    raw_path=write(root,sources['raw']['path'],raw); sources['raw']=ref(root,raw_path)
    seal=write(root,sources['raw_seal']['path'],pilot_seal(raw_path.parent))
    sources['raw_seal']=ref(root,seal)


def test_declarations_and_unrelated_child_do_not_establish_control_completion(prepared):
    from treeneuron.protocol import eligible_design
    root,_,sources=prepared; raw=json.loads((root/sources['raw']['path']).read_text())
    record=raw['accepted_records'][0]
    record['methods']={name:{'method_id':method['method_id'],'status':'completed'}
        for name,method in record['methods'].items()}
    reseal_pair(root,sources,raw,record)
    with pytest.raises(ValueError,match='control|stage|checkpoint'):
        eligible_design(sources)


@pytest.mark.parametrize('part',list('FGV'))
def test_self_consistent_development_counts_must_match_recipe(candidate,part):
    from dataclasses import replace
    import numpy as np
    from treeneuron.data.types import array_hash
    from treeneuron.protocol import freeze
    root,decision,sources,manifests=candidate; unit=sources['cohort']['units'][0]
    manifest=manifests[0]
    hashes=manifest.hashes
    if part=='F':
        fit=np.array([0.,1.,2.]); path=root/unit['fit_targets']['path']; np.save(path,fit)
        unit['fit_targets']=ref(root,path); hashes=(('F.y',array_hash(fit)),)
    manifest=replace(manifest,row_id=manifest.row_id+(999,),group_id=manifest.group_id+(999,),
        assignments=manifest.assignments+(part,),hashes=hashes)
    unit['manifest']=manifest.to_dict()
    with pytest.raises(ValueError,match='count|recipe'):
        freeze(decision,sources,root/'protocol/frozen.yaml')
    assert not (root/'protocol/frozen.yaml').exists()


@pytest.mark.parametrize('change',['regime','nuisance','prevalences','gates','probabilities'])
def test_valid_teacher_spec_must_match_declared_generation_recipe(candidate,change):
    from dataclasses import replace
    from treeneuron.data.synthetic import TeacherSpec,make_teacher
    from treeneuron.protocol import freeze
    root,decision,sources,manifests=candidate; unit=sources['cohort']['units'][0]
    path=root/unit['teacher']['path']; old=TeacherSpec.from_dict(json.loads(path.read_text()))
    kwargs={'regime':{'regime':'axis_aligned'},'nuisance':{'nuisance_dimensions':1},
        'prevalences':{'gate_prevalences':(.2,)},'gates':{'gate_types':('hard',)}}
    teacher=(replace(old,gate_type_probabilities=(.2,.3,.5)) if change=='probabilities' else
        make_teacher(old.seed,old.d,reference_seed=old.reference_seed,**kwargs[change]))
    path.write_text(json.dumps(teacher.to_dict())); unit['teacher']=ref(root,path)
    unit['manifest']=replace(manifests[0],teacher_id=teacher.teacher_id).to_dict()
    with pytest.raises(ValueError,match='recipe|generation'):
        freeze(decision,sources,root/'protocol/frozen.yaml')
    assert not (root/'protocol/frozen.yaml').exists()


def simulated_pair_graph(root,record,config):
    """Small producer-shaped JSON/payload graph; deliberately no trained model."""
    import hashlib
    import numpy as np
    from treeneuron.data.types import array_hash
    identities=dict(code=record['source_sha256'],lock='simulated-lock',raw='simulated-raw',
        split=record['data_hash'],transform='simulated-transform',runtime='simulated-runtime',
        config=canonical_hash(config.to_dict()))
    record['child_runs']=[]
    def run(name,kind='continuation',start=1,metadata=None,coordinator=None,allowance=10):
        directory=root/'runs'/name
        metadata=metadata or {}
        events=[dict(type='update',update=start+1,row_ids=[1,2],loss_standardized_mse=1.),
            dict(type='evaluation',update=start+1,improved=True),
            dict(type='selection',selected_update=start+1,executed_updates=1,g_mse=1.)]
        state=dict(identity_hashes=identities,config=config.to_dict(),update=start+1,
            stage=dict(complete=True,kind=kind,start_update=start,updates_completed=1,best_mse=1.,
                policy=dict(max_updates=allowance,checkpoint_interval=1)),growth_metadata=metadata)
        checkpoint=directory/'checkpoints'; checkpoint.mkdir(parents=True)
        version='checkpoint-'+'0'*32
        payload=checkpoint/(version+'.pt'); payload.write_bytes(b'simulated non-model payload')
        desc=write(checkpoint,version+'.meta.json',dict(schema_version=1,payload=payload.name,
            sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),state=state,best_state=None,history=[]))
        pointer=write(checkpoint,'best.json',dict(schema_version=1,metadata=desc.name,sha256=ref(checkpoint,desc)['sha256']))
        manifest=dict(identity=dict(run_id=name,status='completed',phase='exploratory',
            config_hash=canonical_hash(config.to_dict()),code_revision=identities['code'],lock_hash=identities['lock']),
            data=dict(test_labels_accessed=False,raw_hash=identities['raw'],split_hash=identities['split'],
                transform_hash=identities['transform']),method=dict(scope='single_model',resolved_config=config.to_dict(),
                weight_checkpoint='checkpoints/best.json',weight_hash=ref(checkpoint,desc)['sha256']))
        if coordinator: manifest['identity']['coordinator_run_id']=coordinator
        else: record['child_runs'].append(dict(run_id=name,path=str(directory),growth_metadata=metadata))
        write(directory,'manifest.json',manifest)
        (directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        stage=dict(run_id=name,path=str(directory.relative_to(root)),selected_update=start+1,executed_updates=1,
            max_updates=allowance,checkpoint_interval=1,hit_ceiling=allowance==1,improving_at_ceiling=allowance==1,
            evaluations=[events[1]],training_loss=[dict(update=start+1,value=1.)],
            row_hashes=[array_hash(np.array([1,2],dtype=np.int64))])
        return stage,pointer
    parent,pointer=run('parent',kind='base',start=0,allowance=config.training.base_updates)
    record.update(parent_run_id='parent',parent_checkpoint_sha256=ref(root,pointer)['sha256'],
        parent_identity_hashes=identities,parent_pairing={'update':1})
    allocations={name:dict(requested_count=1,effective_count=1,table_hash='simulated-table',ids=[0],
        contrasts_float64=[[1.,0.]],policy=policy) for name,policy in [('top','adam'),('random','random'),('greedy','greedy')]}
    for name,method in record['methods'].items():
        warm=name not in ('ordinary_lower','ordinary_upper','static_partial')
        recipe={'top':{'variant':'adam'},'widen_upper':{'variant':'widen'},
            'ordinary_lower':{'variant':'lower'},'ordinary_upper':{'variant':'upper'},
            'static_partial':{'variant':'partial','coordinates':'centered'}}.get(name,{})
        method.update(recipe=recipe,stages=[],selected_g_mse=1.)
        metadata={'causal_adapter':{'method_id':method['method_id'],'recipe':recipe}}
        start=1 if warm else 0
        for j,allowance in enumerate([config.training.continuation_updates] if warm else
                [config.training.base_updates,config.training.continuation_updates]):
            stage,_=run(f'{name}-{j}',kind='continuation' if warm or j else 'base',start=start,
                metadata=metadata,allowance=allowance)
            method['stages'].append(stage); start=stage['selected_update']
        method['executed_row_ids']=method['stages'][-1]['row_hashes']
        if warm: method.update(parent_run_id='parent',parent_checkpoint_sha256=record['parent_checkpoint_sha256'],
            initial_pairing=record['parent_pairing'])
        if name in allocations: method['allocation']=allocations[name]
    coordinator=root/'runs'/'coordinator'; coordinator.mkdir()
    write(coordinator,'manifest.json',dict(identity=dict(status='completed',phase='exploratory',run_id='coordinator'),
        data={'test_labels_accessed':False},method={'scope':'coordinator'}))
    record['child_runs'].append(dict(run_id='coordinator',path=str(coordinator),growth_metadata={}))
    import shutil
    # The coordinator's parent is a separate pointer/descriptor publication in the real producer.
    parent_pointer=json.loads(pointer.read_text()); parent_desc=json.loads((pointer.parent/parent_pointer['metadata']).read_text())
    for filename in (parent_pointer['metadata'],parent_desc['payload']):
        shutil.copyfile(pointer.parent/filename,coordinator/filename)
    write(coordinator,'parent.json',parent_pointer)
    trials=[]
    for candidate,unit in [('keep',None),('unit_0',0),('unit_1',1)]:
        stage,_=run('probe-'+candidate,start=1,metadata={'greedy_probe':dict(candidate=candidate,
            coordinator_run_id='coordinator',parent_checkpoint_sha256=parent_pointer['sha256'],score_hash='simulated-table')},
            coordinator='coordinator',allowance=config.pilot.probe_updates)
        directory=root/stage['path']; target=coordinator/'probes'/directory.name
        target.parent.mkdir(exist_ok=True); shutil.move(str(directory),target)
        trials.append(dict(candidate=candidate,unit_id=unit,run_id=directory.name,
            path=str(target.relative_to(coordinator)),method_id='B6',outcome='completed',
            selected_update=2,executed_updates=1,g_mse=1.))
    score=coordinator/'score-table.npz'; score.write_bytes(b'simulated score table')
    greedy=write(coordinator,'greedy.json',dict(complete=True,method='B6',not_attempted=[],trials=trials,
        eligible_ids=[0,1],probe_updates=config.pilot.probe_updates,selection_split='G',V_accessed=False,T_accessed=False,
        parent_identity_hashes=identities,parent_checkpoint_sha256=parent_pointer['sha256'],score_hash='simulated-table',
        score_artifact=ref(coordinator,score),allocation=dict(ids=[0],table_hash='simulated-table',
            requested_count=1,effective_count=1)))
    record['greedy_report']=str(greedy)


@pytest.mark.parametrize('change',['missing_stage','wrong_role','missing_child','incomplete_child',
    'wrong_checkpoint','incomplete_checkpoint','missing_probe','incomplete_probe','selected_update'])
def test_resealed_role_graph_must_retain_completed_selected_evidence(prepared,change):
    from treeneuron.protocol import eligible_design
    root,_,sources=prepared; raw=json.loads((root/sources['raw']['path']).read_text())
    record=raw['accepted_records'][0]; method=record['methods']['ordinary_upper']
    pair=root/'evidence/pilot/teacher-0/repetition-0'; stage=method['stages'][-1]
    path=pair/stage['path']
    if change=='missing_stage': method['stages'].pop()
    elif change=='wrong_role': method['stages']=copy.deepcopy(record['methods']['ordinary_lower']['stages'])
    elif change=='missing_child': record['child_runs']=[r for r in record['child_runs'] if r['run_id']!=stage['run_id']]
    elif change=='incomplete_child':
        manifest=json.loads((path/'manifest.json').read_text()); manifest['identity']['status']='failed'
        write(path,'manifest.json',manifest)
    elif change in ('wrong_checkpoint','incomplete_checkpoint'):
        pointer=json.loads((path/'checkpoints/best.json').read_text())
        descriptor_path=path/'checkpoints'/pointer['metadata']; descriptor=json.loads(descriptor_path.read_text())
        if change=='wrong_checkpoint': descriptor['state']['growth_metadata']['causal_adapter']['recipe']['variant']='lower'
        else: descriptor['state']['stage']['complete']=False
        descriptor_path.write_text(json.dumps(descriptor)); pointer['sha256']=ref(root,descriptor_path)['sha256']
        write(path,'checkpoints/best.json',pointer)
        manifest=json.loads((path/'manifest.json').read_text()); manifest['method']['weight_hash']=pointer['sha256']
        write(path,'manifest.json',manifest)
    elif change in ('missing_probe','incomplete_probe'):
        gp=Path(record['greedy_report']); greedy=json.loads(gp.read_text()); trial=greedy['trials'][-1]
        if change=='missing_probe': greedy['trials'].pop(); gp.write_text(json.dumps(greedy))
        else:
            mp=gp.parent/trial['path']/'manifest.json'; manifest=json.loads(mp.read_text())
            manifest['identity']['status']='failed'; mp.write_text(json.dumps(manifest))
    else: stage['selected_update']+=1
    reseal_pair(root,sources,raw,record)
    with pytest.raises(ValueError,match='control|stage|checkpoint|pilot|probe|selection'):
        eligible_design(sources)


def test_scientific_primary_claim_must_match_pilot_population(candidate):
    from treeneuron.protocol import _scientific_sources,pilot_seal
    root,decision,sources,_=candidate; raw_path=root/sources['raw']['path']
    raw=json.loads(raw_path.read_text()); raw['config']['data']['population']='pure_noise'
    raw_path.write_text(json.dumps(raw)); sources['raw']=ref(root,raw_path)
    seal=write(root,sources['raw_seal']['path'],pilot_seal(raw_path.parent)); sources['raw_seal']=ref(root,seal)
    with pytest.raises(ValueError,match='population'):
        _scientific_sources(sources,decision)


@pytest.mark.parametrize('prevalences,valid', [([.1,.5],True),([.01],True),([.2],False)])
def test_rare_recipe_retains_versioned_prevalence_normalization(candidate,prevalences,valid):
    from dataclasses import replace
    from treeneuron.data.synthetic import make_teacher
    from treeneuron.protocol import _scientific_sources
    root,decision,sources,manifests=candidate
    raw_path=root/sources['raw']['path']; raw=json.loads(raw_path.read_text())
    raw['config']['data'].update(regime='rare',gate_prevalences=prevalences)
    raw_path.write_text(json.dumps(raw)); sources['raw']=ref(root,raw_path)
    teacher=make_teacher(999,raw['config']['model']['d'],regime='rare')
    for unit,manifest in zip(sources['cohort']['units'],manifests):
        value=replace(teacher,seed=unit['seeds']['teacher'],reference_seed=unit['seeds']['reference'])
        path=root/unit['teacher']['path']; path.write_text(json.dumps(value.to_dict())); unit['teacher']=ref(root,path)
        unit['manifest']=replace(manifest,teacher_id=value.teacher_id).to_dict()
    if valid:
        assert _scientific_sources(sources,decision)['study_id']=='toy'
    else:
        with pytest.raises(ValueError,match='rare|prevalence|recipe'):
            _scientific_sources(sources,decision)


@pytest.mark.parametrize('change',['fresh_data','probe_policy'])
def test_resealed_checkpoint_graph_preserves_common_data_and_probe_allowance(prepared,change):
    from treeneuron.protocol import eligible_design
    root,_,sources=prepared; raw=json.loads((root/sources['raw']['path']).read_text())
    record=raw['accepted_records'][0]; pair=root/'evidence/pilot/teacher-0/repetition-0'
    if change=='fresh_data':
        paths=[pair/stage['path'] for stage in record['methods']['ordinary_upper']['stages']]
    else:
        greedy_path=Path(record['greedy_report']); greedy=json.loads(greedy_path.read_text())
        paths=[greedy_path.parent/greedy['trials'][-1]['path']]
    for path in paths:
        pointer=json.loads((path/'checkpoints/best.json').read_text())
        descriptor_path=path/'checkpoints'/pointer['metadata']; descriptor=json.loads(descriptor_path.read_text())
        manifest=json.loads((path/'manifest.json').read_text())
        if change=='fresh_data':
            descriptor['state']['identity_hashes']['raw']='different-population'
            manifest['data']['raw_hash']='different-population'
        else: descriptor['state']['stage']['policy']['max_updates']+=1
        descriptor_path.write_text(json.dumps(descriptor)); pointer['sha256']=ref(root,descriptor_path)['sha256']
        write(path,'checkpoints/best.json',pointer); manifest['method']['weight_hash']=pointer['sha256']
        write(path,'manifest.json',manifest)
    reseal_pair(root,sources,raw,record)
    with pytest.raises(ValueError,match='control|probe|identity'):
        eligible_design(sources)
