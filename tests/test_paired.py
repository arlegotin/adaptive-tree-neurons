"""End-to-end causal pairing, retained controls and artifact-only rebuilding."""
import copy
import json
import weakref
from dataclasses import replace

import pytest
import torch

from test_train import cpu_training_evidence
from treeneuron.config import resolve_experiment


def small_config(root, receipt):
    return resolve_experiment({'study_id': 'paired-contract',
        'model': {'d': 2, 'h': 4},
        'data': {'population': 'conditional_slope_2d', 'n_fit': 16, 'n_growth': 8, 'n_validation': 8},
        'training': {'batch_size': 8, 'base_updates': 2, 'continuation_updates': 2,
                     'validation_interval': 1, 'patience': 20},
        'backend': {'cpu_threads': 1},
        'execution': {'backend_report': str(receipt), 'artifacts_root': str(root)}})


@pytest.fixture(scope='module')
def study(tmp_path_factory, cpu_training_evidence):
    from treeneuron.experiments import paired
    config = small_config(tmp_path_factory.mktemp('paired'), cpu_training_evidence)
    original = paired.load_checkpoint
    refs = []
    def sequential_load(*args, **kwargs):
        assert all(ref() is None for ref in refs), 'previous parent fork still live'
        state = original(*args, **kwargs)
        refs.append(weakref.ref(state))
        return state
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(paired, 'load_checkpoint', sequential_load)
        directory = paired.run_paired(config)
    assert len(refs) >= 6
    return directory


def test_complete_study_retains_all_branches_and_equal_budgets(study):
    from treeneuron.analysis.exploratory import summarize_exploratory
    report = summarize_exploratory(study)
    assert report['status'] == 'completed'
    methods = report['methods']
    assert {'parent', 'keep', 'sham', 'top', 'random', 'gaussian', 'mean', 'ridge',
            'static_partial_centered', 'static_full_direct_leaf', 'ordinary_lower',
            'ordinary_upper', 'widen_lower', 'widen_upper'} <= methods.keys()
    assert methods['top']['allocation']['requested_count'] == methods['random']['allocation']['requested_count'] == 16
    assert methods['top']['allocation']['effective_count'] == methods['random']['allocation']['effective_count']
    assert report['pairing_audit']['passed']
    assert methods['static_full_direct_leaf']['declared_total_updates'] == 4
    assert methods['top']['declared_total_updates'] == 4
    assert report['paired_differences']['sham_minus_keep']['G']['mse'] == 0
    assert report['paired_differences']['sham_minus_keep']['V']['max_prediction_abs'] == 0
    assert report['costs']['total_study_ns'] > report['costs']['shared_scoring_ns'] > 0
    assert report['costs']['standalone_random_scoring_ns'] is None
    assert report['tiny_gate_probe']['table_predicted_decrease'] > 0
    assert report['tiny_gate_probe']['table_gradient_max_abs'] < 1e-10
    for name, method in methods.items():
        assert method['status'] == 'completed', name
        assert method['enclosing_ns'] > 0
        assert method['metrics']['V']['mse'] >= 0


def test_pairing_audit_rejects_sampler_cursor_mutation(study):
    from treeneuron.experiments.paired import audit_pairing
    report = json.loads((study / 'study.json').read_text())
    report['methods']['random']['initial_pairing']['cursor'] += 1
    with pytest.raises(ValueError, match='pairing'):
        audit_pairing(report)


def test_shared_per_id_contrasts_and_executed_row_streams(study):
    report = json.loads((study / 'study.json').read_text())
    table = json.loads((study / 'score-table.json').read_text())
    for name in ('top', 'random'):
        method = report['methods'][name]
        allocation = method['allocation']
        for unit, contrast in zip(allocation['ids'], allocation['contrasts_float64']):
            assert contrast == table['contrasts'][unit]
        assert method['initial_pairing'] == report['parent_pairing']
    top = report['methods']['top']['executed_row_ids']
    random = report['methods']['random']['executed_row_ids']
    assert top[:min(len(top), len(random))] == random[:min(len(top), len(random))]


def test_exploration_never_loads_T(tmp_path, cpu_training_evidence, monkeypatch):
    from treeneuron.experiments.paired import run_paired
    from treeneuron.data import types
    def forbidden(*args, **kwargs):
        raise AssertionError('Protected T loader called')
    monkeypatch.setattr(types.DevelopmentData, 'T', property(forbidden), raising=False)
    path = run_paired(small_config(tmp_path, cpu_training_evidence))
    assert json.loads((path / 'study.json').read_text())['test_labels_accessed'] is False


def test_zero_score_teacher_is_retained(tmp_path, cpu_training_evidence, monkeypatch):
    from treeneuron.experiments import paired
    actual_score = paired.score_units
    def zero_residual_teacher(model, X, y, **kwargs):
        # Exact zero residual teacher on this F: real scorer, no invented table.
        with torch.no_grad(): target = model(X).clone()
        table = actual_score(model, X, target, **kwargs)
        assert not table.eligible.any()
        return table
    monkeypatch.setattr(paired, 'score_units', zero_residual_teacher)
    path = paired.run_paired(small_config(tmp_path, cpu_training_evidence))
    report = json.loads((path / 'study.json').read_text())
    top, random = [report['methods'][name]['allocation'] for name in ('top', 'random')]
    assert top['requested_count'] == random['requested_count'] == 16
    assert top['effective_count'] == random['effective_count'] == 0
    assert report['status'] == 'completed'


def test_summary_detects_modified_prediction_artifact(study, tmp_path):
    import shutil
    from treeneuron.analysis.exploratory import summarize_exploratory
    shutil.copytree(study, tmp_path / 'copy')
    copied = tmp_path / 'copy'
    target = copied / 'predictions' / 'top-G.json'
    target.write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        summarize_exploratory(copied)


def test_failed_fork_retains_completed_losers_and_terminal_cost(tmp_path, cpu_training_evidence, monkeypatch):
    from treeneuron.experiments import paired
    build = paired.build_causal
    def fail_top(method, recipe, **kwargs):
        if method == 'B5' and recipe.get('variant') == 'adam':
            raise RuntimeError('intentional factory failure')
        return build(method, recipe, **kwargs)
    monkeypatch.setattr(paired, 'build_causal', fail_top)
    with pytest.raises(RuntimeError, match='intentional factory failure'):
        paired.run_paired(small_config(tmp_path, cpu_training_evidence))
    report = json.loads(next(tmp_path.glob('*/study.json')).read_text())
    assert report['status'] == 'failed'
    assert report['methods']['keep']['status'] == 'completed'
    assert report['methods']['sham']['status'] == 'completed'
    assert report['methods']['top']['status'] == 'failed'
    assert report['methods']['top']['enclosing_ns'] > 0
    assert report['failures'][0]['message'] == 'intentional factory failure'


def test_completed_coordinator_and_children_publish_observed_manifests(study):
    from treeneuron.config import canonical_hash
    report=json.loads((study/'study.json').read_text())
    manifest=json.loads((study/'manifest.json').read_text())
    splits=json.loads((study/'splits.json').read_text())
    teacher=json.loads((study/'teacher.json').read_text())
    assert manifest['method']['scope']=='coordinator'
    assert manifest['data']['raw_hash']==splits['dataset_id']
    assert manifest['data']['split_hash']==report['data']['split_hash']
    assert manifest['method']['backend']['actual']['device']=='cpu'
    assert manifest['resources']['rss_bytes']==report['resources']['process_peak_rss_bytes']
    assert manifest['costs']['enclosing']['nanoseconds']==report['costs']['total_study_ns']
    for field in ('total_parameters','trainable_parameters','weight_checkpoint','weight_hash','serialized_bytes','context_bytes','cache_bytes','weight_terms'):
        assert manifest['method'][field]['value'] is None
        assert 'coordinator' in manifest['method'][field]['reason'].lower()
    all_manifests=[manifest]
    for name,method in report['methods'].items():
        for i,stage in enumerate(method['stages']):
            child=json.loads((study/stage['path']/'manifest.json').read_text());all_manifests.append(child)
            assert child['identity']['status']=='completed'
            assert child['identity']['coordinator_run_id']==study.name
            assert child['method']['scope']=='single_model'
            assert child['method']['total_parameters']==method['total_parameters']
            assert child['resources']['rss_bytes']>0
            if method.get('parent_run_id'): assert child['identity']['parent_run_id']==method['parent_run_id']
            elif i: assert child['identity']['parent_run_id']==method['stages'][i-1]['run_id']
            if name in ('top','random','gaussian','sgd'):
                assert child['identity']['growth_hash']==canonical_hash(method['growth_metadata'])
                assert child['identity']['score_hash']==method['allocation']['table_hash']
            if i==len(method['stages'])-1:
                assert child['method']['prediction_artifacts']==method['predictions']
                assert json.loads((study/stage['path']/'metrics.json').read_text())==method['metrics']
    for record in all_manifests:
        assert record['data']['boundaries']['V']=='descriptive_only'
        for part,n in [('F',16),('G',8),('V',8)]:assert record['data']['counts'][part]=={'rows':n,'groups':n}
        for key in ('reference_mean','reference_std','reference_rows','reference_seed'):
            assert record['data']['teacher_normalization'][key]==teacher[key]
        assert record['data']['teacher_normalization']['target_normalization']=='none'
        assert record['data']['test_labels_accessed'] is False
        assert record['resources']['pressure']['value'] is None
        assert record['resources']['swap_delta_bytes']['value'] is None


def test_prediction_failure_finalizes_open_child_with_known_observations(tmp_path,cpu_training_evidence,monkeypatch):
    from treeneuron.experiments import paired
    predict=paired._predictions
    def fail(predictor,data,root,name):
        if name=='top':raise RuntimeError('prediction publication failed')
        return predict(predictor,data,root,name)
    monkeypatch.setattr(paired,'_predictions',fail)
    with pytest.raises(RuntimeError,match='prediction publication'):
        paired.run_paired(small_config(tmp_path,cpu_training_evidence))
    root=next(tmp_path.glob('*/study.json')).parent
    report=json.loads((root/'study.json').read_text())
    child=json.loads((root/report['methods']['top']['stages'][0]['path']/'manifest.json').read_text())
    assert child['identity']['status']=='failed'
    assert child['data']['counts']['V']=={'rows':8,'groups':8}
    assert child['data']['boundaries']['V']=='descriptive_only'
    assert child['identity']['score_hash']==report['methods']['top']['allocation']['table_hash']
    assert json.loads((root/'manifest.json').read_text())['data']['counts']['V']=={'rows':8,'groups':8}


@pytest.mark.parametrize('field,value', [('policy','keep'), ('policy','sgd'), ('policy','random'),
    ('direction','gaussian'), ('projection_rms_floor',1.0), ('projection_rms_floor',1e-13)])
def test_unsupported_growth_request_rejected_before_artifacts(tmp_path, cpu_training_evidence, field, value):
    from treeneuron.experiments.paired import run_paired
    config = small_config(tmp_path, cpu_training_evidence)
    config = replace(config, growth=replace(config.growth, **{field:value}))
    with pytest.raises(ValueError, match='fixed.*growth'):
        run_paired(config)
    assert list(tmp_path.iterdir()) == []


def test_fixed_inventory_semantics_are_explicit(study):
    report=json.loads((study/'study.json').read_text())
    semantics=report['execution_semantics']
    assert semantics['proposed'] == {'policy':'adam','direction':'eigenvector'}
    assert semantics['projection_rms_floor'] == 1e-12
    assert semantics['inventory'] == 'fixed_causal_comparison'


@pytest.mark.parametrize('fail_count', [1,4])
def test_ineligible_diagnostic_score_export_and_summary(tmp_path, cpu_training_evidence, monkeypatch, fail_count):
    from treeneuron.experiments import paired
    from treeneuron.analysis.exploratory import summarize_exploratory
    score=paired.score_units
    original_eigh=torch.linalg.eigh
    identities=[]
    def score_with_eigen_failure(*args, **kwargs):
        calls=0
        def eigh(moment):
            nonlocal calls
            calls+=1
            if calls<=fail_count: raise torch.linalg.LinAlgError('injected unavailable eigenvalues')
            return original_eigh(moment)
        with monkeypatch.context() as patch:
            patch.setattr(torch.linalg, 'eigh', eigh)
            table=score(*args, **kwargs)
        identities.append(table.table_hash)
        return table
    monkeypatch.setattr(paired,'score_units',score_with_eigen_failure)
    directory=paired.run_paired(small_config(tmp_path,cpu_training_evidence))
    raw=(directory/'score-table.json').read_text()
    assert 'NaN' not in raw and 'Infinity' not in raw
    table=json.loads(raw)
    assert table['table_hash']==identities[0]
    assert table['status'][:fail_count]==['eigendecomposition_failed']*fail_count
    assert table['eigenvalues'][:fail_count]==[[None]*3]*fail_count
    assert all(item['reason']=='eigendecomposition_failed' for item in table['unavailable_diagnostics'])
    report=summarize_exploratory(directory)
    assert report['status']=='completed'
    for method in ('top','random','gaussian','sgd'):
        allocation=report['methods'][method]['allocation']
        assert allocation['effective_count']=={1:2,4:0}[fail_count]
        assert allocation['table_hash']==identities[0]


def test_invalid_json_leaves_no_export_file(tmp_path):
    from treeneuron.experiments.paired import write_json
    path=tmp_path/'score-table.json'
    with pytest.raises(ValueError,match='finite'):
        write_json(path,{'unavailable':float('nan')})
    assert not path.exists()


@pytest.mark.parametrize('target,failure', [('parent',KeyboardInterrupt), ('top',FloatingPointError),
    ('ordinary_lower',RuntimeError)])
def test_training_failure_retains_registered_child_context(tmp_path,cpu_training_evidence,monkeypatch,target,failure):
    from treeneuron import train
    from treeneuron.experiments import paired
    from treeneuron.config import canonical_hash
    observe=train._observe_initial
    step=torch.optim.AdamW.step
    active=False
    def mark_stage(state,data,paths,*args):
        nonlocal active
        observe(state,data,paths,*args)
        if target=='parent': active=True
        elif target=='top': active=len(state.optimizer.param_groups)>1
        else: active=state.growth_metadata.get('causal_adapter',{}).get('recipe',{}).get('variant')=='lower'
    def fail_step(optimizer,*args,**kwargs):
        if active: raise failure('injected actual training failure')
        return step(optimizer,*args,**kwargs)
    monkeypatch.setattr(train,'_observe_initial',mark_stage)
    monkeypatch.setattr(torch.optim.AdamW,'step',fail_step)
    with pytest.raises(failure,match='actual training failure'):
        paired.run_paired(small_config(tmp_path,cpu_training_evidence))
    root=next(tmp_path.glob('*/study.json')).parent
    study=json.loads((root/'study.json').read_text())
    expected_status='interrupted' if target=='parent' else 'failed'
    assert study['status']==expected_status
    method=study['methods'][target]
    assert method['status']==expected_status
    assert len(method['stages'])==1
    stage=method['stages'][0]
    child=json.loads((root/stage['path']/'manifest.json').read_text())
    assert child['identity']['status']==expected_status
    assert child['identity']['coordinator_run_id']==root.name
    assert child['data']['boundaries']['V']=='descriptive_only'
    assert child['data']['counts']['V']=={'rows':8,'groups':8}
    assert child['data']['teacher_normalization']['reference_rows']>0
    assert child['data']['teacher_normalization']['target_normalization']=='none'
    assert child['data']['test_labels_accessed'] is False
    events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    assert any(e['type']=='child_registered' and e['method']==target and e['run_id']==stage['run_id'] for e in events)
    if target=='top':
        assert child['identity']['parent_run_id']==study['methods']['parent']['stages'][0]['run_id']
        assert child['identity']['score_hash']==method['allocation']['table_hash']
        assert child['identity']['growth_hash']==canonical_hash(method['growth_metadata'])
        pointer=json.loads((root/stage['path']/'checkpoints'/'last.json').read_text())
        metadata=json.loads((root/stage['path']/'checkpoints'/pointer['metadata']).read_text())
        assert metadata['state']['growth_metadata']==method['growth_metadata']


@pytest.mark.parametrize('mutation',['empty','truncated','changed','count'])
def test_pairing_rejects_duplicate_stream_corruption(study,tmp_path,mutation):
    import shutil
    from treeneuron.experiments.paired import audit_pairing
    from treeneuron.analysis.exploratory import summarize_exploratory
    report=json.loads((study/'study.json').read_text())
    method=report['methods']['random']
    if mutation=='empty': method['executed_row_ids']=[]
    elif mutation=='truncated': method['executed_row_ids']=method['executed_row_ids'][:-1]
    elif mutation=='changed': method['executed_row_ids'][0]='0'*64
    else: method['stages'][-1]['executed_updates']+=1
    with pytest.raises(ValueError,match='pairing'):
        audit_pairing(report)
    root=tmp_path/'corrupted'
    shutil.copytree(study,root)
    (root/'study.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='pairing'):
        summarize_exploratory(root)


def test_pairing_accepts_consistent_unequal_stopping_lengths(study):
    from treeneuron.experiments.paired import audit_pairing
    report=json.loads((study/'study.json').read_text())
    method=report['methods']['random']
    method['executed_row_ids']=method['executed_row_ids'][:1]
    method['stages'][-1]['row_hashes']=method['stages'][-1]['row_hashes'][:1]
    method['stages'][-1]['executed_updates']=1
    assert audit_pairing(report)['passed']


@pytest.mark.parametrize('phase',['observation','index','study','terminal'])
@pytest.mark.parametrize('body_failure',[False,True])
def test_final_publication_failure_restores_process_and_primary_error(tmp_path,cpu_training_evidence,monkeypatch,phase,body_failure):
    from treeneuron import train
    from treeneuron.experiments import paired
    from treeneuron.config import canonical_hash
    receipt=train.BACKEND_REPORT_PATH
    rng=train.capture_rng()
    threads=torch.get_num_threads()
    expected_rng=canonical_hash(paired._json_tree(rng))
    build=paired.build_causal
    observe=paired.update_run_observations
    hash_file=paired.file_hash
    write=paired.write_json
    finish=paired.finish_run
    finalizing=False
    def fail_factory(method,recipe,**kwargs):
        if method=='B5' and recipe.get('variant')=='adam': raise RuntimeError('primary experiment failure')
        return build(method,recipe,**kwargs)
    def observe_final(paths,values):
        nonlocal finalizing
        if 'enclosing' in values.get('costs',{}):
            finalizing=True
            if phase=='observation': raise OSError('secondary observation publication failure')
        return observe(paths,values)
    def hash_final(path):
        if finalizing and phase=='index': raise OSError('secondary index publication failure')
        return hash_file(path)
    def write_final(path,value):
        if path.name=='study.json' and phase=='study': raise OSError('secondary study publication failure')
        return write(path,value)
    def finish_final(paths,*args):
        if paths.directory.parent==tmp_path and phase=='terminal': raise OSError('secondary terminal publication failure')
        return finish(paths,*args)
    if body_failure: monkeypatch.setattr(paired,'build_causal',fail_factory)
    monkeypatch.setattr(paired,'update_run_observations',observe_final)
    monkeypatch.setattr(paired,'file_hash',hash_final)
    monkeypatch.setattr(paired,'write_json',write_final)
    monkeypatch.setattr(paired,'finish_run',finish_final)
    try:
        with pytest.raises(RuntimeError if body_failure else OSError,
                match='primary experiment failure' if body_failure else 'secondary') as caught:
            paired.run_paired(small_config(tmp_path,cpu_training_evidence))
        assert train.BACKEND_REPORT_PATH==receipt
        assert torch.get_num_threads()==threads
        assert canonical_hash(paired._json_tree(train.capture_rng()))==expected_rng
        if body_failure:
            assert any('secondary' in note for note in caught.value.__notes__)
        sidecar=json.loads(next(tmp_path.glob('*/publication-failure-*.json')).read_text())
        assert 'secondary' in sidecar['message']
    finally:
        train.BACKEND_REPORT_PATH=receipt
        train.restore_rng(rng)
        torch.set_num_threads(threads)
