"""PRE-FREEZE toy/trap fixtures; none are scientific observations."""
import json
import shutil
from pathlib import Path
import pytest
from test_protocol import repository, prepared, candidate


def test_live_frozen_identity_survives_restored_checkout(candidate, tmp_path):
    from treeneuron.protocol import freeze, validate_frozen
    root, decision, sources, _ = candidate
    frozen = freeze(decision, sources, root/'protocol/frozen.yaml')
    restored = tmp_path/'restored'
    shutil.copytree(root, restored)
    root.rename(tmp_path/'unavailable-original')
    checked = validate_frozen(restored/'protocol/frozen.yaml', {'root': str(restored)})
    assert checked.protocol_hash == frozen.protocol_hash
    (restored/'src/model.py').write_text('drift')
    with pytest.raises(ValueError):
        validate_frozen(restored/'protocol/frozen.yaml', {'root': str(restored)})


def test_audit_missing_study_is_blocked_and_report_is_deterministic(candidate, tmp_path):
    from treeneuron.protocol import freeze
    from treeneuron.analysis.audit import audit
    from treeneuron.analysis.report import render_report
    root, decision, sources, _ = candidate
    frozen = freeze(decision, sources, root/'protocol/frozen.yaml')
    result = audit(root/'protocol/confirmation'/frozen.protocol_hash, frozen)
    assert result['valid'] is False and result['status'] == 'blocked'
    assert result['reasons'] and result['conclusion']['superiority'] is False
    first, second = tmp_path/'first.md', tmp_path/'second.md'
    render_report(result, first); render_report(result, second)
    assert first.read_bytes() == second.read_bytes()
    assert 'blocked' in first.read_text()


def test_archive_api_is_distinct_from_live_capability():
    from treeneuron.analysis.audit import audit_archive, seal_archive
    assert callable(audit_archive) and callable(seal_archive)


def test_negative_complete_rows_and_missing_pair_have_different_conclusions():
    from test_primary import rows, policy
    from treeneuron.analysis.primary import summarize_cohort
    from treeneuron.analysis.audit import conclude
    from treeneuron.protocol import EVALUATION_METHODS
    predictions, runs = rows((-1., -2., -3.), 2)
    for saved in predictions.values():
        saved['predictions'].update({role: [0., 0.] for role in EVALUATION_METHODS if role not in ('top', 'random')})
    payload = {'design': {'K': 3, 'statistical_policy': policy(), 'branch': 'estimation'},
               'sources': {'scientific': {}}}
    ledger = {'status': 'completed', 'costs': {}}
    result = summarize_cohort(payload, ledger, runs, predictions, 'toy')
    complete = conclude(result, [])
    assert complete['status'] == 'completed' and complete['superiority'] is False
    assert complete['interpretation'] == 'negative'
    del predictions['0-1']
    result = summarize_cohort(payload, ledger, runs, predictions, 'toy')
    missing = conclude(result, [])
    assert missing['status'] == 'incomplete' and missing['interpretation'] == 'incomplete'
    assert result['teachers'][0]['D'] is None

from test_train import cpu_training_evidence


def test_real_incomplete_audit_relocation_archive_and_tamper(prepared,cpu_training_evidence,monkeypatch,tmp_path):
    """Actual eight-role fits for one teacher/two reps; 31 missing teachers.

    Pytest substitutes only its numerical receipt and interop setting. The
    temporary source/readiness pilot is the existing simulated protocol fixture.
    """
    import importlib
    import torch
    from test_protocol import git, ref
    from treeneuron.protocol import eligible_design, freeze, amend, validate_frozen
    from treeneuron.analysis.audit import audit, seal_archive, audit_archive
    from treeneuron.analysis.report import render_report
    from treeneuron.evaluation import protected
    module=importlib.import_module('treeneuron.evaluation.confirm')
    root,config,sources=prepared; decision=eligible_design(sources)
    sources=module.prepare_cohort(decision,sources,root/'cohort',cohort_id='audit-toy',root_seed=17)
    frozen=freeze(decision,sources,root/'protocol/frozen.yaml')
    original_resolve=module.resolve_experiment
    def fixture_runtime(value):
        value=dict(value); value['execution']=dict(value['execution'],backend_report=str(cpu_training_evidence))
        value['backend']=dict(value['backend'],interop_threads=torch.get_num_interop_threads())
        return original_resolve(value)
    monkeypatch.setattr(module,'resolve_experiment',fixture_runtime)
    monkeypatch.setattr(module,'_runtime_preflight',lambda c: {})
    original_fit=module._fit_run
    def fit(p,payload,unit,rep,directory):
        if unit['unit_id']!='unit-0': raise RuntimeError('Deliberately omitted toy numerical work')
        return original_fit(p,payload,unit,rep,directory)
    monkeypatch.setattr(module,'_fit_run',fit)
    directory=module.run_confirmation(frozen,'synthetic')
    def forbidden(*args,**kwargs): raise AssertionError('Audit attempted new fit/predict/T access')
    monkeypatch.setattr(module,'_fit_run',forbidden)
    monkeypatch.setattr(module,'load_predictors',forbidden)
    monkeypatch.setattr(protected,'_test_labels',forbidden)
    monkeypatch.setattr(protected,'_rng',forbidden)
    first=audit(directory,frozen)
    assert first['status']=='incomplete' and not first['valid'], first['reasons']
    assert not first['reasons'] and first['primary']['valid_runs']==2
    assert first['accounting']['failed_attempts']==124
    second=audit(directory,frozen)
    assert first==second
    work=[json.loads(p.read_text()) for p in (directory/'audit-work').glob('*.json')]
    assert len(work)==2 and all(w['seconds']>0 for w in work)
    assert all(len(w['nested_analysis_receipts'])==1 for w in work)
    latest=max(work,key=lambda w:w['total_study_work_seconds'])
    assert latest['total_study_work_seconds']==pytest.approx(first['accounting']['costs']['combined_seconds']+sum(w['seconds'] for w in work))
    render_report(first,tmp_path/'a.md'); render_report(second,tmp_path/'b.md')
    assert (tmp_path/'a.md').read_bytes()==(tmp_path/'b.md').read_bytes()
    # All attempt records are authoritative, including excluded/failed attempts.
    ledger_path=directory/'confirmation.json'; ledger_bytes=ledger_path.read_bytes()
    ledger=json.loads(ledger_bytes); ledger['runs'].pop()
    ledger_path.write_text(json.dumps(ledger))
    assert 'cohort' in ' '.join(audit(directory,frozen)['reasons'])
    ledger_path.write_bytes(ledger_bytes)
    ledger=json.loads(ledger_bytes); ledger['costs']['seconds']+=1
    ledger_path.write_text(json.dumps(ledger))
    assert 'cost' in ' '.join(audit(directory,frozen)['reasons'])
    ledger_path.write_bytes(ledger_bytes)
    # Changed predictions are detected independently of a hand-edited metric.
    prediction=protected._directory(frozen)/'runs'/sources['cohort']['units'][0]['run_ids'][0]/'predictions.json'
    original_prediction=prediction.read_bytes(); value=json.loads(original_prediction)
    value['predictions']['top'][0]+=1; prediction.write_text(json.dumps(value))
    assert audit(directory,frozen)['status']=='blocked'
    prediction.write_bytes(original_prediction)
    # A restored checkout needs neither the original filesystem root nor a
    # rewrite of frozen configs, selected models or the protected access chain.
    from treeneuron.protocol import FrozenProtocol
    restored=tmp_path/'restored-real'; shutil.copytree(root,restored)
    hidden=tmp_path/'hidden-real'; root.rename(hidden)
    moved=FrozenProtocol(restored/'protocol/frozen.yaml',frozen.protocol_hash,frozen.pinned_commit,frozen.checkout_commit)
    moved_dir=restored/directory.relative_to(root)
    relocated=audit(moved_dir,moved)
    assert relocated['primary']==first['primary'] and not relocated['reasons']
    assert module.run_confirmation(moved,'synthetic')==moved_dir
    hidden.rename(root)
    archive=seal_archive(directory,frozen,tmp_path/'sealed')
    # New authorized evidence_read events do not invalidate the copied prefix.
    protected.load_evaluation(frozen,run_id=sources['cohort']['units'][0]['run_ids'][0])
    # Only a verified archive for this original identity can bind an amendment.
    from treeneuron.config import canonical_hash
    archive_bytes=archive.read_bytes(); archive_value=json.loads(archive_bytes)
    archive_value['unexpected']=True; archive_value.pop('archive_sha256')
    archive_value['archive_sha256']=canonical_hash(archive_value)
    archive.write_text(json.dumps(archive_value))
    with pytest.raises(ValueError,match='archive'):
        amend(frozen,reason='Unknown archive field',replacement_study_id='toy-next',archive=archive)
    archive.write_bytes(archive_bytes)
    copied_prediction=archive.parent/'files'/prediction.relative_to(root)
    copied_bytes=copied_prediction.read_bytes(); copied_prediction.write_text('{}')
    with pytest.raises(ValueError,match='hash drift'):
        amend(frozen,reason='Damaged archive',replacement_study_id='toy-next',archive=archive)
    copied_prediction.write_bytes(copied_bytes)
    other=json.loads(frozen.path.read_text()); other['payload']['study']['study_id']='other-original'
    other['protocol_hash']=canonical_hash(other['payload'])
    other_path=root/'protocol/other.json'; other_path.write_text(json.dumps(other))
    other_protocol=FrozenProtocol(other_path,other['protocol_hash'],frozen.pinned_commit,frozen.checkout_commit)
    with pytest.raises(ValueError,match='different original protocol'):
        amend(other_protocol,reason='Wrong archive',replacement_study_id='toy-next',archive=archive)
    original_frozen=frozen.path.read_bytes()
    amendment=amend(frozen,reason='Toy analysis amendment',replacement_study_id='toy-next',archive=archive)
    assert json.loads(amendment.read_text())['archive']==dict(eligibility='sealed_saved_evidence_only',
        archive_sha256=json.loads(archive_bytes)['archive_sha256'])
    assert frozen.path.read_bytes()==original_frozen
    with pytest.raises(ValueError,match='amendment'):
        validate_frozen(frozen.path,{'root':str(root)})
    with pytest.raises(ValueError,match='live evidence'):
        seal_archive(directory,frozen,tmp_path/'too-late')
    (root/'src/model.py').write_text('# changed analysis source\n')
    git(root,'add','src/model.py'); git(root,'commit','-qm','toy source amendment')
    archived=audit_archive(archive,analysis_root=root)
    assert not archived['reasons'], archived['reasons']
    assert archived['status']=='incomplete' and archived['conclusion']['superiority'] is False
    assert archived['primary']['valid_runs']==2 and archived['primary']['primary']['mean']==first['primary']['primary']['mean']
    with pytest.raises(ValueError): protected.open_test(archived,None,run_id='unit-0-rep-0')
    copied=tmp_path/'sealed'/'files'/prediction.relative_to(root)
    copied.write_text('{}')
    assert audit_archive(archive,analysis_root=root)['status']=='blocked'


def test_metric_terms_and_resume_evidence_are_not_summary_authority(grant):
    from test_protected import predictors
    from treeneuron.evaluation import protected
    from treeneuron.analysis.audit import _validate_saved
    from treeneuron.protocol import _read, fit_variance
    root,frozen,manifest=grant; rid='unit-0-rep-0'
    protected.open_test(frozen,manifest,run_id=rid).evaluate(predictors())
    saved=protected.load_evaluation(frozen,run_id=rid)
    payload=_read(frozen.path)['payload']; unit=payload['sources']['cohort']['units'][0]
    saved['metrics']['top']['mse']+=1
    with pytest.raises(ValueError,match='metric mismatch'):
        _validate_saved(saved,unit,payload,fit_variance(root,unit))
    del saved['metrics']['top']
    with pytest.raises(ValueError,match='metric terms'):
        _validate_saved(saved,unit,payload,fit_variance(root,unit))
    protected._event(protected._directory(frozen),'operational_resume',run_id=rid,operational={'resume_run_id':'wrong'})
    with pytest.raises(ValueError,match='resume'):
        protected.load_evaluation(frozen,run_id=rid)

from test_protected import grant


def test_checkpoint_inventory_rejects_missing_terms_and_invalid_resume(tmp_path,cpu_training_evidence):
    import torch
    from test_pilot_calibrate import config
    from test_protocol import ref
    from treeneuron.config import resolve_experiment
    from treeneuron.pilot.calibrate import teacher_inputs,fit_repetition
    from treeneuron.analysis.audit import _audit_pair_inventory
    value=config(tmp_path).to_dict()
    value['execution']['backend_report']=str(cpu_training_evidence)
    value['backend']['interop_threads']=torch.get_num_interop_threads()
    c=resolve_experiment(value); _,data,_,tc=teacher_inputs(c,0)
    directory=tmp_path/'fit'; pair=fit_repetition(tc,data,directory,'fixture')
    inventory=_audit_pair_inventory(tmp_path,pair,directory)
    assert inventory['selected_models']==8 and inventory['probe_candidates']>=1
    child=Path(pair['child_runs'][0]['path']); manifest=child/'manifest.json'
    original=manifest.read_bytes(); record=json.loads(original)
    del record['method']['weight_terms']; manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='terms'):
        _audit_pair_inventory(tmp_path,pair,directory)
    manifest.write_bytes(original)
    pointer=child/'checkpoints/last.json'; pointer_value=json.loads(pointer.read_text())
    metadata=pointer.parent/pointer_value['metadata']; meta=json.loads(metadata.read_text())
    meta['state']['cursor']=-1; metadata.write_text(json.dumps(meta))
    pointer_value['sha256']=ref(tmp_path,metadata)['sha256']; pointer.write_text(json.dumps(pointer_value))
    with pytest.raises(ValueError,match='sampler|metadata'):
        _audit_pair_inventory(tmp_path,pair,directory)


def test_cli_audit_and_report_do_not_accept_summary_scores(tmp_path,monkeypatch):
    from treeneuron.cli import main
    assert main(['audit','--protocol',str(tmp_path/'absent.yaml'),'--study-dir',str(tmp_path)])==1
    output=tmp_path/'report.md'
    assert main(['report','--mode','audited','--protocol',str(tmp_path/'absent.yaml'),
                 '--study-dir',str(tmp_path),'--output',str(output)])==1
    assert 'blocked' in output.read_text()


def test_complete_negative_toy_audit_and_missing_probe(prepared,cpu_training_evidence,monkeypatch,tmp_path):
    """Eight-unit toy design seam; actual fitting graph, controlled adverse rows.

    The numeric pilot gate is replaced solely to bound this status/authority
    fixture. Predictor outputs are controlled to guarantee a negative sign; this
    is never model performance evidence. Source, frozen identities, actual model
    graphs, original access chain, metrics and auditing remain real.
    """
    import importlib
    import numpy as np
    import torch
    from dataclasses import replace
    from treeneuron import protocol
    from treeneuron.analysis.audit import audit,seal_archive,audit_archive
    from treeneuron.evaluation import protected
    module=importlib.import_module('treeneuron.evaluation.confirm')
    root,config,sources=prepared
    decision=protocol.eligible_design(sources)
    decision=replace(decision,K=8,statistical_policy=dict(decision.statistical_policy,
        inference_supported=True,bootstrap_iterations=200,endpoint_tolerance=1e20))
    monkeypatch.setattr(protocol,'eligible_design',lambda source:decision)
    monkeypatch.setattr(module,'eligible_design',lambda source:decision)
    sources=module.prepare_cohort(decision,sources,root/'negative',cohort_id='negative-fixture',root_seed=42)
    frozen=protocol.freeze(decision,sources,root/'protocol/frozen.yaml')
    original_resolve=module.resolve_experiment
    def fixture_runtime(value):
        value=dict(value); value['execution']=dict(value['execution'],backend_report=str(cpu_training_evidence))
        value['backend']=dict(value['backend'],interop_threads=torch.get_num_interop_threads())
        return original_resolve(value)
    monkeypatch.setattr(module,'resolve_experiment',fixture_runtime)
    monkeypatch.setattr(module,'_runtime_preflight',lambda c: {})
    monkeypatch.setattr(module,'load_predictors',lambda root,models:
        {role:(lambda X,role=role:np.full(len(X),1e6 if role=='top' else 0.)) for role in protocol.EVALUATION_METHODS})
    directory=module.run_confirmation(frozen,'synthetic')
    def forbidden(*args,**kwargs): raise AssertionError('New T/prediction/fit in audit')
    monkeypatch.setattr(protected,'_test_labels',forbidden)
    monkeypatch.setattr(module,'_fit_run',forbidden)
    monkeypatch.setattr(module,'load_predictors',forbidden)
    result=audit(directory,frozen)
    assert result['valid'] is True and result['status']=='completed',result['reasons']
    assert result['conclusion']['interpretation']=='negative' and result['conclusion']['superiority'] is False
    assert result['primary']['valid_teachers']==8 and result['primary']['valid_runs']==16
    ledger=protocol._read(directory/'confirmation.json')
    pair=protocol._read(protocol._reference(root,ledger['runs'][0]['attempts'][0]['pair']))
    greedy=Path(pair['greedy_report']); original=greedy.read_bytes(); value=json.loads(original)
    value['trials'].pop(); greedy.write_text(json.dumps(value))
    invalid=audit(directory,frozen)
    assert invalid['valid'] is False and invalid['status']=='blocked'
    assert 'probe' in ' '.join(invalid['reasons']).lower()
    greedy.write_bytes(original)
    bundle=seal_archive(directory,frozen,tmp_path/'negative-archive')
    archived=audit_archive(bundle,analysis_root=root)
    assert archived['valid'] is True and archived['status']=='exploratory',archived['reasons']
    assert not archived['conclusion']['superiority']


def test_native_evidence_read_failure_is_a_blocked_audit(candidate,monkeypatch):
    import importlib
    from treeneuron.protocol import freeze
    module=importlib.import_module('treeneuron.analysis.audit')
    root,decision,sources,_=candidate
    frozen=freeze(decision,sources,root/'protocol/frozen.yaml')
    directory=root/'protocol/confirmation'/frozen.protocol_hash; directory.mkdir(parents=True)
    (directory/'confirmation.json').write_text(json.dumps({'protocol_hash':frozen.protocol_hash}))
    def failed_read(*args): raise RuntimeError('Native checkpoint reader rejected malformed tensors')
    monkeypatch.setattr(module,'_account',failed_read)
    result=module.audit(directory,frozen)
    assert result['status']=='blocked' and not result['valid']
    assert 'malformed tensors' in ' '.join(result['reasons'])


def test_amendment_explicitly_records_absent_archive_eligibility(candidate):
    from treeneuron.protocol import freeze,amend
    root,decision,sources,_=candidate
    frozen=freeze(decision,sources,root/'protocol/frozen.yaml')
    path=amend(frozen,reason='Toy amendment without seal',replacement_study_id='toy-next')
    value=json.loads(path.read_text())
    assert value['archive']['eligibility']=='unavailable_no_archive_supplied'
    assert value['archive']['archive_sha256'] is None


def test_amendment_rejects_missing_supplied_archive(candidate,tmp_path):
    from treeneuron.protocol import freeze,amend
    root,decision,sources,_=candidate
    frozen=freeze(decision,sources,root/'protocol/frozen.yaml')
    with pytest.raises(ValueError,match='evidence'):
        amend(frozen,reason='Toy archive binding',replacement_study_id='toy-next',archive=tmp_path/'missing')
    assert not (root/'protocol/amendments').exists()


def test_outer_ledger_status_and_enclosing_work_are_authoritative(tmp_path,monkeypatch):
    import importlib
    module=importlib.import_module('treeneuron.analysis.audit')
    monkeypatch.setattr(module,'preparation_receipt',lambda *args:None)
    root=tmp_path; study=root/'protocol/confirmation/toy'; study.mkdir(parents=True)
    payload={'sources':{'cohort':{'units':[{'unit_id':'unit-0','run_ids':['run-0']}]}},
        'design':{'statistical_policy':{'retries':1},'forecast':{'seconds':10.,'storage_bytes':1000}}}
    from treeneuron.evaluation.confirm import SEED_POLICY
    ledger=dict(schema_version=1,study='synthetic',protocol_hash='toy',seed_policy=SEED_POLICY,
        status='incomplete',runs=[dict(unit_id='unit-0',repetition=0,run_id='run-0',status='failed',
        attempts=[dict(index=0,status='failed',phase='fitting_and_evaluation',
        path=str((study/'run-0/attempt-0').relative_to(root)),seconds=.5,error='Toy failure')])],
        invocations=[{'seconds':1.}],costs=dict(seconds=1.,preparation_seconds=None,combined_seconds=None,
        forecast_seconds=10.,seconds_deviation=None,forecast_storage_bytes=1000,
        hard_interrupt_cost_unknown=False,aggregation='disjoint invocation',boundary='active invocation',
        forecast_boundary='preparation and invocations',retained_bytes=0,storage_deviation=None,
        combined_reason='Preparation not measured'))
    assert module._account(payload,ledger,root,study)['failed_attempts']==1
    ledger['status']='invented'
    with pytest.raises(ValueError,match='ledger status'): module._account(payload,ledger,root,study)
    ledger['status']='completed'
    with pytest.raises(ValueError,match='completed ledger'): module._account(payload,ledger,root,study)
    ledger['status']='incomplete'; ledger['runs'][0]['attempts'][0]['seconds']=2.
    with pytest.raises(ValueError,match='enclosing invocation'): module._account(payload,ledger,root,study)
    ledger['unmeasured_interruption']=True; ledger['costs']['hard_interrupt_cost_unknown']=True
    # Missing old invocation duration is unknown, not a zero-duration bound.
    assert module._account(payload,ledger,root,study)['costs']['combined_seconds'] is None


@pytest.mark.parametrize('outer_status',['running','interrupted','incomplete'])
def test_nonterminal_publication_does_not_claim_complete(outer_status):
    from test_primary import rows,policy
    from treeneuron.analysis.primary import summarize_cohort
    from treeneuron.protocol import EVALUATION_METHODS
    predictions,runs=rows((-1.,-2.,-3.),2)
    for saved in predictions.values():
        saved['predictions'].update({role:[0.,0.] for role in EVALUATION_METHODS if role not in ('top','random')})
    payload={'design':{'K':3,'statistical_policy':policy(),'branch':'estimation'},'sources':{'scientific':{}}}
    result=summarize_cohort(payload,{'status':outer_status,'costs':{}},runs,predictions,'toy')
    assert result['status']=='incomplete' and result['primary']['superiority'] is False
