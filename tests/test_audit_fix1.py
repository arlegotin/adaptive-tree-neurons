"""Scoped Task13 review regressions; bounded engineering fixtures only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from test_protocol import repository,prepared,candidate,git


@pytest.mark.parametrize('absolute',[True,False])
def test_receipt_io_survives_unavailable_original_root(tmp_path,absolute):
    from test_pilot_calibrate import config
    original=(tmp_path/'original').resolve(); original.mkdir()
    receipt=original/'environment/backend.json'; receipt.parent.mkdir()
    source=Path('environment/evidence7-calibration-fix1/search-backend.json').resolve()
    shutil.copyfile(source,receipt)
    value=config(tmp_path).to_dict(); value['growth']['m_requested']=1
    value['execution']['backend_report']=str(receipt) if absolute else 'environment/backend.json'
    (original/'config.json').write_text(json.dumps(value))
    restored=tmp_path/'restored'; shutil.copytree(original,restored)
    original.rename(tmp_path/'unavailable')
    script=r'''
import importlib,json,sys
from pathlib import Path
from treeneuron.config import resolve_experiment,canonical_hash
from treeneuron.locators import relocated,producer_directory
from treeneuron.protocol import FrozenProtocol,_ref
from treeneuron.pilot.calibrate import _runtime_preflight,fit_repetition
from treeneuron.data.synthetic import make_from_config
original,root=map(Path,sys.argv[1:])
value=json.loads((root/'config.json').read_text()); c=resolve_experiment(value)
with relocated(original,root),producer_directory(root):
    preflight=_runtime_preflight(c)
    assert Path(preflight['receipt_path'])==root/'environment/backend.json'
    _,data=make_from_config(c)
    pair=fit_repetition(c,data,root/'fit',canonical_hash(value))
    assert pair['status']=='completed' and len(pair['models'])==8
    # Source/protocol construction is isolated here; actual terminal preflight
    # and its receipt I/O remain unpatched. No label access or further fitting.
    module=importlib.import_module('treeneuron.evaluation.confirm')
    (root/'raw.json').write_text(json.dumps({'config':value}))
    payload={'sources':{'root':str(original),'raw':_ref(root,root/'raw.json'),'cohort':{'units':[]}}}
    path=root/'protocol/frozen.yaml';path.parent.mkdir()
    path.write_text(json.dumps({'payload':payload}))
    frozen=FrozenProtocol(path,'toy','toy','toy')
    directory=root/'protocol/confirmation/toy';directory.mkdir(parents=True)
    (directory/'confirmation.json').write_text(json.dumps({'protocol_hash':'toy','status':'completed','runs':[]}))
    module._payload=lambda protocol:payload
    assert module.run_confirmation(frozen,'synthetic')==directory
    assert json.loads((root/'config.json').read_text())==value
    # A relocated receipt still undergoes ordinary producer validation in fit.
    (root/'environment/backend.json').write_text('{}')
    try: fit_repetition(c,data,root/'bad',canonical_hash(value))
    except (ValueError,KeyError): pass
    else: raise AssertionError('Corrupt receipt accepted')
'''
    env=dict(os.environ,PYTHONPATH=str(Path('src').resolve()),PYTORCH_ENABLE_MPS_FALLBACK='0')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): env[name]='1'
    result=subprocess.run([sys.executable,'-c',script,str(original),str(restored.resolve())],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def analysis_checkout(root):
    """A real committed copy of the imported package, not a source-check bypass."""
    import treeneuron
    shutil.copytree(Path(treeneuron.__file__).parent,root/'src/treeneuron',ignore=shutil.ignore_patterns('__pycache__'))
    git(root,'add','src/treeneuron');git(root,'commit','-qm','same-source installed analysis fixture')
    return root


def test_analysis_identity_binds_imported_bytes_to_verified_checkout(repository):
    from treeneuron.analysis.audit import _analysis_identity
    from treeneuron.train import source_hash
    with pytest.raises(ValueError,match='imported package'):
        _analysis_identity(repository)
    analysis_checkout(repository)
    identity=_analysis_identity(repository)
    assert identity['analysis_package_sha256']==source_hash()
    # A different clean committed source must not attest the running package.
    copied=repository/'src/treeneuron/analysis/audit.py'
    copied.write_text(copied.read_text()+'\n# Different verified source\n')
    git(repository,'add','src/treeneuron/analysis/audit.py');git(repository,'commit','-qm','different package')
    with pytest.raises(ValueError,match='imported package'):
        _analysis_identity(repository)

from test_train import cpu_training_evidence


def test_real_partial_failure_inventory_survives_successful_retry(prepared,cpu_training_evidence,monkeypatch,tmp_path):
    import importlib
    import torch
    from treeneuron import train
    from treeneuron.protocol import eligible_design,freeze
    from treeneuron.analysis.audit import audit,seal_archive
    module=importlib.import_module('treeneuron.evaluation.confirm')
    root,config,sources=prepared;decision=eligible_design(sources)
    sources=module.prepare_cohort(decision,sources,root/'cohort',cohort_id='partial-failure',root_seed=17)
    frozen=freeze(decision,sources,root/'protocol/frozen.yaml')
    resolve=module.resolve_experiment
    def fixture_runtime(value):
        value=dict(value);value['execution']=dict(value['execution'],backend_report=str(cpu_training_evidence))
        value['backend']=dict(value['backend'],interop_threads=torch.get_num_interop_threads())
        return resolve(value)
    monkeypatch.setattr(module,'resolve_experiment',fixture_runtime)
    monkeypatch.setattr(module,'_runtime_preflight',lambda c:{})
    fit=module._fit_run
    def one_run(p,payload,unit,repetition,directory):
        if unit['unit_id']!='unit-0' or repetition!=0: raise RuntimeError('Deliberately omitted toy numerical work')
        return fit(p,payload,unit,repetition,directory)
    monkeypatch.setattr(module,'_fit_run',one_run)
    original_parent=train.fit_parent; first=True
    def partial(*args,**kwargs):
        nonlocal first
        result=original_parent(*args,**kwargs)
        if first:
            first=False
            raise RuntimeError('Injected after real parent checkpoint publication')
        return result
    monkeypatch.setattr(train,'fit_parent',partial)
    directory=module.run_confirmation(frozen,'synthetic')
    ledger=json.loads((directory/'confirmation.json').read_text())
    retried=ledger['runs'][0]
    assert retried['status']=='completed' and [a['status'] for a in retried['attempts']]==['failed','completed']
    failed=retried['attempts'][0];inventory=failed['retained']
    assert inventory['state']=='captured' and inventory['files']
    assert ledger['runs'][1]['attempts'][0]['retained']['state']=='not_materialized'
    baseline=audit(directory,frozen)
    assert baseline['status']=='incomplete' and not baseline['reasons']
    for suffix in ('manifest.json','checkpoints/last.json'):
        name=next(name for name in inventory['files'] if name.endswith(suffix))
        path=root/failed['path']/name;original=path.read_bytes()
        for mutation in ('delete','corrupt'):
            if mutation=='delete':path.unlink()
            else:path.write_bytes(b'{}')
            result=audit(directory,frozen)
            assert result['status']=='blocked' and 'retained' in ' '.join(result['reasons']).lower()
            with pytest.raises(ValueError,match='live evidence'):
                seal_archive(directory,frozen,tmp_path/f'blocked-{suffix.replace("/","-")}-{mutation}')
            path.write_bytes(original)
    # Direct ledger fixture mutation models a hard-stop observation, not a new
    # numerical experiment. Original terminal evidence remains byte-identical.
    unknown_root=tmp_path/'unknown-repo';shutil.copytree(root,unknown_root)
    unknown_dir=unknown_root/directory.relative_to(root)
    unknown_ledger=json.loads((unknown_dir/'confirmation.json').read_text())
    prior=unknown_ledger['runs'][0]['attempts'][0]
    prior.update(status='interrupted',seconds=None,duration_incomplete=True,error='Simulated unknown hard stop')
    prior['retained'].update(state='unknown_hard_stop',reason='Observed after process interruption; earlier materialized inventory is unknown')
    unknown_ledger['unmeasured_interruption']=True
    unknown_ledger['costs'].update(combined_seconds=None,seconds_deviation=None,hard_interrupt_cost_unknown=True,
                                  combined_reason='Hard interruption duration unknown')
    (unknown_dir/'confirmation.json').write_text(json.dumps(unknown_ledger))
    # This copy represents the state before publication of a terminal receipt.
    (unknown_dir/'archive-completion.json').unlink()
    from treeneuron.protocol import FrozenProtocol
    unknown_protocol=FrozenProtocol(unknown_root/'protocol/frozen.yaml',frozen.protocol_hash,frozen.pinned_commit,frozen.checkout_commit)
    unknown=audit(unknown_dir,unknown_protocol)
    assert unknown['status']=='incomplete' and not unknown['reasons']
    assert unknown['accounting']['retention']['status']=='unknown_hard_stop'
    assert unknown['primary']['valid_runs']==1
    unknown_archive=seal_archive(unknown_dir,unknown_protocol,tmp_path/'unknown-archive')
    assert json.loads(unknown_archive.read_text())['live_audit']['accounting']['retention']['status']=='unknown_hard_stop'
    terminal=(directory/'confirmation.json').read_bytes()
    completion=(directory/'archive-completion.json').read_bytes()
    assert baseline['archive_eligibility']['status']=='unsealed'
    external=seal_archive(directory,frozen,tmp_path/'valid-external')
    external_status=audit(directory,frozen)['archive_eligibility']
    assert external_status['status']=='external_unbound' and external_status['reason']
    assert external_status['bindings'][0]['path'] is None
    internal=seal_archive(directory,frozen,root/'audit/archives/partial')
    status=audit(directory,frozen)['archive_eligibility']
    assert status['status']=='bound'
    assert next(b for b in status['bindings'] if b['scope']=='repo_local')['path']=='audit/archives/partial/archive.json'
    assert (directory/'confirmation.json').read_bytes()==terminal
    assert (directory/'archive-completion.json').read_bytes()==completion
    # Binding verification never trusts a digest without its copied inventory.
    archived=next((internal.parent/'files').rglob('teacher.json')); saved=archived.read_bytes()
    archived.write_bytes(b'{}')
    assert audit(directory,frozen)['status']=='blocked'
    archived.write_bytes(saved)
    # A local binding retains authority after relocation with original absent.
    restored=tmp_path/'bound-restored';shutil.copytree(root,restored)
    hidden=tmp_path/'bound-hidden';root.rename(hidden)
    moved=FrozenProtocol(restored/'protocol/frozen.yaml',frozen.protocol_hash,frozen.pinned_commit,frozen.checkout_commit)
    relocated=audit(restored/directory.relative_to(root),moved)
    assert not relocated['reasons'] and relocated['archive_eligibility']['status']=='bound'
    hidden.rename(root)
    # Current analysis executes equal imported source, but never loads an old
    # retained checkpoint. Exact copied bytes and original live audit attest it.
    analysis_checkout(root)
    audit_module=importlib.import_module('treeneuron.analysis.audit')
    monkeypatch.setattr(audit_module,'_validate_checkpoint',lambda *a:pytest.fail('Archive loaded a checkpoint'))
    for bundle in (internal,external,unknown_archive):
        archived_result=audit_module.audit_archive(bundle,analysis_root=root)
        assert archived_result['status']=='incomplete' and not archived_result['reasons'],archived_result['reasons']
        if bundle==unknown_archive:
            assert archived_result['accounting']['retention']['status']=='unknown_hard_stop'
            assert archived_result['primary']['valid_runs']==1 and not archived_result['valid']



def test_unknown_hard_stop_retains_observation_without_certifying_history(tmp_path):
    from treeneuron.evaluation.confirm import _capture_attempt_inventory
    from treeneuron.analysis.audit import _audit_retained_attempt,conclude
    directory=tmp_path/'attempt';directory.mkdir();observed=directory/'observed.json';observed.write_text('{}')
    inventory=_capture_attempt_inventory(directory,hard_stop=True)
    assert _audit_retained_attempt(directory,inventory)=='unknown_hard_stop'
    conclusion=conclude({'status':'completed'},[],{'status':'unknown_hard_stop','unknown_attempts':['run-0:0']})
    assert conclusion['status']=='incomplete' and not conclusion['superiority']
    observed.write_text('changed')
    with pytest.raises(ValueError,match='Retained attempt'):
        _audit_retained_attempt(directory,inventory)


def test_terminal_completion_records_explicit_unsealed_eligibility(tmp_path):
    from treeneuron.archive_binding import completion_record,archive_eligibility
    directory=tmp_path/'protocol/confirmation/toy';directory.mkdir(parents=True)
    ledger=directory/'confirmation.json';ledger.write_text(json.dumps({'protocol_hash':'toy','status':'completed'}))
    original=ledger.read_bytes()
    completion_record(directory,'toy')
    receipt=directory/'archive-completion.json';before=receipt.read_bytes()
    completion_record(directory,'toy')
    assert receipt.read_bytes()==before and ledger.read_bytes()==original
    value=archive_eligibility(directory,tmp_path,'toy')
    assert value['status']=='unsealed' and value['reason'] and value['bindings']==[]
    ledger.write_text('{}')
    with pytest.raises(ValueError,match='completion'):
        archive_eligibility(directory,tmp_path,'toy')


def test_archive_locator_is_idempotent_inside_original_root(tmp_path):
    from treeneuron.locators import relocated,locate
    original=tmp_path/'repo';current=original/'audit/archives/example/files'
    with relocated(original,current):
        mapped=locate(original/'protocol/frozen.yaml')
        assert mapped==current/'protocol/frozen.yaml'
        assert locate(mapped)==mapped


def test_retained_partial_prediction_bytes_are_immutable_but_later_publication_is_allowed(tmp_path):
    from treeneuron.evaluation.confirm import _capture_attempt_inventory
    from treeneuron.analysis.audit import _audit_retained_attempt
    directory=tmp_path/'attempt';access=tmp_path/'access';access.mkdir()
    link=access/'link.json';link.write_text('{}')
    inventory=_capture_attempt_inventory(directory,protected_directory=access)
    # An inference-only retry may add a prediction after the failure snapshot.
    prediction=access/'predictions.json';prediction.write_text('{}')
    assert _audit_retained_attempt(directory,inventory,protected_directory=access)=='not_materialized'
    observed=_capture_attempt_inventory(directory,protected_directory=access,hard_stop=True)
    prediction.write_text('changed')
    with pytest.raises(ValueError,match='Retained protected'):
        _audit_retained_attempt(directory,observed,protected_directory=access)
    prediction.write_text('{}');link.unlink()
    with pytest.raises(ValueError,match='Retained protected'):
        _audit_retained_attempt(directory,inventory,protected_directory=access)


@pytest.mark.parametrize('durations',[
    [4.240696667,4.2385195,4.27692,4.471023208],
    [1e308,1e308],
])
def test_work_receipt_exact_prefix_and_overflow_classification(tmp_path,durations):
    import math,time
    from treeneuron.analysis.audit import _finish_audit_work,conclude
    from treeneuron.config import canonical_hash
    work=tmp_path/'analysis-work';work.mkdir()
    report={};digest=canonical_hash(report);report['report_sha256']=digest
    (tmp_path/f'primary-{digest}.json').write_text(json.dumps(report))
    files=[]
    for index,seconds in enumerate(durations):
        path=work/f'{len(durations)-index}.json';files.append(path)
        path.write_text(json.dumps(dict(schema_version=1,protocol_hash='toy',report_sha256=digest,
                                       previous_analysis_seconds=math.fsum(durations[:index]),seconds=seconds)))
    result=dict(protocol_hash='toy',primary=None,reasons=[],accounting={'costs':{'combined_seconds':0}},
                conclusion=conclude(None,[]),status='incomplete',valid=False)
    result['audit_sha256']=canonical_hash(result)
    _finish_audit_work(tmp_path,result,'clock',time.perf_counter_ns(),set(files))
    if durations[0]==1e308:
        assert result['status']=='blocked' and 'overflow' in ' '.join(result['reasons']).lower()
        assert json.loads((tmp_path/'audit-work/clock.json').read_text())['total_study_work_seconds'] is None
    else:
        assert not result['reasons']


def test_analysis_clock_writer_is_independent_of_filesystem_order(tmp_path):
    import math
    from treeneuron.analysis.primary import _previous_analysis_seconds
    durations=[4.240696667,4.2385195,4.27692]
    paths=[]
    for index,seconds in enumerate(durations):
        path=tmp_path/f'{index}.json';path.write_text(json.dumps({'seconds':seconds}));paths.append(path)
    class Receipts:
        def __init__(self,paths):self.paths=paths
        def glob(self,pattern):return iter(self.paths)
    assert _previous_analysis_seconds(Receipts(paths))==_previous_analysis_seconds(Receipts(paths[::-1]))==math.fsum(durations)


@pytest.mark.parametrize('base,outer,standalone,overflow',[
    (0.,1e308,1e308,True),
    (1e308,1e308,0.,True),
    (2.,3.,5.,False),
])
def test_final_work_total_cross_category_overflow_is_blocked(tmp_path,base,outer,standalone,overflow):
    import math,time
    from treeneuron.analysis.audit import _finish_audit_work,conclude
    from treeneuron.config import canonical_hash
    analysis=tmp_path/'analysis-work';analysis.mkdir()
    audits=tmp_path/'audit-work';audits.mkdir()
    report={};digest=canonical_hash(report);report['report_sha256']=digest
    (tmp_path/f'primary-{digest}.json').write_text(json.dumps(report))
    prior=analysis/'standalone.json'
    prior.write_text(json.dumps(dict(schema_version=1,protocol_hash='toy',report_sha256=digest,
                                    previous_analysis_seconds=0.,seconds=standalone)))
    (audits/'prior.json').write_text(json.dumps(dict(schema_version=1,protocol_hash='toy',seconds=outer,
                                                  nested_analysis_receipts=[])))
    result=dict(protocol_hash='toy',primary=None,reasons=[],accounting={'costs':{'combined_seconds':base}},
                conclusion=conclude(None,[]),status='incomplete',valid=False)
    result['audit_sha256']=canonical_hash(result)
    _finish_audit_work(tmp_path,result,'current',time.perf_counter_ns(),{prior})
    receipt=json.loads((audits/'current.json').read_text())
    snapshot=json.loads((tmp_path/f'audit-{result["audit_sha256"]}.json').read_text())
    assert snapshot==result and receipt['audit_sha256']==result['audit_sha256']
    if overflow:
        assert result['status']=='blocked' and 'overflow' in ' '.join(result['reasons']).lower()
        assert receipt['total_study_work_seconds'] is None and receipt['total_reason']
    else:
        assert not result['reasons'] and receipt['total_reason'] is None
        assert receipt['total_study_work_seconds']==math.fsum((base,outer,standalone,receipt['seconds']))
