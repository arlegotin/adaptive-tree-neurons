"""Closed artifacts need source, recipe, worker and sealed-role lineage."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from treeneuron.config import canonical_hash
from test_native_qualification_uncapped import prepared_config,resources


def api():
    assert importlib.util.find_spec('treeneuron.research.native_generation_v2') is not None,'Missing native generation-chain verifier'
    from treeneuron.research import native_generation_v2
    return native_generation_v2.audit_native_generation


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actual(tmp_path,monkeypatch):
    from treeneuron.research.qualify import qualify
    path,value=prepared_config(tmp_path/'input');resources(monkeypatch)
    out=tmp_path/'qualification';qualify(path,out)
    attempt=next((out/'queue/jobs').glob('*/attempt-0'))
    saved=json.loads((attempt/'native-fit-result.json').read_text())
    request=attempt/'input/request.json';config=json.loads(request.read_text())['config']
    args=dict(request_path=request,expected_request_sha256=sha(request),native_reference=saved['recovery_reference'],
        prepared_root=Path(value['datasets']['energy']['root']),
        expected_manifest_sha256=value['datasets']['energy']['manifest_sha256'],outer_fold=0,
        expected_fit_seed=17,expected_configuration=config)
    return args,attempt


def rebind_native(args,attempt):
    from treeneuron.research.workqueue import _inventory
    args['native_reference']['inventory_sha256']=canonical_hash(_inventory(attempt/'native'))


def test_actual_native_P13_chain_aligns_original_roles_and_closed_model_without_V_labels(tmp_path,monkeypatch):
    fn=api();args,attempt=actual(tmp_path,monkeypatch);out=fn(**args)
    assert out['status']=='authenticated_source_bound_chain' and out['model_generation_chain_authenticated'] is True
    assert out['family_id']=='P13' and out['fit_seed']==17 and out['query_rows']>0
    assert out['prediction']['sha256']==sha(attempt/'native/worker/predictions.npz')
    assert out['quality_eligible'] is out['native_workload_qualified'] is out['T_accessed'] is out['V_labels_read'] is False
    assert out['model_training_independently_reexecuted'] is out['HPO_journal_authenticated'] is False


@pytest.mark.parametrize('change',['seed','recipe','manifest','inventory','source','not_completed','unmanaged_command','secondary'])
def test_changed_or_self_rehashed_native_lineage_cannot_become_a_valid_chain(tmp_path,monkeypatch,change):
    fn=api();args,attempt=actual(tmp_path,monkeypatch)
    if change=='seed':args['expected_fit_seed']=18
    elif change=='recipe':args['expected_configuration']['estimator']='ridge'
    elif change=='manifest':args['expected_manifest_sha256']='0'*64
    elif change=='inventory':(attempt/'native/process/stderr.log').write_text('uncommitted old bytes')
    elif change=='secondary':args['outer_fold']=1
    elif change=='source':
        request=args['request_path'];value=json.loads(request.read_text());value['scientific_source_sha256']='0'*64
        request.write_text(json.dumps(value));args['expected_request_sha256']=sha(request)
    elif change=='not_completed':
        path=attempt/'native/result.json';value=json.loads(path.read_text());value['status']='resource_exhausted'
        path.write_text(json.dumps(value));rebind_native(args,attempt)
    else:
        path=attempt/'native/process/process.json';value=json.loads(path.read_text());value['argv'][2]='unmanaged-code'
        path.write_text(json.dumps(value));native=attempt/'native/result.json';value=json.loads(native.read_text())
        value['process_receipt_sha256']=sha(path);native.write_text(json.dumps(value));rebind_native(args,attempt)
    with pytest.raises(ValueError):fn(**args)


def test_a_real_shortened_neural_fixture_cannot_pass_as_native_generation(tmp_path,monkeypatch):
    import shutil
    import sys
    from treeneuron.research.native_fit_v2 import run_native_fit
    from test_ordinary_neural_contract import config
    fn=api();args,attempt=actual(tmp_path,monkeypatch)
    request=json.loads(args['request_path'].read_text());request['family_id']='P03';request['job_id']='fixture/P03/short'
    request['config']=config('P03','engineering_fixture',max_epochs=2,patience=2)
    request['config']['feature_schema']=args['expected_configuration']['feature_schema']
    inputs=tmp_path/'short-inputs';inputs.mkdir()
    for ref in request['files'].values():shutil.copyfile(args['request_path'].parent/ref['path'],inputs/ref['path'])
    path=inputs/'request.json';path.write_text(json.dumps(request,sort_keys=True)+'\n')
    result=run_native_fit(path,expected_request_sha256=sha(path),output=tmp_path/'short-native',interpreter=sys.executable)
    assert result['status']=='completed_ineligible'
    with pytest.raises(ValueError):fn(**dict(args,request_path=path,expected_request_sha256=sha(path),
        native_reference=result['recovery_reference'],expected_configuration=request['config']))


def test_late_atomic_replacement_of_prepared_role_cannot_escape_end_of_audit_check(tmp_path,monkeypatch):
    from treeneuron.research.transfer import TransferStore
    fn=api();args,attempt=actual(tmp_path,monkeypatch);original=TransferStore._access
    def change_sealed_role_after_initial_hash(store,*values,**kwargs):
        original(store,*values,**kwargs)
        # Atomic replacement separates the original from its staged hard link.
        # The fit input remains valid, but the authoritative prepared bytes drift.
        path=args['prepared_root']/'outer-0/F.npz';temporary=path.with_name('controlled-replacement.tmp')
        temporary.write_bytes(b'Changed after the initial prepared-role verification');temporary.replace(path)
    monkeypatch.setattr(TransferStore,'_access',change_sealed_role_after_initial_hash)
    with pytest.raises(ValueError):fn(**args)
