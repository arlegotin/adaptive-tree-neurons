"""A companion admission cannot inherit authority from an engineering worker."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.qualified_native_trial_v2') is not None,'Missing registry-qualified native trial bridge'
    from treeneuron.research import qualified_native_trial_v2
    return qualified_native_trial_v2


def reference(root):
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    return dict(root=str(root.resolve()),inventory_sha256=canonical_hash(_inventory(root)))


def test_handfilled_registry_cannot_construct_a_production_callback_or_start_a_worker(tmp_path,monkeypatch):
    module=api();root=tmp_path/'pretend-registry';root.mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(schema='research/v2-native-qualification-registry',native_workload_qualified=True)))
    def forbidden(**kw):raise AssertionError('Invalid registry reached native callback construction')
    monkeypatch.setattr(module,'InspectedNativeTrialBridge',forbidden)
    with pytest.raises(ValueError,match='schema|registry|authority'):
        module.QualifiedNativeTrialBridge(qualification_reference=reference(root),plan=None,search_directory=tmp_path/'search')
    assert not (tmp_path/'search').exists()


def test_immutable_prerequisite_byte_guard_rejects_later_drift(tmp_path):
    module=api();root=tmp_path/'partial-proof';root.mkdir();path=root/'facts.json';path.write_text('{}')
    # Explicit partial byte guard, never a fake complete qualified registry.
    ref=reference(root);module._check_references([ref]);path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='bytes|proof'):
        module._check_references([ref])


@pytest.fixture(scope='module')
def candidates(tmp_path_factory):
    cached=[]
    def get():
        if not cached:
            from test_native_trial_bridge import setup,bridge,run
            from treeneuron.research.native_trial_bridge_v2 import InspectedNativeTrialBridge
            root=tmp_path_factory.mktemp('candidate-admission-components')
            with pytest.MonkeyPatch.context() as patch:args,_,_,plan=setup(root,patch)
            search=root/'search';result=run(search,plan,bridge(InspectedNativeTrialBridge,args,plan,search))
            assert result['winner'] is None and result['attempts_paid']==2
            keys=('plan_hash','source_sha256','candidate_id','candidate_origin','raw_configuration',
                'compiled_configuration','official_schedule_reference','native_hpo_seed','fit_seed')
            for i in range(2):
                attempt=search/f'attempt-{i:03}';packet=json.loads((attempt/'bridge.json').read_text())
                cached.append(dict(attempt_directory=attempt,expected_trial={k:packet[k] for k in keys},
                    prepared_root=args['prepared_root'],expected_manifest_sha256=args['expected_manifest_sha256'],outer_fold=0))
        return cached
    return get


@pytest.mark.parametrize('index',[0,1])
def test_actual_trial_component_checks_native_generation_workload_and_V_without_global_admission(candidates,index):
    module=api();row=candidates()[index];path=row['attempt_directory']/'bridge.json';before=path.read_bytes()
    proof=module.inspect_completed_native_trial(**row)
    assert proof['candidate_evidence_authenticated'] is True and proof['inspected_V_score']>=0
    assert proof['native_workload_qualified'] is proof['quality_eligible'] is proof['T_accessed'] is False
    assert path.read_bytes()==before


@pytest.mark.parametrize('fault',['plan_hash','fit_seed','candidate_id','raw_configuration',
    'compiled_configuration','native_request_sha256','quality_eligible'])
def test_changed_trial_proposal_or_unearned_authority_cannot_supply_companion_evidence(candidates,fault):
    module=api();row=candidates()[0];path=row['attempt_directory']/'bridge.json';before=path.read_bytes()
    try:
        packet=json.loads(before)
        if fault=='quality_eligible':packet[fault]=True
        elif fault=='fit_seed':packet[fault]=18
        elif fault in ('raw_configuration','compiled_configuration'):packet[fault]=copy.deepcopy(candidates()[1]['expected_trial'][fault])
        else:packet[fault]='0'*64
        path.write_text(json.dumps(packet))
        with pytest.raises(ValueError,match='trial|proposal|authority|request|configuration'):
            module.inspect_completed_native_trial(**row)
    finally:path.write_bytes(before)


@pytest.mark.parametrize('fault',[None,'root','manifest','secondary','source','runtime','split','support'])
def test_pure_procedure_binding_checks_metadata_before_worker_inputs(tmp_path,fault):
    module=api()
    from treeneuron.data.tabular import DATASETS
    from treeneuron.research.native_verification_v2 import verification_identity
    from treeneuron.research.search_supports_v2 import family_search_spec
    from treeneuron.research.search_controller_v2 import make_search_plan
    identity=verification_identity();schema=DATASETS['energy'];root=tmp_path/'official-working-energy'
    spec=family_search_spec('P13',dict(columns=list(schema.features),categorical=list(schema.categorical)))
    value=dict(job_id='fixture/P13',family_id='P13',search_repeat=0,source_sha256=identity['scientific_source_sha256'],
        runtime_sha256=identity['test_and_lock_files']['locks/research-v2/core.json'],support_sha256=spec['support_hash'],split_sha256='b'*64)
    if fault in ('source','runtime','split','support'):value[fault+'_sha256']='0'*64
    plan=make_search_plan(identity=value,fit_seed=17,hpo_seed32=91,anchors=[a['config'] for a in spec['anchors']],
        mode='finite',support=spec['support'],candidates=spec['finite_configurations'][1:],max_attempts=2,deadline_ns=None,prefixes_ns=[])
    # Metadata boundary only: no complete registry, proof readers or fake admission.
    partial=dict(identity=identity,inputs=dict(prepared_roots=dict(energy=str(root))),
        preparation=dict(datasets=dict(energy=dict(manifest_file_sha256='a'*64,split_hashes={'0':'b'*64}))))
    args=dict(plan=plan,prepared_root=tmp_path/'wrong' if fault=='root' else root,
        expected_manifest_sha256='0'*64 if fault=='manifest' else 'a'*64,outer_fold=1 if fault=='secondary' else 0)
    assert hasattr(module,'_bind_procedure'),'Missing pre-worker procedure metadata boundary'
    if fault is None:assert module._bind_procedure(partial,args)=='energy'
    else:
        with pytest.raises(ValueError,match='source|runtime|working|primary|manifest|split|support'):
            module._bind_procedure(partial,args)
    assert not root.exists()
