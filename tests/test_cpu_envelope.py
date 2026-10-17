"""Source-bound CPU envelope correctness, not predictive study evidence."""
import pytest


def test_new_logical_batch_is_executable_policy_and_causal_probe():
    from treeneuron.train import TrainingPolicy, SelectiveRegressor
    from treeneuron.baselines.causal_evidence import probe_causal_cpu
    policy=TrainingPolicy(256,256,1,1,1)
    assert policy.logical_batch_size==policy.physical_batch_size
    receipt=probe_causal_cpu(SelectiveRegressor(16,64),batch_size=256)
    assert receipt['reload_next_update_max_abs']==0


from test_train import cpu_training_evidence

def test_numerical_receipt_uses_serial_interop_runtime(cpu_training_evidence):
    import json
    report=json.loads(cpu_training_evidence.read_text())
    assert report['backend_probes']['probes']['cpu']['environment']['interop_threads']==1


@pytest.fixture(scope='session')
def cpu_envelope_receipt(tmp_path_factory,cpu_training_evidence):
    import json
    from treeneuron.cpu_envelope import probe_cpu_envelope
    measured=probe_cpu_envelope()
    assert measured['status']=='passed',measured.get('reason')
    report=json.loads(cpu_training_evidence.read_text())
    report['cpu_envelope']=measured
    path=tmp_path_factory.mktemp('cpu-envelope')/'report.json'
    path.write_text(json.dumps(report))
    return path


def test_real_logical_sweep_keeps_density_tail_and_saturation(cpu_envelope_receipt):
    import json
    from treeneuron.cpu_envelope import validate_cpu_envelope
    r=json.loads(cpu_envelope_receipt.read_text())['cpu_envelope']
    validate_cpu_envelope(r)
    for n in (577,2306,9224):
        rows=[v for v in r['measurements'] if v['n_fit']==n and v['phase']=='base_epoch']
        assert [v['logical_batch'] for v in rows]==[128,256,512,1024,2048,4096,8192,16384,32768]
        assert rows[-1]['actual_max_rows']==rows[-2]['actual_max_rows']==n
        assert rows[-1]['updates_per_epoch']==rows[-2]['updates_per_epoch']==1
        assert rows[0]['tail_rows']==n%128
    assert r['selection_claim']=='finite candidates only; no universal throughput optimum'


@pytest.mark.parametrize('batch',[128,256,512,1024,2048,4096,8192,16384])
def test_high_density_parent_growth_control_checkpoint_paths(tmp_path,monkeypatch,cpu_envelope_receipt,batch):
    import copy,json
    import numpy as np
    import torch
    from treeneuron import train
    from treeneuron.artifacts import create_run
    from treeneuron.checkpoint import load_checkpoint
    from treeneuron.config import resolve_experiment
    from treeneuron.data.types import ObservationSet,DevelopmentData
    from treeneuron.baselines.causal import widen_parent
    from treeneuron.growth.allocate import Allocation
    from treeneuron.growth.insert import insert_growth
    c=resolve_experiment(dict(study_id='cpu-envelope-path',model=dict(d=16,h=64),growth=dict(m_requested=16),
        data=dict(n_fit=9224,n_growth=8,n_validation=8),
        training=dict(batch_size=batch,base_updates=1,continuation_updates=1,validation_interval=1,patience=2),
        backend=dict(cpu_threads=1,interop_threads=1),execution=dict(backend_report=str(cpu_envelope_receipt))))
    rng=np.random.default_rng(401)
    def rows(n,start):
        x=rng.normal(size=(n,16));ids=np.arange(start,start+n,dtype=np.int64)
        return ObservationSet(x,x[:,0]-x[:,1],ids,ids)
    data=DevelopmentData(rows(9224,0),rows(8,10000),rows(8,20000),'envelope-fixture')
    monkeypatch.setattr(train,'BACKEND_REPORT_PATH',cpu_envelope_receipt)
    def fit(config,label):
        paths=create_run(tmp_path/label,config.to_dict())
        state=train.fit_parent(data,config,paths)
        loaded=load_checkpoint(paths.directory/'checkpoints/best.json',expected_hashes=state.identity_hashes,device='cpu')
        np.testing.assert_array_equal(state.predict(data.G.X),loaded.predict(data.G.X))
        return loaded
    parent=fit(c,'parent')
    ordinary=c.to_dict();ordinary['model']['h']=95
    fit(resolve_experiment(ordinary),'ordinary')
    static=c.to_dict();static['model'].update(coordinates='centered',static_count=16)
    fit(resolve_experiment(static),'static')
    widened=copy.deepcopy(parent);widen_parent(widened,31,17)
    paths=create_run(tmp_path/'widened',widened.config.to_dict())
    train.continue_fit(widened,data,train.TrainingPolicy.from_config(widened.config,continuation=True),paths)
    load_checkpoint(paths.directory/'checkpoints/best.json',expected_hashes=widened.identity_hashes,device='cpu')
    # Use actual scoring/allocation, including the CPU64 full-density reduction.
    from treeneuron.growth.score import score_units
    from treeneuron.growth.allocate import allocate
    table=score_units(copy.deepcopy(parent.model).double().eval(),
        torch.tensor(parent.preprocessing.transform_X(data.F.X),dtype=torch.float64),
        torch.tensor(parent.preprocessing.transform_y(data.F.y),dtype=torch.float64),
        epsilon=.05,tau=1.,adam_epsilon=1e-8,batch_size=batch,unit_block_size=16)
    insert_growth(parent,allocate(table,16,policy='adam',seed=17))
    paths=create_run(tmp_path/'grown',c.to_dict())
    train.continue_fit(parent,data,train.TrainingPolicy.from_config(c,continuation=True),paths)
    load_checkpoint(paths.directory/'checkpoints/best.json',expected_hashes=parent.identity_hashes,device='cpu')


@pytest.mark.parametrize('mutation',['source','runtime','rows','batch','missing_operation','numerical','tail','resource','fallback'])
def test_envelope_rejects_mismatched_or_incomplete_measurement(cpu_envelope_receipt,mutation):
    import json
    from treeneuron.cpu_envelope import validate_cpu_envelope
    r=json.loads(cpu_envelope_receipt.read_text())['cpu_envelope']
    if mutation=='source': r['source_sha256']='0'*64
    elif mutation=='runtime': r['runtime_sha256']='0'*64
    elif mutation=='rows': r['measurements'][0]['n_fit']=9225
    elif mutation=='batch': r['measurements'][0]['logical_batch']=129
    elif mutation=='missing_operation': r['operations'].pop()
    elif mutation=='numerical': r['operations'][0]['float64_max_normalized_error']=2.
    elif mutation=='tail': r['measurements'][0]['tail_rows']+=1
    elif mutation=='resource': r['resources_after']['peak_rss_bytes']=0
    else: r['environment']['fallback_env']='1'
    with pytest.raises(ValueError): validate_cpu_envelope(r)


@pytest.mark.parametrize('batch',[129,255,32768,True])
def test_unsupported_logical_batch_remains_rejected(batch):
    from treeneuron.train import TrainingPolicy
    with pytest.raises(ValueError): TrainingPolicy(batch,batch,1,1,1)


@pytest.mark.parametrize('change',['fallback','blas','threads'])
def test_extended_receipt_rejects_live_runtime_drift(cpu_envelope_receipt,monkeypatch,change):
    import hashlib,torch
    from treeneuron.cpu_envelope import validate_envelope_reference
    from treeneuron.config import resolve_experiment
    c=resolve_experiment(dict(study_id='runtime-drift',model=dict(d=16,h=64),growth=dict(m_requested=16),
        data=dict(n_fit=9224),training=dict(batch_size=256)))
    reference=dict(path=str(cpu_envelope_receipt),sha256=hashlib.sha256(cpu_envelope_receipt.read_bytes()).hexdigest())
    previous=torch.get_num_threads()
    try:
        if change=='fallback':monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
        elif change=='blas':monkeypatch.setenv('OMP_NUM_THREADS','2')
        else:torch.set_num_threads(2)
        with pytest.raises(ValueError,match='runtime'):
            validate_envelope_reference(reference,c)
    finally:torch.set_num_threads(previous)


def test_extended_receipt_rejects_loss_error(cpu_envelope_receipt):
    import json
    from treeneuron.cpu_envelope import validate_cpu_envelope
    r=json.loads(cpu_envelope_receipt.read_text())['cpu_envelope']
    r['operations'][0]['float64_reference']['loss_max_normalized_error']=2.
    with pytest.raises(ValueError):validate_cpu_envelope(r)


@pytest.mark.parametrize('section,change',[
    ('model',{'d':17}),('model',{'h':96}),('growth',{'m_requested':17}),
    ('data',{'n_fit':9225}),('training',{'batch_size':129}),
    ('model',{'coordinates':'direct_leaf','static_count':16}),
    ('model',{'coordinates':'centered','static_count':17})])
def test_extended_workload_preserves_other_limits(section,change):
    from treeneuron.cpu_envelope import validate_workload
    from treeneuron.config import resolve_experiment
    value=dict(study_id='bounded',model=dict(d=16,h=64),growth=dict(m_requested=16),
        data=dict(n_fit=9224),training=dict(batch_size=256))
    value[section].update(change)
    with pytest.raises(ValueError):validate_workload(resolve_experiment(value))


@pytest.mark.parametrize('field,value',[
    ('schema_version',1),('device','mps'),('parameter_shapes',{}),('total_parameters',1),('seed',1)])
def test_envelope_validates_causal_operation_identity(cpu_envelope_receipt,field,value):
    import json
    from treeneuron.cpu_envelope import validate_cpu_envelope
    r=json.loads(cpu_envelope_receipt.read_text())['cpu_envelope']
    r['operations'][0]['causal'][field]=value
    with pytest.raises(ValueError):validate_cpu_envelope(r)
