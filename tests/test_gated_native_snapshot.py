"""Closed own-family prediction state must bind topology, transforms and bytes."""
from test_gated_native_fit import run_native


def test_all_own_topologies_reopen_exactly_with_trusted_closed_state(tmp_path):
    run_native(tmp_path,'''
for i,(family,variant) in enumerate([('P00',None),('P01',None),('P02','lower'),('P02','upper'),('P05','centered_partial'),('P05','centered_full'),('P05','direct_partial'),('P05','direct_full')]):
    model=fit(family,'fit'+str(i),variant);directory=root/('snapshot'+str(i));before=model.predict(G)
    model.save(directory);path=directory/'manifest.json';sha=hashlib.sha256(path.read_bytes()).hexdigest()
    restored=GatedPredictor.load(directory,expected_manifest_sha256=sha)
    assert np.array_equal(restored.predict(G),before)
    assert restored.resource_summary()['quality_eligible'] is False
    with np.load(directory/'weights.npz',allow_pickle=False) as saved:
        assert all(saved[k].dtype.kind in 'fiub' for k in saved.files)
    try:model.save(directory)
    except FileExistsError:pass
    else:raise AssertionError('snapshot overwritten')
    assert restored.resource_summary()['persistent_training_resume'] is False
print('{}')
''')


def test_closed_loader_rejects_corrupt_tensor_dtype_topology_age_and_prediction_bindings(tmp_path):
    run_native(tmp_path,'''
import shutil
model=fit('P00','fit');original=root/'original';model.save(original)
for case in ('dtype','nonfinite','names','duplicate_ids','age','preprocessing','source','seed','scope'):
    directory=root/case;shutil.copytree(original,directory);path=directory/'manifest.json';manifest=json.loads(path.read_text())
    if case in ('dtype','nonfinite','names'):
        with np.load(directory/'weights.npz',allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
        key=next(k for k in arrays if arrays[k].dtype==np.float32)
        if case=='dtype':arrays[key]=arrays[key].astype(np.float64)
        elif case=='nonfinite':arrays[key].flat[0]=np.nan
        else:arrays['unowned_tensor']=np.zeros(1,np.float32)
        np.savez(directory/'weights.npz',**arrays)
        manifest['weights_sha256']=hashlib.sha256((directory/'weights.npz').read_bytes()).hexdigest()
        manifest['payload']={k:dict(dtype=str(v.dtype),shape=list(v.shape)) for k,v in arrays.items()}
    elif case=='duplicate_ids':manifest['resources']['selected_ids']=[0,0]
    elif case=='age':manifest['resources']['parameter_ages'][0]['age']+=1
    elif case=='preprocessing':
        manifest['preprocessing']['columns'][0]['mean']+=1
        manifest['resources']['preprocessing']['columns'][0]['mean']+=1
    elif case=='source':manifest['source']['sha256']='0'*64
    elif case=='seed':manifest['seed']+=1
    else:manifest['snapshot_scope']='full training resume authorized'
    path.write_text(json.dumps(manifest,sort_keys=True,allow_nan=False)+'\\n');trusted=hashlib.sha256(path.read_bytes()).hexdigest()
    try:GatedPredictor.load(directory,expected_manifest_sha256=trusted)
    except ValueError:pass
    else:raise AssertionError('corrupt closed state accepted: '+case)
print('{}')
''')


def test_save_rejects_live_weight_or_preprocessing_changes_and_load_requires_external_hash(tmp_path):
    run_native(tmp_path,'''
model=fit('P00','fit');model.save(root/'saved')
for trusted in ('0'*64,'invalid',None):
    try:GatedPredictor.load(root/'saved',expected_manifest_sha256=trusted)
    except ValueError:pass
    else:raise AssertionError('untrusted manifest accepted')
with torch.no_grad():next(model._model.parameters()).add_(.1)
try:model.save(root/'changed')
except ValueError:pass
else:raise AssertionError('changed live weights accepted')
assert not (root/'changed').exists()
other=fit('P01','other');other._preprocessing['columns'][0]['mean']+=1
try:other.save(root/'changed-transform')
except ValueError:pass
else:raise AssertionError('changed live preprocessing accepted')
assert not (root/'changed-transform').exists()
print('{}')
''')
