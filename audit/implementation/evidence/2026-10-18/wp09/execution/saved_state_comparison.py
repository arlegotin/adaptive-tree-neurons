"""Complete saved parent/probe states and uninterrupted/recovered update stream."""
import time
started=time.perf_counter()
import hashlib,json,os,resource,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.checkpoint import load_checkpoint
ROOT=Path(os.environ['WP09_ORIGINAL_ROOT']);B=ROOT/'environment/h1-v2/wp09';OUT=ROOT/'artifacts/h1-v2/wp09/clean-replay/repetition-0'
REF=Path('artifacts/h1-v2/complete-resource/pilot-5f6c0cc620bd441db53ed126157fbedf/teacher-0/repetition-0')
report=dict(status='running',comparisons=[],inputs=[],failures=[])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b,path=''):
    if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()),path
    elif isinstance(a,np.ndarray):assert a.dtype==b.dtype and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    else:assert a==b,(path,a,b)
def load(p):
    pointer=json.loads(p.read_text());meta=p.parent/pointer['metadata'];d=json.loads(meta.read_text());payload=p.parent/d['payload']
    report['inputs'] += [dict(path=str(x.resolve()),sha256=sha(x)) for x in (p,meta,payload)]
    return load_checkpoint(p,expected_hashes=d['state']['identity_hashes'],device='cpu')
def compare(a,b,name,F,G):
    x=train.pack_state(a);y=train.pack_state(b);equal(x['tensors'],y['tensors'],name+'/model_optimizer_sampler_full_rng')
    for k in ('config','model','identity_hashes','device','update','epoch','cursor','n_fit','optimizer_kind','optimizer_groups','scheduler','preprocessing','fit_target_variance','data_hashes','stage'):equal(x['metadata'][k],y['metadata'][k],name+'/'+k)
    equal(a.next_row_ids(F),b.next_row_ids(F),name+'/next_rows');equal(a.predict(G),b.predict(G),name+'/G_predictions')
    report['comparisons'].append(dict(name=name,exact=True,maximum_difference=0,selected_update=a.update,stage=a.stage,mps_rng_present=a.rng_states['torch_mps']['state'] is not None))
def updates(path):
    with path.open() as stream:return [v for line in stream if (v:=json.loads(line))['type']=='update']
try:
    ref=json.loads((REF/'pair.json').read_text());new=json.loads((OUT/'pair.json').read_text())
    with np.load(REF.parent/'observations.npz',allow_pickle=False) as z:F=SimpleNamespace(row_id=z['F_row_id']);G=z['G_X']
    compare(load(REF/'runs'/ref['parent_run_id']/'checkpoints/best.json'),load(OUT/'runs'/new['parent_run_id']/'checkpoints/best.json'),'parent',F,G)
    oldg=json.loads(Path(ref['greedy_report']).read_text());newg=json.loads(Path(new['greedy_report']).read_text())
    for a,b in zip(oldg['trials'],newg['trials']):
        assert a['candidate']==b['candidate']
        compare(load(Path(ref['greedy_report']).parent/a['path']/'checkpoints/best.json'),load(Path(new['greedy_report']).parent/b['path']/'checkpoints/best.json'),a['candidate'],F,G)
    assert len(report['comparisons'])==66
    stages=0
    for role in ref['methods']:
        for a,b in zip(ref['methods'][role]['stages'],new['methods'][role]['stages']):
            for k in ('row_hashes','executed_updates','selected_update','max_updates','training_loss','evaluations'):equal(a[k],b[k],role+'/'+k)
            stages+=1
    recovery=json.loads((B/'clean-readout.json').read_text())['controlled_interruption']
    whole=updates(OUT/new['methods']['top']['stages'][0]['path']/'events.jsonl')
    first=updates(Path(recovery['path'])/'events.jsonl');retry=updates(Path(recovery['retry_path'])/'events.jsonl')
    equal(first+retry,whole,'recovery all durable update events, losses, learning rates, row IDs')
    report.update(status='completed',complete_probe_states=65,parent_states=1,role_stages=stages,recovery_updates=dict(interrupted=len(first),retry=len(retry),uninterrupted=len(whole),exact_event_equality=True))
except BaseException:report['status']='failed';report['failures'].append(traceback.format_exc());raise
finally:
    report.update(seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,script_sha256=sha(Path(__file__)))
    (B/'saved-state-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons','inputs')}))
