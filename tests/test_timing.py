"""Timing boundaries, complete batches, warmup and reconstructible summaries."""
import json
import numpy as np
import pytest


def api():
    from treeneuron.evaluation import timing
    return timing


def test_mps_timing_synchronizes_both_boundaries():
    t=api();trace=[]
    def timer(): trace.append('start' if 'start' not in trace else 'stop');return len(trace)
    value,elapsed=t.measured_call(lambda:trace.append('predict'),sync=lambda:trace.append('sync'),timer=timer)
    assert trace==['sync','start','predict','sync','stop']
    assert elapsed==3


def test_timing_complete_batches_retains_warmup_and_rebuilds(tmp_path):
    t=api()
    class Model:
        def predict(self,X,*,batch_size=None):return np.asarray(X)[:,0]
        def resource_summary(self):return {'capabilities':{'backend':'numpy_cpu','dtype':'float64','cache':False,'context':False}}
    policy=t.TimingPolicy(batch_sizes=(2,4),warmup_window=2,max_warmup=4,min_repetitions=4,max_repetitions=8,relative_precision=.5,block_order='abba')
    result=t.measure_predictor(Model(),np.ones((9,2)),policy)
    assert result['rows']==9 and result['raw_blocks']
    assert all(b['rows']==9 for b in result['raw_blocks'])
    assert all(b['phase'] in ('warmup','measurement') for b in result['raw_blocks'])
    assert result['summary']==t.rebuild_summary(result['raw_blocks'],policy)
    assert result['summary']['selected_batch'] in (2,4)
    assert result['mps_allocated_bytes']['value'] is None
    path=tmp_path/'timing.json'; t.write_timing(path,result)
    assert t.read_timing(path)['summary']==result['summary']


def test_batch_selection_smallest_at_95_percent_and_precision():
    t=api();p=t.TimingPolicy(batch_sizes=(1,2,4),min_repetitions=4,max_repetitions=4,warmup_window=2,max_warmup=2)
    raw=[{'phase':'measurement','batch_size':b,'elapsed_ns':n,'rows':100,'status':'ok','block':i//2,'repeat':i%2} for i in range(4) for b,n in ((1,200),(2,104),(4,100))]
    summary=t.rebuild_summary(raw,p)
    assert summary['selected_batch']==2
    assert summary['batches']['2']['median_ns']==104
    assert summary['batches']['2']['tail_ns']==104


def test_unsupported_kernel_cache_and_mps_do_not_fallback():
    t=api()
    class Model:
        def predict(self,X,*,batch_size=None):return np.zeros(len(X))
        def resource_summary(self):return {'capabilities':{'backend':'numpy_cpu','cache':False,'context':False}}
    for kwargs in ({'mode':'resident_kernel'},{'context_mode':'uncached'},{'device':'mps'}):
        with pytest.raises(ValueError):t.measure_predictor(Model(),np.ones((2,2)),t.TimingPolicy(**kwargs))
    with pytest.raises(TypeError):t.TimingPolicy(unknown=True)


def test_cold_prediction_fresh_process_measures_import_load_and_first_predict(tmp_path):
    t=api()
    from treeneuron.baselines.simple import RidgePredictor
    from treeneuron.baselines.api import FitRequest
    x=np.arange(24,dtype=float).reshape(12,2);y=2+x[:,0]
    model=RidgePredictor(1.).fit(FitRequest(x,y,x,y,1,{}, {},tmp_path))
    result=t.measure_cold_predictor(model,x,t.TimingPolicy(batch_sizes=(4,)),tmp_path/'cold')
    assert set(result['stages_ns'])=={'imports','load','first_prediction'}
    assert all(n>0 for n in result['stages_ns'].values())
    assert result['process_id']!=__import__('os').getpid()
    assert result['process_wall_ns']>=sum(result['stages_ns'].values())
    np.testing.assert_array_equal(np.load(tmp_path/'cold'/'prediction.npy',allow_pickle=False),model.predict(x))


def test_resident_causal_path_preserves_predictions_and_excludes_transform(training_case):
    t=api();data,config,paths=training_case
    from treeneuron.train import fit_parent
    from treeneuron.baselines.causal import build_causal
    import torch
    model=build_causal('B1',{},parent=fit_parent(data,config,paths))
    resident=t.ResidentCausalPredictor(model)
    with torch.inference_mode():actual=resident.predict_resident(resident.prepare_timing_input(data.G.X)).cpu().numpy()
    actual=model.state.preprocessing.inverse_y(actual)
    np.testing.assert_allclose(actual,model.predict(data.G.X),rtol=1e-7,atol=1e-7)
    result=t.measure_predictor(resident,data.G.X,t.TimingPolicy(mode='resident_kernel',batch_sizes=(2,4),warmup_window=2,max_warmup=2,min_repetitions=4,max_repetitions=4))
    assert result['excluded_work'][0]['elapsed_ns']>0
    assert all(r['rows']==7 for r in result['raw_blocks'])


from test_train import training_case,cpu_training_evidence


def test_timing_failure_retains_prior_raw_blocks_and_original_error():
    t=api()
    class Failing:
        def __init__(self):self.calls=0
        def resource_summary(self):return {'capabilities':{'backend':'numpy_cpu','cache':False,'context':False}}
        def predict(self,X,*,batch_size=None):
            self.calls+=1
            if self.calls==3:raise RuntimeError('original prediction failure')
            return np.zeros(len(X))
    with pytest.raises(RuntimeError,match='original prediction failure') as caught:
        t.measure_predictor(Failing(),np.ones((1,1)),t.TimingPolicy(warmup_window=2,max_warmup=2,min_repetitions=4,max_repetitions=4))
    assert len(caught.value.timing_record['raw_blocks'])==3
    assert caught.value.timing_record['raw_blocks'][-1]['status']=='failed'


def test_cached_uncached_boundaries_retain_cache_work():
    t=api()
    class Cached:
        def __init__(self):self.cache=None;self.builds=0
        def resource_summary(self):return {'capabilities':{'backend':'numpy_cpu','cache':True,'context':True},'context_bytes':8,'cache_bytes':8}
        def clear_prediction_cache(self):self.cache=None
        def predict(self,X,*,batch_size=None):
            if self.cache is None:self.builds+=1;self.cache=1.
            return np.ones(len(X))*self.cache
    a,b=Cached(),Cached()
    kwargs=dict(warmup_window=2,max_warmup=2,min_repetitions=4,max_repetitions=4,batch_sizes=(2,))
    cached=t.measure_predictor(a,np.ones((3,1)),t.TimingPolicy(context_mode='cached',**kwargs))
    uncached=t.measure_predictor(b,np.ones((3,1)),t.TimingPolicy(context_mode='uncached',**kwargs))
    assert a.builds==1 and b.builds==6
    assert len(cached['raw_blocks'])==len(uncached['raw_blocks'])==6


def test_mixed_early_stable_and_capped_unstable_warmup_terminates(monkeypatch):
    t=api()
    class Model:
        def predict(self,X,*,batch_size=None):return np.zeros(len(X))
        def resource_summary(self):return {'capabilities':{'backend':'numpy_cpu','cache':False,'context':False}}
    durations=iter([100,100,200,100,100,200]+[100]*8)
    monkeypatch.setattr(t,'measured_call',lambda call,**kwargs:(call(),next(durations)))
    original=t._order;calls=[0]
    def bounded_order(*args):
        calls[0]+=1
        if calls[0]>10:raise AssertionError('Warmup loop failed to terminate at explicit cap')
        return original(*args)
    monkeypatch.setattr(t,'_order',bounded_order)
    result=t.measure_predictor(Model(),np.ones((2,1)),t.TimingPolicy(batch_sizes=(1,2),warmup_window=2,max_warmup=4,min_repetitions=4,max_repetitions=4))
    warm=[r for r in result['raw_blocks'] if r['phase']=='warmup']
    assert [r['batch_size'] for r in warm]==[1,2,2,1,2,2]
    assert result['warmup_stable']=={'1':True,'2':False}
    assert result['precision_status']=='unresolved_at_repetition_or_warmup_cap'
    assert len(result['raw_blocks'])==14


def test_interruption_between_blocks_retains_measured_warmup(monkeypatch):
    t=api()
    class Model:
        def predict(self,X,*,batch_size=None):return np.zeros(len(X))
        def resource_summary(self):return {'capabilities':{'backend':'numpy_cpu','cache':False,'context':False}}
    durations=iter([100,200,300,400])
    monkeypatch.setattr(t,'measured_call',lambda call,**kwargs:(call(),next(durations)))
    original=t._order;calls=[0]
    def interrupt(*args):
        calls[0]+=1
        if calls[0]==2:raise KeyboardInterrupt('between blocks')
        return original(*args)
    monkeypatch.setattr(t,'_order',interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        t.measure_predictor(Model(),np.ones((2,1)),t.TimingPolicy(batch_sizes=(1,2),warmup_window=2,max_warmup=4,min_repetitions=4,max_repetitions=4))
    assert len(caught.value.timing_record['raw_blocks'])==4
    assert caught.value.timing_record['error']['type']=='KeyboardInterrupt'


def test_comparison_interleaves_methods_in_complete_shared_blocks(monkeypatch):
    t=api();calls=[]
    class Model:
        def __init__(self,name):self.name=name
        def resource_summary(self):return {'capabilities':{'backend':'numpy_cpu','cache':False,'context':False}}
        def predict(self,X,*,batch_size=None):calls.append(self.name);return np.zeros(len(X))
    monkeypatch.setattr(t,'measured_call',lambda call,**kwargs:(call(),100))
    p=t.TimingPolicy(batch_sizes=(4,),warmup_window=2,max_warmup=2,min_repetitions=4,max_repetitions=4,tail_quantile=.5)
    result=t.measure_comparison({'A':(Model('A'),p),'B':(Model('B'),p)},np.ones((3,1)))
    assert calls==['A','B','B','A','B','A','A','B','A','B','B','A']
    for block in range(3):
        rows=[r for r in result['raw_blocks'] if r['block']==block]
        assert sorted((r['method_id'],r['repeat']) for r in rows)==[('A',0),('A',1),('B',0),('B',1)]
    assert result['summary']['complete_block_ids']==[1,2]
    assert result['summary']['paired_ratios']['B/A']['median_ratio']==1
    assert result['summary']['precision_reached']
    assert result['summary']==t.rebuild_comparison(result['raw_blocks'],result['policies'])


def test_precision_requires_multiple_complete_blocks_and_ignores_repeat_duplicates():
    t=api();p=t.TimingPolicy(min_repetitions=4,max_repetitions=40,tail_quantile=.5)
    one=[dict(phase='measurement',batch_size=1,elapsed_ns=100,rows=10,status='ok',block=0,repeat=i) for i in range(2)]
    assert not t.rebuild_summary(one*10,p)['precision_reached']
    raw=one+[dict(r,block=1,elapsed_ns=200) for r in one]
    base=t.rebuild_summary(raw,p);duplicated=t.rebuild_summary(raw+one*30,p)
    assert base==duplicated
    assert base['batches']['1']['independent_blocks']==2
    partial=raw+[dict(one[0],block=2,elapsed_ns=1)]
    assert t.rebuild_summary(partial,p)['batches']['1']['median_ns']==150
    assert t.rebuild_summary(partial,p)['batches']['1']['independent_blocks']==2


def test_comparison_precision_preserves_pairing_and_drops_partial_blocks():
    t=api();p=t.TimingPolicy(min_repetitions=4,max_repetitions=40,tail_quantile=.5)
    raw=[dict(method_id=m,phase='measurement',batch_size=1,elapsed_ns=n*scale,rows=10,status='ok',block=b,repeat=i)
        for b,n in enumerate((100,200,400,800)) for m,scale in [('A',1),('B',2)] for i in range(2)]
    raw.extend(dict(raw[i],block=10,elapsed_ns=1) for i in (0,1))
    result=t.rebuild_comparison(raw,{'A':p,'B':p})
    assert result['complete_block_ids']==[0,1,2,3]
    pair=result['paired_ratios']['B/A']
    assert pair['median_ratio']==2 and pair['interval']==[2.,2.]
    assert result==t.rebuild_comparison(raw+raw[:4]*30,{'A':p,'B':p})


def test_balanced_random_comparison_roundtrips_with_explicit_ids(tmp_path,monkeypatch):
    t=api()
    class Model:
        def resource_summary(self):return {'capabilities':{'backend':'numpy_cpu','cache':False,'context':False}}
        def predict(self,X,*,batch_size=None):return np.zeros(len(X))
    monkeypatch.setattr(t,'measured_call',lambda call,**kwargs:(call(),100))
    p=t.TimingPolicy(batch_sizes=(4,),warmup_window=2,max_warmup=2,min_repetitions=4,max_repetitions=4,block_order='balanced_random',seed=3)
    cases={k:(Model(),p) for k in ('C','B','A')}
    first=t.measure_comparison(cases,np.ones((3,1)));second=t.measure_comparison(cases,np.ones((3,1)))
    assert first['raw_blocks']==second['raw_blocks']
    for block in (0,1,2):
        order=[r['method_id'] for r in first['raw_blocks'] if r['block']==block]
        assert order==order[::-1] and sorted(order)==['A','A','B','B','C','C']
    assert not first['summary']['precision_reached']  # two blocks cannot assess p95
    path=tmp_path/'comparison.json';t.write_timing(path,first)
    assert t.read_timing(path)==json.loads(json.dumps(first))
