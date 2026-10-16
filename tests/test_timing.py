"""Timing boundaries, complete batches, warmup and reconstructible summaries."""
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
    raw=[{'phase':'measurement','batch_size':b,'elapsed_ns':n,'rows':100,'status':'ok','block':i} for i in range(4) for b,n in ((1,200),(2,104),(4,100))]
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
