"""Observe actual pinned RAM row selections without adding random draws."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def api():
    from treeneuron.research import realmlp_initialization
    return realmlp_initialization


@pytest.mark.parametrize('n',[0,-1,True,20001])
def test_invalid_fitting_count_is_rejected_before_native_hooks(n):
    with pytest.raises(ValueError):
        api().NativeInitializationTrace(n_fit=n,dataset_class=object,fitter_class=object,creator_class=object)


def test_pinned_native_initialization_trace_matches_rows_rng_and_model_tensors(tmp_path):
    assert hasattr(api(),'NativeInitializationTrace')
    root=Path(__file__).resolve().parents[1]
    env=os.environ|dict(PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'environment/research-v2-realmlp-site')]),PYTHONDONTWRITEBYTECODE='1',
        MPLCONFIGDIR=str(tmp_path/'mpl'),XDG_CACHE_HOME=str(tmp_path/'cache'))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(tmp_path/'worker')]
    started=time.perf_counter_ns();result=subprocess.run(argv,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    (tmp_path/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(root),returncode=result.returncode,wall_ns=time.perf_counter_ns()-started),indent=2)+'\n')
    (tmp_path/'worker.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout


def worker(output):
    import numpy as np
    from treeneuron.research import realmlp_native
    output.mkdir(parents=True,exist_ok=False)
    before_dependencies=realmlp_native._dependencies()
    import torch
    from pytabkit.models.data.data import DictDataset,TensorInfo
    from pytabkit.models.nn_models.base import Fitter,SequentialFitter,IdentityLayer
    from pytabkit.models.training.nn_creator import NNCreator
    assert torch.get_num_threads()==1 and torch.get_num_interop_threads()==1
    originals=(Fitter.fit_transform_subsample,DictDataset.get_sub_dataset,NNCreator.create_model)
    def data(n=32):
        return DictDataset(tensors={'x_cont':torch.arange(n,dtype=torch.float32)[:,None],
            'y':torch.arange(n,dtype=torch.float32)[:,None]},
            tensor_infos={'x_cont':TensorInfo(feat_shape=[1]),'y':TensorInfo(feat_shape=[1])})
    class Budget(Fitter):
        def __init__(self,size,fail=False):super().__init__(scope_names=[str(size)]);self.size=size;self.fail=fail
        def get_n_forward(self,infos):return self.size
        def _fit(self,ds):
            if self.fail:raise RuntimeError('controlled native fixture failure')
            return IdentityLayer()
    def run(kind,observed):
        ds=data();fitter=SequentialFitter([Budget(32),Budget(64)]) if kind=='sequential' else Budget(32,kind=='exception')
        ram=1. if kind=='no_subsample' else 4*32*8/1024**3
        torch.manual_seed(781);trace=None;selected=None;error=None
        try:
            if observed:
                with api().NativeInitializationTrace(n_fit=32,dataset_class=DictDataset,fitter_class=Fitter,
                        creator_class=NNCreator,standalone_F_dataset=ds) as trace:
                    _,selected=fitter.fit_transform_subsample(ds,ram_limit_gb=ram,needs_tensors=True)
                    positions=trace.F_positions(selected)
                    assert positions.tolist()==selected.tensors['x_cont'][:,0].numpy().astype(np.int64).tolist()
                    with pytest.raises(ValueError,match='installed'):
                        with api().NativeInitializationTrace(n_fit=32,dataset_class=DictDataset,fitter_class=Fitter,
                                creator_class=NNCreator,standalone_F_dataset=ds):pass
            else:_,selected=fitter.fit_transform_subsample(ds,ram_limit_gb=ram,needs_tensors=True)
        except RuntimeError as caught:
            assert kind=='exception';error=str(caught)
        assert originals==(Fitter.fit_transform_subsample,DictDataset.get_sub_dataset,NNCreator.create_model)
        return selected,torch.get_rng_state().clone(),error,None if trace is None else trace.snapshot()
    cases=[]
    for kind in ('single','sequential','no_subsample','exception'):
        plain,rng,err,_=run(kind,False);observed,observed_rng,observed_err,trace=run(kind,True)
        assert err==observed_err and torch.equal(rng,observed_rng)
        if plain is not None:
            assert plain.tensor_infos.keys()==observed.tensor_infos.keys()
            for name in plain.tensors:assert torch.equal(plain.tensors[name],observed.tensors[name])
        expected={'single':1,'sequential':2,'no_subsample':0,'exception':1}[kind]
        assert len(trace['subsamples'])==expected
        if kind=='sequential':
            first,second=trace['subsamples'];assert second['F_positions']==[first['F_positions'][i] for i in second['input_indices']]
        cases.append(dict(kind=kind,RNG_identical=True,output_tensors_identical=plain is not None,
            native_exception_unchanged=err is not None,trace=trace))
    # One complete native epoch exercises the real creator, distinct from the
    # forced tiny-RAM fitter reference above. No actual benchmark data is used.
    from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Regressor
    from pytabkit.models.alg_interfaces.base import SplitIdxs,InterfaceResources
    from pytabkit.models.training.logging import StdoutLogger
    nF,nG=300,37;x=torch.linspace(-2.,2.,nF+nG)[:,None]
    ds=DictDataset(tensors={'x_cont':x,'x_cat':torch.empty((nF+nG,0),dtype=torch.int64),'y':4+x+.3*x*x},
        tensor_infos={'x_cont':TensorInfo(feat_shape=[1]),'x_cat':TensorInfo(cat_sizes=[]),'y':TensorInfo(feat_shape=[1])})
    split=SplitIdxs(torch.arange(nF)[None],torch.arange(nF,nF+nG)[None],None,split_seed=17,
        sub_split_seeds=[int(np.random.RandomState(17).randint(0,2**31-1))],split_id=0)
    def model(observed):
        interface=RealMLP_TD_Regressor(device='cpu',random_state=17,n_cv=1,n_refit=0,n_repeats=1,
            n_threads=1,verbosity=0,n_epochs=1,batch_size=256,predict_batch_size=1024,n_ens=1,
            train_metric_name='mse',val_metric_name='rmse',use_early_stopping=False)._create_alg_interface(n_cv=1)
        torch.manual_seed(478);trace=None
        if observed:
            with api().NativeInitializationTrace(n_fit=nF,dataset_class=DictDataset,fitter_class=Fitter,creator_class=NNCreator) as trace:
                interface.fit(ds,[split],InterfaceResources(1,[]),StdoutLogger(0),[None],name='observed_creator_fixture')
        else:interface.fit(ds,[split],InterfaceResources(1,[]),StdoutLogger(0),[None],name='plain_creator_fixture')
        return interface,torch.get_rng_state().clone(),None if trace is None else trace.snapshot()
    plain,rng,_=model(False);observed,traced_rng,trace=model(True)
    assert torch.equal(rng,traced_rng)
    a,b=plain.model.model.state_dict(),observed.model.model.state_dict();assert a.keys()==b.keys()
    for name in a:assert torch.equal(a[name],b[name]),name
    assert trace['creator_calls']==1 and trace['root_F_positions']==list(range(nF))
    assert trace['scope']=='native_creator' and trace['subsamples']==[]
    assert originals==(Fitter.fit_transform_subsample,DictDataset.get_sub_dataset,NNCreator.create_model)
    # An actual creator request including a G row must fail before its first
    # data-dependent fitter; static shape transforms still follow native code.
    invalid=SplitIdxs(torch.arange(1,nF+1)[None],torch.arange(nF+1,nF+nG)[None],None,
        split_seed=17,sub_split_seeds=[17],split_id=0)
    refused=RealMLP_TD_Regressor(device='cpu',n_threads=1,n_epochs=1)._create_alg_interface(n_cv=1)
    with pytest.raises(ValueError,match='declared F rows'):
        with api().NativeInitializationTrace(n_fit=nF,dataset_class=DictDataset,
                fitter_class=Fitter,creator_class=NNCreator) as forbidden:
            refused.fit(ds,[invalid],InterfaceResources(1,[]),StdoutLogger(0),[None],name='forbidden_G_row')
    assert forbidden.snapshot()['calls']==[] and forbidden.snapshot()['status']=='failed'
    assert originals==(Fitter.fit_transform_subsample,DictDataset.get_sub_dataset,NNCreator.create_model)
    assert before_dependencies==realmlp_native._dependencies()
    (output/'proof.json').write_text(json.dumps(dict(cases=cases,native_creator_trace=trace,
        native_epoch_count=1,native_model_tensor_count=len(a),model_tensors_identical=True,RNG_identical=True,
        forbidden_G_row_rejected_before_fitter=True,full_native_anchor=False,quality_eligible=False,
        T_accessed=False,dependencies=before_dependencies),indent=2,sort_keys=True)+'\n')


if __name__=='__main__':
    assert sys.argv[1]=='--worker';worker(Path(sys.argv[2]))
