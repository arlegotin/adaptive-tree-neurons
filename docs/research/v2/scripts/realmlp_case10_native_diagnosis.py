"""One exact official native control for the retained PLR/SELU numerical failure.

No changed seed/recipe, data filtering, retry-on-new-seed, fallback or full anchor.
The observer stops on nonfinite loss before native backward; it never sanitizes.
"""
from pathlib import Path
import json
import time
import sys


def main(output):
    output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import pandas as pd
    from treeneuron.research import realmlp_native as audit
    dependencies=audit._dependencies()
    import torch
    import pytorch_lightning as pl
    from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Regressor
    from pytabkit.models.data.conversion import ToDictDatasetConverter
    from pytabkit.models.data.data import DictDataset,TensorInfo
    from pytabkit.models.alg_interfaces.base import SplitIdxs,InterfaceResources
    from pytabkit.models.training.lightning_modules import TabNNModule
    from pytabkit.models.training.logging import StdoutLogger
    rng=np.random.default_rng(2026101819)
    def draw(n):
        x=rng.uniform(-2,2,n)
        return pd.DataFrame(dict(x=x,constant=np.full(n,7.))),np.asarray(4.+x+.3*x*x,np.float32)
    F,yF=draw(300);G,yG=draw(37)
    converter=ToDictDatasetConverter(cat_features=[False,False],verbosity=0)
    fd=converter.fit_transform(F);gd=converter.transform(G)
    data=DictDataset(tensors={**{k:torch.cat([fd.tensors[k],gd.tensors[k]]) for k in fd.tensors},
        'y':torch.from_numpy(np.concatenate([yF,yG])[:,None])},
        tensor_infos=fd.tensor_infos|{'y':TensorInfo(feat_shape=[1])})
    split=SplitIdxs(torch.arange(300)[None],torch.arange(300,337)[None],None,
        split_seed=17,sub_split_seeds=[int(np.random.RandomState(17).randint(0,2**31-1))],split_id=0)
    factory=RealMLP_TD_Regressor(device='cpu',random_state=17,n_cv=1,n_refit=0,n_repeats=1,n_threads=1,
        verbosity=0,n_epochs=3,batch_size=256,predict_batch_size=1024,n_ens=1,
        train_metric_name='mse',val_metric_name='rmse',use_early_stopping=False,
        num_emb_type='plr',act='selu',hidden_sizes=[64]*5,add_front_scale=False)
    observed=dict(first_nonfinite_loss=None,losses=[],applied_updates=0,initial_nonfinite_tensors=[],row_batches=[])
    class NativeFailureObserved(RuntimeError):pass
    class Observer(pl.Callback):
        def on_fit_start(self,trainer,module):
            state={k:v.detach().cpu().numpy().copy() for k,v in module.model.state_dict().items()}
            for key,array in state.items():
                if not np.isfinite(array).all():
                    observed['initial_nonfinite_tensors'].append(dict(name=key,shape=list(array.shape),
                        nan_count=int(np.isnan(array).sum()),inf_count=int(np.isinf(array).sum())))
            np.savez(output/'native-initial-state.npz',**state)
            self.ds=module.train_dl.ds;self.get_batch=self.ds.get_batch
            def trace(idxs):
                observed['row_batches'].append(idxs[0].tolist())
                return self.get_batch(idxs)
            self.ds.get_batch=trace
        def on_before_backward(self,trainer,module,loss):
            value=float(loss.detach())
            observed['losses'].append(value if np.isfinite(value) else str(value))
            if not torch.isfinite(loss):
                observed['first_nonfinite_loss']=str(value)
                raise NativeFailureObserved('Direct pinned native source produced nonfinite loss before backward')
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):observed['applied_updates']+=1
    observer=Observer();original=TabNNModule.create_callbacks
    TabNNModule.create_callbacks=lambda module:[*original(module),observer]
    started=time.perf_counter_ns();error=None
    try:
        factory._create_alg_interface(n_cv=1).fit(data,[split],InterfaceResources(1,[]),StdoutLogger(0),[None],name='direct_case10_same_seed')
    except NativeFailureObserved as caught:error=dict(type=type(caught).__name__,message=str(caught))
    finally:
        if hasattr(observer,'get_batch'):observer.ds.get_batch=observer.get_batch
        TabNNModule.create_callbacks=original
    if error is None:raise AssertionError('Direct official source did not reproduce the retained numerical failure')
    result=dict(schema='research/v2-realmlp-native-failure-diagnosis',status='numerical_failure',
        source=audit._source(),dependencies=dependencies,runtime=audit._runtime(),
        native_factory_config=factory.get_config(),seed=17,fixture_stream=2026101819,variant=10,
        observations=observed,error=error,native_wall_ns=time.perf_counter_ns()-started,
        core_or_overlay_changed=False,full_quality_anchor=False,accessed_roles=['F','G'],
        conclusion='Exact direct pinned native source reproduces the adapter numerical failure; support is not removed')
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],observations=observed),indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
