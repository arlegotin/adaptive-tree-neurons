"""Fresh-process cold timing entry point with an explicit safe-loader registry."""
import time
_IMPORT_START=time.perf_counter_ns()
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from treeneuron.artifacts import _atomic_json
from treeneuron.config import canonical_hash
from treeneuron.evaluation.search import file_hash
from treeneuron.evaluation.timing import TimingPolicy,measured_call
from treeneuron.baselines.simple import MeanPredictor,RidgePredictor,SplineRidgePredictor
from treeneuron.baselines.node import NodePredictor
from treeneuron.baselines.causal import CausalPredictor
_IMPORT_NS=time.perf_counter_ns()-_IMPORT_START


def main():
    directory=Path(sys.argv[1]);envelope=json.loads((directory/'request.json').read_text());request=envelope['payload']
    if set(envelope)!={'payload','sha256'} or canonical_hash(request)!=envelope['sha256']:raise ValueError('Cold request hash mismatch')
    if set(request)!={'schema_version','method','policy','model_sha256','input_sha256','expected_hashes'} or request['schema_version']!=1:
        raise ValueError('Cold request schema mismatch')
    if file_hash(directory/'predictor.json')!=request['model_sha256'] or file_hash(directory/'X.npy')!=request['input_sha256']:
        raise ValueError('Cold artifact hash mismatch')
    policy=TimingPolicy(**request['policy'])
    torch.set_num_threads(policy.cpu_threads);torch.set_num_interop_threads(1)
    sync=torch.mps.synchronize if policy.device=='mps' else lambda:None
    registry={c.__name__:c for c in (MeanPredictor,RidgePredictor,SplineRidgePredictor,NodePredictor,CausalPredictor)}
    cls=registry[request['method']]
    model,load_ns=measured_call(lambda:cls.load(directory/'predictor.json',**({'expected_hashes':request['expected_hashes']} if cls is CausalPredictor else {})),sync=sync)
    X=np.load(directory/'X.npy',allow_pickle=False);batch=policy.batch_sizes[0]
    with torch.inference_mode():
        prediction,predict_ns=measured_call(lambda:np.concatenate([model.predict(X[i:i+batch],batch_size=batch) for i in range(0,len(X),batch)]),sync=sync)
    if prediction.shape!=(len(X),) or not np.isfinite(prediction).all():raise ValueError('Invalid cold prediction')
    np.save(directory/'prediction.npy',prediction,allow_pickle=False)
    result=dict(stages_ns={'imports':_IMPORT_NS,'load':load_ns,'first_prediction':predict_ns},
        process_id=os.getpid(),prediction_sha256=file_hash(directory/'prediction.npy'),threads={'cpu':torch.get_num_threads(),'interop':torch.get_num_interop_threads()})
    _atomic_json(directory/'response.json',dict(payload=result,sha256=canonical_hash(result)))


if __name__=='__main__':main()
