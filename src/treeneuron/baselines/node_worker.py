"""Reproducible CPU full-anchor engineering cost worker; never reads datasets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import resource
import time
import numpy as np
import torch
from treeneuron.baselines.api import FitRequest
from treeneuron.baselines.node import NodePredictor,node_identity


def run(directory):
    directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    rng=np.random.RandomState(1602033)
    x=rng.normal(size=(1025,16)).astype('float32');g=rng.normal(size=(1024,16)).astype('float32')
    y=(100+3*x[:,0]-x[:,1]).astype('float32');gy=(100+3*g[:,0]-g[:,1]).astype('float32')
    started=time.perf_counter();model=NodePredictor({})
    model.fit(FitRequest(x,y,g,gy,1337,{},dict(max_updates=3),directory))
    prediction=model.predict(g);path=directory/'model.json';t=time.perf_counter();model.save(path)
    save_seconds=time.perf_counter()-t;t=time.perf_counter();restored=NodePredictor.load(path)
    reload_seconds=time.perf_counter()-t;reloaded=restored.predict(g)
    np.testing.assert_array_equal(prediction,reloaded)
    assert len(set(model.training_row_indices))==1025
    assert model.initialization_rows==1000 and model.executed_updates==3
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(status='passed',identity=node_identity(),resources=model.resource_summary(),
        peak_process_rss_bytes=int(peak if platform.system()=='Darwin' else peak*1024),
        peak_scope='entire dedicated process including imports, init, fitting, prediction and reload',
        workload='synthetic F1025 G1024 d16; full 2048x1 depth6 channels3 anchor; one complete F epoch',
        initialization_prefix='ordered first 1000 F rows; no G initialization',
        checkpoint={'json':str(path.resolve()),'npz_sha256':json.loads(path.read_text())['sha256'],
                    'bytes':path.stat().st_size+path.with_suffix('.npz').stat().st_size,
                    'save_seconds':save_seconds,'reload_seconds':reload_seconds,'reload_max_abs':0.},
        worker_run_seconds=time.perf_counter()-started,
        G_selection_exercised=False,G_selection_reason='budget 3 < report frequency 100; latest raw model, no selected checkpoint',
        quality_plateau_calibrated=False,default_patience_training_executed=False,
        forecast_scope='measured one full F1025 epoch; no extrapolation to real datasets or convergence')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();report=run(a.directory);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
