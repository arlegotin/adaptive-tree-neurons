"""One predeclared M1 gamma2/sigma.1 instance/repeat for native qualification.

This composes the existing role-safe control worker. It never substitutes for
the full sixteen-cell/four-instance/R2 control matrix or a Gate A decision.
"""
import time
STARTED=time.perf_counter_ns()
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    if os.environ.get(name)!='1':raise SystemExit(f'{name}=1 required before numerical imports')
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise SystemExit('Disable MPS fallback')
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)

from treeneuron import train
from treeneuron.config import canonical_hash
from treeneuron.research.config import expand,load_research_config
from treeneuron.research.control_runner import fit_control_pair
from treeneuron.research.controls import BASE_ROWS,ControlParameters,make_positive_control
from treeneuron.research.ledger import write_new_json
from treeneuron.research.roles import fitting_view
from treeneuron.research.runner import _case_config,_guard,_save_observations,_metrics_and_predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();plan=expand(load_research_config(args.config)).to_dict()
    if plan['config']['kind']!='m1_controls':raise ValueError('Requires the fixed M1 controls config')
    attempts=[a for a in plan['attempts'] if a['population']=='positive' and a['gamma']==2.
              and a['sigma']==.1 and a['teacher_index']==0 and a['repetition']==0]
    if [a['role'] for a in attempts]!=['keep','top','random','oracle_current','oracle_true']:
        raise ValueError('Unexpected fixed control qualification attempt list')
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    config=_case_config(plan['config']['base_experiment'],attempts[0],root/'pair')
    manifest=dict(schema='research/v2-positive-control-measurement',status='running',
        source_sha256=train.source_hash(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        execution_identity=plan['execution_identity'],argv=sys.argv,cwd=str(Path.cwd()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parent_plan_hash=plan['plan_hash'],attempts=attempts,protected_labels_accessed=False,
        purpose='native resource/control qualification; first declared instance/repeat at gamma2/sigma.1')
    write_new_json(root/'manifest.json',manifest);write_new_json(root/'parent-plan.json',plan)
    try:
        _guard(root)
        a=attempts[0]
        # Commit the prescribed raw teacher before drawing any observations.
        factors=np.random.default_rng(a['teacher_seed']).uniform(.95,1.05,8)
        factors[1]=factors[0]
        parameters=ControlParameters(np.asarray(BASE_ROWS,dtype=np.float64),
            np.asarray((.5,.5,1.,.2,-.2,.1,-.1,.1))*factors,2.)
        write_new_json(root/'teacher.json',dict(base=parameters.base_rows.tolist(),head=parameters.head.tolist(),
            contrasts=parameters.contrasts.tolist(),gates=parameters.gates.tolist(),
            gamma=2.,sigma=.1,instance_seed=a['teacher_seed'],data_seed=a['data_seed'],
            generating_ids=list(parameters.generating_ids),committed_before_observation_generation=True))
        control=make_positive_control(instance_seed=a['teacher_seed'],
            data_seed=a['data_seed'],gamma=2.,sigma=.1)
        if any(not np.array_equal(getattr(parameters,k),getattr(control.parameters,k))
                for k in ('base_rows','head','contrasts','gates')):
            raise ValueError('Generated teacher differs from its committed raw coefficients')
        data,evaluation=control.development,control.evaluation
        _save_observations(root/'observations.npz',data,evaluation)
        write_new_json(root/'split.json',data.manifest.to_dict())
        print('Starting predeclared planted instance0/repeat0 gamma2/sigma.1; five roles',flush=True)
        report=fit_control_pair(config,fitting_view(data),root/'pair',[a['role'] for a in attempts],
            control=control.parameters,guard=lambda:_guard(root))
        metrics=_metrics_and_predictions(root/'pair',report,data,evaluation)
        files=[p for p in root.rglob('*') if p.is_file()]
        transaction=max((p.stat().st_size for p in files if p.suffix=='.pt'),default=0)+max(
            (p.stat().st_size for p in files if p.name.endswith('.meta.json')),default=0)+max(
            (p.stat().st_size for p in files if p.name.startswith('history-') and p.suffix=='.jsonl'),default=0)+4096
        rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
        measurement=dict(schema='research/v2-resource-measurement',source_sha256=train.source_hash(),
            execution_identity_hash=canonical_hash(plan['execution_identity']),
            workload=dict(d=2,h=8,n_fit=512,n_growth=256,n_validation=256,batch_size=config.training.batch_size),
            phase='continuation',elapsed_ns=max(m['enclosing_ns'] for m in report['methods'].values()
                if m['status']=='completed'),peak_rss_bytes=rss,
            training_contract_hash=canonical_hash({'training':config.to_dict()['training']}),
            retained_bytes=sum(p.stat().st_size for p in files),max_checkpoint_transaction_bytes=transaction,
            status='completed_eligible')
        write_new_json(root/'measurement.json',measurement)
        manifest.update(status='completed',metrics=metrics,
            sampled_noise_NMSE=float(np.mean((evaluation.y-evaluation.noise_free_target)**2)/np.var(data.F.y)),
            measurement=dict(path=str((root/'measurement.json').resolve()),
                sha256=hashlib.sha256((root/'measurement.json').read_bytes()).hexdigest()),
            transaction_estimator='conservative sum of largest payload, metadata, complete history journal and4096-byte publication allowance; not measured write volume',
            elapsed_ns=time.perf_counter_ns()-STARTED,
            limitations=['single predeclared constructed instance/repeat; no Gate A or population efficacy claim',
                         'information-advantaged true-contrast oracle is diagnostic only',
                         'timings subject to contention; parent is deliberately unfitted at update0',
                         'all V/E outcomes are development information'])
    except BaseException as error:
        manifest.update(status='failed_or_interrupted',reason=f'{type(error).__name__}: {error}',
            elapsed_ns=time.perf_counter_ns()-STARTED)
        raise
    finally:
        active=sys.exc_info()[1]
        try:write_new_json(root/'result.json',manifest)
        except BaseException as error:
            if active is None:raise
            active.add_note(f'Qualification result publication also failed: {error}')
    print(json.dumps(dict(status=manifest['status'],elapsed_ns=manifest['elapsed_ns'],
        retained_bytes=measurement['retained_bytes'],peak_rss_bytes=rss)),flush=True)


if __name__=='__main__':main()
