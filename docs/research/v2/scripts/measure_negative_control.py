"""Measure only fixed M1 globally-affine teacher0/repeat0 before matrix admission."""
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


def select_attempts(plan):
    if plan['config']['kind']!='m1_controls':raise ValueError('Requires fixed M1 controls')
    attempts=[a for a in plan['attempts'] if a['population']=='globally_affine'
        and a['teacher_index']==0 and a['repetition']==0]
    if [a['role'] for a in attempts]!=['keep','top','random']:
        raise ValueError('Unexpected fixed negative-control attempt list')
    return attempts


def measurements(plan,config,report,files,*,peak_rss_bytes):
    from treeneuron.config import canonical_hash
    transaction=max((p.stat().st_size for p in files if p.suffix=='.pt'),default=0)+max(
        (p.stat().st_size for p in files if p.name.endswith('.meta.json')),default=0)+max(
        (p.stat().st_size for p in files if p.name.startswith('history-') and p.suffix=='.jsonl'),default=0)+4096
    own=max(m['enclosing_ns'] for m in report['methods'].values() if m['status']=='completed')
    common=dict(schema='research/v2-resource-measurement',source_sha256=plan['source_sha256'],
        execution_identity_hash=canonical_hash(plan['execution_identity']),
        workload=dict(d=config['model']['d'],h=config['model']['h'],
            **{k:config['data'][k] for k in ('n_fit','n_growth','n_validation')},batch_size=config['training']['batch_size']),
        peak_rss_bytes=peak_rss_bytes,retained_bytes=sum(p.stat().st_size for p in files),
        max_checkpoint_transaction_bytes=transaction,training_contract_hash=canonical_hash({'training':config['training']}),
        status='completed_eligible')
    return [dict(**common,phase=phase,elapsed_ns=elapsed) for phase,elapsed in
        (('parent',report['costs']['parent_ns']),('continuation',own))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError(f'{name}=1 required before numerical imports')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise ValueError('Disable MPS fallback')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from treeneuron import train
    from treeneuron.data.synthetic import make_from_config,sample_teacher
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.control_runner import fit_control_pair
    from treeneuron.research.ledger import write_new_json
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.runner import _case_config,_guard,_save_observations,_metrics_and_predictions
    plan=expand(load_research_config(args.config)).to_dict();attempts=select_attempts(plan)
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    config=_case_config(plan['config']['base_experiment'],attempts[0],root/'pair')
    manifest=dict(schema='research/v2-negative-control-measurement',status='running',
        source_sha256=train.source_hash(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        execution_identity=plan['execution_identity'],argv=sys.argv,cwd=str(Path.cwd()),attempts=attempts,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_plan_hash=plan['plan_hash'],
        protected_labels_accessed=False,purpose='fixed first globally-affine case; native resources, no efficacy gate')
    write_new_json(root/'manifest.json',manifest);write_new_json(root/'parent-plan.json',plan)
    try:
        _guard(root);teacher,data=make_from_config(config)
        E=sample_teacher(teacher,plan['config']['evaluation_rows'],attempts[0]['evaluation_seed'],config.data.noise_sigma)
        _save_observations(root/'observations.npz',data,E)
        write_new_json(root/'teacher.json',teacher.to_dict());write_new_json(root/'split.json',data.manifest.to_dict())
        print('Starting fixed globally-affine teacher0/repeat0; keep/top/random',flush=True)
        report=fit_control_pair(config,fitting_view(data),root/'pair',[a['role'] for a in attempts],guard=lambda:_guard(root))
        metrics=_metrics_and_predictions(root/'pair',report,data,E)
        if train.source_hash()!=manifest['source_sha256']:raise ValueError('Scientific source changed during qualification')
        files=[p for p in root.rglob('*') if p.is_file()]
        rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
        records=measurements(plan,config.to_dict(),report,files,peak_rss_bytes=rss)
        references=[]
        for item in records:
            path=root/(item['phase']+'-measurement.json');write_new_json(path,item)
            references.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest.update(status='completed',metrics=metrics,measurements=references,elapsed_ns=time.perf_counter_ns()-STARTED,
            transaction_estimator='conservative largest payload + metadata + complete journal +4096; not measured I/O volume',
            retained_bytes_scope='before qualification measurement/result summaries',
            limitations=['single fixed negative-control teacher/repeat; no Gate A or efficacy claim',
                         'all V/E outcomes are development information; timings subject to contention'])
    except BaseException as error:
        manifest.update(status='failed_or_interrupted',reason=f'{type(error).__name__}: {error}',elapsed_ns=time.perf_counter_ns()-STARTED)
        raise
    finally:
        active=sys.exc_info()[1]
        try:write_new_json(root/'result.json',manifest)
        except BaseException as error:
            if active is None:raise
            active.add_note(f'Qualification result publication also failed: {error}')
    print(json.dumps(dict(status=manifest['status'],elapsed_ns=manifest['elapsed_ns'],peak_rss_bytes=rss)),flush=True)


if __name__=='__main__':main()
