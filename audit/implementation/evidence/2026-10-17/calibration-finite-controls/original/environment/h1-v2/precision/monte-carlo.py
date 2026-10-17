"""Numerical precision/cost on declared E-block inputs, never teacher power."""
import json,time,resource,math
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.pilot.design import _power
from treeneuron.experiments.paired import file_hash,write_json

base=Path('environment/h1-v2/precision');work=json.loads((base/'workload.json').read_text());data=json.loads((base/'loss-timing.json').read_text())
assert train.source_hash()==data['source_sha256']=='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c'
report=dict(scope=work['scope'],input_unit='disjoint E-row block conditional on one fixed fitted development teacher; NOT independent teachers',
    source_sha256=train.source_hash(),input_sha256=file_hash(base/'loss-timing.json'),workload_sha256=file_hash(base/'workload.json'),
    wrapper_sha256=file_hash(Path(__file__)),bootstrap_target=work['bootstrap_declared_relative_endpoint_mcse_target'],
    bootstrap_cells=[],power_target=work['power_declared_mc_half_width_target'],power_cells=[])
started=time.perf_counter_ns();chosen=None
for iterations in work['bootstrap_iterations']:
    current=[]
    for p in data['problems']:
        for n in work['bootstrap_shape_inputs']:
            h=np.asarray(p['block_inputs']['h1_means'][:n],dtype=np.float64)
            s=np.asarray(p['block_inputs']['static_means'][:n],dtype=np.float64)
            for seed in work['bootstrap_seeds']:
                tick=time.perf_counter_ns();indices=np.random.default_rng(seed).integers(n,size=(iterations,n))
                dh=h[indices];ds=s[indices]
                samples=[dh.std(axis=1,ddof=1),dh.mean(axis=1),ds.mean(axis=1)]
                bounds=[np.quantile(a,[.025,.975]) for a in samples]
                sections=np.array([np.concatenate([np.quantile(a[part],[.025,.975]) for a in samples])
                    for part in np.array_split(np.arange(iterations),10)])
                errors=sections.std(axis=0,ddof=1)/math.sqrt(10)
                widths=np.repeat(np.array([v[1]-v[0] for v in bounds]),2)
                nondegenerate=widths>0
                relative=float(np.max(errors[nondegenerate]/widths[nondegenerate])) if nondegenerate.any() else None
                cell=dict(problem=p['problem'],input_blocks=n,iterations=iterations,seed=seed,
                    numeric_inputs=dict(h1=h.tolist(),static=s.tolist()),metric_order=['conditional_block_SD_H1','conditional_block_mean_H1','conditional_block_mean_static'],
                    intervals=[v.tolist() for v in bounds],ten_section_endpoints=sections.tolist(),endpoint_mcse=errors.tolist(),
                    maximum_absolute_mcse=float(errors.max()),maximum_relative_to_interval_width=relative,
                    degenerate_endpoint_indices=np.flatnonzero(~nondegenerate).tolist(),elapsed_ns=time.perf_counter_ns()-tick)
                report['bootstrap_cells'].append(cell);current.append(cell)
                del indices,dh,ds,samples,sections
    if all(c['maximum_relative_to_interval_width'] is not None and c['maximum_relative_to_interval_width']<=report['bootstrap_target'] for c in current):
        chosen=iterations;break
report['bootstrap_choice']=dict(iterations=chosen or work['bootstrap_iterations'][-1],relative_target_reached=chosen is not None,
    absolute_endpoint_tolerance=2*max(c['maximum_absolute_mcse'] for c in report['bootstrap_cells'] if c['iterations']==(chosen or work['bootstrap_iterations'][-1])),
    limitation='Empirical numerical calibration margin on these block arrays. It neither supplies teacher variance nor validates population coverage/unseen tails; future genuinepilot endpoint error must pass its own declared gate.')
values=data['problems'][0]['block_inputs']['h1_means'];sd=float(np.std(values,ddof=1));chosen_power=None
for iterations in work['power_iterations']:
    current=[]
    for ratio in work['power_diagnostic_shift_sd_ratios']:
        for seed in work['power_seeds']:
            tick=time.perf_counter_ns();payload=_power(values,16,ratio*sd,sd,.05,iterations,seed)
            cell=dict(iterations=iterations,seed=seed,diagnostic_shift_sd_ratio=ratio,
                input_unit='conditional disjoint E-block mean; no prospective study-power interpretation',
                raw_existing_function_payload=payload,elapsed_ns=time.perf_counter_ns()-tick)
            report['power_cells'].append(cell);current.append(cell)
    if all(c['raw_existing_function_payload'].get('supported') and c['raw_existing_function_payload']['mc_half_width']<=report['power_target'] for c in current):
        chosen_power=iterations;break
report['power_choice']=dict(iterations=chosen_power or work['power_iterations'][-1],declared_mc_half_width_target=report['power_target'],
    target_reached=chosen_power is not None,inputs=values,diagnostic_input_sd=sd,
    limitation='Finite Monte Carlo iteration/precision policy only. The future planning effect/K/power must come from genuine teacher evidence and prescribed signed-independent rules.')
report.update(enclosing_ns=time.perf_counter_ns()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write_json(base/'monte-carlo.json',report);print('completed numeric MC calibration',report['bootstrap_choice'],report['power_choice']['iterations'],flush=True)
