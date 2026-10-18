"""Inventory a finished M1 producer and retain exact no-op/counter checks."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True,type=Path);parser.add_argument('--expected-plan-hash',required=True)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    from treeneuron.config import canonical_hash,canonical_json
    from treeneuron.research.control_audit import _inactive_study
    root=args.study.resolve();args.output.mkdir(parents=True,exist_ok=False)
    with _inactive_study(root):
        plan=json.loads((root/'plan.json').read_text())
        if (plan['plan_hash']!=args.expected_plan_hash or plan['plan_hash']!=canonical_hash(
                {k:v for k,v in plan.items() if k!='plan_hash'})):
            raise ValueError('Producer plan commitment differs')
        terminal=json.loads((root/'result-0.json').read_text())
        if terminal['status']!='completed' or terminal['completed_final_fits']!=256:
            raise ValueError('Requires the complete pinned M1 run')
        groups={}
        for a in plan['attempts']:groups.setdefault((a['cell'],a['teacher_index'],a['repetition']),[]).append(a)
        records=[]
        for (cell,t,r),cases in groups.items():
            pair=root/cell/f'teacher-{t}'/f'repeat-{r}-attempt-0'
            report=json.loads((pair/'control-pair.json').read_text())
            with np.load(pair/'predictions.npz',allow_pickle=False) as values:
                identical={a['role']:{split:bool(np.array_equal(values[a['role']+'_'+split],values['keep_'+split]))
                    for split in ('V','E')} for a in cases}
            records.append(dict(cell=cell,teacher_index=t,repetition=r,
                roles={k:{f:v[f] for f in ('selected_stage_update','executed_stage_updates','total_parameters')}
                    for k,v in report['methods'].items()},predictions_identical_to_keep=identical))
        bycell={}
        for cell in dict.fromkeys(r['cell'] for r in records):
            rows=[r for r in records if r['cell']==cell];roles=list(rows[0]['roles'])
            bycell[cell]=dict(selected_stage_zero={role:sum(r['roles'][role]['selected_stage_update']==0 for r in rows) for role in roles},
                E_prediction_vectors_identical_to_keep={role:sum(r['predictions_identical_to_keep'][role]['E'] for r in rows) for role in roles})
        checks=dict(schema='research/v2-M1-retained-execution-checks',records=records,by_cell=bycell,model_reexecuted=False,T_accessed=False)
        (args.output/'execution-checks.json').write_text(canonical_json(checks)+'\n')
        files=[]
        for path in sorted(root.rglob('*')):
            if path.is_symlink():raise ValueError('Unqualified symlink')
            if path.is_file():files.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        inventory=dict(schema='research/v2-complete-producer-inventory',root=str(root),files=files,
            total_files=len(files),total_bytes=sum(x['bytes'] for x in files),source_sha256=plan['source_sha256'],
            plan_hash=plan['plan_hash'],T_accessed=False)
        (args.output/'producer-inventory.json').write_text(canonical_json(inventory)+'\n')
    receipt=dict(schema='research/v2-M1-execution-check-invocation',argv=sys.argv,cwd=str(Path.cwd()),
        python=sys.version,executable=sys.executable,platform=platform.platform(),numpy_version=np.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()},
        plan_hash=plan['plan_hash'],source_sha256=plan['source_sha256'],model_reexecuted=False,T_accessed=False)
    (args.output/'invocation.json').write_text(canonical_json(receipt)+'\n')
    print(json.dumps(dict(total_files=inventory['total_files'],total_bytes=inventory['total_bytes'],by_cell=bycell),indent=2))


if __name__=='__main__':main()
