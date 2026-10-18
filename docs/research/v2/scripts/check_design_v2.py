"""Retain analytic/simulated fixtures; these contain no research observations."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from treeneuron.analysis import design_v2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter_ns()
    effects=np.array([-.007,-.002,.002,.007]*4)
    calibration=design_v2.calibrate_variance(effects,runtime_prediction=dict(p50_seconds=1.,p90_seconds=1.),seed=3)
    normal_case=design_v2.choose_teacher_count(calibration,effects,seed=4,feasible_maximum=32)
    sparse=design_v2.simulate_t_gate([0.]*15+[-1.],sd_upper=.1,K=32,seed=88)
    zero={str(n):design_v2.calibrate_variance(np.zeros(n),
        runtime_prediction=dict(p50_seconds=1.,p90_seconds=1.),seed=7) for n in (16,32)}
    package_root=Path(design_v2.__file__).parents[1];digest=hashlib.sha256()
    for path in sorted(package_root.rglob('*.py')):
        digest.update(path.relative_to(package_root).as_posix().encode());digest.update(path.read_bytes())
    result=dict(schema='research/v2-design-software-fixtures',calibration=calibration,
        symmetric_toy=normal_case,sparse_noop_toy=sparse,zero_variance_toys=zero,
        module_sha256=hashlib.sha256(Path(design_v2.__file__).read_bytes()).hexdigest(),
        source_sha256=digest.hexdigest(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),argv=sys.argv,cwd=str(Path.cwd()),
        python=sys.version,executable=sys.executable,platform=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in ('numpy','scipy')},
        elapsed_ns=time.perf_counter_ns()-started,independent_scientific_teachers=0,
        actual_calibration=False,protected_predictions=False,freeze_eligible=False,T_accessed=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(symmetric_toy_status=normal_case['status'],
        sparse_noop_null=sparse['laws']['empirical']['null'],
        sparse_noop_status=sparse['status'],independent_scientific_teachers=0),indent=2))


if __name__=='__main__':main()
