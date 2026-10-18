"""Expand actual sealed metadata with explicitly unqualified support placeholders.

No role array, target or prediction is opened. This is not production admission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from treeneuron.backend import lock_hash, runtime_hash
from treeneuron.config import canonical_hash
from treeneuron.research.ledger import write_new_json
from treeneuron.research.workload import expand_practical
from treeneuron.train import source_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    datasets=('airfoil','concrete','ccpp','parkinsons','bike','energy')
    manifests={name:json.loads((args.data_root/name/'manifest.json').read_text()) for name in datasets}
    support=dict(schema='research/v2-unqualified-support-placeholder',
        families={f'P{i:02}':'Full native quality anchors/support qualification pending' for i in range(14)},
        search_execution_authorized=False)
    write_new_json(args.output/'unqualified-supports.json',support)
    value=expand_practical(manifests,support_lock_sha256=canonical_hash(support),
        source_sha256=source_hash(),runtime_sha256=runtime_hash(),lock_sha256=lock_hash()).to_dict()
    write_new_json(args.output/'plan.json',value)
    proof=dict(schema='research/v2-practical-plan-proof',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        argv=sys.argv,cwd=str(Path.cwd()),source_sha256=source_hash(),plan_hash=value['plan_hash'],
        jobs=len(value['jobs']),tranches=value['tranches'],limits=value['limits'],
        real_transfer_models_fit=0,role_arrays_read=0,protected_labels_read=False,
        raw_files_verified_this_invocation=False,support_status='explicit_unqualified_placeholder',
        execution_authorized=False)
    write_new_json(args.output/'proof.json',proof)
    print(json.dumps(proof,sort_keys=True))


if __name__=='__main__':main()
