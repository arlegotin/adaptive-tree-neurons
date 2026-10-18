"""Profile bounded checkpoint IO; never fit or evaluate predictive labels."""
import argparse
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    if os.environ.get(key) != '1':
        raise SystemExit(f'{key}=1 required')

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from treeneuron import train
from treeneuron.checkpoint import _descriptor, load_checkpoint, save_checkpoint, prune_checkpoints
from treeneuron.research.ledger import write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markers', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    _, descriptor, _ = _descriptor(args.checkpoint)
    state = load_checkpoint(args.checkpoint, expected_hashes=descriptor['state']['identity_hashes'], device='cpu')
    if args.markers < 0:
        raise ValueError('markers must be nonnegative')
    from treeneuron.history import History
    original_history = state.history
    history = History(state.history)
    for index in range(args.markers):
        history.append(dict(type='checkpoint_profile_marker', index=index, payload='x'*200))
    results = []
    for label, value in (('journal_history', history), ('equivalent_inline_history', list(history)), ('empty_history', [])):
        root = args.output/label
        root.mkdir()
        state.history = value
        profile = cProfile.Profile()
        clock = time.perf_counter_ns()
        profile.enable()
        for _ in range(10):
            save_checkpoint(state, root/'last.json')
            prune_checkpoints(root)
        profile.disable()
        elapsed = time.perf_counter_ns()-clock
        profile.dump_stats(str(root/'profile.pstats'))
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(30)
        (root/'profile.txt').write_text(stream.getvalue())
        results.append(dict(case=label, publications=10, elapsed_ns=elapsed,
            history_events=len(value), retained_bytes=sum(p.stat().st_size for p in root.iterdir())))
    state.history = original_history
    write_new_json(args.output/'result.json', dict(schema='research/v2-checkpoint-profile',
        source_sha256=train.source_hash(), checkpoint=str(args.checkpoint.resolve()),
        argv=__import__('sys').argv, cwd=str(Path.cwd()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        synthetic_profile_marker_events=args.markers,
        pointer_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        selected_update=state.update, results=results,
        limitation='Contended diagnostic; no training, predictive target access or formal timing claim'))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
