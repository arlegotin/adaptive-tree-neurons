"""Explicit treatment changes; no optimizer or absolute-schedule restart."""
import hashlib

import torch

from treeneuron.data.types import array_hash


def reset_stream(state, *, seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('New-stream seed must be a63-bit integer')
    if not state.stage.get('complete') or state.best_snapshot is not None:
        raise ValueError('Stream reset requires a complete selected snapshot')
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(state.order.numel(), generator=generator)
    phases = list(state.growth_metadata.get('stream_phases', []))
    event = dict(schema='research/v2-stream-reset', phase=len(phases)+1,
        absolute_update=state.update, seed=seed, discarded_epoch=state.epoch,
        discarded_cursor=state.cursor, discarded_order=state.order.tolist(),
        discarded_order_sha256=array_hash(state.order.numpy()),
        discarded_generator_sha256=hashlib.sha256(state.sampler_generator.get_state().numpy().tobytes()).hexdigest(),
        new_order_sha256=array_hash(order.numpy()), new_cursor=0, phase_epoch=0,
        optimizer_retained=True, absolute_schedule_retained=True)
    state.order, state.cursor, state.epoch, state.sampler_generator = order, 0, 0, generator
    state.growth_metadata = {**state.growth_metadata, 'stream_phases': [*phases, event]}
    return event
