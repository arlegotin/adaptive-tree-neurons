"""Observe the pinned native initialization sampler without drawing randomness.

Install only in a dedicated method worker. This is a scoped observer, not a
replacement initialization policy or a quality-qualification decision.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import math
import threading
import weakref

import numpy as np


_INSTALL_LOCK = threading.Lock()


def _mapping_hash(positions):
    return hashlib.sha256(np.asarray(positions, dtype='<i8').tobytes()).hexdigest()


class NativeInitializationTrace:
    """Trace actual F positions selected by the native RAM-limited fitter."""

    def __init__(self, *, n_fit, dataset_class, fitter_class, creator_class,
                 standalone_F_dataset=None):
        if type(n_fit) is not int or not 2 <= n_fit <= 20000:
            raise ValueError('Initialization observation requires 2..20000 F rows')
        self.n_fit = n_fit
        self.dataset_class = dataset_class
        self.fitter_class = fitter_class
        self.creator_class = creator_class
        self._standalone = standalone_F_dataset
        self._entered = False
        self._used = False
        self._active = False
        self._maps = weakref.WeakKeyDictionary()
        self._stack = []
        self._record = dict(schema='research/v2-realmlp-initialization',
            scope='native_creator' if standalone_F_dataset is None else 'standalone_F_engineering',
            fit_rows=n_fit, creator_calls=0, root_F_positions=None,
            calls=[], subsamples=[], status='not_started', quality_eligible=False)

    def __enter__(self):
        if self._used or not _INSTALL_LOCK.acquire(blocking=False):
            raise ValueError('An initialization observer is already installed or was used')
        try:
            self._originals = (
                self.dataset_class.get_sub_dataset,
                self.fitter_class.fit_transform_subsample,
                self.creator_class.create_model)
            self._creator_code = self._originals[2].__code__
            self._sampler_code = self.fitter_class._fit_transform_subsample.__code__
            if self._standalone is not None:
                if self._standalone.n_samples != self.n_fit:
                    raise ValueError('Standalone initialization fixture is not the declared F table')
                self._maps[self._standalone] = np.arange(self.n_fit, dtype=np.int64)
                self._record['root_F_positions'] = list(range(self.n_fit))
                self._active = True

            @functools.wraps(self._originals[0])
            def subset(ds, indices):
                if not self._active:
                    return self._originals[0](ds, indices)
                frame = inspect.currentframe()
                try:
                    caller = frame.f_back.f_code
                finally:
                    del frame
                return self._subset(ds, indices, caller)

            @functools.wraps(self._originals[1])
            def fitter(fit, ds, ram_limit_gb, needs_tensors=True):
                if not self._active:
                    return self._originals[1](fit, ds, ram_limit_gb, needs_tensors)
                return self._fit(fit, ds, ram_limit_gb, needs_tensors)

            @functools.wraps(self._originals[2])
            def creator(instance, *args, **kwargs):
                if self._standalone is not None or self._record['creator_calls'] or self._active:
                    raise ValueError('Initialization observer supports exactly one native creator')
                self._record['creator_calls'] += 1
                self._active = True
                try:
                    result = self._originals[2](instance, *args, **kwargs)
                    if self._record['root_F_positions'] is None:
                        raise ValueError('Native creator did not expose its declared F selection')
                    return result
                finally:
                    self._active = False
                    self._maps.clear()

            self.dataset_class.get_sub_dataset = subset
            self.fitter_class.fit_transform_subsample = fitter
            self.creator_class.create_model = creator
            self._used = self._entered = True
            self._record['status'] = 'running'
            return self
        except BaseException:
            _INSTALL_LOCK.release()
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.dataset_class.get_sub_dataset = self._originals[0]
            self.fitter_class.fit_transform_subsample = self._originals[1]
            self.creator_class.create_model = self._originals[2]
            self._record['status'] = 'completed' if exc_type is None else 'failed'
            if exc_type is not None:
                self._record['error_type'] = exc_type.__module__ + '.' + exc_type.__qualname__
        finally:
            self._active = self._entered = False
            self._maps.clear()
            self._stack.clear()
            self._standalone = None
            _INSTALL_LOCK.release()
        return False

    def F_positions(self, ds):
        if not self._entered or ds not in self._maps:
            raise ValueError('Initialization dataset has no observed F lineage')
        return self._maps[ds].copy()

    def snapshot(self):
        if self._entered:
            raise ValueError('Close initialization observation before publishing its trace')
        return copy.deepcopy(self._record)

    def _subset(self, ds, indices, caller):
        local = indices.detach().cpu().numpy().copy()
        if (local.ndim != 1 or local.dtype.kind not in 'iu'
                or np.any(local < 0) or np.any(local >= ds.n_samples)
                or len(np.unique(local)) != len(local)):
            raise ValueError('Unexpected native initialization row selector')
        local = local.astype(np.int64)
        if caller is self._creator_code:
            if self._record['root_F_positions'] is not None or self._stack:
                raise ValueError('Unexpected repeated native F selection')
            if not np.array_equal(local, np.arange(self.n_fit, dtype=np.int64)):
                raise ValueError('Native initialization selection differs from the declared F rows')
            positions = local.copy()
        elif caller is self._sampler_code and self._stack and ds in self._maps:
            positions = self._maps[ds][local]
        else:
            raise ValueError('Unrecognized native initialization row-selection lineage')
        result = self._originals[0](ds, indices)
        if result.n_samples != len(positions):
            raise ValueError('Native row selector returned an unexpected row count')
        self._maps[result] = positions
        if caller is self._creator_code:
            self._record['root_F_positions'] = positions.tolist()
        else:
            self._stack[-1]['positions'] = positions
            self._record['subsamples'].append(dict(call_id=self._stack[-1]['id'],
                input_rows=ds.n_samples, output_rows=len(positions),
                input_indices=local.tolist(), F_positions=positions.tolist(),
                input_mapping_sha256=_mapping_hash(self._maps[ds]),
                output_mapping_sha256=_mapping_hash(positions)))
        return result

    def _fit(self, fitter, ds, ram_limit_gb, needs_tensors):
        if ds not in self._maps:
            raise ValueError('Native fitter input has unobserved F lineage')
        positions = self._maps[ds]
        if len(positions) != ds.n_samples:
            raise ValueError('Native fitter row count differs from its F lineage')
        budget = float(ram_limit_gb)
        if math.isnan(budget) or budget < 0:
            raise ValueError('Invalid native initialization RAM budget')
        call = dict(id=len(self._record['calls']),
            parent_call_id=self._stack[-1]['id'] if self._stack else None,
            fitter_type=type(fitter).__module__ + '.' + type(fitter).__qualname__,
            scope_names=list(fitter.scope_names), input_rows=len(positions),
            input_mapping_sha256=_mapping_hash(positions),
            ram_limit_gb=budget if math.isfinite(budget) else 'infinity',
            needs_tensors=bool(needs_tensors), status='running')
        self._record['calls'].append(call)
        context = dict(id=call['id'], positions=positions)
        self._stack.append(context)
        try:
            layer, result = self._originals[1](fitter, ds, ram_limit_gb, needs_tensors)
            output_positions = context['positions']
            if result.n_samples != len(output_positions):
                raise ValueError('Native fitter changed rows without an observed selector')
            if result in self._maps and not np.array_equal(self._maps[result], output_positions):
                raise ValueError('Native fitter output has ambiguous F lineage')
            self._maps[result] = output_positions
            call.update(status='completed', output_rows=result.n_samples,
                output_mapping_sha256=_mapping_hash(output_positions))
            if len(self._stack) > 1:
                self._stack[-2]['positions'] = output_positions
            return layer, result
        except BaseException as error:
            call.update(status='failed', error_type=type(error).__module__ + '.' + type(error).__qualname__)
            raise
        finally:
            self._stack.pop()
