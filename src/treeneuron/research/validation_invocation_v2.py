"""Private prerequisite reuse, discarded at the end of one public invocation.

Only a successful real loader can populate an entry. Frozen JSON prevents a
consumer from changing the validated value; source and retained input guards
run before reuse and when the outer invocation closes. This is neither a
persistent qualification cache nor a caller-supplied authorization context.
"""
from contextvars import ContextVar
from functools import wraps
import json
import os
import threading

from treeneuron.config import canonical_json
from .deployment_worker_v2 import _source


_ACTIVE = ContextVar('research_v2_validation_invocation', default=None)


class _Invocation:
    def __init__(self):
        self.source = _source()
        self.owner = (os.getpid(), threading.get_ident())
        self.active = True
        self.entries = {}

    def check(self):
        if not self.active or self.owner != (os.getpid(), threading.get_ident()):
            raise ValueError('Closed or foreign validation invocation')
        if _source() != self.source:
            raise ValueError('Source changed during validation invocation')

    def recheck(self):
        self.check()
        for frozen, guard in tuple(self.entries.values()):
            guard(json.loads(frozen))
        self.check()

    def close(self):
        try:
            self.recheck()
        finally:
            self.active = False
            self.entries.clear()


def _within_validation(function):
    @wraps(function)
    def invoke(*args, **kwargs):
        current = _ACTIVE.get()
        if current is not None:
            current.check()
            return function(*args, **kwargs)
        context = _Invocation(); token = _ACTIVE.set(context)
        try:
            return function(*args, **kwargs)
        finally:
            try:
                context.close()
            finally:
                _ACTIVE.reset(token)
    return invoke


def _recheck_validation():
    """Finish all retained-input guards before immutable output publication."""
    context = _ACTIVE.get()
    if context is None:
        raise ValueError('Actual public validation invocation required')
    context.recheck()


def _load_once(kind, arguments, load, *, guard, check=None):
    context = _ACTIVE.get()
    if context is None:
        raise ValueError('Actual public validation invocation required')
    context.check()
    key = canonical_json(dict(kind=kind, arguments=arguments))
    entry = context.entries.get(key)
    if entry is None:
        value = load()
        frozen = canonical_json(value)
        guard(json.loads(frozen))
        context.check()
        context.entries[key] = (frozen, guard)
    else:
        frozen, _ = entry
    value = json.loads(frozen)
    if check is not None:
        check(value)
    context.check()
    return value
