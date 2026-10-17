"""Translate historical locators without changing frozen scientific identities.

This context carries locations only. It never disables source, receipt, config,
checkpoint or access validation. Relative producer paths use the frozen root;
stage-relative paths are still resolved by their enclosing graph validators.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import os

_LOCATION = ContextVar('evidence_location', default=None)


def locate(path, *, producer=False):
    path = Path(path)
    location = _LOCATION.get()
    if location is None:
        return path
    original, current = location
    if not path.is_absolute():
        return current/path if producer else path
    if path.is_relative_to(current):
        return path
    try:
        return current/path.relative_to(original)
    except ValueError:
        return path


@contextmanager
def relocated(original, current):
    token = _LOCATION.set((Path(original).resolve(), Path(current).resolve()))
    try:
        yield
    finally:
        _LOCATION.reset(token)


def located(function):
    """Establish locations for a protocol operation, without granting authority."""
    @wraps(function)
    def wrapped(protocol, *args, **kwargs):
        from .protocol import FrozenProtocol, _read
        if not isinstance(protocol, FrozenProtocol):
            raise ValueError('Live FrozenProtocol identity required')
        value = _read(protocol.path)
        with relocated(value['payload']['sources']['root'], protocol.path.resolve().parent.parent):
            return function(protocol, *args, **kwargs)
    return wrapped


@contextmanager
def producer_directory(root):
    """Serial launcher CWD for unchanged relative backend receipt configuration."""
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)
