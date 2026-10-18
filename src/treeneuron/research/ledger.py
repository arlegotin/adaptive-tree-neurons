"""Append-only attempt transitions, paid costs, and identity-checked recovery."""
import json
import os
from pathlib import Path
import time

from treeneuron.config import canonical_hash, canonical_json

TERMINAL = frozenset(('completed_eligible', 'completed_ineligible', 'timeout',
    'resource_exhausted', 'numerical_failure', 'input_or_dependency_failure',
    'interrupted_recoverable', 'irrecoverable_missing'))


class StudyLease:
    """OS lease releases on process death; lease metadata is not a stale lock."""
    def __init__(self, directory):
        self.path = Path(directory)/'worker-lease.json'
        self.stream = None

    def __enter__(self):
        import fcntl
        self.stream = self.path.open('a+')
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.stream.close()
            self.stream = None
            raise ValueError('Study has an active worker lease') from error
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(canonical_json(dict(pid=os.getpid(), acquired_utc_ns=time.time_ns()))+'\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return self

    def __exit__(self, *error):
        import fcntl
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(canonical_json(value)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


class AttemptLedger:
    def __init__(self, directory, plan):
        self.directory, self.plan = Path(directory), plan
        self.ids = tuple(a['attempt_id'] for a in plan['attempts'])
        if len(self.ids) != len(set(self.ids)):
            raise ValueError('Duplicate planned attempt IDs')
        self.events = []
        previous = '0'*64
        for line in (self.directory/'events.jsonl').read_text().splitlines():
            event = json.loads(line)
            digest = event.pop('event_hash')
            if event['previous_hash'] != previous or canonical_hash(event) != digest:
                raise ValueError('Attempt event commitment mismatch')
            self._validate(event)
            event['event_hash'] = digest
            self.events.append(event)
            previous = digest

    @classmethod
    def create(cls, directory, plan):
        directory = Path(directory)
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ValueError('Study output already exists; use identity-checked resume') from error
        write_new_json(directory/'plan.json', plan)
        (directory/'events.jsonl').touch(exist_ok=False)
        return cls(directory, plan)

    @classmethod
    def resume(cls, directory, plan):
        saved = json.loads((Path(directory)/'plan.json').read_text())
        if canonical_json(saved) != canonical_json(plan):
            raise ValueError('Cannot resume changed source/config/attempt plan')
        return cls(directory, plan)

    def _validate(self, event):
        if event['attempt_id'] not in self.ids or event['status'] not in TERMINAL | {'running'}:
            raise ValueError('Unknown attempt or status')
        previous = [e for e in self.events if e['attempt_id'] == event['attempt_id']]
        if previous and previous[-1]['status'] in TERMINAL-{'interrupted_recoverable'}:
            raise ValueError('Terminal attempt cannot be rewritten')
        if previous and previous[-1]['status'] == 'interrupted_recoverable':
            if (event['status'] != 'running' or event.get('recovery_attempt') != 1
                    or any(e.get('recovery_attempt') == 1 and e['status'] == 'running' for e in previous)):
                raise ValueError('Interrupted recovery requires the one declared same-seed retry')
        if type(event['elapsed_ns']) is not int or event['elapsed_ns'] < 0:
            raise ValueError('Invalid attempt cost')
        if previous and event['elapsed_ns'] < previous[-1]['elapsed_ns']:
            raise ValueError('Paid attempt time cannot decrease')

    def record(self, attempt_id, status, *, elapsed_ns, **fields):
        reserved = {'attempt_id', 'status', 'elapsed_ns', 'previous_hash', 'event_hash'}
        if reserved.intersection(fields):
            raise ValueError('Reserved attempt fields')
        event = dict(attempt_id=attempt_id, status=status, elapsed_ns=elapsed_ns,
            previous_hash=self.events[-1]['event_hash'] if self.events else '0'*64, **fields)
        self._validate(event)
        event['event_hash'] = canonical_hash(event)
        with (self.directory/'events.jsonl').open('a') as stream:
            stream.write(canonical_json(event)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)

    @property
    def pending_ids(self):
        latest = {e['attempt_id']: e['status'] for e in self.events}
        done = {i for i, status in latest.items() if status in TERMINAL-{'interrupted_recoverable'}}
        return tuple(i for i in self.ids if i not in done)

    @property
    def charged_ns(self):
        return sum(max((e['elapsed_ns'] for e in self.events if e['attempt_id']==i), default=0)
                   for i in self.ids)
