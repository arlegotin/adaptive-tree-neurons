"""Containing native M3 handler observations; no scientific admission authority."""
import copy
import hashlib
from pathlib import Path
import resource
import sys
import time

import torch

from treeneuron import train
from treeneuron.config import canonical_hash
from .identity import execution_identity
from .joint_workers import JointWorkers
from .ledger import write_new_json
from .workqueue import _inventory


class ObservedJointWorkers:
    def __init__(self,plan,*,guard=lambda:None,handler=None):
        self.identity=copy.deepcopy(plan['identity'])
        if 'source_sha256' not in self.identity or 'execution_identity' not in self.identity:
            raise ValueError('M3 observations require explicit source/runtime identities')
        if handler is not None and not callable(handler):raise ValueError('Callable native qualification handler required')
        self.handler=JointWorkers(plan,guard=guard) if handler is None else handler

    def _check(self):
        if (self.identity['source_sha256']!=train.source_hash()
                or execution_identity(self.identity['study_config'])!=self.identity['execution_identity']):
            raise ValueError('Observed M3 source/runtime/receipt identity changed')
        if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
            raise ValueError('M3 native observations require one intra/inter-op thread')

    def __call__(self,job,directory,dependencies,attempt):
        self._check();started=time.perf_counter_ns()
        value=self.handler(job,directory,dependencies,attempt)
        self._check();directory=Path(directory);inventory=_inventory(directory);files=inventory['files']
        transaction=(max((f['bytes'] for f in files if f['path'].endswith('.pt')),default=0)
            +max((f['bytes'] for f in files if f['path'].endswith('.meta.json')),default=0)
            +max((f['bytes'] for f in files if Path(f['path']).name.startswith('history-') and f['path'].endswith('.jsonl')),default=0)+4096)
        report=value.get('report',{})
        updates=(report.get('executed_updates',0) if job['kind']=='parent' else
            report.get('executed_stage_updates',0) if job['kind'] in ('fit','probe') else 0)
        prefix=report.get('parent_update',0) if job['kind'] in ('fit','probe') else 0
        if type(updates) is not int or updates<0 or type(prefix) is not int or prefix<0:
            raise ValueError('Explicit nonnegative native execution/prefix counters required')
        observation=dict(schema='research/v2-M3-resource-observation',job_id=job['job_id'],
            job_sha256=canonical_hash(job),source_sha256=self.identity['source_sha256'],
            execution_identity_hash=canonical_hash(self.identity['execution_identity']),attempt=attempt,
            procedure_status=value['status'],executed_updates=updates,parent_selected_update=prefix,
            handler_and_observation_ns=time.perf_counter_ns()-started,
            peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
            retained_bytes_before_observation=sum(f['bytes'] for f in files),raw_inventory=inventory,
            estimated_checkpoint_transaction_bytes=transaction,
            transaction_estimator='Largest retained payload +metadata +complete history journal +4096; not measured write volume',
            elapsed_scope='Native handler/source/file observations before publication; outer queue/preparation/process clocks separate',
            memory_scope='Whole process lifetime peak own RSS; not incremental or aggregate descendant RAM',
            production_admission=False,T_accessed=False)
        path=directory/'resource-observation.json';write_new_json(path,observation)
        return dict(value,resource_observation=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
