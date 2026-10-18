"""Source-bound observations around real M2 workers; no production admission."""
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
from .ledger import write_new_json
from .opportunity_workers import OpportunityWorkers
from .workqueue import _inventory


class ObservedOpportunityWorkers:
    """Add timing/file observations without changing native fitting or RNG.

    Handler clocks exclude outer queue dependency preparation/publication;
    native profile qualification must retain those queue and invocation clocks.
    A completed ineligible procedure is never promoted to an eligible model.
    """
    def __init__(self, queue_plan, *, guard=lambda:None):
        self.identity=copy.deepcopy(queue_plan['identity'])
        if 'source_sha256' not in self.identity or 'execution_identity' not in self.identity:
            raise ValueError('Resource observations require explicit source/runtime identities')
        self.handler=OpportunityWorkers(queue_plan,guard=guard)

    def _check(self):
        if self.identity['source_sha256']!=train.source_hash():raise ValueError('Observed worker source differs')
        if execution_identity(self.identity['study_config'])!=self.identity['execution_identity']:
            raise ValueError('Observed worker runtime/receipt identity differs')
        if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
            raise ValueError('Observed numerical workers require one thread')

    def __call__(self, job, directory, dependencies, attempt):
        self._check();started=time.perf_counter_ns()
        value=self.handler(job,directory,dependencies,attempt)
        self._check();directory=Path(directory)
        inventory=_inventory(directory)
        files=inventory['files']
        transaction=(max((f['bytes'] for f in files if f['path'].endswith('.pt')),default=0)
            +max((f['bytes'] for f in files if f['path'].endswith('.meta.json')),default=0)
            +max((f['bytes'] for f in files if Path(f['path']).name.startswith('history-')
                and f['path'].endswith('.jsonl')),default=0)+4096)
        report=value.get('report',{})
        updates=report.get('executed_updates',0) if job['kind']=='parent' else (
            report.get('executed_stage_updates',0) if job['kind']=='fit' else 0)
        if type(updates) is not int or updates<0:raise ValueError('Invalid observed execution counter')
        observation=dict(schema='research/v2-queue-resource-observation',job_id=job['job_id'],
            job_sha256=canonical_hash(job),source_sha256=self.identity['source_sha256'],
            execution_identity_hash=canonical_hash(self.identity['execution_identity']),
            attempt=attempt,procedure_status=value['status'],executed_updates=updates,
            handler_and_observation_ns=time.perf_counter_ns()-started,
            peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                *(1 if sys.platform=='darwin' else 1024),
            retained_bytes_before_observation=sum(f['bytes'] for f in files),raw_inventory=inventory,
            estimated_checkpoint_transaction_bytes=transaction,
            transaction_estimator='largest retained payload +metadata +complete history journal +4096; not measured write volume',
            elapsed_scope='native handler plus identity/file observation before observation publication; outer queue and invocation clocks separate',
            memory_scope='process lifetime peak own RSS; not incremental or whole-job aggregate RAM',
            production_admission=False,T_accessed=False)
        path=directory/'resource-observation.json';write_new_json(path,observation)
        return dict(value,resource_observation=dict(path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
