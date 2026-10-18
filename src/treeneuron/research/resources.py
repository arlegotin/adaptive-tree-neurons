"""Resource admission is evidence-based and separate from efficacy outcomes."""
import json
import hashlib
from pathlib import Path
import shutil
import time

from treeneuron.config import canonical_hash
from .ledger import AttemptLedger,write_new_json


class ResourceAvailabilityError(ValueError):
    def __init__(self,decision):
        super().__init__('Resource admission forecast exceeds free disk including margin')
        self.decision=decision


def case_id(attempt):
    return canonical_hash({k:attempt[k] for k in ('cell','teacher_index','repetition')})


def _binding(plan,admission_bytes):
    return dict(schema='research/v2-resource-admission-binding',source_sha256=plan['source_sha256'],
        plan_hash=plan['plan_hash'],admission_sha256=hashlib.sha256(admission_bytes).hexdigest(),
        admission_record=json.loads(admission_bytes))


def _check_binding(output,plan,admission_bytes,ledger):
    path=Path(output)/'resource-admission-binding.json'
    if path.exists():
        if path.is_symlink() or json.loads(path.read_text())!=_binding(plan,admission_bytes):
            raise ValueError('Original resource admission commitment differs')
    elif ledger.events:
        raise ValueError('Missing original resource admission commitment after paid work')


def publish_availability(output,decision,ledger):
    """Caller holds StudyLease; readiness records never rewrite paid attempts."""
    value=dict(decision)
    event_hash=ledger.events[-1]['event_hash'] if ledger.events else '0'*64
    if value['ledger_event_hash'] not in (None,event_hash):
        raise ValueError('Availability decision no longer matches the locked ledger')
    value['ledger_event_hash']=event_hash
    root=Path(output)
    admission_bytes=Path(ledger.plan['config']['resource_admission']).read_bytes()
    if hashlib.sha256(admission_bytes).hexdigest()!=value['admission_sha256']:
        raise ValueError('Original resource admission commitment changed before publication')
    _check_binding(root,ledger.plan,admission_bytes,ledger)
    if not (root/'resource-admission-binding.json').exists():
        write_new_json(root/'resource-admission-binding.json',_binding(ledger.plan,admission_bytes))
    index=len(list(root.glob('resource-availability-*.json')))
    write_new_json(root/f'resource-availability-{index}.json',value)


def validate_resource_admission(plan, output, *, ledger=None):
    if plan.get('plan_hash')!=canonical_hash({k:v for k,v in plan.items() if k!='plan_hash'}):
        raise ValueError('Resource admission plan digest differs')
    config = plan['config']
    t = config['base_experiment']['training']
    model = config['base_experiment']['model']
    data = config['base_experiment']['data']
    # Deliberately tiny software fixtures do not stand in for measured scientific
    # workloads. This exception bounds both architecture and all update ceilings.
    fixture = (model['d'] <= 2 and model['h'] <= 8 and config['teachers'] <= 1
               and config['repetitions'] <= 1 and config['evaluation_rows'] <= 32
               and data['n_fit'] <= 512 and data['n_growth'] <= 256 and data['n_validation'] <= 256
               and max(t['base_updates'], t['continuation_updates']) <= 25)
    if fixture:
        return dict(status='software_fixture_only', scientific_resource_forecast=False)
    locator = config.get('resource_admission')
    if not locator or not Path(locator).is_file():
        raise ValueError('Measured resource admission is required before a production matrix; '
                         'retain native anchor/probe timings and disk/RAM forecast first')
    admission_bytes=Path(locator).read_bytes()
    record = json.loads(admission_bytes)
    keys = {'schema', 'source_sha256', 'config_hash', 'measurements',
            'forecast_worker_seconds_p50', 'forecast_worker_seconds_p90',
            'forecast_remaining_bytes', 'free_disk_margin_bytes', 'peak_rss_bytes',
            'available_worker_seconds', 'execution_identity_hash', 'max_checkpoint_transaction_bytes',
            'case_retained_bytes','retry_reserve_bytes'}
    if set(record) != keys or record['schema'] != 'research/v2-resource-admission':
        raise ValueError('Invalid resource admission schema')
    if (record['source_sha256'] != plan['source_sha256']
            or record['config_hash'] != canonical_hash(config)
            or record['execution_identity_hash'] != canonical_hash(plan['execution_identity'])):
        raise ValueError('Resource admission source/recipe differs from matrix')
    if not isinstance(record['measurements'], list) or not record['measurements']:
        raise ValueError('Resource admission requires native measured anchors/probes')
    measured = []
    for ref in record['measurements']:
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
            raise ValueError('Invalid resource measurement reference')
        path = Path(ref['path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('Resource measurement bytes missing or changed')
        item = json.loads(path.read_text())
        fields = {'schema', 'source_sha256', 'execution_identity_hash', 'workload', 'phase',
                  'elapsed_ns', 'peak_rss_bytes', 'retained_bytes', 'max_checkpoint_transaction_bytes',
                  'training_contract_hash', 'status'}
        if (set(item) != fields or item['schema'] != 'research/v2-resource-measurement'
                or item['source_sha256'] != plan['source_sha256']
                or item['execution_identity_hash'] != record['execution_identity_hash']
                or item['training_contract_hash'] != canonical_hash({'training': t})
                or item['status'] != 'completed_eligible'):
            raise ValueError('Resource measurement identity/schema/eligibility mismatch')
        if any(type(item[k]) is not int or item[k] <= 0 for k in
               ('elapsed_ns', 'peak_rss_bytes', 'retained_bytes', 'max_checkpoint_transaction_bytes')):
            raise ValueError('Resource measurement quantities must be measured positive integers')
        measured.append(item)
    required = {(model['d'], model['h'], data['n_fit'], data['n_growth'], data['n_validation'], t['batch_size'])}
    if config['kind'] == 'm1_controls':
        required.add((2, 8, 512, 256, 256, t['batch_size']))
    covered = set()
    phases = set()
    for item in measured:
        workload = item['workload']
        if not isinstance(workload, dict) or set(workload) != {'d', 'h', 'n_fit', 'n_growth', 'n_validation', 'batch_size'}:
            raise ValueError('Resource measurement workload schema mismatch')
        covered.add(tuple(workload[k] for k in ('d','h','n_fit','n_growth','n_validation','batch_size')))
        phases.add(item['phase'])
    if not required <= covered or not {'parent', 'continuation'} <= phases or (config['kind'] == 'm0_reference' and 'singleton_probe' not in phases):
        raise ValueError('Resource measurement coverage does not qualify matrix workload/stages')
    for name in keys-{'schema', 'source_sha256', 'config_hash', 'measurements', 'execution_identity_hash','case_retained_bytes'}:
        value = record[name]
        if type(value) not in (int, float) or not 0 < value < float('inf'):
            raise ValueError(f'Invalid resource admission quantity: {name}')
    if (record['peak_rss_bytes'] < max(m['peak_rss_bytes'] for m in measured)
            or record['max_checkpoint_transaction_bytes'] < max(m['max_checkpoint_transaction_bytes'] for m in measured)
            or record['free_disk_margin_bytes'] < record['max_checkpoint_transaction_bytes']):
        raise ValueError('Resource admission understates measured RAM/checkpoint transaction')
    if (record['forecast_worker_seconds_p90'] < record['forecast_worker_seconds_p50']
            or record['forecast_worker_seconds_p90'] > record['available_worker_seconds']
            or record['peak_rss_bytes'] > 16*1024**3):
        raise ValueError('Resource admission exceeds available time/RAM envelope')
    groups={}
    for attempt in plan['attempts']:
        groups.setdefault(case_id(attempt),[]).append(attempt['attempt_id'])
    allocations=record['case_retained_bytes']
    if (not isinstance(allocations,dict) or set(allocations)!=set(groups)
            or any(type(v) is not int or v<record['max_checkpoint_transaction_bytes'] for v in allocations.values())
            or type(record['retry_reserve_bytes']) is not int
            or record['retry_reserve_bytes']<max(allocations.values(),default=0)
            or record['forecast_remaining_bytes']<sum(allocations.values())+record['retry_reserve_bytes']):
        raise ValueError('Invalid transaction-level resource allocations or retry reserve')
    terminal=set();event_hash=None
    if ledger is not None:
        if (not isinstance(ledger,AttemptLedger) or ledger.directory.resolve()!=Path(output).resolve()
                or canonical_hash(ledger.plan)!=canonical_hash(plan)):
            raise ValueError('Resource availability ledger/plan differs')
        # Re-read the committed plan and complete hash chain, never mutable
        # caller event objects or arbitrary retained-byte counts.
        ledger=AttemptLedger.resume(output,plan)
        _check_binding(output,plan,admission_bytes,ledger)
        pending=set(ledger.pending_ids)
        terminal={name for name,ids in groups.items() if not pending.intersection(ids)}
        event_hash=ledger.events[-1]['event_hash'] if ledger.events else '0'*64
    extra_retry=record['retry_reserve_bytes']
    if ledger is not None and ledger.events:
        attempted={event['attempt_id'] for event in ledger.events}
        # A resumed transaction already consumes its only permitted rerun.
        # Only untouched pending transactions can still require an extra retry.
        extra_retry=max((allocations[name] for name,ids in groups.items()
            if name not in terminal and not attempted.intersection(ids)),default=0)
    remaining=(record['forecast_remaining_bytes']-sum(allocations[name] for name in terminal)
        -record['retry_reserve_bytes']+extra_retry)
    parent = Path(output).resolve()
    while not parent.exists():
        parent = parent.parent
    free=shutil.disk_usage(parent).free
    decision=dict(schema='research/v2-resource-availability',source_sha256=plan['source_sha256'],
        plan_hash=plan['plan_hash'],admission_sha256=hashlib.sha256(admission_bytes).hexdigest(),
        ledger_event_hash=event_hash,required_additional_bytes=remaining,
        additional_retry_reserve_bytes=extra_retry,
        free_disk_bytes=free,free_disk_margin_bytes=record['free_disk_margin_bytes'],
        terminal_transaction_count=len(terminal),pending_transaction_count=len(groups)-len(terminal),
        basis='frozen whole-transaction allocations; partial/failed-recoverable work gets no credit',
        checked_utc_ns=time.time_ns(),status='admitted' if free>=remaining+record['free_disk_margin_bytes']
        else 'insufficient_free_disk')
    if decision['status']!='admitted':raise ResourceAvailabilityError(decision)
    return {**record,'availability':decision}
