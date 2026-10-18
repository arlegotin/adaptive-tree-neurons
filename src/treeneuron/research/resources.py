"""Resource admission is evidence-based and separate from efficacy outcomes."""
import json
import hashlib
from pathlib import Path
import shutil

from treeneuron.config import canonical_hash


def validate_resource_admission(plan, output):
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
    record = json.loads(Path(locator).read_text())
    keys = {'schema', 'source_sha256', 'config_hash', 'measurements',
            'forecast_worker_seconds_p50', 'forecast_worker_seconds_p90',
            'forecast_remaining_bytes', 'free_disk_margin_bytes', 'peak_rss_bytes',
            'available_worker_seconds', 'execution_identity_hash', 'max_checkpoint_transaction_bytes'}
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
    for name in keys-{'schema', 'source_sha256', 'config_hash', 'measurements', 'execution_identity_hash'}:
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
    parent = Path(output).resolve()
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < record['forecast_remaining_bytes']+record['free_disk_margin_bytes']:
        raise ValueError('Resource admission forecast exceeds free disk including margin')
    return record
