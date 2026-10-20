"""Authenticate a closed native fit's operational outcome, including failures."""
import copy
import os
from pathlib import Path
import signal
import sys

from treeneuron.config import canonical_hash
from .method_worker import _ROOT, _THREADS, _read_json, _sha, _validate_request, scientific_source_sha256
from .native_anchor_v2 import _closed_reference
from .process_receipt_v2 import inspect_closed_process
from .trial_bridge_v2 import EngineeringTrialBridge
from .workqueue import _inventory


_FIELDS={'schema','status','reason','request_sha256','source_sha256','family_id','fit_seed',
    'child_reaped','child_started','process_receipt_sha256','worker_terminal_sha256','recovery',
    'quality_eligible','V_labels_read','T_accessed','elapsed_before_recovery_inventory_ns','clock_scope'}
_CLOCK='Function entry through result preparation; containing queue must charge final publication/inventory/return'
_WORKER_STATUSES={'completed_ineligible','timeout','resource_exhausted','numerical_failure',
    'input_or_dependency_failure','interrupted_recoverable'}


def _recovery(value,family):
    if type(value) is not dict or type(value.get('resume')) is not bool:
        raise ValueError('Exact native recovery disposition required')
    if value['resume']:
        if (family!='P12' or set(value)!={'resume','reason','source','checkpoint_inventory_sha256','scope'}
                or value['reason']!='verified_independent_checkpoint_copy'):
            raise ValueError('Only NODE can claim retained native optimizer recovery')
        # The exact search journal must separately bind this source to its
        # preceding paid attempt. This reader never grants that lineage.
        ref=value['source']
        if type(ref) is not dict or set(ref)!={'root','inventory_sha256','request_sha256'}:
            raise ValueError('Exact prior native recovery commitment required')
        _closed_reference({k:ref[k] for k in ('root','inventory_sha256')})
        source=Path(ref['root'])/'training-checkpoints'
        if canonical_hash(_inventory(source))!=value['checkpoint_inventory_sha256']:
            raise ValueError('Prior NODE recovery bytes differ')
    elif (set(value)!={'resume','reason'} or value['reason'] not in
            ('fresh_native_attempt','same_seed_restart_without_optimizer_checkpoint','no_committed_checkpoint_same_seed_restart')
            or family!='P12' and value['reason']=='no_committed_checkpoint_same_seed_restart'):
        raise ValueError('Unknown native same-seed restart disposition')


def inspect_native_fit_outcome(*,request_path,expected_request_sha256,native_reference):
    """Reopen an actual managed fit without fitting, inference or V/T labels.

    A complete result and process closure are mandatory. An ENOSPC-truncated
    record stays incomplete; this reader cannot manufacture its missing tail.
    Successful execution still needs full native workload/model inspection.
    """
    reference=copy.deepcopy(native_reference);path=Path(request_path)
    if (type(reference) is not dict or set(reference)!={'root','inventory_sha256','request_sha256'}
            or reference['request_sha256']!=expected_request_sha256):
        raise ValueError('Exact closed native/request commitment required')
    raw=_read_json(path);request,_=_validate_request(path,expected_request_sha256,
        recoverable_training=raw.get('family_id')=='P12')
    if request['budget']!={'native_stopping':True}:raise ValueError('Explicit native stopping request required')
    root,inventory=_closed_reference({k:reference[k] for k in ('root','inventory_sha256')})
    saved=_read_json(root/'result.json');process_path=root/'process/process.json';process=_read_json(process_path)
    source=scientific_source_sha256();family=request['family_id']
    if (type(saved) is not dict or set(saved)!=_FIELDS or saved['schema']!='research/v2-native-fit-result'
            or saved['request_sha256']!=expected_request_sha256 or saved['source_sha256']!=source
            or source!=request['scientific_source_sha256'] or saved['family_id']!=family
            or type(saved['fit_seed']) is not int or saved['fit_seed']!=request['fit_seed']
            or any(saved[k] is not False for k in ('quality_eligible','V_labels_read','T_accessed'))
            or saved['process_receipt_sha256']!=_sha(process_path)
            or saved['clock_scope']!=_CLOCK
            or type(saved['elapsed_before_recovery_inventory_ns']) is not int
            or saved['elapsed_before_recovery_inventory_ns']<=0):
        raise ValueError('Native result source/request/seed/authority/clock differs')
    _recovery(saved['recovery'],family)
    argv=process.get('argv',[]);env=process.get('environment_overrides',{})
    if (type(argv) is not list or not argv or type(argv[0]) is not str or type(env) is not dict
            or type(env.get('PYTHONPATH')) is not str):
        raise ValueError('Exact managed native invocation required')
    expected_argv=[argv[0],'-B','-m','treeneuron.research.method_worker','--request',str(path.resolve()),
        '--expected-request-sha256',expected_request_sha256,'--output',str(root/'worker')]
    if family=='P12':
        expected_argv+=['--training-checkpoints',str(root/'training-checkpoints')]
        if saved['recovery']['resume']:expected_argv+=['--resume']
    paths=env.get('PYTHONPATH','').split(os.pathsep)
    if (Path(argv[0]).resolve()!=Path(sys.executable).resolve() or not paths or paths[0]!=str(_ROOT/'src')
            or any(not Path(p).is_absolute() or not Path(p).is_dir()
                or not Path(p).resolve().is_relative_to((_ROOT/'environment').resolve()) for p in paths[1:])):
        raise ValueError('Native interpreter or pinned environment overlay differs')
    expected_env={key:'1' for key in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}
    expected_env.update(PYTHONPATH=os.pathsep.join(paths),PYTORCH_ENABLE_MPS_FALLBACK='0',KMP_DUPLICATE_LIB_OK=None,
        MPLCONFIGDIR=str(root/'cache/mpl'),XDG_CACHE_HOME=str(root/'cache/xdg'))
    identity=dict(request_sha256=expected_request_sha256,scientific_source_sha256=source,
        family_id=family,fit_seed=request['fit_seed'],native_stopping=True,recovery=saved['recovery'])
    observed=inspect_closed_process(dict(root=str(root/'process'),inventory_sha256=canonical_hash(_inventory(root/'process'))),
        expected_argv=expected_argv,expected_cwd=str(_ROOT),expected_environment_overrides=expected_env,expected_identity=identity)
    if (process['timeout_ns'] is not None
            or any(type(saved[k]) is not bool or saved[k]!=process[k] for k in ('child_started','child_reaped'))
            or saved['elapsed_before_recovery_inventory_ns']<observed['elapsed_ns']):
        raise ValueError('Native result contradicts its no-deadline process closure/cost')
    status=process['execution_status'];reason=process['stop_reason'];worker_sha=None
    if status=='exited' and process['returncode'] in (-signal.SIGINT,-signal.SIGTERM,-signal.SIGKILL):
        status='interrupted_recoverable';reason='owned_child_terminated_by_signal_before_closed_worker_publication'
    elif status=='exited':
        worker=EngineeringTrialBridge._worker_terminal(root/'worker',request,expected_request_sha256)
        status=worker['status'];worker_sha=_sha(root/'worker/terminal.json')
        if status not in _WORKER_STATUSES or (process['returncode']==0)!=(status=='completed_ineligible'):
            raise ValueError('Native worker outcome contradicts its process exit')
    if saved['status']!=status or saved['reason']!=reason or saved['worker_terminal_sha256']!=worker_sha:
        raise ValueError('Saved native outcome differs from actual process/worker disposition')
    if (_inventory(root)!=inventory or _sha(path)!=expected_request_sha256 or scientific_source_sha256()!=source):
        raise ValueError('Native outcome source/request/closed evidence changed during reading')
    return dict(schema='research/v2-checked-native-fit-outcome',native_outcome_authenticated=True,
        native_reference=reference,request_sha256=expected_request_sha256,source_sha256=source,family_id=family,
        fit_seed=request['fit_seed'],status=status,reason=reason,process=observed,
        worker_terminal_sha256=worker_sha,recovery=copy.deepcopy(saved['recovery']),
        elapsed_before_recovery_inventory_ns=saved['elapsed_before_recovery_inventory_ns'],
        native_workload_qualified=False,quality_eligible=False,recovery_lineage_authenticated=False,
        whole_process_cost_authenticated=False,T_accessed=False,V_labels_read=False,
        scope='Existing request F/G/V-features and managed native result/process/worker closure only. Failures remain availability outcomes, without imputed risk. Full workload, exact search recovery lineage, model/validation/qualification, outer wall time and protected authority remain separate.')
