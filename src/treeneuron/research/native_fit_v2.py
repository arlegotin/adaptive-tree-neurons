"""Deadline-free engineering fits with owned RSS and immutable recovery copies.

This grants no native qualification, V scoring, HPO or protected-data authority.
The containing queue supplies paid attempts and a trusted recovery commitment.
"""
import os
from pathlib import Path
import signal
import shutil
import time

from treeneuron.config import canonical_hash
from .ledger import TERMINAL,write_new_json
from .method_worker import (_ROOT,_THREADS,_read_json,_sha,_digest,_validate_request,scientific_source_sha256)
from .process_supervision_v2 import supervise_process
from .trial_bridge_v2 import EngineeringTrialBridge
from .workqueue import _inventory


def _recovery_source(reference,expected_request_sha256,destination):
    if (type(reference) is not dict or set(reference)!={'root','inventory_sha256','request_sha256'}
            or type(reference['root']) is not str or not _digest(reference['inventory_sha256'])
            or not _digest(expected_request_sha256) or reference['request_sha256']!=expected_request_sha256):
        raise ValueError('Recovery must preserve the externally committed request/source/data/seed')
    root=Path(reference['root']);destination=Path(destination).resolve()
    if (not root.is_absolute() or root.resolve()!=root or root.is_symlink() or not root.is_dir()
            or destination.is_relative_to(root) or root.is_relative_to(destination)
            or canonical_hash(_inventory(root))!=reference['inventory_sha256']):
        raise ValueError('Recovery source bytes or destination differ from its immutable commitment')
    saved=_read_json(root/'result.json')
    if (saved.get('schema')!='research/v2-native-fit-result' or saved.get('request_sha256')!=expected_request_sha256
            or saved.get('status') not in TERMINAL
            or not (saved.get('child_reaped') is True or saved.get('child_started') is False)):
        raise ValueError('Recovery requires a closed owned worker, not a live/unknown writer')
    return root


def clone_native_checkpoint(reference,destination,*,expected_request_sha256):
    """Copy bytes only; NODE validates actual tensor/source/state semantics later."""
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():raise ValueError('Fresh recovery destination required')
    root=_recovery_source(reference,expected_request_sha256,destination)
    source=root/'training-checkpoints'
    if not (source/'latest.json').is_file():
        return dict(resume=False,reason='no_committed_checkpoint_same_seed_restart')
    before=_inventory(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,destination,copy_function=shutil.copyfile)
    if (_inventory(destination)!=before or _inventory(source)!=before
            or canonical_hash(_inventory(root))!=reference['inventory_sha256']):
        raise ValueError('Checkpoint transfer changed bytes; old attempt retained, new copy ineligible')
    return dict(resume=True,reason='verified_independent_checkpoint_copy',source=reference,
        checkpoint_inventory_sha256=canonical_hash(before),
        scope='Byte integrity only; source/runtime/data/tensor validation remains with NODE loader')


def run_native_fit(request_path,*,expected_request_sha256,output,interpreter,
                   extra_pythonpath=(),rss_limit_bytes=4*1024**3,recovery_reference=None):
    """One native engineering attempt; no deadline, retry or eligibility upgrade."""
    started=time.perf_counter_ns();request_path=Path(request_path);output=Path(output)
    raw=_read_json(request_path);is_node=raw.get('family_id')=='P12'
    request,_=_validate_request(request_path,expected_request_sha256,recoverable_training=is_node)
    if request['budget']!={'native_stopping':True}:raise ValueError('Explicit native_stopping budget required; capped requests unchanged')
    interpreter=Path(interpreter)
    if (not interpreter.is_absolute() or not interpreter.is_file() or type(rss_limit_bytes) is not int or rss_limit_bytes<=0
            or not isinstance(extra_pythonpath,(list,tuple)) or output.is_symlink() or output.exists()):
        raise ValueError('Fresh native attempt, qualified interpreter and explicit RSS guard required')
    overlays=[]
    for value in extra_pythonpath:
        path=Path(value).resolve()
        if not path.is_dir() or not path.is_relative_to((_ROOT/'environment').resolve()):
            raise ValueError('Native overlays must use the pinned project environment')
        overlays.append(str(path))
    if recovery_reference is not None and not is_node:
        _recovery_source(recovery_reference,expected_request_sha256,output)
    output.mkdir(parents=True,exist_ok=False)
    recovery=dict(resume=False,reason='fresh_native_attempt' if recovery_reference is None else 'same_seed_restart_without_optimizer_checkpoint')
    if recovery_reference is not None and is_node:
        recovery=clone_native_checkpoint(recovery_reference,output/'training-checkpoints',expected_request_sha256=expected_request_sha256)
    argv=[str(interpreter),'-B','-m','treeneuron.research.method_worker','--request',str(request_path.resolve()),
        '--expected-request-sha256',expected_request_sha256,'--output',str((output/'worker').resolve())]
    if is_node:
        argv+=['--training-checkpoints',str((output/'training-checkpoints').resolve())]
        if recovery['resume']:argv.append('--resume')
    environment={key:'1' for key in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}
    environment.update(PYTHONPATH=os.pathsep.join([str(_ROOT/'src'),*overlays]),PYTORCH_ENABLE_MPS_FALLBACK='0',KMP_DUPLICATE_LIB_OK=None,
        MPLCONFIGDIR=str((output/'cache/mpl').resolve()),XDG_CACHE_HOME=str((output/'cache/xdg').resolve()))
    identity=dict(request_sha256=expected_request_sha256,scientific_source_sha256=request['scientific_source_sha256'],
        family_id=request['family_id'],fit_seed=request['fit_seed'],native_stopping=True,recovery=recovery)
    process=supervise_process(argv=argv,cwd=_ROOT,env_overrides=environment,output=output/'process',
        timeout_ns=None,rss_limit_bytes=rss_limit_bytes,identity=identity)
    worker=None;status=process['execution_status'];reason=process['stop_reason']
    if status=='exited' and process['returncode'] in (-signal.SIGINT,-signal.SIGTERM,-signal.SIGKILL):
        status='interrupted_recoverable';reason='owned_child_terminated_by_signal_before_closed_worker_publication'
    elif status=='exited':
        worker=EngineeringTrialBridge._worker_terminal(output/'worker',request,expected_request_sha256)
        status=worker['status']
        if (process['returncode']==0)!=(status=='completed_ineligible'):
            raise ValueError('Native worker eligibility and process exit contradict each other')
    if scientific_source_sha256()!=request['scientific_source_sha256'] or _sha(request_path)!=expected_request_sha256:
        raise ValueError('Native request/source changed during fitting')
    _validate_request(request_path,expected_request_sha256,recoverable_training=is_node)
    result=dict(schema='research/v2-native-fit-result',status=status,reason=reason,request_sha256=expected_request_sha256,
        source_sha256=request['scientific_source_sha256'],family_id=request['family_id'],fit_seed=request['fit_seed'],
        child_reaped=process['child_reaped'],child_started=process['child_started'],
        process_receipt_sha256=_sha(output/'process/process.json'),
        worker_terminal_sha256=_sha(output/'worker/terminal.json') if worker is not None else None,
        recovery=recovery,quality_eligible=False,V_labels_read=False,T_accessed=False,
        elapsed_before_recovery_inventory_ns=time.perf_counter_ns()-started,
        clock_scope='Function entry through result preparation; containing queue must charge final publication/inventory/return')
    write_new_json(output/'result.json',result)
    reference=dict(root=str(output.resolve()),inventory_sha256=canonical_hash(_inventory(output)),request_sha256=expected_request_sha256)
    return dict(result,recovery_reference=reference,callback_elapsed_ns=time.perf_counter_ns()-started)
