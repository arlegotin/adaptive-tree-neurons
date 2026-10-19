"""Byte-preserving representation change for committed immutable training logs.

Never stage in an inventoried queue. Only producer-published, already completed
job files are admitted; a currently open writer could otherwise keep appending
to an old inode after rename. This is external storage maintenance, not fitting.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
from contextlib import contextmanager

from treeneuron.config import canonical_hash
from .ledger import write_new_json

_LOG=re.compile(r'(?:artifact/runs/[0-9a-f]{32}/events\.jsonl|artifact/(?:[A-Za-z0-9_-]+/)*history-[0-9a-f]{32}\.jsonl)\Z')


def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _fingerprint(path):
    s=path.stat()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,
        mode=stat.S_IMODE(s.st_mode),uid=s.st_uid,gid=s.st_gid)


def _sync(directory):
    fd=os.open(directory,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _ditto(source,destination):
    argv=['/usr/bin/ditto','--hfsCompression','--noclone',str(source),str(destination)]
    result=subprocess.run(argv,capture_output=True,text=True,timeout=120)
    return dict(argv=argv,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)


def _committed_file(ref,root):
    if (not isinstance(ref,dict) or set(ref)!={'path','sha256','bytes','job_id','result_reference'}
            or type(ref['bytes']) is not int or ref['bytes']<0):raise ValueError('Committed training-log reference required')
    source=Path(ref['path']);reference=ref['result_reference'];marker=Path(reference['path'])
    if (source.is_symlink() or marker.is_symlink() or not marker.is_file() or not source.is_file()
            or not marker.resolve().is_relative_to(root/'jobs') or marker.name!='job-result.json'
            or _sha(marker)!=reference['sha256']):raise ValueError('Completed job result commitment differs')
    result=json.loads(marker.read_text())
    if (result.get('job_id')!=ref['job_id'] or result.get('value',{}).get('status') not in ('completed_eligible','completed_ineligible')
            or not source.resolve().is_relative_to(marker.parent.resolve())):raise ValueError('File belongs to an unfinished/different job')
    relative=source.resolve().relative_to(marker.parent.resolve()).as_posix()
    if (not _LOG.fullmatch(relative) or source.stat().st_nlink!=1
            or dict(path=relative,bytes=ref['bytes'],sha256=ref['sha256']) not in result.get('files',[])
            or source.stat().st_size!=ref['bytes'] or _sha(source)!=ref['sha256']):
        raise ValueError('Immutable training-log bytes/path/ownership differ')
    return source.resolve()


@contextmanager
def history_maintenance_lease(protected_root):
    """Exclude the producer and its coordinated readers; do not rewrite metadata."""
    import fcntl
    path=Path(protected_root)/'worker-lease.json'
    if path.is_symlink() or not path.is_file():raise ValueError('Protected queue lease required for history maintenance')
    with path.open('r') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PermissionError('Checkpoint history compression requires an inactive producer and readers') from error
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def compress_training_log(ref,*,protected_root,output,staging_root,min_saving_bytes=4096,
                          free_disk_floor_bytes=512*1024**2):
    kwargs=dict(protected_root=protected_root,output=output,staging_root=staging_root,
        min_saving_bytes=min_saving_bytes,free_disk_floor_bytes=free_disk_floor_bytes)
    path=ref.get('path') if isinstance(ref,dict) else None
    if type(path) is str and re.fullmatch(r'history-[0-9a-f]{32}\.jsonl',Path(path).name):
        with history_maintenance_lease(protected_root):return _compress_training_log(ref,**kwargs)
    return _compress_training_log(ref,**kwargs)


def _compress_training_log(ref,*,protected_root,output,staging_root,min_saving_bytes,
                           free_disk_floor_bytes):
    started=time.perf_counter_ns();root=Path(protected_root).resolve();output=Path(output);staging=Path(staging_root)
    if (output.resolve().is_relative_to(root) or staging.resolve().is_relative_to(root)
            or type(min_saving_bytes) is not int or min_saving_bytes<0
            or type(free_disk_floor_bytes) is not int or free_disk_floor_bytes<0):
        raise ValueError('Staging/receipts must be outside protected inventories with explicit nonnegative limits')
    source=_committed_file(ref,root);output.mkdir(parents=True,exist_ok=False);staging.mkdir(parents=True,exist_ok=True)
    if source.stat().st_dev!=staging.stat().st_dev:raise ValueError('Atomic publication requires the same filesystem')
    stage=Path(tempfile.mkdtemp(prefix='compression-',dir=staging));temporary=stage/'compressed';backup=stage/'original-link'
    before=_fingerprint(source);before_blocks=source.stat().st_blocks;published=False
    write_new_json(output/'started.json',dict(reference=ref,source_fingerprint=before,staging_path=str(stage),
        min_saving_bytes=min_saving_bytes,free_disk_floor_bytes=free_disk_floor_bytes,started_utc_ns=time.time_ns()))
    try:
        if shutil.disk_usage(stage).free<ref['bytes']+free_disk_floor_bytes:
            raise OSError('Insufficient free disk for a full staging copy plus the declared margin')
        command=_ditto(source,temporary);write_new_json(output/'compressor.json',command)
        if command.get('returncode')!=0:raise ValueError('Native compressor failed; original retained')
        if (temporary.is_symlink() or not temporary.is_file() or temporary.stat().st_size!=ref['bytes']
                or _sha(temporary)!=ref['sha256']):raise ValueError('Native compressed copy bytes differ; original retained')
        if _fingerprint(source)!=before or _sha(source)!=ref['sha256']:
            raise ValueError('Immutable source changed during compression; no replacement made')
        saving=(before_blocks-temporary.stat().st_blocks)*512
        if saving>=min_saving_bytes:
            os.chmod(temporary,before['mode']);os.utime(temporary,ns=(source.stat().st_atime_ns,before['mtime_ns']))
            with temporary.open('rb') as stream:os.fsync(stream.fileno())
            # A retained old inode permits diagnosis/recovery if publication is
            # interrupted. Its name stays outside every protected inventory.
            os.link(source,backup);_sync(stage)
            if _fingerprint(source)!=before or _sha(source)!=ref['sha256']:
                raise ValueError('Immutable source changed before publication; no replacement made')
            os.replace(temporary,source);published=True;_sync(source.parent)
            if source.stat().st_size!=ref['bytes'] or _sha(source)!=ref['sha256']:
                raise ValueError('Post-publication bytes differ; old inode retained for recovery')
        result=dict(schema='research/v2-training-log-storage-operation',
            status='published_identical_bytes' if published else 'no_measured_block_saving',
            reference=ref,source_before=before,source_after=_fingerprint(source),
            reported_per_file_block_saving_bytes=saving if published else 0,
            elapsed_ns=time.perf_counter_ns()-started,logical_bytes_changed=False,
            volume_saving_attributed=False,model_or_scientific_source_changed=False,T_accessed=False)
        write_new_json(output/'result.json',result);_sync(output)
        # Delete only our scratch copies after the durable matching-byte receipt.
        shutil.rmtree(stage);_sync(staging)
        return result
    except BaseException as error:
        write_new_json(output/'failure.json',dict(exception_type=type(error).__name__,message=str(error),
            reference=ref,staging_path=str(stage),published=published,elapsed_ns=time.perf_counter_ns()-started,
            source_current_sha256=_sha(source) if source.is_file() else None,
            old_inode_backup_retained=backup.is_file(),automatic_retry=False))
        raise


def plan_completed_training_logs(directory,*,expected_plan_hash):
    """Snapshot only a valid newline-terminated prefix while its worker may run.

    It is safe to read published result manifests from that prefix: terminal jobs
    cannot be rewritten. No fitting/scoring outcomes affect log selection.
    Each selected file is reauthenticated immediately before publication.
    """
    root=Path(directory).resolve();plan=json.loads((root/'plan.json').read_text())
    if (plan.get('schema')!='research/v2-workqueue' or plan.get('plan_hash')!=expected_plan_hash
            or canonical_hash({k:v for k,v in plan.items() if k!='plan_hash'})!=expected_plan_hash):
        raise ValueError('External protected queue plan commitment differs')
    raw=(root/'events.jsonl').read_bytes();raw=raw[:raw.rfind(b'\n')+1]
    previous='0'*64;latest={};jobs={j['job_id']:j for j in plan['attempts']}
    for line in raw.splitlines():
        event=json.loads(line)
        if (event.get('previous_hash')!=previous or event.get('attempt_id') not in jobs
                or canonical_hash({k:v for k,v in event.items() if k!='event_hash'})!=event.get('event_hash')):
            raise ValueError('Protected queue snapshot ledger commitment differs')
        previous=event['event_hash'];latest[event['attempt_id']]=event
    files=[];completed=0
    for name,event in latest.items():
        if event['status'] not in ('completed_eligible','completed_ineligible'):continue
        reference=event['result'];path=Path(reference['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(root) or _sha(path)!=reference['sha256']:
            raise ValueError('Completed result commitment differs')
        result=json.loads(path.read_text())
        if result.get('job_id')!=name or result.get('job_hash')!=canonical_hash(jobs[name]):
            raise ValueError('Completed result job binding differs')
        completed+=1
        for entry in result['files']:
            if not _LOG.fullmatch(entry['path']) or entry['bytes']<16384:continue
            files.append(dict(path=str((path.parent/entry['path']).resolve()),bytes=entry['bytes'],sha256=entry['sha256'],
                job_id=name,result_reference=reference))
    files.sort(key=lambda r:r['path'])
    value=dict(schema='research/v2-completed-log-storage-plan',protected_root=str(root),protected_plan_hash=expected_plan_hash,
        frozen_frontier_hash=previous,frozen_prefix_bytes=len(raw),committed_job_count=completed,files=files,
        selection_uses_outcomes=False,live_files_selected=False,source_sha256=plan['identity'].get('source_sha256'))
    value['plan_hash']=canonical_hash(value)
    return value
