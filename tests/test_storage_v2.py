"""Storage changes representations only, after authenticating immutable logs."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest


def api():
    from treeneuron.research import storage_v2
    return storage_v2


def setup(tmp_path):
    study=tmp_path/'study';file=study/'jobs'/'frozen'/'attempt-0'/'artifact'/'runs'/('a'*32)/'events.jsonl'
    file.parent.mkdir(parents=True);file.write_bytes(b'{"update":1,"rows":[1,2,3]}\n'*3000)
    ref=dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),bytes=file.stat().st_size)
    marker=file.parents[3]/'job-result.json'
    marker.write_text(json.dumps(dict(job_id='frozen',value={'status':'completed_eligible'},
        files=[dict(path=str(file.relative_to(marker.parent)),sha256=ref['sha256'],bytes=ref['bytes'])])))
    ref.update(job_id='frozen',result_reference=dict(path=str(marker),sha256=hashlib.sha256(marker.read_bytes()).hexdigest()))
    return study,file,ref


def plain(source,destination):
    shutil.copy2(source,destination);return dict(argv=['test_plain_copy'],returncode=0,stdout='',stderr='')


def compress(tmp_path,ref,**overrides):
    return api().compress_training_log(ref,protected_root=tmp_path/'study',
        output=tmp_path/'operation',staging_root=tmp_path/'stage',min_saving_bytes=0,**overrides)


def test_atomic_publication_keeps_bytes_and_already_open_readers_identical(tmp_path,monkeypatch):
    _,file,ref=setup(tmp_path);before=file.read_bytes();old_inode=file.stat().st_ino
    monkeypatch.setattr(api(),'_ditto',plain)
    with file.open('rb') as old_reader:
        result=compress(tmp_path,ref)
        assert old_reader.read()==before
    assert result['status']=='published_identical_bytes' and result['logical_bytes_changed'] is False
    assert file.read_bytes()==before and file.stat().st_ino!=old_inode
    assert file.stat().st_nlink==1
    assert not list((tmp_path/'stage').iterdir())
    assert set(file.parent.iterdir())=={file}  # No temporary names enter frozen inventories.


def test_changed_source_is_never_silently_overwritten(tmp_path,monkeypatch):
    _,file,ref=setup(tmp_path)
    def concurrent_change(source,destination):
        plain(source,destination);source.write_bytes(b'changed elsewhere');return dict(returncode=0)
    monkeypatch.setattr(api(),'_ditto',concurrent_change)
    with pytest.raises(ValueError,match='source changed'):
        compress(tmp_path,ref)
    assert file.read_bytes()==b'changed elsewhere'


def test_empty_or_bad_compressed_copy_cannot_replace_original(tmp_path,monkeypatch):
    _,file,ref=setup(tmp_path);before=file.read_bytes();inode=file.stat().st_ino
    def broken(source,destination):destination.write_bytes(b'');return dict(returncode=0)
    monkeypatch.setattr(api(),'_ditto',broken)
    with pytest.raises(ValueError,match='compressed copy'):
        compress(tmp_path,ref)
    assert file.read_bytes()==before and file.stat().st_ino==inode
    assert (tmp_path/'operation'/'failure.json').is_file()


def test_nonzero_compressor_status_never_publishes_even_if_bytes_match(tmp_path,monkeypatch):
    _,file,ref=setup(tmp_path);inode=file.stat().st_ino
    def failed(source,destination):plain(source,destination);return dict(returncode=1)
    monkeypatch.setattr(api(),'_ditto',failed)
    with pytest.raises(ValueError,match='compressor'):
        compress(tmp_path,ref)
    assert file.stat().st_ino==inode


def test_files_without_measured_block_savings_keep_the_original_inode(tmp_path,monkeypatch):
    _,file,ref=setup(tmp_path);inode=file.stat().st_ino
    monkeypatch.setattr(api(),'_ditto',plain)
    result=api().compress_training_log(ref,protected_root=tmp_path/'study',output=tmp_path/'operation',
        staging_root=tmp_path/'stage',min_saving_bytes=4096)
    assert result['status']=='no_measured_block_saving' and file.stat().st_ino==inode
    assert not list((tmp_path/'stage').iterdir())


@pytest.mark.parametrize('change',['wrong_hash','wrong_size','outside','symlink','wrong_name','hardlink'])
def test_invalid_or_non_training_log_inputs_fail_before_calling_compressor(tmp_path,monkeypatch,change):
    _,file,ref=setup(tmp_path)
    if change=='wrong_hash':ref['sha256']='0'*64
    elif change=='wrong_size':ref['bytes']+=1
    elif change=='outside':
        other=tmp_path/'outside'/'events.jsonl';other.parent.mkdir();shutil.copy2(file,other);ref['path']=str(other)
    elif change=='symlink':
        link=file.parent/'alias';link.symlink_to(file);ref['path']=str(link)
    elif change=='wrong_name':
        new=file.with_name('checkpoint.pt');file.rename(new);ref['path']=str(new)
    else:
        import os
        os.link(file,file.with_name('alias'))
    monkeypatch.setattr(api(),'_ditto',lambda *args:(_ for _ in ()).throw(AssertionError('must not run')))
    with pytest.raises(ValueError):compress(tmp_path,ref)


def test_staging_and_operator_receipts_cannot_enter_the_protected_queue(tmp_path,monkeypatch):
    study,file,ref=setup(tmp_path);monkeypatch.setattr(api(),'_ditto',plain)
    with pytest.raises(ValueError):
        api().compress_training_log(ref,protected_root=study,output=study/'operator',staging_root=tmp_path/'stage')
    with pytest.raises(ValueError):
        api().compress_training_log(ref,protected_root=study,output=tmp_path/'operator',staging_root=study/'stage')
    assert set(study.iterdir())=={study/'jobs'}


def test_plan_selects_only_committed_completed_job_logs_and_binds_snapshot(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    jobs=[dict(job_id=n,kind='fixture',payload={},depends_on=[]) for n in ('done','future')]
    plan=make_queue_plan({'source_sha256':'a'*64},jobs);study=tmp_path/'queue'
    def handler(job,directory,*_):
        p=directory/'artifact'/'runs'/('a'*32)/'events.jsonl';p.parent.mkdir(parents=True)
        p.write_bytes(b'{"row":1}\n'*3000)
        return dict(status='completed_eligible')
    run_queue(plan,study,handler,check_identity=lambda _:None,admitted_job_ids=['done'])
    result=api().plan_completed_training_logs(study,expected_plan_hash=plan['plan_hash'])
    assert len(result['files'])==1 and result['committed_job_count']==1
    assert result['protected_plan_hash']==plan['plan_hash'] and result['selection_uses_outcomes'] is False
    assert result['files'][0]['job_id']=='done'
    assert not result['live_files_selected']


def test_corrupted_completed_result_cannot_be_admitted_by_the_storage_plan(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    job=dict(job_id='done',kind='fixture',payload={},depends_on=[])
    plan=make_queue_plan({'source_sha256':'a'*64},[job]);study=tmp_path/'queue'
    run_queue(plan,study,lambda *_:dict(status='completed_eligible'),check_identity=lambda _:None)
    next(study.rglob('job-result.json')).write_text('{}')
    with pytest.raises(ValueError):api().plan_completed_training_logs(study,expected_plan_hash=plan['plan_hash'])
