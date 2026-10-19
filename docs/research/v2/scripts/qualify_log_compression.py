"""Qualify byte/seek/mmap/atomic-reader behavior on disposable native log copies."""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import sys
import time

from treeneuron.research import storage_v2
from treeneuron.research.ledger import write_new_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--expected-plan-hash',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter_ns();args.output.mkdir(parents=True,exist_ok=False)
    source_plan=storage_v2.plan_completed_training_logs(args.study,expected_plan_hash=args.expected_plan_hash)
    original=max(source_plan['files'],key=lambda r:(r['bytes'],r['path']))
    before=storage_v2._sha(original['path']);assert before==original['sha256']
    carrier=args.output/'disposable-study';job=carrier/'jobs'/'copy'/'attempt-0'
    file=job/'artifact'/'runs'/('a'*32)/'events.jsonl';file.parent.mkdir(parents=True)
    shutil.copy2(original['path'],file)
    entry=dict(path=file.relative_to(job).as_posix(),bytes=file.stat().st_size,sha256=before)
    marker=job/'job-result.json';write_new_json(marker,dict(job_id='disposable-qualification-copy',
        value={'status':'completed_eligible'},files=[entry],source_copy_only=original))
    reference=dict(path=str(file.resolve()),bytes=entry['bytes'],sha256=before,job_id='disposable-qualification-copy',
        result_reference=dict(path=str(marker.resolve()),sha256=storage_v2._sha(marker)))
    with file.open('rb') as old_reader:
        result=storage_v2.compress_training_log(reference,protected_root=carrier,
            output=args.output/'operation',staging_root=args.output/'staging')
        old_reader.seek(0);assert hashlib.file_digest(old_reader,'sha256').hexdigest()==before
    assert result['status']=='published_identical_bytes' and storage_v2._sha(file)==before
    with file.open('rb') as handle,mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ) as mapped:
        assert hashlib.sha256(mapped).hexdigest()==before
    with file.open('rb') as reader,Path(original['path']).open('rb') as source:
        for offset in (0,1,4093,65521,file.stat().st_size//2,max(0,file.stat().st_size-123)):
            reader.seek(offset);source.seek(offset);assert reader.read(4096)==source.read(4096)
    with file.open() as reader:lines=sum(1 for line in reader if isinstance(json.loads(line),dict))
    assert storage_v2._sha(original['path'])==before
    assert set(file.parent.iterdir())=={file}
    proof=dict(schema='research/v2-atomic-log-compression-qualification',argv=sys.argv,cwd=str(Path.cwd()),
        script_sha256=storage_v2._sha(Path(__file__)),operator_module_sha256=storage_v2._sha(storage_v2.__file__),
        compression_tool_sha256=storage_v2._sha('/usr/bin/ditto'),platform=sys.platform,
        source_plan_hash=source_plan['plan_hash'],source_log_reference=original,
        source_frozen_frontier_hash=source_plan['frozen_frontier_hash'],original_unchanged=True,
        source_device=Path(original['path']).stat().st_dev,destination_device=file.stat().st_dev,
        disposable_copy_only=True,logical_bytes=entry['bytes'],original_reader_identical=True,
        random_seek_reads_identical=True,readonly_mmap_identical=True,JSON_lines_parsed=lines,
        no_temporary_names_in_protected_inventory=True,reported_block_saving_bytes=result['reported_per_file_block_saving_bytes'],
        compression_flags=getattr(file.stat(),'st_flags',None),all_checks_passed=True,
        elapsed_ns=time.perf_counter_ns()-started,native_escalated_execution=True,
        live_scientific_files_mutated=False,model_or_source_changed=False,T_accessed=False,
        volume_saving_attributed=False)
    write_new_json(args.output/'qualification.json',proof)
    print(json.dumps(proof,sort_keys=True))


if __name__=='__main__':main()
