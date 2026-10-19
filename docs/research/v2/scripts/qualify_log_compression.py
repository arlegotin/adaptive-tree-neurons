"""Qualify byte/seek/mmap/atomic-reader behavior on disposable native log copies."""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import resource
import sys
import time

from treeneuron.research import storage_v2
from treeneuron.research.ledger import write_new_json
from treeneuron.history import History


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--expected-plan-hash',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter_ns();args.output.mkdir(parents=True,exist_ok=False)
    source_plan=storage_v2.plan_completed_training_logs(args.study,expected_plan_hash=args.expected_plan_hash)
    checks=[];carrier=args.output/'disposable-study';carrier.mkdir()
    write_new_json(carrier/'worker-lease.json',dict(software_only=True))
    for kind in ('events','history'):
        choices=[r for r in source_plan['files'] if (Path(r['path']).name=='events.jsonl')==(kind=='events')]
        original=max(choices,key=lambda r:(r['bytes'],r['path']))
        before=storage_v2._sha(original['path']);assert before==original['sha256']
        job=carrier/'jobs'/kind/'attempt-0'
        file=(job/'artifact'/'runs'/('a'*32)/'events.jsonl' if kind=='events'
            else job/'artifact'/('history-'+'b'*32+'.jsonl'))
        file.parent.mkdir(parents=True)
        # Materialize ordinary bytes, without copying a compressed source's flags.
        with Path(original['path']).open('rb') as src,file.open('xb') as dst:shutil.copyfileobj(src,dst)
        with file.open() as reader:lines=sum(1 for line in reader if isinstance(json.loads(line),dict))
        history_reference=dict(schema_version=2,encoding='canonical_json_lines_utf8',file=file.name,
            bytes=file.stat().st_size,count=lines,sha256=before)
        loaded_before=History.load(file.parent,history_reference) if kind=='history' else None
        entry=dict(path=file.relative_to(job).as_posix(),bytes=file.stat().st_size,sha256=before)
        marker=job/'job-result.json';write_new_json(marker,dict(job_id='disposable-'+kind+'-copy',
            value={'status':'completed_eligible'},files=[entry],source_copy_only=original))
        reference=dict(path=str(file.resolve()),bytes=entry['bytes'],sha256=before,job_id='disposable-'+kind+'-copy',
            result_reference=dict(path=str(marker.resolve()),sha256=storage_v2._sha(marker)))
        with file.open('rb') as old_reader:
            result=storage_v2.compress_training_log(reference,protected_root=carrier,
                output=args.output/('operation-'+kind),staging_root=args.output/'staging')
            old_reader.seek(0);assert hashlib.file_digest(old_reader,'sha256').hexdigest()==before
        assert result['status']=='published_identical_bytes' and storage_v2._sha(file)==before
        with file.open('rb') as handle,mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ) as mapped:
            assert hashlib.sha256(mapped).hexdigest()==before
        with file.open('rb') as reader,Path(original['path']).open('rb') as source:
            for offset in (0,1,4093,65521,file.stat().st_size//2,max(0,file.stat().st_size-123)):
                reader.seek(offset);source.seek(offset);assert reader.read(4096)==source.read(4096)
        if kind=='history':
            loaded_after=History.load(file.parent,history_reference)
            assert len(loaded_before)==len(loaded_after) and all(a==b for a,b in zip(loaded_before,loaded_after))
        assert storage_v2._sha(original['path'])==before and set(file.parent.iterdir())=={file}
        checks.append(dict(kind=kind,source_log_reference=original,logical_bytes=entry['bytes'],JSON_lines_parsed=lines,
            original_unchanged=True,original_reader_identical=True,random_seek_reads_identical=True,readonly_mmap_identical=True,
            history_values_identical=True if kind=='history' else None,
            reported_block_saving_bytes=result['reported_per_file_block_saving_bytes'],compression_flags=getattr(file.stat(),'st_flags',None)))
    proof=dict(schema='research/v2-atomic-log-compression-qualification',argv=sys.argv,cwd=str(Path.cwd()),
        script_sha256=storage_v2._sha(Path(__file__)),operator_module_sha256=storage_v2._sha(storage_v2.__file__),
        compression_tool_sha256=storage_v2._sha('/usr/bin/ditto'),platform=sys.platform,
        source_plan_hash=source_plan['plan_hash'],checks=checks,qualified_log_kinds=['events','history'],
        source_frozen_frontier_hash=source_plan['frozen_frontier_hash'],original_unchanged=True,
        source_device=Path(original['path']).stat().st_dev,destination_device=file.stat().st_dev,
        disposable_copy_only=True,logical_bytes=sum(r['logical_bytes'] for r in checks),original_reader_identical=True,
        random_seek_reads_identical=True,readonly_mmap_identical=True,JSON_lines_parsed=sum(r['JSON_lines_parsed'] for r in checks),
        no_temporary_names_in_protected_inventory=True,reported_block_saving_bytes=sum(r['reported_block_saving_bytes'] for r in checks),
        history_values_identical=True,history_requires_inactive_producer=True,
        history_reader_module_sha256=storage_v2._sha(sys.modules[History.__module__].__file__),all_checks_passed=True,
        elapsed_ns=time.perf_counter_ns()-started,native_escalated_execution=True,
        live_scientific_files_mutated=False,model_or_source_changed=False,T_accessed=False,
        volume_saving_attributed=False,operator_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        child_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
    write_new_json(args.output/'qualification.json',proof)
    print(json.dumps(proof,sort_keys=True))


if __name__=='__main__':main()
