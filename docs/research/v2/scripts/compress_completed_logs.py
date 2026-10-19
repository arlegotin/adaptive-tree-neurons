"""Maintain completed M2 log representations under a native qualified operator."""
import argparse
import json
from pathlib import Path
import shutil
import stat
import sys
import time

from treeneuron.config import canonical_hash,canonical_json
from treeneuron.research import storage_v2
from treeneuron.research.ledger import StudyLease,write_new_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--expected-plan-hash',required=True);parser.add_argument('--qualification',type=Path,required=True)
    parser.add_argument('--expected-qualification-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--events-only',action='store_true',help='Allow only finished event logs while producer runs; history requires inactive producer')
    args=parser.parse_args();started=time.perf_counter_ns()
    if args.output.resolve().is_relative_to(args.study.resolve()):raise ValueError('Maintenance must not add files to the frozen queue')
    if args.qualification.is_symlink() or storage_v2._sha(args.qualification)!=args.expected_qualification_sha256:
        raise ValueError('Native qualification external commitment differs')
    proof=json.loads(args.qualification.read_text())
    required=('all_checks_passed','disposable_copy_only','original_reader_identical',
        'random_seek_reads_identical','readonly_mmap_identical','no_temporary_names_in_protected_inventory')
    if (proof.get('schema')!='research/v2-atomic-log-compression-qualification'
            or proof.get('qualified_log_kinds')!=['events','history']
            or proof.get('history_values_identical') is not True
            or proof.get('history_requires_inactive_producer') is not True
            or proof.get('history_reader_module_sha256')!=storage_v2._sha(Path(storage_v2.__file__).parents[1]/'history.py')
            or any(proof.get(k) is not True for k in required) or proof.get('live_scientific_files_mutated') is not False
            or proof.get('T_accessed') is not False or proof.get('platform')!=sys.platform or sys.platform!='darwin'
            or proof.get('operator_module_sha256')!=storage_v2._sha(storage_v2.__file__)
            or proof.get('compression_tool_sha256')!=storage_v2._sha('/usr/bin/ditto')
            or proof.get('source_device')!=args.study.stat().st_dev):
        raise ValueError('Native compression/reader/atomic-publication qualification differs')
    plan=storage_v2.plan_completed_training_logs(args.study,expected_plan_hash=args.expected_plan_hash)
    original_snapshot=plan.pop('plan_hash');plan['base_storage_snapshot_hash']=original_snapshot
    plan['file_selection']='events_only' if args.events_only else 'events_and_history'
    if args.events_only:plan['files']=[ref for ref in plan['files'] if Path(ref['path']).name=='events.jsonl']
    else:
        # Refuse before any mutation; each history operation also holds its own
        # exclusive lease, so a producer starting later cannot race publication.
        with storage_v2.history_maintenance_lease(args.study):pass
    plan['plan_hash']=canonical_hash(plan)
    args.output.mkdir(parents=True,exist_ok=False)
    with StudyLease(args.output):
        write_new_json(args.output/'plan.json',plan)
        write_new_json(args.output/'invocation.json',dict(argv=sys.argv,cwd=str(Path.cwd()),
            script_sha256=storage_v2._sha(__file__),operator_module_sha256=storage_v2._sha(storage_v2.__file__),
            expected_qualification_sha256=args.expected_qualification_sha256,qualification_path=str(args.qualification),
            producer_source_sha256=plan['source_sha256'],started_utc_ns=time.time_ns(),
            protected_source_and_parameters_unchanged=True,scope='Completed immutable training logs only; active queue/fit logs excluded',T_accessed=False))
        free_before=shutil.disk_usage(args.study).free;counts={};saving=0;last=None;failure=None
        try:
            with (args.output/'events.jsonl').open('x') as stream:
                for index,ref in enumerate(plan['files']):
                    last=index;path=storage_v2._committed_file(ref,args.study.resolve())
                    if getattr(path.stat(),'st_flags',0)&getattr(stat,'UF_COMPRESSED',0x20):
                        record=dict(index=index,path=ref['path'],sha256=ref['sha256'],status='already_compressed_verified',reported_per_file_block_saving_bytes=0)
                    else:
                        value=storage_v2.compress_training_log(ref,protected_root=args.study,
                            output=args.output/f'file-{index:06}',staging_root=args.output/'staging')
                        record=dict(index=index,path=ref['path'],sha256=ref['sha256'],status=value['status'],
                            reported_per_file_block_saving_bytes=value['reported_per_file_block_saving_bytes'])
                    counts[record['status']]=counts.get(record['status'],0)+1;saving+=record['reported_per_file_block_saving_bytes']
                    stream.write(canonical_json(record)+'\n');stream.flush()
                    if (index+1)%64==0:print(json.dumps(dict(processed=index+1,planned=len(plan['files']),counts=counts)),flush=True)
        except BaseException as error:
            failure=dict(type=type(error).__name__,message=str(error),last_index=last)
            raise
        finally:
            write_new_json(args.output/'result.json',dict(status='completed' if failure is None else 'failed',
                failure=failure,planned_files=len(plan['files']),counts=counts,
                reported_per_file_block_saving_bytes=saving,volume_free_before_bytes=free_before,
                volume_free_after_bytes=shutil.disk_usage(args.study).free,volume_saving_attributed=False,
                elapsed_ns=time.perf_counter_ns()-started,logical_bytes_changed=False,
                scientific_source_parameters_and_attempt_queue_changed=False,
                cost_scope='External storage maintenance overlapping the contended scientific cohort; do not add overlapping containing clocks',T_accessed=False))
        print(json.dumps(dict(status='completed',planned=len(plan['files']),counts=counts,
            reported_per_file_block_saving_bytes=saving),sort_keys=True))


if __name__=='__main__':main()
