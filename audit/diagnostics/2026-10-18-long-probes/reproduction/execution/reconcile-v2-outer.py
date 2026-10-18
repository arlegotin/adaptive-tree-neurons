"""One serial reconciliation launch, Python outer clock/resources; no retries."""
import datetime,json,os,resource,shutil,subprocess,sys,time
from pathlib import Path
B=Path('environment/long-probes-20261018/audit')
def usage():
 r=resource.getrusage(resource.RUSAGE_CHILDREN)
 return {k:getattr(r,k) for k in ('ru_utime','ru_stime','ru_maxrss','ru_minflt','ru_majflt','ru_inblock','ru_oublock','ru_nvcsw','ru_nivcsw')}
threads=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
argv=[str(Path('.venv/bin/python').absolute()),str(B/'reconcile-v2.py')]
with (B/'reconcile-v2-invocation.json').open('x') as f:json.dump({'argv':argv,'environment':threads,'automatic_retries':0,'elapsed_time_kill':False},f,indent=2);f.write('\n')
before=usage();free_before=shutil.disk_usage('.').free;start=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.perf_counter_ns()
with (B/'reconcile-v2-stdout.txt').open('xb') as stdout,(B/'reconcile-v2-stderr.txt').open('xb') as stderr:
 child=subprocess.Popen(argv,env=dict(os.environ,**threads),stdout=stdout,stderr=stderr);status=child.wait()
elapsed=time.perf_counter_ns()-tick;after=usage()
with (B/'reconcile-v2-outer-measurement.json').open('x') as f:
 json.dump({'returncode':status,'started_utc':start,'outer_elapsed_ns':elapsed,'usage_before':before,'usage_after':after,'peak_child_rss_bytes':after['ru_maxrss'],'free_disk_before':free_before,'free_disk_after':shutil.disk_usage('.').free,'boundary':'perf_counter_ns before fresh interpreter Popen through wait completion; analysis and two report publications included','automatic_retries':0},f,indent=2);f.write('\n')
sys.exit(status)
