"""Explicitly resumed first cost member; receipt reused; no retries/cohort."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT=Path.cwd()
ENV=ROOT/'environment/long-probes-20261018'
OUT=ROOT/'artifacts/long-probes-20261018/cost'
PYTHON=str(ROOT/'.venv/bin/python')
EXPECTED_SOURCE='089b03d13e1950b8e0c1ff54a147a121cf78db66acaaa2dcedd9ebf29afc3543'
THREADS=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTORCH_ENABLE_MPS_FALLBACK='0')

def write(name,value):
    with (ENV/name).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def native():
    result={'free_disk_bytes':shutil.disk_usage(ROOT).free}
    for label,argv in [('power',['pmset','-g','batt']),
            ('memory_pressure',['memory_pressure','-Q']),('vm_stat',['vm_stat'])]:
        p=subprocess.run(argv,capture_output=True,text=True)
        result[label]=dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    return result

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tracked():
    names=subprocess.check_output(['git','ls-files','-z']).split(b'\0')
    return {os.fsdecode(p):sha(ROOT/os.fsdecode(p)) for p in names
            if p and (ROOT/os.fsdecode(p)).is_file()}

def usage():
    r=resource.getrusage(resource.RUSAGE_CHILDREN)
    return {key:getattr(r,key) for key in ('ru_utime','ru_stime','ru_maxrss','ru_ixrss','ru_idrss',
        'ru_isrss','ru_minflt','ru_majflt','ru_nswap','ru_inblock','ru_oublock','ru_msgsnd','ru_msgrcv',
        'ru_nsignals','ru_nvcsw','ru_nivcsw')}

assert not OUT.exists(), 'Cost path must be new; no retry'
assert sha(ENV/'cpu128-receipt.json')=='e506511d50107138c0e94b1341687c5d9ff97e2fecac5b500a0d1946d1a93f43'
prior={str(p.relative_to(ROOT)):sha(p) for p in ENV.rglob('*') if p.is_file() and p.name!='cost-only.py'}
old_report=ROOT/'.superpowers/sdd/2026-10-18-long-probe-allocation-diagnostic/task-2-report.md'
prior[str(old_report.relative_to(ROOT))]=sha(old_report)
baseline=tracked()
bootstrap=("import runpy,torch; torch.set_num_threads(1); torch.set_num_interop_threads(1); "
    "from treeneuron import train; assert train.source_hash()=='"+EXPECTED_SOURCE+"'; "
    "runpy.run_module('treeneuron.pilot.probe_diagnostic',run_name='__main__')")
argv=[PYTHON,'-c',bootstrap,'measure','--spec','protocol/diagnostics/2026-10-18-long-probes.json',
    '--output','artifacts/long-probes-20261018/cost']
write('cost-only-invocation.json',dict(argv=argv,cwd=str(ROOT),environment=THREADS,
    logical_cli=[PYTHON,'-m','treeneuron.pilot.probe_diagnostic','measure','--spec',
        'protocol/diagnostics/2026-10-18-long-probes.json','--output','artifacts/long-probes-20261018/cost'],
    expected_source=EXPECTED_SOURCE,receipt_reused=True,receipt_producer_rerun=False,
    automatic_retries=0,cohort_run=False,
    bootstrap='Fresh interpreter CPU1/inter-op1 before module entry; BLAS1/fallback0 before imports.'))
write('cost-only-preflight.json',dict(utc=utc(),native=native(),tracked_sha256=baseline,
    prior_evidence_sha256=prior,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    git_status=subprocess.check_output(['git','status','--porcelain'],text=True)))
awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
status=1
try:
    before=native()
    with (ENV/'cost-only-stdout.txt').open('xb') as stdout, (ENV/'cost-only-stderr.txt').open('xb') as stderr:
        usage_before=usage()
        started_utc=utc()
        tick=time.perf_counter_ns()
        p=subprocess.run(argv,env=dict(os.environ,**THREADS),stdout=stdout,stderr=stderr)
        elapsed=time.perf_counter_ns()-tick
        finished_utc=utc()
        usage_after=usage()
    status=p.returncode
    files=sorted(p for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else []
    post_bytes=sum(p.stat().st_size for p in files)
    after=native()
    write('cost-only-outer-measurement.json',dict(returncode=status,started_utc=started_utc,
        finished_utc=finished_utc,outer_elapsed_ns=elapsed,
        elapsed_boundary='perf_counter_ns immediately around subprocess.run launch through process exit; all final measure publication included; pre/post native counters excluded',
        final_post_publication_retained_bytes=post_bytes,final_post_publication_file_count=len(files),
        bytes_boundary='All regular files below cost output after measure process exit; environment execution evidence excluded',
        children_usage_before=usage_before,children_usage_after=usage_after,
        child_peak_rss_bytes=int(usage_after['ru_maxrss'])*(1 if sys.platform=='darwin' else 1024),
        child_peak_rss_boundary='RUSAGE_CHILDREN maximum over completed children of this fresh wrapper; before baseline retained; final observation taken before post-native subprocesses or caffeinate termination',
        child_user_seconds=usage_after['ru_utime']-usage_before['ru_utime'],
        child_system_seconds=usage_after['ru_stime']-usage_before['ru_stime'],
        before=before,after=after,automatic_retries=0))
    print(json.dumps(dict(phase='cost',returncode=status,outer_elapsed_ns=elapsed,
        retained_bytes=post_bytes,files=len(files))),flush=True)
finally:
    awake.terminate(); awake.wait()
    now=tracked()
    changed_prior=[p for p,h in prior.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    write('cost-only-stop.json',dict(utc=utc(),returncode=status,cohort_run=False,
        receipt_producer_rerun=False,automatic_retries=0,tracked_unchanged=(now==baseline),
        changed_tracked=[p for p in set(now)|set(baseline) if now.get(p)!=baseline.get(p)],
        prior_evidence_unchanged=not changed_prior,changed_prior_evidence=changed_prior,
        caffeinate_process_scoped=True))
sys.exit(status)
