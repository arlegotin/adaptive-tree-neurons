"""One receipt then one cost comparison, no retries and no cohort invocation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path.cwd()
ENV = ROOT/'environment/long-probes-20261018'
OUT = ROOT/'artifacts/long-probes-20261018/cost'
PYTHON = str(ROOT/'.venv/bin/python')
THREADS = dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
CHILD_ENV = dict(os.environ,**THREADS)

def write(name,value):
    (ENV/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def native():
    result={}
    for label,cmd in [('power',['pmset','-g','batt']),('memory_pressure',['memory_pressure','-Q']),
            ('vm_stat',['vm_stat']),('swap',['sysctl','vm.swapusage']),('disk',['df','-k',str(ROOT)])]:
        p=subprocess.run(cmd,capture_output=True,text=True)
        result[label]=dict(argv=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    return result

def tracked():
    names=subprocess.check_output(['git','ls-files','-z']).split(b'\0')
    return {os.fsdecode(p):hashlib.sha256((ROOT/os.fsdecode(p)).read_bytes()).hexdigest()
        for p in names if p and (ROOT/os.fsdecode(p)).is_file()}

def invoke(label,argv):
    write(label+'-invocation.json',dict(argv=argv,cwd=str(ROOT),environment=THREADS,automatic_retries=0,
        logical_cli=([PYTHON,'-m','treeneuron.pilot.probe_diagnostic','measure','--spec',
            'protocol/diagnostics/2026-10-18-long-probes.json','--output','artifacts/long-probes-20261018/cost']
            if label=='cost' else None),
        bootstrap='Fresh Python sets torch CPU1/inter-op1 before runpy module entry; BLAS/fallback exported before interpreter startup.'))
    before=native()
    start_utc=utc(); start=time.perf_counter_ns()
    with (ENV/(label+'-stdout.txt')).open('wb') as stdout, (ENV/(label+'-stderr.txt')).open('wb') as stderr:
        p=subprocess.run(['/usr/bin/time','-l',*argv],env=CHILD_ENV,stdout=stdout,stderr=stderr)
    elapsed=time.perf_counter_ns()-start; end_utc=utc()
    after=native()
    result=dict(returncode=p.returncode,started_utc=start_utc,finished_utc=end_utc,outer_elapsed_ns=elapsed,
        boundary='subprocess /usr/bin/time -l plus fresh interpreter from launch through exit; all final publication inside',
        before=before,after=after,automatic_retries=0)
    if label=='cost' and OUT.exists():
        files=[p for p in OUT.rglob('*') if p.is_file()]
        result.update(final_post_publication_retained_bytes=sum(p.stat().st_size for p in files),
            final_post_publication_file_count=len(files),
            retained_bytes_boundary='all regular files beneath cost output after process exit; environment harness excluded')
    write(label+'-outer-measurement.json',result)
    print(json.dumps(dict(phase=label,returncode=p.returncode,outer_elapsed_ns=elapsed)),flush=True)
    return p.returncode

assert not OUT.exists(), 'Cost output must be new'
assert not (ENV/'cpu128-receipt.json').exists(), 'Fresh receipt path must be new'
baseline=tracked()
write('preflight.json',dict(utc=utc(),native=native(),tracked_sha256=baseline,
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    git_status=subprocess.check_output(['git','status','--porcelain'],text=True),
    python=PYTHON,environment=THREADS,automatic_retries=0,
    authorization='Task2 PHASE A ONLY: receipt and one complete cost comparison, then STOP.'))
awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
status=1
try:
    status=invoke('receipt',[PYTHON,str(ENV/'receipt-producer.py')])
    if status==0:
        assert tracked()==baseline, 'Tracked bytes changed before cost'
        bootstrap="import runpy,torch; torch.set_num_threads(1); torch.set_num_interop_threads(1); runpy.run_module('treeneuron.pilot.probe_diagnostic',run_name='__main__')"
        status=invoke('cost',[PYTHON,'-c',bootstrap,'measure','--spec',
            'protocol/diagnostics/2026-10-18-long-probes.json','--output','artifacts/long-probes-20261018/cost'])
finally:
    awake.terminate(); awake.wait()
    now=tracked()
    write('phase-a-stop.json',dict(utc=utc(),returncode=status,cohort_run=False,automatic_retries=0,
        tracked_unchanged=(now==baseline),changed_tracked=[p for p in set(now)|set(baseline) if now.get(p)!=baseline.get(p)],
        caffeinate_process_scoped=True,native=native()))
sys.exit(status)
