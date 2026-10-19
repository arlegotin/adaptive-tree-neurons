"""Real own-child processes exercise deadlines, logs and observed RSS guards."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pytest


def api():
    from treeneuron.research import process_supervision_v2
    return process_supervision_v2


def run(tmp_path,code,**overrides):
    values=dict(argv=[sys.executable,'-B','-c',code],cwd=str(tmp_path),env_overrides={'ATN_TOY_VALUE':'fixed'},
        output=tmp_path/'supervisor',timeout_ns=4000000000,rss_limit_bytes=512*1024**2,
        poll_interval_ns=20000000,terminate_grace_ns=100000000,log_limit_bytes=1024**2,
        identity=dict(study='artificial-process-qualification',native_qualified=False))
    return api().supervise_process(**(values|overrides))


def test_normal_process_retains_exact_command_environment_output_and_external_clock(tmp_path):
    r=run(tmp_path,"import os,sys; print(os.environ['ATN_TOY_VALUE']); print('err',file=sys.stderr)")
    assert r['execution_status']=='exited' and r['returncode']==0 and r['child_started']
    assert r['elapsed_ns']>0 and r['eligible_model'] is False
    assert r['environment_overrides']=={'ATN_TOY_VALUE':'fixed'}
    assert (tmp_path/'supervisor'/'stdout.log').read_text()=='fixed\n'
    assert (tmp_path/'supervisor'/'stderr.log').read_text()=='err\n'
    receipt=json.loads((tmp_path/'supervisor'/'process.json').read_text())
    assert receipt==r
    assert all(hashlib.sha256((tmp_path/'supervisor'/ref['path']).read_bytes()).hexdigest()==ref['sha256'] for ref in r['files'])
    assert r['rss_scope']=='Observed root worker RSS samples; descendants and unobserved peaks are not measured'


def test_timeout_terminates_the_own_process_group_and_preserves_partial_output(tmp_path):
    started=time.perf_counter()
    r=run(tmp_path,"import time; print('started',flush=True); time.sleep(30)",timeout_ns=300000000)
    assert r['execution_status']=='timeout' and r['returncode']!=0
    assert r['child_reaped'] and r['termination']['term_sent']
    assert (tmp_path/'supervisor'/'stdout.log').read_text()=='started\n'
    assert time.perf_counter()-started<4


def test_a_child_ignoring_TERM_is_killed_after_the_bounded_grace(tmp_path):
    r=run(tmp_path,"import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(30)",timeout_ns=300000000)
    assert r['execution_status']=='timeout' and r['termination']['kill_sent'] and r['child_reaped']


def test_observed_RSS_excess_stops_the_trial_without_changing_its_configuration(tmp_path):
    r=run(tmp_path,"import time; x=bytearray(80*1024*1024); print('allocated',flush=True); time.sleep(30)",rss_limit_bytes=40*1024**2)
    assert r['execution_status']=='resource_exhausted' and r['stop_reason']=='observed_root_rss_limit'
    assert r['observed_peak_root_rss_bytes']>40*1024**2 and r['child_reaped']
    assert r['strict_peak_memory_limit_qualified'] is False


def test_excess_log_volume_is_retained_and_stops_the_worker(tmp_path):
    r=run(tmp_path,"import sys,time; sys.stdout.write('x'*100000); sys.stdout.flush(); time.sleep(30)",log_limit_bytes=1024)
    assert r['execution_status']=='resource_exhausted' and r['stop_reason']=='log_bytes_limit'
    assert (tmp_path/'supervisor'/'stdout.log').stat().st_size>1024


def test_nonzero_exit_is_a_retained_process_result_not_a_predictive_loss(tmp_path):
    r=run(tmp_path,"import sys; print('native diagnostic',file=sys.stderr); sys.exit(7)")
    assert r['execution_status']=='exited' and r['returncode']==7 and r['eligible_model'] is False


def test_unknown_RSS_measurement_fails_closed_and_does_not_start_the_worker(tmp_path,monkeypatch):
    def fail(pid,timeout_ns):raise RuntimeError('probe unavailable')
    monkeypatch.setattr(api(),'root_rss_bytes',fail)
    r=run(tmp_path,"raise AssertionError('must not run')")
    assert r['execution_status']=='input_or_dependency_failure' and not r['child_started']
    assert r['stop_reason']=='rss_probe_unavailable' and 'probe unavailable' in r['diagnostic']


def test_lost_RSS_probe_after_start_stops_the_owned_child_and_retains_failure(tmp_path,monkeypatch):
    calls=[]
    def fail(pid,timeout_ns):
        calls.append(pid)
        if len(calls)>1:raise RuntimeError('probe lost')
        return 1
    monkeypatch.setattr(api(),'root_rss_bytes',fail)
    r=run(tmp_path,"import time; time.sleep(30)")
    assert r['execution_status']=='input_or_dependency_failure' and r['child_started'] and r['child_reaped']
    assert r['stop_reason']=='rss_probe_unavailable'


def test_process_that_leaves_background_descendants_is_not_a_clean_completion(tmp_path):
    r=run(tmp_path,"import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print('parent exits')")
    assert r['returncode']==0 and r['execution_status']=='input_or_dependency_failure'
    assert r['stop_reason']=='unexpected_live_process_group' and r['termination']['term_sent']


@pytest.mark.parametrize('change',[
    dict(argv='shell command'),dict(argv=['relative-python']),dict(argv=[sys.executable,True]),
    dict(timeout_ns=0),dict(timeout_ns=True),dict(rss_limit_bytes=0),dict(poll_interval_ns=0),
    dict(terminate_grace_ns=-1),dict(log_limit_bytes=0),dict(env_overrides={'ATN_X':1}),
    dict(env_overrides={'HOME':'repurposed'}),dict(identity={})])
def test_invalid_execution_contract_does_not_create_output_or_start_a_child(tmp_path,change):
    with pytest.raises(ValueError):run(tmp_path,'pass',**change)
    assert not (tmp_path/'supervisor').exists()


def test_existing_output_is_not_overwritten_or_silently_retried(tmp_path):
    r=run(tmp_path,"print('first')")
    original=(tmp_path/'supervisor'/'process.json').read_bytes()
    with pytest.raises(ValueError):run(tmp_path,"print('second')")
    assert (tmp_path/'supervisor'/'process.json').read_bytes()==original


def test_fast_exit_cannot_evade_the_retained_log_volume_guard(tmp_path):
    r=run(tmp_path,"import sys; sys.stdout.write('x'*100000); sys.stdout.flush()",log_limit_bytes=1024,poll_interval_ns=100000000)
    assert r['execution_status']=='resource_exhausted' and r['stop_reason']=='log_bytes_limit'
    assert (tmp_path/'supervisor'/'stdout.log').stat().st_size>1024


def test_macos_zombie_group_probe_is_rechecked_after_reaping_the_own_child(monkeypatch):
    calls=[]
    class Child:
        pid=123456;returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout):calls.append('reap');self.returncode=-15;return self.returncode
    child=Child()
    def group(pid):
        calls.append('probe')
        if child.returncode is None:raise PermissionError(1,'zombie-only group probe')
        return False
    monkeypatch.setattr(api(),'_group_exists',group)
    monkeypatch.setattr(api().os,'killpg',lambda pid,sig:calls.append('signal'))
    receipt=api()._terminate(child,100000000)
    assert child.returncode==-15 and receipt['group_absent_observed']
    assert calls.index('reap')<len(calls)-1 and calls.count('signal')==1
