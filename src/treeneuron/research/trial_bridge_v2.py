"""Supervised engineering-only trials for the immutable paid search controller.

No V labels or scores are accepted here. Native quality and recovery admission
remain separate; this bridge cannot turn an engineering model into a winner.
"""
import errno
import json
import os
from pathlib import Path
import shutil
import time

from treeneuron.config import canonical_hash,canonical_json
from .method_worker import (_ROOT,_THREADS,_ADAPTERS,_sha,_read_json,_write,
                            _validate_request,scientific_source_sha256)
from .search_controller_v2 import _checked_plan
from .process_supervision_v2 import supervise_process


class EngineeringTrialBridge:
    def __init__(self,*,plan,template_path,expected_template_sha256,search_directory,
                 interpreter,extra_pythonpath=(),rss_limit_bytes=512*1024**2):
        started=time.perf_counter_ns()
        bound=_checked_plan(plan);template_path=Path(template_path)
        template,_=_validate_request(template_path,expected_template_sha256)
        identity=bound['identity'];interpreter=Path(interpreter)
        if (identity['family_id'] not in _ADAPTERS or identity['family_id']!=template['family_id']
                or identity['source_sha256']!=template['scientific_source_sha256']
                or identity['split_sha256']!=template['observation_identity']['split_sha256']
                or bound['fit_seed']!=template['fit_seed'] or len(identity['job_id'])>230
                or not interpreter.is_absolute() or not interpreter.is_file()
                or type(rss_limit_bytes) is not int or rss_limit_bytes<=0
                or not isinstance(extra_pythonpath,(list,tuple))):
            raise ValueError('Engineering bridge source/seed/method/split/runtime binding differs')
        overlays=[]
        for value in extra_pythonpath:
            path=Path(value).resolve()
            if not path.is_dir() or not path.is_relative_to((_ROOT/'environment').resolve()):
                raise ValueError('Engineering overlays must use the separately pinned project environment')
            overlays.append(str(path))
        self._binding=canonical_json(dict(plan=bound,template=template,
            template_path=str(template_path.resolve()),template_sha256=expected_template_sha256,
            search_directory=str(Path(search_directory).resolve()),interpreter=str(interpreter),
            extra_pythonpath=overlays,rss_limit_bytes=rss_limit_bytes))
        self.preparation_ns=time.perf_counter_ns()-started

    def _unchanged(self,binding):
        path=Path(binding['template_path'])
        if (path.is_symlink() or not path.is_file() or _sha(path)!=binding['template_sha256']
                or scientific_source_sha256()!=binding['template']['scientific_source_sha256']):
            raise ValueError('Engineering bridge template or source changed')
        for ref in binding['template']['files'].values():
            file=path.parent/ref['path']
            if (file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(path.parent)
                    or _sha(file)!=ref['sha256']):
                raise ValueError('Engineering bridge committed role archive changed')

    @staticmethod
    def _files(attempt,search):
        result=[]
        for p in sorted(attempt.rglob('*')):
            if p.is_symlink():raise ValueError('Unexpected engineering artifact symlink')
            if p.is_file():result.append(dict(path=p.relative_to(search).as_posix(),sha256=_sha(p)))
        return result

    @staticmethod
    def _worker_terminal(worker,request,request_sha):
        path=worker/'terminal.json'
        if not path.is_file() or path.is_symlink():raise ValueError('Missing engineering worker terminal; diagnosis required')
        terminal=_read_json(path)
        if (terminal.get('purpose')!='engineering_qualification'
                or terminal.get('expected_request_sha256')!=request_sha
                or terminal.get('scientific_source_sha256')!=request['scientific_source_sha256']
                or terminal.get('fit_seed')!=request['fit_seed'] or terminal.get('family_id')!=request['family_id']
                or terminal.get('job_id')!=request['job_id']
                or terminal.get('observation_identity')!=request['observation_identity']
                or terminal.get('accessed_roles')!=['F_labels','G_labels','V_features']
                or any(terminal.get(k) is not False for k in
                    ('quality_eligible','checkpoint_eligible','protected_access_authorized','T_accessed'))):
            raise ValueError('Engineering worker identity/role/eligibility differs from its request')
        expected={};root=worker.resolve()
        for ref in terminal['files']:
            relative=Path(ref['path']);p=worker/relative
            if (relative.is_absolute() or '..' in relative.parts or p.is_symlink()
                    or not p.is_file() or not p.resolve().is_relative_to(root)
                    or ref['path'] in expected or p.stat().st_size!=ref['bytes'] or _sha(p)!=ref['sha256']):
                raise ValueError('Engineering worker artifact commitment differs')
            expected[ref['path']]=ref['sha256']
        actual={p.relative_to(worker).as_posix() for p in worker.rglob('*') if p.is_file()}
        if actual!=set(expected)|{'terminal.json'}:
            raise ValueError('Engineering worker artifact inventory is incomplete')
        return terminal

    def __call__(self,request):
        started=time.perf_counter_ns();binding=json.loads(self._binding)
        template=binding['template'];plan=binding['plan'];self._unchanged(binding)
        search=Path(binding['search_directory']);attempt=Path(request['attempt_directory'])
        index=request['attempt_index'];remaining=request['budget_remaining_ns']
        if (type(index) is not int or not 0<=index<plan['max_attempts']
                or attempt.is_symlink() or not attempt.is_dir() or any(attempt.iterdir())
                or attempt.resolve().parent!=search or attempt.name!=f'attempt-{index:03}'
                or Path(request['search_directory']).resolve()!=search
                or type(request['fit_seed']) is not int or request['fit_seed']!=plan['fit_seed']
                or request['retry_index']!=0 or request['checkpoint_files']!=[]
                or type(remaining) is not int or not 0<remaining<=plan['deadline_ns']
                or type(request['config']) is not dict or request['config'].get('family')!=template['family_id']
                or request['config'].get('feature_schema')!=template['config']['feature_schema']):
            raise ValueError('Engineering callback differs from its bound queue or requests unqualified recovery')
        input_root=attempt/'input';input_root.mkdir();staged={};material=[]
        for role,ref in template['files'].items():
            original=Path(binding['template_path']).parent/ref['path'];target=input_root/(role+'.npz')
            try:os.link(original,target);method='hard_link'
            except OSError as error:
                if error.errno!=errno.EXDEV:raise
                shutil.copyfile(original,target);method='verified_copy'
            if _sha(target)!=ref['sha256']:raise ValueError('Staged engineering role bytes changed')
            staged[role]=dict(path=target.name,sha256=ref['sha256'])
            material.append(dict(role=role,path=target.name,source_path=str(original),sha256=ref['sha256'],
                logical_bytes=target.stat().st_size,method=method))
        _write(input_root/'materialization.json',dict(files=material,
            policy='Immutable byte-authenticated role archives; hard links or disclosed cross-device copies'))
        elapsed=lambda:time.perf_counter_ns()-started
        def finish(status,reason,process=None,worker=None):
            self._unchanged(binding)
            for ref in staged.values():
                if _sha(input_root/ref['path'])!=ref['sha256']:raise ValueError('Staged role changed during engineering attempt')
            _write(attempt/'bridge.json',dict(schema='research/v2-engineering-trial-bridge',
                plan_hash=plan['plan_hash'],template_sha256=binding['template_sha256'],
                status=status,reason=reason,fit_seed=request['fit_seed'],config_sha256=canonical_hash(request['config']),
                source_sha256=template['scientific_source_sha256'],budget_remaining_at_entry_ns=remaining,
                callback_body_before_final_hashing_ns=elapsed(),
                process_receipt_sha256=_sha(attempt/'process/process.json') if process is not None else None,
                worker_terminal_sha256=_sha(attempt/'worker/terminal.json') if worker is not None else None,
                supervisor_external_span_ns=process_span if process is not None else None,
                V_labels_read=False,V_score=None,quality_eligible=False,protected_access_authorized=False))
            return dict(status=status,V_score=None,files=self._files(attempt,search),checkpoint_files=[],reason=reason)
        available=remaining-elapsed()
        if available<=0:return finish('timeout','Enclosing budget exhausted by paid input staging before worker launch')
        worker_request=dict(template,job_id=plan['identity']['job_id']+f'/attempt-{index:03}',
            config=request['config'],budget=dict(max_seconds=available/1e9),files=staged)
        path=input_root/'request.json';_write(path,worker_request);request_sha=_sha(path)
        env={k:'1' for k in _THREADS}
        env.update(PYTHONPATH=os.pathsep.join([str(_ROOT/'src'),*binding['extra_pythonpath']]),
            PYTHONDONTWRITEBYTECODE='1',KMP_DUPLICATE_LIB_OK=None,
            MPLCONFIGDIR=str(attempt/'cache/mpl'),XDG_CACHE_HOME=str(attempt/'cache/xdg'))
        available=remaining-elapsed()
        if available<=0:return finish('timeout','Enclosing budget exhausted by paid request preparation before worker launch')
        process_started=time.perf_counter_ns()
        process=supervise_process(argv=[binding['interpreter'],'-B','-m','treeneuron.research.method_worker',
            '--request',str(path),'--expected-request-sha256',request_sha,'--output',str(attempt/'worker')],
            cwd=_ROOT,env_overrides=env,output=attempt/'process',timeout_ns=available,
            rss_limit_bytes=binding['rss_limit_bytes'],
            identity=dict(purpose='engineering_qualification',plan_hash=plan['plan_hash'],
                request_sha256=request_sha,scientific_source_sha256=template['scientific_source_sha256'],
                fit_seed=request['fit_seed']))
        process_span=time.perf_counter_ns()-process_started
        status=process['execution_status']
        if status in ('timeout','resource_exhausted','input_or_dependency_failure'):
            return finish(status,'Supervisor: '+process['stop_reason'],process=process)
        if status!='exited':raise ValueError('Interrupted native recovery is not qualified by the engineering bridge')
        worker=self._worker_terminal(attempt/'worker',worker_request,request_sha)
        status=worker['status']
        if status not in ('completed_ineligible','timeout','resource_exhausted','numerical_failure','input_or_dependency_failure'):
            raise ValueError('Unqualified or unknown engineering worker terminal status')
        if (status=='completed_ineligible')!=(process['returncode']==0):
            # The worker also permits a retained observed-after-fit timeout at
            # exit0. It remains ineligible; no process exit can promote it.
            if not (status=='timeout' and process['returncode']==0):
                raise ValueError('Worker terminal contradicts its process exit')
        return finish(status,worker.get('reason') or 'Engineering model; native quality qualification pending',
            process=process,worker=worker)
