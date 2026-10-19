"""Atomic TPE to supervised native engineering trials; no V-score authority."""
import json
from pathlib import Path
import time

from treeneuron.config import canonical_hash,canonical_json
from .ledger import write_new_json
from .search_controller_v2 import _checked_plan
from .search_supports_v2 import family_search_spec,compile_search_draw
from .trial_bridge_v2 import EngineeringTrialBridge


class CompiledEngineeringTrialBridge:
    def __init__(self,*,plan,template_path,expected_template_sha256,search_directory,
            interpreter,extra_pythonpath=(),rss_limit_bytes=512*1024**2):
        started=time.perf_counter_ns();bound=_checked_plan(plan)
        self._native=EngineeringTrialBridge(plan=plan,template_path=template_path,
            expected_template_sha256=expected_template_sha256,search_directory=search_directory,
            interpreter=interpreter,extra_pythonpath=extra_pythonpath,rss_limit_bytes=rss_limit_bytes)
        native=json.loads(self._native._binding);family=bound['identity']['family_id']
        spec=family_search_spec(family,native['template']['config']['feature_schema'])
        if (spec['search_kind']!='tpe' or bound['mode']!='tpe' or bound['support']!=spec['support']
                or bound['identity']['support_sha256']!=spec['support_hash']
                or bound['anchors']!=[a['config'] for a in spec['anchors']]):
            raise ValueError('Typed bridge requires the exact declared family support and native anchor sequence')
        self._binding=canonical_json(dict(plan=bound,spec=spec,native=native))
        self.preparation_ns=time.perf_counter_ns()-started

    def __call__(self,request):
        started=time.perf_counter_ns();binding=json.loads(self._binding)
        plan=binding['plan'];spec=binding['spec'];native=binding['native']
        keys={'candidate_id','candidate_origin','config','retry_index','checkpoint_files',
            'attempt_index','fit_seed','budget_remaining_ns','attempt_directory','search_directory'}
        if type(request) is not dict or set(request)!=keys:raise ValueError('Exact controller proposal required')
        search=Path(native['search_directory']);attempt=Path(request['attempt_directory'])
        index=request['attempt_index'];remaining=request['budget_remaining_ns']
        if (type(index) is not int or not 0<=index<plan['max_attempts']
                or type(remaining) is not int or not 0<remaining<=plan['deadline_ns']
                or type(request['fit_seed']) is not int or request['fit_seed']!=plan['fit_seed']
                or type(request['retry_index']) is not int or request['retry_index']!=0 or request['checkpoint_files']!=[]
                or attempt.is_symlink() or not attempt.is_dir() or any(attempt.iterdir())
                or attempt.resolve().parent!=search or attempt.name!=f'attempt-{index:03}'
                or Path(request['search_directory']).resolve()!=search
                or request['candidate_origin'] not in ('anchor','tpe') or type(request['candidate_id']) is not str):
            raise ValueError('Typed proposal differs from its bound queue or requests unqualified recovery')
        origin=request['candidate_origin'];prefix,separator,number=request['candidate_id'].rpartition('-')
        if (not separator or prefix!=origin or not number.isdigit() or str(int(number))!=number):
            raise ValueError('Candidate ID/origin linkage differs')
        candidate_index=int(number)
        self._native._unchanged(native)
        if origin=='anchor':
            if candidate_index>=len(plan['anchors']) or request['config']!=plan['anchors'][candidate_index]:
                raise ValueError('Native anchor changed')
            compiled=json.loads(canonical_json(request['config']))
        else:
            if candidate_index>=plan['max_attempts']-len(plan['anchors']):raise ValueError('Adaptive proposal limit exceeded')
            compiled=compile_search_draw(plan['identity']['family_id'],spec['feature_schema'],request['config'])
        compilation=time.perf_counter_ns()-started;worker_budget=max(0,remaining-compilation)
        if worker_budget:
            result=self._native(dict(request,config=compiled,budget_remaining_ns=worker_budget))
        else:
            result=dict(status='timeout',V_score=None,files=[],checkpoint_files=[],
                reason='Enclosing trial budget exhausted by source/compilation preparation before worker launch')
        raw=request['config'];inactive=[]
        if origin=='tpe':
            inactive=[name[:-5]+'_value' for name,value in raw.items() if name.endswith('_mode') and value=='zero']
            if plan['identity']['family_id']=='P07' and raw['variant']=='plain':inactive+=['n_bins','d_embedding']
        self._native._unchanged(native)
        write_new_json(attempt/'translation.json',dict(schema='research/v2-compiled-engineering-proposal',
            source_sha256=plan['identity']['source_sha256'],family_support_hash=spec['support_hash'],
            candidate_id=request['candidate_id'],candidate_origin=origin,raw_configuration=raw,
            compiled_configuration=compiled,raw_configuration_sha256=canonical_hash(raw),
            compiled_configuration_sha256=canonical_hash(compiled),inactive_conditional_coordinates=sorted(inactive),
            compilation_ns=compilation,compilation_clock_scope='Wrapper entry through identity checks and compilation',
            remaining_budget_at_entry_ns=remaining,remaining_worker_budget_ns=worker_budget,
            V_labels_read=False,quality_eligible=False,protected_access_authorized=False))
        self._native._unchanged(native)
        return dict(result,files=self._native._files(attempt,search))
