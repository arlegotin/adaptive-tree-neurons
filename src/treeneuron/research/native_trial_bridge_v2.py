"""Native proposal -> managed fit -> fresh replay -> primary V, without promotion.

This composes engineering evidence. A separate qualified production registry
must admit the procedure before these inspected models can enter selection.
"""
import json
from pathlib import Path
import shutil
import time

from treeneuron.config import canonical_hash,canonical_json
from .ledger import write_new_json
from .method_worker import _ROOT,_read_json,_sha,_validate_request,scientific_source_sha256
from .native_fit_v2 import run_native_fit
from .native_candidate_v2 import inspect_primary_native_candidate
from .search_controller_v2 import _checked_plan,_files
from .search_supports_v2 import family_search_spec,compile_search_draw
from .transfer import TransferStore
from .trial_bridge_v2 import EngineeringTrialBridge


class InspectedNativeTrialBridge:
    def __init__(self,*,plan,template_path,expected_template_sha256,prepared_root,
            expected_manifest_sha256,outer_fold,search_directory,interpreter,
            extra_pythonpath=(),rss_limit_bytes=4*1024**3,official_schedule_reference=None,native_hpo_seed=None):
        started=time.perf_counter_ns();plan=_checked_plan(plan)
        template_path=Path(template_path).resolve();family=plan['identity']['family_id']
        template,_=_validate_request(template_path,expected_template_sha256,recoverable_training=family=='P12')
        prepared_root=Path(prepared_root).resolve();manifest=prepared_root/'manifest.json'
        if manifest.is_symlink() or _sha(manifest)!=expected_manifest_sha256:raise ValueError('Prepared manifest bytes differ')
        store=TransferStore(prepared_root);split=store._split(outer_fold)
        spec=family_search_spec(family,template['config']['feature_schema']);identity=plan['identity']
        expected_candidates=spec['finite_configurations'][1:] if family=='P13' else []
        expected_mode=spec['search_kind'];self.native_sampler_minimum_prerequisite_ns=0
        if family=='P06':
            from .native_schedule_v2 import load_realmlp_native_schedule
            official=load_realmlp_native_schedule(official_schedule_reference,
                expected_hpo_seed=native_hpo_seed,expected_feature_schema=spec['feature_schema'])
            if official['configurations'][:1]!=[a['config'] for a in spec['anchors']]:raise ValueError('Official P06 anchor differs')
            expected_candidates=official['configurations'][1:];expected_mode='finite'
            # The caller must charge its measured whole preparation call once,
            # including publication/return. This recorded span is a lower bound.
            self.native_sampler_minimum_prerequisite_ns=official['paid_preparation_ns']
        elif official_schedule_reference is not None or native_hpo_seed is not None:
            raise ValueError('Official native sampler reference belongs only to P06')
        if (plan['deadline_ns'] is not None or template['budget']!={'native_stopping':True}
                or family!=template['family_id'] or identity['source_sha256']!=template['scientific_source_sha256']
                or identity['source_sha256']!=scientific_source_sha256()
                or identity['split_sha256']!=split['split_hash'] or plan['fit_seed']!=template['fit_seed']
                or identity['support_sha256']!=spec['support_hash'] or len(identity['job_id'])>230
                or expected_mode not in ('finite','tpe') or plan['mode']!=expected_mode
                or plan['support']!=spec['support'] or plan['candidates']!=expected_candidates
                or plan['anchors']!=[a['config'] for a in spec['anchors']]
                or template['config']!=spec['anchors'][0]['config']
                or template['observation_identity']!=dict(dataset=store.manifest['dataset'],
                    split_kind='temporal' if store.manifest['dataset']=='bike' else 'grouped',split_sha256=split['split_hash'])
                or spec['feature_schema']!=dict(columns=store.manifest['feature_names'],categorical=store.manifest['categorical_features'])):
            raise ValueError('Native bridge requires exact current primary source/seed/support/anchor/native schedule')
        interpreter=Path(interpreter)
        if (not interpreter.is_absolute() or not interpreter.is_file() or type(rss_limit_bytes) is not int
                or rss_limit_bytes<=0 or not isinstance(extra_pythonpath,(list,tuple))):
            raise ValueError('Explicit interpreter, pinned overlays and owned RSS guard required')
        overlays=[]
        for value in extra_pythonpath:
            path=Path(value).resolve()
            if not path.is_dir() or not path.is_relative_to((_ROOT/'environment').resolve()):
                raise ValueError('Pinned project native overlay required')
            overlays.append(str(path))
        self._binding=canonical_json(dict(plan=plan,template=template,template_path=str(template_path),
            template_sha256=expected_template_sha256,prepared_root=str(prepared_root),
            expected_manifest_sha256=expected_manifest_sha256,outer_fold=outer_fold,
            search_directory=str(Path(search_directory).resolve()),interpreter=str(interpreter),
            extra_pythonpath=overlays,rss_limit_bytes=rss_limit_bytes,spec=spec,
            official_schedule_reference=official_schedule_reference,native_hpo_seed=native_hpo_seed))
        self._unchanged(json.loads(self._binding))
        self.preparation_ns=time.perf_counter_ns()-started

    @staticmethod
    def _unchanged(binding):
        template=binding['template'];path=Path(binding['template_path']);prepared=Path(binding['prepared_root'])
        if (path.is_symlink() or _sha(path)!=binding['template_sha256'] or scientific_source_sha256()!=template['scientific_source_sha256']
                or _sha(prepared/'manifest.json')!=binding['expected_manifest_sha256']):
            raise ValueError('Native bridge template/source/prepared manifest changed')
        manifest=TransferStore(prepared).manifest;fold=binding['outer_fold']
        if binding['official_schedule_reference'] is not None:
            from .native_schedule_v2 import load_realmlp_native_schedule
            official=load_realmlp_native_schedule(binding['official_schedule_reference'],
                expected_hpo_seed=binding['native_hpo_seed'],expected_feature_schema=binding['spec']['feature_schema'])
            if binding['plan']['candidates']!=official['configurations'][1:]:raise ValueError('Official P06 schedule changed')
        for role,ref in template['files'].items():
            relative=f'outer-{fold}/'+('V-features.npz' if role=='V_features' else role+'.npz')
            original=path.parent/ref['path'];sealed=prepared/relative
            if (original.is_symlink() or sealed.is_symlink() or _sha(original)!=ref['sha256']
                    or _sha(sealed)!=ref['sha256'] or manifest['files'][relative]['sha256']!=ref['sha256']):
                raise ValueError('Original or template primary role bytes changed')

    @staticmethod
    def _configuration(binding,request):
        plan=binding['plan'];origin=request['candidate_origin'];name=request['candidate_id']
        prefix,separator,index=name.rpartition('-') if type(name) is str else ('','','')
        if (not separator or prefix!=origin or origin not in ('anchor','finite','tpe')
                or not index.isdigit() or str(int(index))!=index):raise ValueError('Exact candidate identity required')
        index=int(index)
        if origin in ('anchor','finite'):
            values=plan['anchors'] if origin=='anchor' else plan['candidates']
            if index>=len(values) or request['config']!=values[index]:raise ValueError('Native finite/anchor proposal changed')
            return json.loads(canonical_json(values[index]))
        if plan['mode']!='tpe' or index>=plan['max_attempts']-len(plan['anchors']):raise ValueError('Adaptive proposal exceeds support')
        return compile_search_draw(plan['identity']['family_id'],binding['spec']['feature_schema'],request['config'])

    def __call__(self,request):
        started=time.perf_counter_ns();binding=json.loads(self._binding);plan=binding['plan'];self._unchanged(binding)
        keys={'candidate_id','candidate_origin','config','retry_index','checkpoint_files','attempt_index',
            'fit_seed','budget_remaining_ns','attempt_directory','search_directory'}
        if type(request) is not dict or set(request)!=keys:raise ValueError('Exact native controller request required')
        search=Path(binding['search_directory']);attempt=Path(request['attempt_directory']);index=request['attempt_index']
        if (type(index) is not int or not 0<=index<plan['max_attempts'] or request['budget_remaining_ns'] is not None
                or type(request['fit_seed']) is not int or request['fit_seed']!=plan['fit_seed']
                or type(request['retry_index']) is not int or request['retry_index'] not in (0,1)
                or attempt.is_symlink() or not attempt.is_dir() or any(attempt.iterdir())
                or attempt.resolve().parent!=search or attempt.name!=f'attempt-{index:03}'
                or Path(request['search_directory']).resolve()!=search):raise ValueError('Native proposal/queue/seed differs')
        compiled=self._configuration(binding,request);compilation_ns=time.perf_counter_ns()-started
        template=binding['template'];worker_request=dict(template,job_id=plan['identity']['job_id']+f'/attempt-{index:03}',
            config=compiled,files={role:dict(path=role+'.npz',sha256=ref['sha256']) for role,ref in template['files'].items()})
        recovery=None
        if request['retry_index']:
            refs=_files(search,request['checkpoint_files'])
            if len(refs)!=1 or Path(refs[0]['path']).name!='recovery.json':raise ValueError('One committed native recovery packet required')
            packet=_read_json(search/refs[0]['path'])
            if (set(packet)!={'plan_hash','initial_attempt_index','request','request_canonical_hash','native_reference'}
                    or packet['plan_hash']!=plan['plan_hash'] or type(packet['initial_attempt_index']) is not int
                    or not 0<=packet['initial_attempt_index']<index):raise ValueError('Native recovery identity changed')
            worker_request['job_id']=plan['identity']['job_id']+f"/attempt-{packet['initial_attempt_index']:03}"
            if packet['request']!=worker_request or canonical_hash(worker_request)!=packet['request_canonical_hash']:
                raise ValueError('Recovery cannot change the candidate, data, source or seed')
            recovery=packet['native_reference']
        elif request['checkpoint_files']!=[]:raise ValueError('Fresh native attempt cannot reuse undeclared state')
        generation=attempt/'generation';inputs=generation/'input';inputs.mkdir(parents=True)
        for role,ref in template['files'].items():
            target=inputs/(role+'.npz');shutil.copyfile(Path(binding['template_path']).parent/ref['path'],target)
            if _sha(target)!=ref['sha256']:raise ValueError('Native staged role copy changed')
        path=inputs/'request.json';write_new_json(path,worker_request);request_sha=_sha(path)
        native=run_native_fit(path,expected_request_sha256=request_sha,output=generation/'native',
            interpreter=binding['interpreter'],extra_pythonpath=binding['extra_pythonpath'],
            rss_limit_bytes=binding['rss_limit_bytes'],recovery_reference=recovery)
        write_new_json(generation/'native-fit-result.json',native)
        inspection=None;inspection_span=0;status=native['status'];reason=native.get('reason')
        if status=='completed_ineligible':
            inspection_started=time.perf_counter_ns()
            inspection=inspect_primary_native_candidate(generation_arguments=dict(request_path=path,
                expected_request_sha256=request_sha,native_reference=native['recovery_reference'],
                prepared_root=Path(binding['prepared_root']),expected_manifest_sha256=binding['expected_manifest_sha256'],
                outer_fold=binding['outer_fold'],expected_fit_seed=plan['fit_seed'],expected_configuration=compiled),
                immutable_generation_root=generation,output=attempt/'inspection',interpreter=binding['interpreter'],
                extra_pythonpath=binding['extra_pythonpath'],rss_limit_bytes=binding['rss_limit_bytes'])
            inspection_span=time.perf_counter_ns()-inspection_started
            if inspection['status']!='completed_candidate_inspection':status=inspection['status']
        checkpoints=[]
        if status in ('resource_exhausted','interrupted_recoverable'):
            status='interrupted_recoverable';reason='Owned resource/interruption stop; one identical paid recovery requires renewed admission'
            if request['retry_index']:initial=packet['initial_attempt_index']
            else:initial=index
            write_new_json(attempt/'recovery.json',dict(plan_hash=plan['plan_hash'],initial_attempt_index=initial,
                request=worker_request,request_canonical_hash=canonical_hash(worker_request),native_reference=native['recovery_reference']))
            checkpoints=[dict(path=str((attempt/'recovery.json').relative_to(search)),sha256=_sha(attempt/'recovery.json'))]
        elif status=='completed_ineligible':reason='Inspected engineering model; native workload/support qualification and production admission pending'
        else:reason=reason or 'Native trial did not complete its inspected model'
        self._unchanged(binding)
        write_new_json(attempt/'bridge.json',dict(schema='research/v2-inspected-native-trial',plan_hash=plan['plan_hash'],
            source_sha256=template['scientific_source_sha256'],candidate_id=request['candidate_id'],
            candidate_origin=request['candidate_origin'],raw_configuration=request['config'],compiled_configuration=compiled,
            official_schedule_reference=binding['official_schedule_reference'],native_hpo_seed=binding['native_hpo_seed'],
            fit_seed=plan['fit_seed'],native_request_sha256=request_sha,status=status,reason=reason,
            inspected_V_score=inspection['validation']['score'] if inspection and inspection['validation'] else None,
            compilation_ns=compilation_ns,inspection_span_ns=inspection_span,callback_body_ns=time.perf_counter_ns()-started,
            clock_scope='Callback entry through packet preparation; native fit and inspection spans are nested; outer search pays publication/return',
            quality_eligible=False,native_workload_qualified=False,HPO_journal_authenticated=False,T_accessed=False))
        return dict(status=status,V_score=None,reason=reason,files=EngineeringTrialBridge._files(attempt,search),checkpoint_files=checkpoints)
