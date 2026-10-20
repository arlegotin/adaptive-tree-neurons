"""Serial outer search accounting, independent of any model or label loader.

The coordinator callback must perform qualified F/G fitting and V evaluation.
This controller enforces paid slots and post-completion eligibility. Hard child
deadlines, process startup accounting, native qualification and protected-role
authorization belong to the production supervisor; this module grants none.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
import traceback

from treeneuron.config import canonical_hash,canonical_json
from treeneuron.analysis.practical_v2 import select_completed_trial
from .ledger import TERMINAL,StudyLease,write_new_json


def _copy(value):return json.loads(canonical_json(dict(value=value)))['value']


def _integer(value,low=0,high=None):
    return type(value) is int and value>=low and (high is None or value<=high)


def _finite(value):
    try:return type(value) in (int,float) and value>=0 and math.isfinite(value)
    except OverflowError:return False


def _remaining(plan,total):
    return None if plan['deadline_ns'] is None else plan['deadline_ns']-total


def _expired(plan,total):
    return plan['deadline_ns'] is not None and total>=plan['deadline_ns']


@dataclass(frozen=True)
class SearchPlan:
    resolved_json:str

    def to_dict(self):return json.loads(self.resolved_json)


def make_search_plan(*,identity,fit_seed,hpo_seed32,anchors,mode,support,candidates,
                     max_attempts=32,deadline_ns=7200000000000,
                     prefixes_ns=(600000000000,1800000000000)):
    """Bind supplied support; explicit deadline=None allows finite native quality.

    This grants neither support/native-worker qualification nor label access.
    Integer deadlines retain their legacy budget and completion semantics.
    """
    fields={'job_id','family_id','search_repeat','source_sha256','runtime_sha256','support_sha256','split_sha256'}
    if (not isinstance(identity,dict) or set(identity)!=fields or type(identity['job_id']) is not str
            or not identity['job_id'] or identity['family_id'] not in tuple(f'P{i:02}' for i in range(14))
            or type(identity['search_repeat']) is not int or identity['search_repeat'] not in (0,1)
            or any(type(identity[k]) is not str or re.fullmatch('[0-9a-f]{64}',identity[k]) is None
                for k in fields-{'job_id','family_id','search_repeat'})):
        raise ValueError('Explicit immutable search identity required')
    if (not _integer(fit_seed,0,2**63-1) or not _integer(hpo_seed32,0,2**32-1)
            or not _integer(max_attempts,1,32) or deadline_ns is not None and not _integer(deadline_ns,1,7200000000000)):
        raise ValueError('Invalid bounded search budget or seed')
    if (not isinstance(prefixes_ns,(list,tuple)) or any(not _integer(n,1,7200000000000 if deadline_ns is None else deadline_ns) for n in prefixes_ns)
            or list(prefixes_ns)!=sorted(set(prefixes_ns))):raise ValueError('Invalid descriptive prefixes')
    if (not isinstance(anchors,list) or not 1<=len(anchors)<=max_attempts
            or any(not isinstance(p,dict) or not p for p in anchors)
            or not isinstance(candidates,list) or any(not isinstance(p,dict) or not p for p in candidates)):
        raise ValueError('Explicit native anchors and candidate configurations required')
    if mode=='tpe':
        from .search_v2 import _support
        support=_support(support)
        if candidates:raise ValueError('TPE support cannot silently add a finite candidate schedule')
    elif mode=='finite':
        if support is not None:raise ValueError('Finite schedules have no TPE support')
    else:raise ValueError('Unknown search schedule')
    value=dict(schema='research/v2-outer-search-plan',identity=_copy(identity),fit_seed=fit_seed,
        hpo_seed32=hpo_seed32,anchors=_copy(anchors),mode=mode,support=_copy(support),
        candidates=_copy(candidates),max_attempts=max_attempts,deadline_ns=deadline_ns,
        prefixes_ns=list(prefixes_ns),anchor_feedback='outside_generic_tpe',
        recovery_policy='one_identical_candidate_recovery_paid_as_another_slot',
        protected_access_authorized=False)
    value['plan_hash']=canonical_hash(value)
    return SearchPlan(canonical_json(value))


def _checked_plan(plan):
    if not isinstance(plan,SearchPlan):raise ValueError('Immutable SearchPlan required')
    saved=plan.to_dict()
    keys=('identity','fit_seed','hpo_seed32','anchors','mode','support','candidates','max_attempts','deadline_ns','prefixes_ns')
    try:rebuilt=make_search_plan(**{k:saved[k] for k in keys}).to_dict()
    except (KeyError,TypeError) as error:raise ValueError('Invalid search plan') from error
    if canonical_json(saved)!=canonical_json(rebuilt):raise ValueError('Search plan commitment or policy differs')
    return rebuilt


def _files(directory,refs):
    if not isinstance(refs,list):raise ValueError('Explicit artifact file inventory required')
    output=[];seen=set();root=directory.resolve()
    for ref in refs:
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or type(ref['path']) is not str:
            raise ValueError('Invalid artifact file commitment')
        path=Path(ref['path']);path=path if path.is_absolute() else directory/path
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Artifact file commitment escapes its search or is absent')
        relative=str(path.resolve().relative_to(root))
        if (relative in seen or type(ref['sha256']) is not str
                or hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']):
            raise ValueError('Artifact file commitment changed or duplicated')
        seen.add(relative);output.append(dict(path=relative,sha256=ref['sha256']))
    return output


def _result(directory,value):
    keys={'status','V_score','files','checkpoint_files','reason'}
    if (not isinstance(value,dict) or set(value)!=keys or value['status'] not in TERMINAL
            or (value['status']=='completed_eligible' and (not _finite(value['V_score']) or value['reason'] is not None))
            or (value['status']!='completed_eligible' and (value['V_score'] is not None
                or type(value['reason']) is not str or not value['reason']))):
        raise ValueError('Coordinator result contradicts its eligibility or terminal status')
    files=_files(directory,value['files']);checkpoints=_files(directory,value['checkpoint_files'])
    if (value['status']=='completed_eligible' and not files
            or value['status']=='interrupted_recoverable' and not checkpoints
            or any(ref not in files for ref in checkpoints)):
        raise ValueError('Eligible model/recoverable checkpoint file commitments required')
    return _copy(value)|dict(files=files,checkpoint_files=checkpoints)


class _Cursor:
    def __init__(self,plan):
        self.plan=plan;self.anchor=0;self.index=0;self.current=None;self.recovery_exhausted=False
        self.sampler=None
        if plan['mode']=='tpe' and len(plan['anchors'])<plan['max_attempts']:
            from .search_v2 import ReplayTPEV2
            self.sampler=ReplayTPEV2(seed=plan['hpo_seed32'],support=plan['support'],
                identity=plan['identity'],max_suggestions=plan['max_attempts']-len(plan['anchors']))

    @property
    def dependencies(self):return None if self.sampler is None else self.sampler.dependency_identity

    def take(self):
        if self.current is not None:return _copy(self.current)
        if self.anchor<len(self.plan['anchors']):
            index=self.anchor;self.anchor+=1;origin='anchor';config=self.plan['anchors'][index]
        elif self.plan['mode']=='finite':
            if self.index>=len(self.plan['candidates']):return None
            index=self.index;self.index+=1;origin='finite';config=self.plan['candidates'][index]
        else:
            index=self.index;self.index+=1;origin='tpe';config=self.sampler.ask()
        self.current=dict(candidate_id=f'{origin}-{index}',candidate_origin=origin,config=_copy(config),
            retry_index=0,checkpoint_files=[])
        return _copy(self.current)

    def finish(self,result):
        if result['status']=='interrupted_recoverable' and self.current['retry_index']==0:
            self.current|=dict(retry_index=1,checkpoint_files=_copy(result['checkpoint_files']))
            return
        if result['status']=='interrupted_recoverable':self.recovery_exhausted=True
        if self.current['candidate_origin']=='tpe':
            self.sampler.tell(result['V_score'] if result['status']=='completed_eligible' else None)
        self.current=None


def _read_events(directory,plan,cursor):
    events=[];previous='0'*64;total=0;session=False;active=None;records=[]
    admission_identity=None;admission_permission=None
    fields={'kind','total_charged_ns','previous_hash','event_hash','payload'}
    for line in (directory/'events.jsonl').read_text().splitlines():
        event=json.loads(line)
        if (not isinstance(event,dict) or set(event)!=fields or event['previous_hash']!=previous
                or canonical_hash({k:v for k,v in event.items() if k!='event_hash'})!=event['event_hash']
                or not _integer(event['total_charged_ns'],total)):
            raise ValueError('Search event commitment or paid clock changed')
        row=event['payload'];kind=event['kind']
        if kind=='session_started':
            base={'plan_hash','sampler_dependencies','invocation_preparation_ns'}
            if (session or set(row) not in (base,base|{'trial_admission_identity'})
                    or row['plan_hash']!=plan['plan_hash'] or not _integer(row['invocation_preparation_ns'])
                    or canonical_json(dict(value=row['sampler_dependencies']))!=canonical_json(dict(value=cursor.dependencies))):
                raise ValueError('Search session plan or sampler dependencies changed')
            admission_identity=row.get('trial_admission_identity');admission_permission=None
            if 'trial_admission_identity' in row and (type(admission_identity) is not dict or not admission_identity):
                raise ValueError('Explicit trial admission identity required')
            session=True
        elif kind=='trial_admission':
            if (not session or active is not None or admission_identity is None or admission_permission is not None
                    or set(row)!={'identity_sha256','decision'} or row['identity_sha256']!=canonical_hash(admission_identity)
                    or type(row['decision']) is not dict or type(row['decision'].get('admitted')) is not bool):
                raise ValueError('Invalid pre-proposal resource admission event')
            admission_permission=row['decision']['admitted']
        elif kind=='attempt_started':
            if (not session or active is not None or len(records)>=plan['max_attempts']
                    or admission_identity is not None and admission_permission is not True):
                raise ValueError('Invalid outer attempt transition')
            admission_permission=None
            expected=cursor.take()
            request=dict(**expected,attempt_index=len(records),fit_seed=plan['fit_seed'],
                budget_remaining_ns=_remaining(plan,event['total_charged_ns'])) if expected is not None else None
            if request is None or _expired(plan,event['total_charged_ns']) or canonical_json(row)!=canonical_json(request):
                raise ValueError('Archived candidate/seed/budget differs from exact search replay')
            active=row
        elif kind=='attempt_finished':
            if not session or active is None or set(row)!={'record','result'}:
                raise ValueError('Invalid outer result transition')
            result=row['result'];raw={k:v for k,v in result.items() if k in {'status','V_score','files','checkpoint_files','reason'}}
            if _result(directory,raw)!=raw:raise ValueError('Archived artifact file commitment changed')
            record=_record(plan,active,result,event['total_charged_ns'],records)
            if canonical_json(record)!=canonical_json(row['record']):raise ValueError('Archived V record or cost differs')
            cursor.finish(result);records.append(record);active=None
        elif kind=='session_finished':
            if not session or active is not None or set(row)!={'stop_reason'}:
                raise ValueError('Invalid search session completion')
            if ((row['stop_reason']=='resource_pause')!=(admission_permission is False)):
                raise ValueError('Resource pause requires the exact denied pre-proposal decision')
            session=False
        else:raise ValueError('Unknown search journal event')
        events.append(event);previous=event['event_hash'];total=event['total_charged_ns']
    if session or active is not None:
        raise ValueError('Unclosed search session: authenticated external invocation cost/disposition required before recovery')
    return events,records,total


def _record(plan,request,result,total,records):
    return dict(attempt_id=f"{plan['identity']['job_id']}/attempt-{request['attempt_index']:03}",
        attempt_index=request['attempt_index'],family_id=plan['identity']['family_id'],
        search_repeat=plan['identity']['search_repeat'],candidate_id=request['candidate_id'],
        candidate_origin=request['candidate_origin'],retry_index=request['retry_index'],
        config=_copy(request['config']),fit_seed=request['fit_seed'],status=result['status'],
        V_score=result['V_score'],reported_status=result.get('reported_status',result['status']),
        reported_V_score=result.get('reported_V_score',result['V_score']),
        unexpected_exception=result.get('unexpected_exception',False),files=_copy(result['files']),
        completion_ns=total,charged_ns=total-(records[-1]['completion_ns'] if records else 0))


def run_search(directory,*,plan,evaluate_trial,resume=False,clock_ns=time.perf_counter_ns,
               max_attempts_this_invocation=32,invocation_preparation_ns=0,expected_journal_hash=None,
               invocation_origin_ns=None,trial_admission=None,trial_admission_identity=None):
    """Execute serial callback attempts, or identical recovery after a clean pause.

    An unclosed process-death journal fails closed: its unmeasured tail cannot be
    assigned zero cost. The external crash supervisor/recovery receipt is pending.
    Returned V scores are caller-supplied, not independently authenticated here.
    An externally authenticated origin includes cold startup/preparation through
    this entry without a measurement-to-call gap. Additional preparation may pay
    a preceding closed episode's tail; the controller authenticates neither.
    """
    started=clock_ns();plan=_checked_plan(plan);directory=Path(directory)
    if (not _integer(started) or not _integer(max_attempts_this_invocation,1,32)
            or not _integer(invocation_preparation_ns) or type(resume) is not bool or not callable(evaluate_trial)):
        raise ValueError('Invalid search invocation')
    if ((trial_admission is None)!=(trial_admission_identity is None)
            or trial_admission is not None and (not callable(trial_admission)
                or type(trial_admission_identity) is not dict or not trial_admission_identity)):
        raise ValueError('Paired resource admission callback and explicit identity required')
    if trial_admission_identity is not None:trial_admission_identity=_copy(trial_admission_identity)
    if invocation_origin_ns is not None:
        if not _integer(invocation_origin_ns,0,started):
            raise ValueError('Invalid or future invocation clock origin')
        invocation_preparation_ns+=started-invocation_origin_ns
    if directory.is_symlink():raise ValueError('Search directory cannot be a symbolic link')
    if not resume:
        if directory.exists():raise ValueError('Search already exists; use unchanged-plan resume')
        directory.mkdir(parents=True);write_new_json(directory/'plan.json',plan)
        (directory/'events.jsonl').touch(exist_ok=False)
    with StudyLease(directory):
        if canonical_json(json.loads((directory/'plan.json').read_text()))!=canonical_json(plan):
            raise ValueError('Cannot resume a changed search plan')
        cursor=_Cursor(plan);events,records,base=_read_events(directory,plan,cursor)
        if resume and (not events or type(expected_journal_hash) is not str
                or events[-1]['event_hash']!=expected_journal_hash):
            raise ValueError('Resume requires the externally retained search journal commitment')
        if events and events[-1]['payload']['stop_reason'] not in ('operator_pause','interrupted_recoverable','resource_pause'):
            raise ValueError('Search is terminal; no additional attempts are permitted')
        def now():
            value=clock_ns()
            if not _integer(value,started):raise ValueError('Monotonic search clock moved backwards')
            return base+invocation_preparation_ns+value-started
        def append(kind,payload,total=None):
            event=dict(kind=kind,total_charged_ns=now() if total is None else total,payload=_copy(payload),
                previous_hash=events[-1]['event_hash'] if events else '0'*64)
            if events and event['total_charged_ns']<events[-1]['total_charged_ns']:
                raise ValueError('Paid search clock cannot decrease')
            event['event_hash']=canonical_hash(event)
            with (directory/'events.jsonl').open('a') as stream:
                stream.write(canonical_json(event)+'\n');stream.flush()
                import os
                os.fsync(stream.fileno())
            events.append(event)
        session_payload=dict(plan_hash=plan['plan_hash'],sampler_dependencies=cursor.dependencies,
            invocation_preparation_ns=invocation_preparation_ns)
        if trial_admission is not None:session_payload['trial_admission_identity']=_copy(trial_admission_identity)
        append('session_started',session_payload)
        count=0
        while True:
            total=now()
            if _expired(plan,total):reason='deadline';break
            if len(records)>=plan['max_attempts']:reason='attempt_limit';break
            if count>=max_attempts_this_invocation:reason='operator_pause';break
            if trial_admission is not None:
                decision=trial_admission()
                if type(decision) is not dict or type(decision.get('admitted')) is not bool:
                    raise ValueError('Resource admission must retain a strict boolean decision')
                append('trial_admission',dict(identity_sha256=canonical_hash(trial_admission_identity),decision=decision))
                if not decision['admitted']:reason='resource_pause';break
            candidate=cursor.take()
            if candidate is None:reason='support_exhausted';break
            # Candidate generation/replay/preparation is paid before admission.
            total=now()
            if _expired(plan,total):reason='deadline';break
            request=dict(**candidate,attempt_index=len(records),fit_seed=plan['fit_seed'],
                budget_remaining_ns=_remaining(plan,total))
            append('attempt_started',request,total)
            attempt=directory/f"attempt-{len(records):03}";attempt.mkdir(exist_ok=False)
            callback_request=_copy(request)|dict(attempt_directory=str(attempt.resolve()),search_directory=str(directory.resolve()))
            unexpected=False
            try:result=_result(directory,evaluate_trial(callback_request))
            except Exception as error:
                unexpected=True;path=attempt/'unexpected-error.json'
                write_new_json(path,dict(exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
                partial_files=[]
                for saved in sorted(attempt.rglob('*')):
                    if saved.is_symlink():raise ValueError('Unexpected trial artifact symlink; audit required')
                    if saved.is_file():
                        with saved.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                        partial_files.append(dict(path=str(saved.relative_to(directory)),sha256=digest))
                result=dict(status='input_or_dependency_failure',V_score=None,
                    files=partial_files,
                    checkpoint_files=[],reason='Unexpected coordinator exception; diagnosis required',unexpected_exception=True)
            total=now()
            if plan['deadline_ns'] is not None and total>plan['deadline_ns'] and result['status']=='completed_eligible':
                result|=dict(reported_status=result['status'],reported_V_score=result['V_score'],status='timeout',
                    V_score=None,reason='Completed after the enclosing dedicated-worker deadline')
            record=_record(plan,request,result,total,records)
            append('attempt_finished',dict(record=record,result=result),total)
            cursor.finish(result);records.append(record);count+=1
            if unexpected:reason='unexpected_coordinator_error';break
            if cursor.recovery_exhausted:reason='recovery_exhausted';break
            if result['status']=='interrupted_recoverable':
                reason=('recovery_budget_exhausted' if len(records)>=plan['max_attempts'] or _expired(plan,now())
                    else 'interrupted_recoverable')
                break
        # Prefix selection/summary preparation itself belongs to this invocation.
        selected=select_completed_trial(records,deadline_ns=now() if plan['deadline_ns'] is None else plan['deadline_ns'])
        prefixes={str(n):select_completed_trial(records,deadline_ns=n) for n in plan['prefixes_ns']}
        append('session_finished',dict(stop_reason=reason))
        total=events[-1]['total_charged_ns']
        return dict(schema='research/v2-outer-search-result',plan_hash=plan['plan_hash'],stop_reason=reason,
            records=_copy(records),attempts_paid=len(records),winner=selected['winner'],prefixes=prefixes,
            total_charged_ns=total,final_unassigned_coordinator_ns=total-sum(r['charged_ns'] for r in records),
            sampler_history=[] if cursor.sampler is None else cursor.sampler.history,
            pending_suggestion=None if cursor.sampler is None else cursor.sampler.pending_suggestion,
            journal_hash=events[-1]['event_hash'],protected_access_authorized=False,native_worker_qualified=False,
            hard_worker_deadline_implemented=False,whole_process_cost_authenticated=False,
            clock_scope='Supplied invocation preparation plus function entry through final clock read; final publication/return and caller startup need external supervisor receipt')
