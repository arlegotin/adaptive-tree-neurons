"""Read-only exact native-quality search replay; no model or V authority."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

from treeneuron.analysis.practical_v2 import select_completed_trial
from treeneuron.config import canonical_json
from .deployment_worker_v2 import _bytes,_json,_source
from .method_worker import _sha
from .search_controller_v2 import SearchPlan,_checked_plan,_Cursor,_read_events,_files


@contextmanager
def _inactive(root):
    if root.is_symlink() or not root.is_dir():raise ValueError('Ordinary closed search directory required')
    lease=root/'worker-lease.json'
    if lease.is_symlink() or not lease.is_file():raise ValueError('Existing search writer lease required')
    with lease.open('rb') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as error:raise PermissionError('Cannot audit an active search writer') from error
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def _stops(events,plan):
    records=[];finishes=[];previous_total=0
    for index,event in enumerate(events):
        if (event['kind']=='session_started'
                and event['total_charged_ns']-previous_total<event['payload']['invocation_preparation_ns']):
            raise ValueError('Declared search preparation exceeds its charged session clock')
        previous_total=event['total_charged_ns']
        if event['kind']=='attempt_finished':
            result=event['payload']['result'];fields={'status','V_score','files','checkpoint_files','reason'}
            if (set(result)!=fields and (set(result)!=fields|{'unexpected_exception'}
                    or result['unexpected_exception'] is not True)):
                raise ValueError('Native controller result has unknown or impossible fields')
            if ('unexpected_exception' in result and (result['status']!='input_or_dependency_failure'
                    or result['V_score'] is not None or result['checkpoint_files']!=[]
                    or result['reason']!='Unexpected coordinator exception; diagnosis required')):
                raise ValueError('Unexpected native controller result contradicts its failure disposition')
            records.append(event['payload']['record'])
        if event['kind']!='session_finished':continue
        reason=event['payload']['stop_reason'];count=len(records);last=records[-1] if records else None
        if reason=='attempt_limit':valid=count==plan['max_attempts']
        elif reason=='support_exhausted':
            valid=(plan['mode']=='finite' and count<plan['max_attempts']
                and sum(r['retry_index']==0 for r in records)==len(plan['anchors'])+len(plan['candidates']))
        elif reason=='operator_pause':
            valid=(0<count<plan['max_attempts'] and last['status']!='interrupted_recoverable'
                and last['unexpected_exception'] is False)
        elif reason=='resource_pause':
            before=events[index-1] if index>0 else None
            valid=(count<plan['max_attempts'] and before is not None and before['kind']=='trial_admission'
                and before['payload']['decision']['admitted'] is False)
        elif reason in ('interrupted_recoverable','recovery_budget_exhausted'):
            valid=(last is not None and last['status']=='interrupted_recoverable' and last['retry_index']==0
                and (count<plan['max_attempts'])==(reason=='interrupted_recoverable'))
        elif reason=='recovery_exhausted':valid=last is not None and last['status']=='interrupted_recoverable' and last['retry_index']==1
        elif reason=='unexpected_coordinator_error':valid=last is not None and last['unexpected_exception'] is True
        else:valid=False
        if not valid:raise ValueError('Native search stop reason contradicts its paid history')
        required=None
        if last is not None:
            if last['unexpected_exception']:required='unexpected_coordinator_error'
            elif last['status']=='interrupted_recoverable':
                required=('recovery_exhausted' if last['retry_index']==1 else
                    'recovery_budget_exhausted' if count==plan['max_attempts'] else 'interrupted_recoverable')
        if required is not None and reason not in (required,'resource_pause'):
            raise ValueError('Native stop bypasses its required exception/recovery disposition')
        finishes.append(reason)
    if (not finishes or any(reason not in ('operator_pause','interrupted_recoverable','resource_pause') for reason in finishes[:-1])):
        raise ValueError('A terminal native search was illegally continued')
    return finishes


def audit_native_search_journal(*,search_directory,expected_plan_sha256,result_path,
        expected_result_sha256,expected_journal_sha256):
    """Authenticate declared bytes and reproduce the actual paid proposal law.

    Raw V scores are journal inputs, not independently verified predictions.
    Supply their candidate inspection/qualification separately before selection.
    """
    root=Path(search_directory);result_path=Path(result_path);source=_source()
    with _inactive(root):
        plan_path=root/'plan.json';journal_path=root/'events.jsonl'
        plan_raw=_bytes(plan_path,expected_plan_sha256);result_raw=_bytes(result_path,expected_result_sha256)
        journal_raw=_bytes(journal_path,expected_journal_sha256)
        plan=_checked_plan(SearchPlan(canonical_json(_json(plan_raw))));saved=_json(result_raw)
        if plan['deadline_ns'] is not None:raise ValueError('This audit requires explicit native-quality deadline=None')
        if plan['identity']['source_sha256']!=source:raise ValueError('Native search declared source differs from the current frozen auditor')
        if not journal_raw or not journal_raw.endswith(b'\n'):raise ValueError('Complete newline-terminated native journal required')
        # The shared controller replays fields. First reject duplicate/nonfinite
        # JSON so alternate consumers cannot interpret the same bytes differently.
        for line in journal_raw.splitlines():_json(line)
        cursor=_Cursor(plan);events,records,total=_read_events(root,plan,cursor)
        reasons=_stops(events,plan);stop=reasons[-1]
        history=[] if cursor.sampler is None else cursor.sampler.history
        pending=None if cursor.sampler is None else cursor.sampler.pending_suggestion
        selected=select_completed_trial(records,deadline_ns=total)
        prefixes={str(n):select_completed_trial(records,deadline_ns=n) for n in plan['prefixes_ns']}
        expected=dict(schema='research/v2-outer-search-result',plan_hash=plan['plan_hash'],stop_reason=stop,
            records=records,attempts_paid=len(records),winner=selected['winner'],prefixes=prefixes,
            total_charged_ns=total,final_unassigned_coordinator_ns=total-sum(r['charged_ns'] for r in records),
            sampler_history=history,pending_suggestion=pending,journal_hash=events[-1]['event_hash'],
            protected_access_authorized=False,native_worker_qualified=False,hard_worker_deadline_implemented=False,
            whole_process_cost_authenticated=False,
            clock_scope='Supplied invocation preparation plus function entry through final clock read; final publication/return and caller startup need external supervisor receipt')
        if canonical_json(saved)!=canonical_json(expected):raise ValueError('Closed result differs from exact native search history/winners/costs')
        for row in records:_files(root,row['files'])
        if (_bytes(plan_path,expected_plan_sha256)!=plan_raw or _bytes(result_path,expected_result_sha256)!=result_raw
                or _bytes(journal_path,expected_journal_sha256)!=journal_raw or _source()!=source):
            raise ValueError('Native search source or declared evidence changed during replay')
        return dict(schema='research/v2-native-search-journal-audit',search_journal_authenticated=True,
            completed_search=stop in ('attempt_limit','support_exhausted'),stop_reason=stop,sessions=len(reasons),
            attempts_paid=len(records),sampler_feedback_count=len(history),
            failed_sampler_feedback_count=sum(row['value'] is None for row in history),
            journal_event_hash=events[-1]['event_hash'],journal_file_sha256=expected_journal_sha256,
            plan_file_sha256=expected_plan_sha256,result_file_sha256=expected_result_sha256,source_sha256=source,
            winner=selected['winner'],prefixes=prefixes,total_charged_ns=total,
            native_workload_qualified=False,model_generation_authenticated=False,V_arithmetic_authenticated=False,
            whole_process_cost_authenticated=False,protected_access_authorized=False,T_accessed=False,
            arrays_or_labels_decoded=False,
            scope='Exact paid native-quality cursor/sampler/file/clock replay under its current source. This authenticates the declared journal, not native fit qualification, model generation, V arithmetic, external wall time or protected access.')
