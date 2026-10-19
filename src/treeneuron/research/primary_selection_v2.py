"""Immutable primary model/selection commitments, without data-release authority."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time

from treeneuron.config import canonical_hash,canonical_json
from treeneuron.analysis.practical_v2 import select_completed_trial
from .deployment_worker_v2 import _json,_source
from .ledger import write_new_json
from .search_controller_v2 import SearchPlan,_checked_plan
from .workload import PracticalPlan,_DATASETS,_FAMILIES,practical_search_budget,_paired_search_seeds


_FLAGS=('secondary_release_authorized','protected_access_authorized','native_qualification_authenticated',
    'model_generation_authenticated','search_journal_authenticated','V_arithmetic_authenticated')


def _digest(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def _ref(value,references,*,decode=False):
    if type(value) is not dict or set(value)!={'path','sha256'} or type(value['path']) is not str or not _digest(value['sha256']):
        raise ValueError('Explicit absolute artifact byte reference required')
    path=Path(value['path'])
    if (not path.is_absolute() or path.resolve()!=path or path.is_symlink() or not path.is_file()):
        raise ValueError('Committed ordinary canonical artifact path required')
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=value['sha256'] or references.get(str(path),digest)!=digest:raise ValueError('Committed artifact bytes changed')
    references[str(path)]=digest
    if decode:
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Artifact changed while decoding metadata')
        return _json(raw)
    return str(path)


def _primary(plan,expected):
    if not isinstance(plan,PracticalPlan) or not _digest(expected):raise ValueError('Externally committed PracticalPlan required')
    value=plan.to_dict()
    if (type(value) is not dict or value.get('schema')!='research/v2-practical-plan' or value.get('plan_hash')!=expected
            or canonical_hash({k:v for k,v in value.items() if k!='plan_hash'})!=expected
            or value.get('source_sha256')!=_source() or value.get('protected_access_authorized') is not False):
        raise ValueError('Practical plan/source commitment changed')
    jobs=value.get('jobs')
    if type(jobs) is not list or len(jobs)!=840:raise ValueError('All840fixed practical jobs required')
    policy=value.get('budget_policy','legacy_bounded')
    if policy not in ('legacy_bounded','native_quality'):raise ValueError('Unknown practical budget policy')
    seeds=_paired_search_seeds(value.get('study_id'),value.get('seed_root'),native_quality=policy=='native_quality')
    seen=set();primary=[]
    for job in jobs:
        if type(job) is not dict:raise ValueError('Invalid fixed practical job')
        dataset=job.get('dataset');fold=job.get('outer_fold');repeat=job.get('search_repeat');family=job.get('family_id')
        if (type(dataset) is not str or dataset not in _DATASETS or type(fold) is not int or fold not in range(5)
                or type(repeat) is not int or repeat not in (0,1) or type(family) is not str or family not in _FAMILIES):
            raise ValueError('Invalid practical dataset/fold/family/repetition')
        key=(dataset,fold,repeat,family);job_id=f'{dataset}/outer-{fold}/repeat-{repeat}/{family}'
        is_primary=fold==(4 if dataset=='bike' else 0)
        if (key in seen or job.get('job_id')!=job_id or job.get('tranche')!=('primary' if is_primary else 'secondary')
                or any(job.get(k)!=value[k] for k in ('source_sha256','runtime_sha256','support_lock_sha256','lock_sha256'))
                or job.get('budget')!=practical_search_budget(family,native_quality=policy=='native_quality')
                or any(type(job.get(k)) is not type(v) or job[k]!=v for k,v in seeds[dataset,fold,repeat].items())
                or job.get('seed_selection_allowed') is not False):
            raise ValueError('Practical primary/secondary partition or binding changed')
        seen.add(key)
        if is_primary:primary.append(job)
    if len(primary)!=168:raise ValueError('Exactly168fixed primary searches required')
    return value,primary


def _proposal_lineage(plan,records,stop):
    """Check proposal order/identity; TPE suggestion law still needs its journal."""
    candidates=[('anchor',f'anchor-{i}',config) for i,config in enumerate(plan['anchors'])]
    if plan['mode']=='finite':candidates += [('finite',f'finite-{i}',config) for i,config in enumerate(plan['candidates'])]
    index=0;previous=None
    for paid,row in enumerate(records):
        retry=row.get('retry_index')
        if (type(retry) is not int or retry not in (0,1)
                or row.get('attempt_id')!=f"{plan['identity']['job_id']}/attempt-{paid:03}"
                or type(row.get('config')) is not dict or not row['config']):
            raise ValueError('Changed paid proposal identity/configuration')
        if retry:
            if previous is None or previous['status']!='interrupted_recoverable' or previous['retry_index']!=0:
                raise ValueError('Unqualified or repeated candidate recovery')
            expected=tuple(previous[k] for k in ('candidate_origin','candidate_id','config'))
        else:
            if previous is not None and previous['status']=='interrupted_recoverable':
                raise ValueError('Interrupted candidate was replaced instead of recovered')
            if index<len(candidates):expected=candidates[index]
            elif plan['mode']=='tpe':expected=('tpe',f"tpe-{index-len(plan['anchors'])}",row['config'])
            else:raise ValueError('Search contains a proposal outside its finite support')
            index+=1
        if tuple(row.get(k) for k in ('candidate_origin','candidate_id','config'))!=expected:
            raise ValueError('Anchor/candidate proposal order or configuration changed')
        previous=row
    if (previous['status']=='interrupted_recoverable'
            or stop=='support_exhausted' and index!=len(candidates)):
        raise ValueError('Incomplete candidate history cannot claim terminal support exhaustion')


def _assemble(plan,expected,procedure,selections,created):
    value,jobs=_primary(plan,expected);references={}
    if type(created) is not int or created<=0:raise ValueError('Immutable publication time required')
    if type(selections) is not dict or set(selections)!={j['job_id'] for j in jobs}:
        raise ValueError('Exactly all168primary selections required; no secondary or seed selection')
    _ref(procedure,references,decode=True)
    models=[]
    for job in jobs:
        entry=selections[job['job_id']]
        if type(entry) is not dict or set(entry)!={'search_plan','search_result','selected_model'}:
            raise ValueError('Exact search-plan/result/selected-model references required')
        saved_plan=_ref(entry['search_plan'],references,decode=True)
        search_plan=_checked_plan(SearchPlan(canonical_json(saved_plan)));identity=search_plan['identity']
        budget=job['budget'];deadline=budget['deadline_ns'];prefixes=budget['descriptive_prefixes_ns']
        if (identity['job_id']!=job['job_id'] or identity['family_id']!=job['family_id']
                or identity['search_repeat']!=job['search_repeat'] or identity['source_sha256']!=value['source_sha256']
                or identity['runtime_sha256']!=value['runtime_sha256'] or identity['split_sha256']!=job['split_hash']
                or search_plan['hpo_seed32']!=job['hpo_seed32'] or search_plan['max_attempts']!=budget['max_attempts']
                or search_plan['deadline_ns']!=deadline or search_plan['prefixes_ns']!=prefixes):
            raise ValueError('Search source/family/split/repeat/HPO/budget differs from its primary job')
        if value.get('budget_policy')=='native_quality' and search_plan['fit_seed']!=job['fit_seed']:
            raise ValueError('Search model seed differs from the fixed paired practical job')
        result=_ref(entry['search_result'],references,decode=True)
        stops=('attempt_limit','support_exhausted') if deadline is None else ('attempt_limit','deadline','support_exhausted')
        if (type(result) is not dict or result.get('schema')!='research/v2-outer-search-result'
                or result.get('plan_hash')!=search_plan['plan_hash'] or result.get('stop_reason') not in stops
                or result.get('protected_access_authorized') is not False):
            raise ValueError('Complete unchanged terminal search required, not paused/error execution')
        records=result.get('records')
        if (type(records) is not list or not records or type(result.get('attempts_paid')) is not int
                or result['attempts_paid']!=len(records)
                or any(type(r) is not dict or r.get('family_id')!=job['family_id']
                    or type(r.get('search_repeat')) is not int or r['search_repeat']!=job['search_repeat']
                    or type(r.get('fit_seed')) is not int or r['fit_seed']!=search_plan['fit_seed'] for r in records)):
            raise ValueError('Paid attempt inventory/repetition differs')
        _proposal_lineage(search_plan,records,result['stop_reason'])
        total=result.get('total_charged_ns')
        if type(total) is not int or total<0:raise ValueError('Actual primary search endpoint required')
        selection=select_completed_trial(records,deadline_ns=total if deadline is None else deadline)
        winner=selection['winner']
        if (winner is None or winner!=result.get('winner') or type(total) is not int
                or total<records[-1]['completion_ns'] or total<selection['total_charged_ns']
                or result.get('final_unassigned_coordinator_ns')!=total-selection['total_charged_ns']
                or result['stop_reason']=='attempt_limit' and len(records)!=32
                or result['stop_reason']=='deadline' and total<deadline
                or result['stop_reason']=='support_exhausted' and search_plan['mode']!='finite'):
            raise ValueError('Primary winner/deadline/tie/cost result contradicts retained trials')
        if result.get('prefixes')!={str(n):select_completed_trial(records,deadline_ns=n) for n in prefixes}:
            raise ValueError('Primary descriptive prefix results contradict retained completion times')
        search_root=Path(entry['search_plan']['path']).parent
        winner_files=set()
        for row in records:
            files=row.get('files')
            if type(files) is not list or not files:raise ValueError('Every paid attempt needs retained file commitments')
            local=set()
            for record in files:
                if type(record) is not dict or set(record)!={'path','sha256'} or type(record['path']) is not str:
                    raise ValueError('Invalid paid trial file commitment')
                relative=Path(record['path']);path=search_root/relative
                if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(search_root):
                    raise ValueError('Paid trial evidence escapes its search directory')
                file_ref=dict(path=str(path),sha256=record['sha256']);_ref(file_ref,references)
                pair=(str(path),record['sha256'])
                if pair in local:raise ValueError('Duplicate paid trial file commitment')
                local.add(pair)
            if row['attempt_index']==winner['attempt_index']:winner_files=local
        model=_ref(entry['selected_model'],references)
        if (model,entry['selected_model']['sha256']) not in winner_files:
            raise ValueError('Selected model is not one of its V-selected winner artifacts')
        model_record=dict(job_id=job['job_id'],dataset=job['dataset'],outer_fold=job['outer_fold'],
            family_id=job['family_id'],search_repeat=job['search_repeat'],split_hash=job['split_hash'],
            manifest_hash=job['manifest_hash'],winner_attempt_index=winner['attempt_index'],V_score=winner['V_score'],
            attempts_paid=len(records),paid_search_ns=total,model=entry['selected_model'])
        if value.get('budget_policy')=='native_quality':
            model_record.update(fit_seed=job['fit_seed'],fit_seed64=job['fit_seed64'],
                fit_seed32_collision_counter=job['fit_seed32_collision_counter'])
        models.append(model_record)
    for path,digest in list(references.items()):_ref(dict(path=path,sha256=digest),{})
    if _source()!=value['source_sha256']:raise ValueError('Scientific source changed during primary sealing')
    return dict(schema='research/v2-primary-selection-payload',created_utc_ns=created,plan_hash=expected,
        source_sha256=value['source_sha256'],support_lock_sha256=value['support_lock_sha256'],
        global_procedure=procedure,selections=selections,models=models,
        artifact_references=[dict(path=p,sha256=d) for p,d in sorted(references.items())],
        **{flag:False for flag in _FLAGS},
        scope='Primary selection and committed artifact integrity only; final protocol/readiness and release require separate authentication')


@dataclass(frozen=True)
class PrimarySelectionInventory:
    path:Path
    file_sha256:str
    _payload_json:str

    @property
    def payload(self):return json.loads(self._payload_json)


def seal_primary_selection_inventory(*,plan,expected_plan_hash,global_procedure,selections,output):
    output=Path(output)
    if output.exists() or output.is_symlink():raise ValueError('Primary selection output already exists')
    payload=_assemble(plan,expected_plan_hash,global_procedure,selections,time.time_ns())
    value=dict(schema='research/v2-primary-selection-inventory',payload=payload,inventory_hash=canonical_hash(payload))
    write_new_json(output,value)
    return PrimarySelectionInventory(output,hashlib.sha256(output.read_bytes()).hexdigest(),canonical_json(payload))


def load_primary_selection_inventory(path,*,expected_file_sha256,plan,expected_plan_hash):
    path=Path(path);value=_ref(dict(path=str(path),sha256=expected_file_sha256),{},decode=True)
    if (type(value) is not dict or set(value)!={'schema','payload','inventory_hash'}
            or value['schema']!='research/v2-primary-selection-inventory' or type(value['payload']) is not dict
            or canonical_hash(value['payload'])!=value['inventory_hash']):raise ValueError('Changed primary inventory envelope')
    saved=value['payload']
    try:rebuilt=_assemble(plan,expected_plan_hash,saved['global_procedure'],saved['selections'],saved['created_utc_ns'])
    except (KeyError,TypeError) as error:raise ValueError('Malformed primary inventory payload') from error
    if canonical_json(saved)!=canonical_json(rebuilt):raise ValueError('Primary inventory semantics or authority changed')
    return PrimarySelectionInventory(path,expected_file_sha256,canonical_json(rebuilt))
