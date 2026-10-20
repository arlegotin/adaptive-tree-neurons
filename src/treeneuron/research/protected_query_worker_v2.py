"""Owned cold saved-query leaf; coordinator admission is a separate gate.

Only standard-library code precedes the cooperative nonce/grant handshake.
This leaf cannot authenticate native qualification or authorize test labels.
"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

from .deployment_worker_v2 import _source,_digest,_json,_ROOT,_THREADS
from .protected_journal_v2 import _path,_reference,_hash,_read_metadata_study,_write_new

_FALSE=dict(native_qualification_authenticated=False,protected_access_authorized=False,T_accessed=False)


def protected_query_worker_identity(request_path,expected_request_sha256):
    path=_path(request_path)
    if not _digest(expected_request_sha256):raise ValueError('Exact owned saved-query request digest required')
    return dict(purpose='owned_saved_protected_feature_query',request_path=str(path),request_sha256=expected_request_sha256,source_sha256=_source())


def _delegate(path,expected,*,output,require_active):
    """Read metadata only; actual parent protocol admission cannot be replaced."""
    path=_path(path);_,raw=_reference(dict(path=str(path),sha256=expected));value=_json(raw)
    fields={'schema','source_sha256','protocol_reference','study_directory','study_binding_reference','genesis_sha256','query_id',
        'attempt_index','journal_sha256','query_started_event_hash','core_request','core_request_sha256','worker_output'}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-owned-saved-query-delegate'
            or value['source_sha256']!=_source() or type(value['attempt_index']) is not int or value['attempt_index']<0
            or any(not _digest(value[k]) for k in ('genesis_sha256','query_id','journal_sha256','query_started_event_hash','core_request_sha256'))
            or type(value['core_request']) is not dict or _hash(value['core_request'])!=value['core_request_sha256']):
        raise ValueError('Strict source/request/attempt bound saved-query delegation required')
    study=_path(value['study_directory']);episode=study/'queries'/value['query_id']/f"episode-{value['attempt_index']:04}"
    if (path!=episode/'request.json' or _path(output)!=episode/'worker' or value['worker_output']!=str(_path(output))
            or type(value['study_binding_reference']) is not dict or value['study_binding_reference'].get('path')!=str(study/'binding.json')
            or type(value['protocol_reference']) is not dict or not str(value['protocol_reference'].get('path','')).endswith('.json')):
        raise ValueError('Delegation must preserve exact study/request/worker/binding/protocol paths')
    _reference(value['study_binding_reference']);_reference(value['protocol_reference'])
    state=_read_metadata_study(study);binding=state['binding'];core=value['core_request']
    from .protected_query_v2 import _metadata_request
    query,_=_metadata_request(core)
    if (binding['genesis_sha256']!=value['genesis_sha256'] or binding['source_sha256']!=value['source_sha256']
            or binding['protocol_sha256']!=value['protocol_reference']['sha256'] or core['protocol_sha256']!=binding['protocol_sha256']
            or core['source_sha256']!=value['source_sha256'] or query['query_id']!=value['query_id']
            or core['prediction_manifest_sha256']!=binding['prediction_manifest_sha256']):
        raise ValueError('Delegated query no longer matches its exact witnessed owner/protocol identity')
    prefix=hashlib.sha256();matched=False
    for line in (study/'events.jsonl').read_bytes().splitlines(keepends=True):
        prefix.update(line)
        if prefix.hexdigest()==value['journal_sha256']:
            event=_json(line)
            matched=(event['event_hash']==value['query_started_event_hash'] and event['kind']=='query_started'
                and event['details']['query_id']==value['query_id'] and event['details']['attempt_index']==value['attempt_index'])
    if not matched:raise ValueError('Exact witnessed query-start journal prefix is absent')
    if require_active and (state['phase']!='owners_ready' or state['active']!=(value['query_id'],value['attempt_index'])
            or state['last_event_hash']!=value['query_started_event_hash'] or state['journal_sha256']!=value['journal_sha256']):
        raise ValueError('Delegated query is no longer the sole exact active journal stage')
    _reference(dict(path=str(path),sha256=expected))
    return copy.deepcopy(value)


def _checked_delegate(path,expected,*,output):
    return _delegate(path,expected,output=output,require_active=True)


def _checked_delegate_prefix(path,expected,*,output):
    """Historical readonly prefix; never the live leaf's access check."""
    return _delegate(path,expected,output=output,require_active=False)


def _run_delegated_body(path,expected,*,output):
    """Private body for the trusted owned CLI, not a native/protected capability."""
    from .cooperative_process_v2 import _write
    control=_path(os.environ['ATN_SEARCH_CONTROL_ROOT']);started=time.perf_counter_ns()
    outcome=dict(schema='research/v2-owned-saved-query-body',status='failed_saved_query_body',source_sha256=_source(),
        request_reference=dict(path=str(path),sha256=expected),**_FALSE,
        clock_scope='Delegated body entry through validation/readback; own body publication/return and containing process excluded')
    try:
        delegate=_checked_delegate(path,expected,output=output)
        outcome.update(**{k:delegate[k] for k in ('query_id','attempt_index','protocol_reference','study_binding_reference','genesis_sha256',
            'journal_sha256','query_started_event_hash','core_request_sha256')})
        from .protected_query_v2 import _saved_query,inspect_saved_query_artifact
        result=_saved_query(delegate['core_request'],output=output)
        terminal=Path(output)/'terminal.json'
        terminal_reference=dict(path=str(terminal),sha256=hashlib.sha256(terminal.read_bytes()).hexdigest())
        readback=inspect_saved_query_artifact(delegate['core_request'],terminal_reference=terminal_reference)
        _checked_delegate(path,expected,output=output)
        target=Path(output)/'readback.json';_write_new(target,readback)
        outcome.update(status='closed_saved_query_body',terminal_reference=terminal_reference,
            prediction_reference=result['prediction_reference'],readback_reference=dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest()),
            scope='Actual owned saved-state body and numerical readback only; actual coordinator native/protocol/journal commit gates remain separate')
        return outcome
    except BaseException as error:
        outcome.update(error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        terminal=Path(output)/'terminal.json'
        if terminal.is_file() and not terminal.is_symlink():
            outcome['terminal_reference']=dict(path=str(terminal),sha256=hashlib.sha256(terminal.read_bytes()).hexdigest())
        raise
    finally:
        outcome['body_elapsed_ns']=time.perf_counter_ns()-started
        _write(control/'body-outcome.json',outcome)



def inspect_owned_saved_query(process_reference):
    """Join actual closed owned bytes to finite output; no native/T authority."""
    from .cooperative_process_v2 import inspect_cooperative_process
    from .protected_query_v2 import inspect_saved_query_artifact
    fields={'process_root','process_receipt_sha256','process_started_sha256'}
    if (type(process_reference) is not dict or set(process_reference)!=fields
            or any(not _digest(process_reference[k]) for k in fields-{'process_root'})):
        raise ValueError('Exact owned query process reference required')
    root=_path(process_reference['process_root'])
    if root.name!='process':raise ValueError('Exact local owned query process directory required')
    proof=inspect_cooperative_process(root=root,expected_receipt_sha256=process_reference['process_receipt_sha256'],
        expected_started_sha256=process_reference['process_started_sha256'])
    if proof['returncode']!=0 or proof['child_started'] is not True or proof['worker_protocol_closed'] is not True:
        raise ValueError('Only an actually closed granted query can supply complete numerical output')
    _,_,start=_read_process_json(root/'started.json',process_reference['process_started_sha256'])
    argv=start['argv'];episode=root.parent
    if (type(argv) is not list or len(argv)!=11 or argv[:6]!=[sys.executable,'-P','-B','-m','treeneuron.research.protected_query_worker_v2','--request']
            or argv[6]!=str(episode/'request.json') or argv[7]!='--expected-request-sha256' or not _digest(argv[8])
            or argv[9]!='--output' or argv[10]!=str(episode/'worker') or start['cwd']!=str(_ROOT)
            or start['identity']!=protected_query_worker_identity(Path(argv[6]),argv[8])
            or any(start['environment_overrides'].get(k)!='1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE'))
            or start['environment_overrides'].get('KMP_DUPLICATE_LIB_OK') is not None):
        raise ValueError('Owned source/command/interpreter/thread identity differs')
    delegate=_checked_delegate_prefix(Path(argv[6]),argv[8],output=Path(argv[10]))
    committed={row['path']:row for row in proof['files']}
    if not {'body-outcome.json','worker-closure.json'}<=set(committed):raise ValueError('Closed owned query body/closure bytes are missing')
    body_ref=dict(path=str(root/'body-outcome.json'),sha256=committed['body-outcome.json']['sha256'])
    _,body_raw=_reference(body_ref);body=_json(body_raw)
    _,_,closure=_read_process_json(root/'worker-closure.json',committed['worker-closure.json']['sha256'])
    expected_fields={'schema','status','source_sha256','request_reference','clock_scope','query_id','attempt_index','protocol_reference',
        'study_binding_reference','genesis_sha256','journal_sha256','query_started_event_hash','core_request_sha256','terminal_reference',
        'prediction_reference','readback_reference','scope','body_elapsed_ns',*_FALSE}
    if (type(body) is not dict or set(body)!=expected_fields or body['schema']!='research/v2-owned-saved-query-body'
            or body['status']!='closed_saved_query_body' or body['source_sha256']!=_source() or any(body[k] is not False for k in _FALSE)
            or type(body['attempt_index']) is not int or type(body['body_elapsed_ns']) is not int or body['body_elapsed_ns']<=0
            or body['request_reference']!=dict(path=argv[6],sha256=argv[8])
            or closure['body_entered'] is not True or closure['exit_kind']!='returned'
            or body['clock_scope']!='Delegated body entry through validation/readback; own body publication/return and containing process excluded'
            or body['scope']!='Actual owned saved-state body and numerical readback only; actual coordinator native/protocol/journal commit gates remain separate'
            or any(body[k]!=delegate[k] for k in ('query_id','attempt_index','protocol_reference','study_binding_reference','genesis_sha256',
                'journal_sha256','query_started_event_hash','core_request_sha256'))):
        raise ValueError('Exact successful query body does not match its owned delegated request')
    for name,filename in (('terminal_reference','terminal.json'),('readback_reference','readback.json'),('prediction_reference','prediction.npz')):
        if type(body[name]) is not dict or body[name].get('path')!=str(episode/'worker'/filename):
            raise ValueError('Owned query output cannot substitute another worker or artifact role')
    actual=inspect_saved_query_artifact(delegate['core_request'],terminal_reference=body['terminal_reference'])
    _,readback_raw=_reference(body['readback_reference'])
    if _json(readback_raw)!=actual or body['prediction_reference']!=actual['prediction_reference']:
        raise ValueError('Owned saved-query body/readback/prediction bindings differ')
    _checked_delegate_prefix(Path(argv[6]),argv[8],output=Path(argv[10]));_reference(body_ref)
    again=inspect_cooperative_process(root=root,expected_receipt_sha256=process_reference['process_receipt_sha256'],
        expected_started_sha256=process_reference['process_started_sha256'])
    if again!=proof or body['source_sha256']!=_source():raise ValueError('Owned query proof/source changed during readback')
    return dict(schema='research/v2-owned-saved-query-readback',query_id=delegate['query_id'],attempt_index=delegate['attempt_index'],
        source_sha256=delegate['source_sha256'],protocol_reference=delegate['protocol_reference'],
        study_binding_reference=delegate['study_binding_reference'],genesis_sha256=delegate['genesis_sha256'],
        journal_prefix_sha256=delegate['journal_sha256'],query_started_event_hash=delegate['query_started_event_hash'],
        core_request_sha256=delegate['core_request_sha256'],process_reference=copy.deepcopy(process_reference),body_reference=body_ref,
        terminal_reference=body['terminal_reference'],prediction_reference=body['prediction_reference'],readback_reference=body['readback_reference'],
        owned_saved_query_body_checked=True,finite_aligned_prediction_artifact_checked=True,observed_root_elapsed_ns=proof['elapsed_ns'],
        clock_scope=proof['clock_scope'],**_FALSE,
        scope='Actual closed owned query and numerical file join only; actual native/protocol/coordinator journal and whole-cost gates remain separate')


def _read_process_json(path,digest):
    target,raw=_reference(dict(path=str(path),sha256=digest))
    return target,raw,_json(raw)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--expected-request-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        identity=protected_query_worker_identity(args.request,args.expected_request_sha256)
        from .cooperative_process_v2 import cooperative_worker
        with cooperative_worker(identity=identity):
            _run_delegated_body(args.request,args.expected_request_sha256,output=args.output)
    except BaseException as error:
        print(f'owned-saved-query/v2: {type(error).__name__}: {error}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
