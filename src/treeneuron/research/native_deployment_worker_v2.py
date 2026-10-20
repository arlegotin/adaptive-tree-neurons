"""Owned measurements of original saved snapshots; no native/cost authority.

Only standard-library imports precede the cooperative ownership handshake.
Actual native-model and F-projection qualification belongs to the controller.
"""
import argparse
import hashlib
import importlib
import os
from pathlib import Path
import sys
import time
import traceback

from .deployment_worker_v2 import _source,_digest,_json,_ROOT,_THREADS
from .protected_journal_v2 import _path,_reference,_hash,_write_new

_FALSE=dict(native_models_authenticated=False,cost_claim_authorized=False,protected_access_authorized=False,T_accessed=False,
    original_F_role_authenticated=False,quality_precision_equivalence_authenticated=False,dedicated_timing_qualified=False)


def native_measurement_identity(path,expected):
    if not _digest(expected):raise ValueError('Exact owned measurement request digest required')
    return dict(purpose='owned_original_snapshot_measurement',request_path=str(_path(path)),request_sha256=expected,source_sha256=_source())


def _checked_measurement_request(path,expected):
    """Validate original byte bindings/feature-only arrays before method import."""
    from .protected_query_v2 import _feature_schema,_snapshot_inventory
    from .protected_label_evaluator_v2 import _npz
    import numpy as np
    _,raw=_reference(dict(path=str(_path(path)),sha256=expected));value=_json(raw)
    fields={'schema','source_sha256','family_id','logical_job_id','logical_fit_seed','feature_schema','model_binding',
        'query_reference','query_provenance_reference','batch_sizes','warmup_calls','samples_per_batch'}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-original-snapshot-measurement-request'
            or value['source_sha256']!=_source() or value['family_id'] not in tuple(f'P{i:02}' for i in range(14))
            or type(value['logical_job_id']) is not str or not value['logical_job_id']
            or type(value['logical_fit_seed']) is not int or not 0<=value['logical_fit_seed']<2**63
            or value['batch_sizes']!=[1,1024] or any(type(n) is not int for n in value['batch_sizes'])
            or type(value['warmup_calls']) is not int or value['warmup_calls']!=3
            or type(value['samples_per_batch']) is not int or value['samples_per_batch']!=32):
        raise ValueError('Strict current-source original-snapshot measurement request required')
    _feature_schema(value['feature_schema']);binding=value['model_binding']
    if (type(binding) is not dict or set(binding)!={'attempt_index','selected_metadata','snapshot_root','snapshot_inventory_sha256','generation_sha256'}
            or type(binding['attempt_index']) is not int or not 0<=binding['attempt_index']<32
            or any(not _digest(binding[k]) for k in ('snapshot_inventory_sha256','generation_sha256'))):
        raise ValueError('Complete original selected snapshot binding required')
    snapshot=_path(binding['snapshot_root']);tree=value['family_id'] in ('P08','P09','P10','P11')
    metadata=snapshot/('state.json' if tree else 'manifest.json')
    if binding['selected_metadata']['path']!=str(metadata):raise ValueError('Measurement must retain the original selected model path')
    _,raw=_reference(binding['selected_metadata']);saved=_json(raw);state=saved.get('payload') if tree else saved
    if (type(state) is not dict or type(state.get('config')) is not dict or state['config'].get('family')!=value['family_id']
            or state['config'].get('feature_schema')!=value['feature_schema']
            or _hash(_snapshot_inventory(snapshot))!=binding['snapshot_inventory_sha256']):
        raise ValueError('Original full snapshot bytes/family/feature schema changed')
    _,raw=_reference(value['query_provenance_reference']);provenance=_json(raw)
    pfields={'schema','source_sha256','source_F_reference','query_reference','row_policy','source_rows','workload_rows',
        'source_role','original_role_verified_by_native_controller'}
    if (type(provenance) is not dict or set(provenance)!=pfields or provenance['schema']!='research/v2-F-derived-deployment-query'
            or provenance['source_sha256']!=value['source_sha256'] or provenance['query_reference']!=value['query_reference']
            or provenance['row_policy']!='cycle_original_F_order_to_1024' or provenance['source_role']!='F'
            or type(provenance['source_rows']) is not int or not 2<=provenance['source_rows']<=20000
            or type(provenance['workload_rows']) is not int or provenance['workload_rows']!=1024
            or provenance['original_role_verified_by_native_controller'] is not False
            or type(provenance['source_F_reference']) is not dict or set(provenance['source_F_reference'])!={'path','sha256'}
            or not _digest(provenance['source_F_reference']['sha256']) or _path(provenance['source_F_reference']['path']).name!='F.npz'):
        raise ValueError('Explicit F-derived query metadata required; actual F-role derivation remains a controller gate')
    query,raw=_reference(value['query_reference'])
    if query.name!='F-deployment-features.npz':raise ValueError('Explicit F deployment-feature role required')
    arrays=_npz(raw,['X','row_id']);X=arrays['X'];rows=arrays['row_id']
    if (X.dtype!=np.float32 or X.shape!=(1024,len(value['feature_schema']['columns'])) or not np.isfinite(X).all()
            or rows.dtype!=np.int64 or not np.array_equal(rows,np.arange(1024,dtype=np.int64))):
        raise ValueError('Finite1024-row float32 workload and explicit unique workload IDs required')
    _reference(dict(path=str(_path(path)),sha256=expected))
    return value,snapshot,state,arrays,provenance


def _worker_output(path,output,value):
    from .primary_program_v2 import _check_program_location
    root=_path(output)
    if root!=_path(path).parent/'worker' or root.exists():raise ValueError('Fresh local measurement worker output required')
    return _check_program_location(root,working_roots=[value['model_binding']['snapshot_root'],str(Path(value['query_reference']['path']).parent)],
        proof_roots=[],input_files=[str(path),value['query_provenance_reference']['path']])


def _run_body(path,expected,*,output):
    from .cooperative_process_v2 import _write
    control=_path(os.environ['ATN_SEARCH_CONTROL_ROOT']);started=time.perf_counter_ns()
    outcome=dict(schema='research/v2-owned-native-measurement-body',status='failed_measurement_body',source_sha256=_source(),
        request_reference=dict(path=str(path),sha256=expected),**_FALSE)
    try:
        begin=time.perf_counter_ns()
        import numpy as np
        import pandas as pd
        common=time.perf_counter_ns()-begin
        value,snapshot,state,arrays,_=_checked_measurement_request(path,expected);root=_worker_output(path,output,value)
        if not sys.dont_write_bytecode or any(os.environ.get(k)!='1' for k in _THREADS):raise ValueError('Declared single-thread/bytecode-disabled measurement required')
        root.mkdir(parents=True,exist_ok=False)
        begin=time.perf_counter_ns();frame=pd.DataFrame(arrays['X'],columns=value['feature_schema']['columns']);frame_ns=time.perf_counter_ns()-begin
        begin=time.perf_counter_ns()
        from .method_worker import _ADAPTERS,_initialize_native_threads,loaded_method_modules
        threads=_initialize_native_threads(value['family_id']);module,name,_=_ADAPTERS[value['family_id']]
        cls=getattr(importlib.import_module('treeneuron.research.'+module),name);imports=time.perf_counter_ns()-begin
        argument='expected_state_sha256' if value['family_id'] in ('P08','P09','P10','P11') else 'expected_manifest_sha256'
        begin=time.perf_counter_ns();model=cls.load(snapshot,**{argument:value['model_binding']['selected_metadata']['sha256']});load_ns=time.perf_counter_ns()-begin
        batches=[]
        for size in value['batch_sizes']:
            query=frame.iloc[:size].copy(deep=True)
            def predict():
                begin=time.perf_counter_ns();prediction=model.predict(query,batch_size=size);elapsed=time.perf_counter_ns()-begin
                a=np.asarray(prediction)
                if a.dtype not in (np.dtype('float32'),np.dtype('float64')) or a.shape!=(size,) or not np.isfinite(a).all() or elapsed<=0:
                    raise FloatingPointError('Invalid finite original-unit native measurement output/clock')
                return elapsed,a
            first,golden=predict();dtype=golden.dtype;bits=golden.tobytes()
            def sample():
                elapsed,a=predict()
                if a.dtype!=dtype or a.tobytes()!=bits:raise FloatingPointError('Identical resident measurement queries changed prediction bits')
                return elapsed
            warmups=[sample() for _ in range(3)];resident=[sample() for _ in range(32)]
            batches.append(dict(batch_size=size,first_prediction_ns=first,warmup_prediction_ns=warmups,resident_prediction_ns=resident,
                predictions_equal_exact=True,prediction_dtype=str(dtype),prediction_values_sha256=hashlib.sha256(bits).hexdigest(),
                workload_row_ids_sha256=hashlib.sha256(arrays['row_id'][:size].tobytes()).hexdigest()))
        _checked_measurement_request(path,expected)
        result=dict(schema='research/v2-original-snapshot-measurement',status='completed_ineligible',source_sha256=value['source_sha256'],
            request_sha256=expected,family_id=value['family_id'],logical_job_id=value['logical_job_id'],logical_fit_seed=value['logical_fit_seed'],
            saved_native_seed=state.get('seed'),feature_schema=value['feature_schema'],model_binding=value['model_binding'],
            query_reference=value['query_reference'],query_provenance_reference=value['query_provenance_reference'],query_rows=1024,input_dtype='float32',
            batch_sizes=[1,1024],warmup_calls=3,samples_per_batch=32,batches=batches,common_numeric_import_ns=common,
            frame_construction_ns=frame_ns,method_import_ns=imports,model_load_ns=load_ns,torch_threads=threads,
            loaded_method_modules=loaded_method_modules(),prediction_includes_native_preprocessing=True,**_FALSE,
            scope='Observed original-snapshot measurements only; controller must authenticate original F derivation/native workload, paired timing/precision/quality/cost conditions separately')
        _write_new(root/'measurement.json',result)
        outcome.update(status='closed_measurement_body',measurement_reference=dict(path=str(root/'measurement.json'),sha256=hashlib.sha256((root/'measurement.json').read_bytes()).hexdigest()))
        return result
    except BaseException as error:
        outcome.update(error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc());raise
    finally:
        outcome['body_elapsed_ns']=time.perf_counter_ns()-started
        outcome['clock_scope']='Body entry through measurement publication; own body-outcome publication/return and containing process excluded'
        _write(control/'body-outcome.json',outcome)


def inspect_owned_native_measurement(reference):
    from .cooperative_process_v2 import inspect_cooperative_process
    if type(reference) is not dict or set(reference)!={'process_root','process_receipt_sha256','process_started_sha256'}:raise ValueError('Exact owned measurement reference required')
    root=_path(reference['process_root']);episode=root.parent
    proof=inspect_cooperative_process(root=root,expected_receipt_sha256=reference['process_receipt_sha256'],expected_started_sha256=reference['process_started_sha256'])
    if proof['returncode']!=0 or not proof['child_started'] or not proof['worker_protocol_closed']:raise ValueError('Closed granted owned measurement required')
    _,raw=_reference(dict(path=str(root/'started.json'),sha256=reference['process_started_sha256']));start=_json(raw);argv=start['argv']
    if (root.name!='process' or len(argv)!=11 or argv[:6]!=[sys.executable,'-P','-B','-m','treeneuron.research.native_deployment_worker_v2','--request']
            or argv[6]!=str(episode/'request.json') or argv[7]!='--expected-request-sha256' or argv[9:]!=['--output',str(episode/'worker')]
            or start['identity']!=native_measurement_identity(Path(argv[6]),argv[8]) or start['cwd']!=str(_ROOT)
            or any(start['environment_overrides'].get(k)!='1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE'))
            or start['environment_overrides'].get('KMP_DUPLICATE_LIB_OK') is not None):raise ValueError('Exact owned measurement source/command/thread identity required')
    committed={f['path']:f for f in proof['files']}
    if not {'body-outcome.json','worker-closure.json'}<=set(committed):raise ValueError('Closed measurement body/closure bytes missing')
    _,raw=_reference(dict(path=str(root/'worker-closure.json'),sha256=committed['worker-closure.json']['sha256']));closure=_json(raw)
    if closure['body_entered'] is not True or closure['exit_kind']!='returned':raise ValueError('Measurement body must actually enter and return normally')
    _,raw=_reference(dict(path=str(root/'body-outcome.json'),sha256=committed['body-outcome.json']['sha256']));body=_json(raw)
    body_fields={'schema','status','source_sha256','request_reference','measurement_reference','body_elapsed_ns','clock_scope',*_FALSE}
    if (type(body) is not dict or set(body)!=body_fields or body.get('schema')!='research/v2-owned-native-measurement-body' or body.get('status')!='closed_measurement_body'
            or body.get('source_sha256')!=_source() or body.get('request_reference')!=dict(path=argv[6],sha256=argv[8])
            or type(body.get('body_elapsed_ns')) is not int or not 0<body['body_elapsed_ns']<=proof['elapsed_ns']
            or body.get('clock_scope')!='Body entry through measurement publication; own body-outcome publication/return and containing process excluded'
            or any(body.get(k) is not False for k in _FALSE) or body.get('measurement_reference',{}).get('path')!=str(episode/'worker/measurement.json')):
        raise ValueError('Exact closed measurement body binding required')
    value,_,state,arrays,_=_checked_measurement_request(Path(argv[6]),argv[8]);_,raw=_reference(body['measurement_reference']);measurement=_json(raw)
    measurement_fields={'schema','status','source_sha256','request_sha256','family_id','logical_job_id','logical_fit_seed',
        'saved_native_seed','feature_schema','model_binding','query_reference','query_provenance_reference','query_rows','input_dtype',
        'batch_sizes','warmup_calls','samples_per_batch','batches','common_numeric_import_ns','frame_construction_ns','method_import_ns',
        'model_load_ns','torch_threads','loaded_method_modules','prediction_includes_native_preprocessing','scope',*_FALSE}
    if (type(measurement) is not dict or set(measurement)!=measurement_fields
            or measurement.get('schema')!='research/v2-original-snapshot-measurement' or measurement.get('status')!='completed_ineligible'
            or measurement.get('source_sha256')!=value['source_sha256'] or measurement.get('request_sha256')!=argv[8]
            or measurement.get('saved_native_seed')!=state.get('seed') or type(measurement.get('query_rows')) is not int or measurement['query_rows']!=1024
            or measurement.get('input_dtype')!='float32' or measurement.get('prediction_includes_native_preprocessing') is not True
            or type(measurement.get('loaded_method_modules')) is not list or any(type(m) is not str for m in measurement['loaded_method_modules'])
            or any(measurement.get(k)!=value[k] for k in ('family_id','logical_job_id','logical_fit_seed','feature_schema','model_binding',
                'query_reference','query_provenance_reference','batch_sizes','warmup_calls','samples_per_batch'))
            or any(measurement.get(k) is not False for k in _FALSE) or len(measurement.get('batches',[]))!=2):
        raise ValueError('Observed measurement differs from its fixed source/request/model/query')
    nested_ns=0
    for size,row in zip((1,1024),measurement['batches']):
        values=[row.get('first_prediction_ns'),*row.get('warmup_prediction_ns',[]),*row.get('resident_prediction_ns',[])]
        if (set(row)!={'batch_size','first_prediction_ns','warmup_prediction_ns','resident_prediction_ns','predictions_equal_exact',
                'prediction_dtype','prediction_values_sha256','workload_row_ids_sha256'}
                or row.get('batch_size')!=size or len(row.get('warmup_prediction_ns',[]))!=3 or len(row.get('resident_prediction_ns',[]))!=32
                or any(type(n) is not int or n<=0 for n in values) or row.get('predictions_equal_exact') is not True
                or row.get('prediction_dtype') not in ('float32','float64') or not _digest(row.get('prediction_values_sha256'))
                or row.get('workload_row_ids_sha256')!=hashlib.sha256(arrays['row_id'][:size].tobytes()).hexdigest()):
            raise ValueError('Complete finite exact-repeat measurement samples required')
        nested_ns+=sum(values)
    for key in ('common_numeric_import_ns','frame_construction_ns','method_import_ns','model_load_ns'):
        if type(measurement.get(key)) is not int or measurement[key]<=0:raise ValueError('Observed import/frame/load clocks required')
        nested_ns+=measurement[key]
    if nested_ns>body['body_elapsed_ns']:raise ValueError('Disjoint measurement clocks cannot exceed the containing body')
    if inspect_cooperative_process(root=root,expected_receipt_sha256=reference['process_receipt_sha256'],expected_started_sha256=reference['process_started_sha256'])!=proof:
        raise ValueError('Owned measurement changed during readback')
    _reference(body['measurement_reference'])
    return dict(schema='research/v2-owned-native-measurement-readback',process_reference=reference,
        request_reference=dict(path=argv[6],sha256=argv[8]),measurement_reference=body['measurement_reference'],model_binding=value['model_binding'],
        finite_repeated_measurement_checked=True,observed_root_elapsed_ns=proof['elapsed_ns'],**_FALSE,
        scope='Owned original-snapshot byte/sample/closure join only; actual native/F-role, dedicated/precision/quality/full-cost gates remain separate')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--expected-request-sha256',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
    try:
        from .cooperative_process_v2 import cooperative_worker
        with cooperative_worker(identity=native_measurement_identity(args.request,args.expected_request_sha256)):
            _run_body(args.request,args.expected_request_sha256,output=args.output)
    except BaseException as error:
        print(f'owned-native-measurement/v2: {type(error).__name__}: {error}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
