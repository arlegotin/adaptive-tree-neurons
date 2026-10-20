"""Join fixed native anchor software and execution proofs, without promotion."""
import copy
from pathlib import Path
import time

from treeneuron.config import canonical_hash
from .deployment_worker_v2 import _ROOT,_source,_json
from .ledger import write_new_json
from .native_candidate_v2 import inspect_primary_native_candidate
from .native_verification_v2 import verification_profiles,load_native_implementation_verification
from .search_supports_v2 import family_search_spec
from .workqueue import _inventory


def qualify_native_anchor(*,implementation_reference,generation_arguments,immutable_generation_root,
        output,interpreter,extra_pythonpath=(),rss_limit_bytes=4*1024**3):
    """Authenticate a native anchor's source/software/state/replay/primary V.

    This does not authenticate raw acquisition or whole support/workload bounds.
    A complete qualified registry and production protocol remain prerequisites.
    """
    started=time.perf_counter_ns();args=copy.deepcopy(generation_arguments);output=Path(output)
    fields={'request_path','expected_request_sha256','native_reference','prepared_root','expected_manifest_sha256',
        'outer_fold','expected_fit_seed','expected_configuration'}
    if type(args) is not dict or set(args)!=fields:raise ValueError('Exact native generation arguments required')
    container=Path(immutable_generation_root).resolve();prepared=Path(args['prepared_root']).resolve()
    if (not container.is_dir() or output.exists() or output.is_symlink() or output.resolve().is_relative_to(container)
            or output.resolve().is_relative_to(prepared)
            or any(output.resolve().is_relative_to(_ROOT/p) for p in ('src','tests','locks','configs','protocol'))):
        raise ValueError('Fresh external qualification output must preserve immutable/executable inputs')
    cfg=args['expected_configuration']
    if type(cfg) is not dict or 'family' not in cfg or 'feature_schema' not in cfg:
        raise ValueError('Exact mandatory native anchor configuration required')
    spec=family_search_spec(cfg['family'],cfg['feature_schema'])
    matches=[a['anchor_id'] for a in spec['anchors'] if a['config']==cfg]
    if len(matches)!=1 or type(args['expected_fit_seed']) is not int or args['expected_fit_seed']!=17:
        raise ValueError('Qualification requires one declared native anchor and fixed seed17')
    profile=next(name for name,row in verification_profiles().items() if cfg['family'] in row['families'])
    software=load_native_implementation_verification(implementation_reference,expected_profile=profile);source=_source()
    interpreter=Path(interpreter)
    if (not interpreter.is_absolute() or not interpreter.is_file() or type(rss_limit_bytes) is not int or rss_limit_bytes<=0):
        raise ValueError('Qualified interpreter and owned replay RSS guard required')
    output.mkdir(parents=True,exist_ok=False);output=output.resolve()
    record=dict(schema='research/v2-native-anchor-execution',status='input_or_dependency_failure',
        native_anchor_execution_authenticated=False,source_sha256=source,family_id=cfg['family'],anchor_id=matches[0],
        dataset=None,software_verified=True,implementation_reference=copy.deepcopy(implementation_reference),
        configuration_sha256=canonical_hash(cfg),support_sha256=spec['support_hash'],fit_seed=17,
        prediction_reload_exact=False,inspection_span_ns=0,inspection_reference=None,
        V_score_used_for_qualification=False,raw_acquisition_authenticated=False,native_workload_qualified=False,
        quality_eligible=False,HPO_journal_authenticated=False,protected_access_authorized=False,T_accessed=False,
        scope='Fixed native anchor software plus managed current-source generation, closed prediction replay and primary V. No raw acquisition, complete support/workload registry or production/protected authority.')
    try:
        inspection_started=time.perf_counter_ns()
        inspection=inspect_primary_native_candidate(generation_arguments=args,immutable_generation_root=container,
            output=output/'inspection',interpreter=interpreter,extra_pythonpath=extra_pythonpath,rss_limit_bytes=rss_limit_bytes)
        record['inspection_span_ns']=time.perf_counter_ns()-inspection_started
        saved=_json((output/'inspection/result.json').read_bytes())
        if saved!=inspection:raise ValueError('Returned and published native inspection differ')
        if (load_native_implementation_verification(implementation_reference,expected_profile=profile)!=software
                or _source()!=source):raise ValueError('Native anchor software/source changed during inspection')
        record['inspection_reference']=dict(root=str(output/'inspection'),inventory_sha256=canonical_hash(_inventory(output/'inspection')))
        record['dataset']=inspection['generation']['dataset']
        record['status']=inspection['status']
        if inspection['status']=='completed_candidate_inspection':
            record.update(status='authenticated_native_anchor_execution',native_anchor_execution_authenticated=True,
                prediction_reload_exact=inspection['replay']['prediction_reload_exact'])
        return record
    except BaseException as error:
        record.update(error_type=type(error).__name__,reason=str(error))
        raise
    finally:
        record['elapsed_ns']=time.perf_counter_ns()-started
        record['clock_scope']='Function entry through result preparation; inspection is nested; caller pays final publication/return'
        write_new_json(output/'result.json',record)
