"""Provisional pre-calibration identity; this is not a confirmation capability.

Source and byte bindings here do not qualify a procedure, authorize calibration,
or open protected labels. Those require the remaining v2 readiness/freeze gates.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time

from treeneuron.config import canonical_hash,canonical_json
from treeneuron.protocol import execution_snapshot,_execution,_reference
from treeneuron.analysis import design_v2,primary_v2
from . import seeds
from .ledger import write_new_json


_CLAIMS=('placement','pruning_selection','residual_capacity')
_REFERENCES={'decision','runtime','development_registry','search_support'}
_SOURCE_FIELDS={'pinned_commit','checkout_commit','files','execution_sha256'}
_PAYLOAD_FIELDS={'phase','study_id','created_utc_ns','repository_locator','source','code_bindings',
    'method','method_sha256','references','references_sha256','analysis_policy','analysis_sha256',
    'calibration_execution_authorized','protected_access_authorized','configuration_semantics_qualified'}


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def _copy(value):return json.loads(canonical_json(dict(value=value)))['value']


def _method(value):
    fields={'procedure_id','claim','configuration','primary_roles','ordinary_control_roles'}
    if (type(value) is not dict or set(value)!=fields or type(value['procedure_id']) is not str
            or not 0<len(value['procedure_id'])<=128 or value['claim'] not in _CLAIMS
            or type(value['configuration']) is not dict or not value['configuration']
            or type(value['primary_roles']) is not dict or type(value['ordinary_control_roles']) is not dict):
        raise ValueError('Strict provisional procedure/claim/configuration/roles required')
    roles=value['primary_roles'];controls=value['ordinary_control_roles']
    if value['claim']=='residual_capacity':
        if set(roles)!={'gated_residual','ordinary_lower','ordinary_upper'} or controls:
            raise ValueError('Residual capacity has its own mandatory bracket roles, not placement roles')
    elif set(roles)!={'selected','random'} or not set(controls)<={'keep','warm','fresh'}:
        raise ValueError('Placement/pruning requires matched selected/random and declared ordinary roles')
    names=[*roles.values(),*controls.values()]
    if (any(type(name) is not str or not 0<len(name)<=128 for name in names)
            or len(names)!=len(set(names))):raise ValueError('Every provisional fitting role must be distinct')
    return _copy(value)


def _analysis(method,study_id):
    residual=method['claim']=='residual_capacity'
    matrix='C2-calibration' if residual else 'C0'
    bootstrap_seeds={stage:seeds.seed64(2026101804,matrix,study_id,'variance-bootstrap',stage)
        for stage in ('initial16','extended32')}
    simulation_seed=seeds.seed64(2026101804,matrix,study_id,'power-validation')
    if len(set([*bootstrap_seeds.values(),simulation_seed]))!=3:
        raise ValueError('Provisional calibration seed namespaces collided')
    hypotheses=(['capacity_vs_lower','capacity_vs_upper'] if residual else
        ['growth_vs_'+name for name in ('keep','warm','fresh') if name in method['ordinary_control_roles']])
    return dict(schema='research/v2-provisional-analysis-policy',repetitions=2,
        independent_unit='teacher',risk='original_unit_MSE_divided_by_F_variance_ddof0',
        repeat_aggregation='average_risks_within_teacher_before_log_ratio',
        primary_estimand='V_selected_ordinary_residual_over_gated_residual_teacher_log_risk' if residual else
            'matched_random_over_selected_teacher_log_risk',
        useful_relative_risk_reduction=.02,design_alternative_relative_risk_reduction=.05,
        useful_margin=design_v2.USEFUL_MARGIN,design_alternative=design_v2.DESIGN_ALTERNATIVE,
        primary_CI='two_sided_95_percent_teacher_Student_t',one_sided_alpha=.025,
        useful_decision='primary_lower_bound_strictly_above_useful_margin',
        teacher_grid=list(design_v2.TEACHER_GRID),calibration_teacher_counts=[16,32],
        extension='once_for_declared_variance_or_runtime_precision_failure_only',
        bootstrap_SD_resamples=10000,SD_upper='max_normal_theory_90_upper_and_bootstrap_90_percentile',
        variance_seed_root=2026101804,calibration_matrix=matrix,
        bootstrap_seeds=bootstrap_seeds,simulation_seed=simulation_seed,
        power_simulation_cohorts=100000,power_laws=['centered_empirical','normal','t5'],
        target_power=.9,power_binomial_lower_minimum=.9,null_binomial_upper_maximum=.03,
        secondary_hypotheses=hypotheses,secondary_multiplicity='Holm_fixed_declared_family',
        compatible_adjusted_lower_bounds=dict(construction='teacher_t_guilbaud2008_choice2_strict',alpha=.025,
            marginal_lower_endpoint='included',rejected_margin_endpoint='excluded',
            partial_rejection='no_numeric_gap_for_rejected_effect',all_rejected='common_minimum_marginal_lower'),
        missingness='missing_planned_prediction_blocks_complete_primary_claim',
        protected_sample_size_looks='none',refit='retain_F_fit_G_selected_model',
        effect_sign_or_significance_used_for_sample_size=False)


def _code_bindings(root,snapshot):
    expected={
        'method_lock':('src/treeneuron/research/protocol_v2.py',Path(__file__)),
        'variance_decision':('src/treeneuron/analysis/design_v2.py',Path(design_v2.__file__)),
        'secondary_bounds':('src/treeneuron/analysis/primary_v2.py',Path(primary_v2.__file__)),
        'seed_derivation':('src/treeneuron/research/seeds.py',Path(seeds.__file__)),
    }
    result={}
    for name,(relative,running) in expected.items():
        digest=_sha(running)
        if snapshot['files'].get(relative)!=digest or _sha(root/relative)!=digest:
            raise ValueError('Committed provisional/variance decision source differs from the running code')
        result[name]=dict(path=relative,sha256=digest)
    return result


def _references(root,values):
    if type(values) is not dict or set(values)!=_REFERENCES:
        raise ValueError('Explicit decision/runtime/development/support byte references required')
    for value in values.values():_reference(root,value)
    return _copy(values)


@dataclass(frozen=True)
class ProvisionalMethodLock:
    path:Path
    file_sha256:str
    lock_hash:str
    _payload_json:str

    @property
    def payload(self):return json.loads(self._payload_json)


def lock_provisional_method(*,repository_root,pinned_commit,study_id,method,references,output):
    """Publish immutable chosen-method identity; no data access is authorized."""
    root=Path(repository_root).resolve();output=Path(output)
    if type(study_id) is not str or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',study_id) is None:
        raise ValueError('Stable provisional study ID required')
    if output.exists() or output.is_symlink():raise ValueError('Provisional method output already exists')
    if output.resolve().is_relative_to(root) and _execution(output.resolve().relative_to(root).as_posix()):
        raise ValueError('Method lock output must not add/change executable inputs')
    chosen=_method(method);refs=_references(root,references)
    snapshot=execution_snapshot(root,pinned_commit);code=_code_bindings(root,snapshot);analysis=_analysis(chosen,study_id)
    payload=dict(phase='provisional_method',study_id=study_id,created_utc_ns=time.time_ns(),
        repository_locator=str(root),source=snapshot,code_bindings=code,method=chosen,
        method_sha256=canonical_hash(chosen),references=refs,references_sha256=canonical_hash(refs),
        analysis_policy=analysis,analysis_sha256=canonical_hash(analysis),
        calibration_execution_authorized=False,protected_access_authorized=False,configuration_semantics_qualified=False)
    value=dict(schema='research/v2-provisional-method-lock',payload=payload,lock_hash=canonical_hash(payload))
    write_new_json(output,value)
    return ProvisionalMethodLock(output,_sha(output),value['lock_hash'],canonical_json(payload))


def load_provisional_method(path,*,expected_file_sha256,repository_root=None):
    """Authenticate the external commitment, fixed policy and current source."""
    path=Path(path)
    if (not _digest(expected_file_sha256) or path.is_symlink() or not path.is_file()
            or _sha(path)!=expected_file_sha256):raise ValueError('Trusted provisional method file commitment differs')
    from treeneuron.protocol import _read
    value=_read(path)
    if (type(value) is not dict or set(value)!={'schema','payload','lock_hash'}
            or value['schema']!='research/v2-provisional-method-lock'
            or type(value['payload']) is not dict or set(value['payload'])!=_PAYLOAD_FIELDS
            or canonical_hash(value['payload'])!=value['lock_hash']):
        raise ValueError('Provisional method envelope or payload differs')
    p=value['payload'];source=p['source']
    if (p['phase']!='provisional_method' or type(p['study_id']) is not str
            or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',p['study_id']) is None
            or type(p['created_utc_ns']) is not int or p['created_utc_ns']<=0
            or type(p['repository_locator']) is not str or not Path(p['repository_locator']).is_absolute()
            or any(p[k] is not False for k in ('calibration_execution_authorized','protected_access_authorized',
                'configuration_semantics_qualified'))
            or type(source) is not dict or set(source)!=_SOURCE_FIELDS):
        raise ValueError('Provisional phase/source/authority declaration differs')
    chosen=_method(p['method']);analysis=_analysis(chosen,p['study_id'])
    if (canonical_hash(chosen)!=p['method_sha256'] or analysis!=p['analysis_policy']
            or canonical_hash(analysis)!=p['analysis_sha256']):
        raise ValueError('Provisional method or fixed analysis policy changed')
    root=Path(repository_root if repository_root is not None else p['repository_locator']).resolve()
    current=execution_snapshot(root,source['pinned_commit'])
    if (any(source[k]!=current[k] for k in ('pinned_commit','files','execution_sha256'))
            or type(source['checkout_commit']) is not str or re.fullmatch('[0-9a-f]{40}',source['checkout_commit']) is None
            or _code_bindings(root,current)!=p['code_bindings']):
        raise ValueError('Provisional source changed; documentation-only commits may differ')
    if source['checkout_commit']!=current['checkout_commit']:
        original_checkout=execution_snapshot(root,source['checkout_commit'])
        if any(original_checkout[k]!=source[k] for k in ('files','execution_sha256')):
            raise ValueError('Recorded checkout does not contain the locked execution bytes')
    refs=_references(root,p['references'])
    if canonical_hash(refs)!=p['references_sha256']:raise ValueError('Provisional reference inventory changed')
    return ProvisionalMethodLock(path,expected_file_sha256,value['lock_hash'],canonical_json(p))
