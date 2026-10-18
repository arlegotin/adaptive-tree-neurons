"""Final development-screen disposition after authenticated M1/M2 collection."""
from treeneuron.config import canonical_hash
from .control_analysis import positive_sensitivity
from .opportunity_analysis import rebuild_opportunity_summary


def gate_a_decision(control_audit, initial_summary, final_summary, terminal_statuses, *,
                    expected_job_ids, extension_status, opportunity_contract):
    """Callers must authenticate producer artifacts and the complete frozen DAG.

    This pure decision records consumed inputs. It neither authenticates files
    itself nor converts a development screen into predictive confirmation.
    """
    expected = list(expected_job_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Gate A requires the complete unique frozen job inventory')
    bindings = dict(control_audit_sha256=canonical_hash(control_audit),
        initial_summary_sha256=canonical_hash(initial_summary),final_summary_sha256=canonical_hash(final_summary),
        job_statuses_sha256=canonical_hash(terminal_statuses),expected_jobs_sha256=canonical_hash({'jobs':expected}),
        extension_status_sha256=canonical_hash({'status':extension_status}),
        opportunity_contract_sha256=canonical_hash(opportunity_contract))

    def result(decision, reason, *, advance=False, scope=None, gated=False):
        return dict(schema='research/v2-gate-A',decision=decision,reason=reason,
            screen_passed=advance,advance_joint_selector=advance,scope=scope,
            gated_growth_opportunity=gated,efficacy_claim=False,input_bindings=bindings,
            expected_job_count=len(expected))

    if set(terminal_statuses) != set(expected) or any(v not in ('completed_eligible','completed_ineligible')
                                                  for v in terminal_statuses.values()):
        return result('blocked','Every initial/conditional diagnostic requires a valid terminal status')
    if terminal_statuses.get('extension_decision:primary') != extension_status:
        return result('blocked','Extension status differs from its frozen queue node')
    if (control_audit.get('status') != 'completed'
            or control_audit.get('evidence_level') != 'retained_raw_prediction_arithmetic'):
        return result('blocked','Complete retained M1 prediction arithmetic is required')
    control = control_audit.get('positive_sensitivity',{})
    rows = control_audit.get('rows')
    if (not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows)
            or canonical_hash(control) != canonical_hash(positive_sensitivity(rows))):
        return result('blocked','Control sensitivity differs from its retained per-repeat risks')
    count = control.get('qualifying_instances')
    if (control.get('status') != 'completed_eligible' or control.get('coordinate') != dict(gamma=2.,sigma=.1)
            or control.get('required_instances') != 4 or control.get('minimum_qualifying_instances') != 2
            or control.get('minimum_relative_reduction') != .02
            or type(count) is not int or not 0 <= count <= 4
            or type(control.get('sensitive')) is not bool or control['sensitive'] != (count >= 2)):
        return result('blocked','The fixed four-instance/R2 true-oracle control is incomplete or inconsistent')
    for summary in (initial_summary,final_summary):
        if summary.get('status') != 'completed_eligible':
            return result('blocked','Primary opportunity evidence is incomplete')
        try:
            rebuilt = rebuild_opportunity_summary(summary,opportunity_contract)
        except (ValueError,KeyError,TypeError):
            return result('blocked','Retained primary opportunity risks or contract are invalid')
        if canonical_hash(summary) != canonical_hash(rebuilt):
            return result('blocked','Opportunity summary differs from its retained per-teacher arithmetic')
        n = len(summary.get('teacher_ids',[]))
        counts = [summary.get(k) for k in ('maximum_qualifying_teachers',
            'gated_growth_maximum_qualifying_teachers','ordinary_capacity_maximum_qualifying_teachers')]
        if any(type(v) is not int or not 0 <= v <= n for v in counts) or counts[0] != max(counts[1:]):
            return result('blocked','Fixed-arm qualification counts are inconsistent')
    if (initial_summary.get('teacher_ids') != list(range(4))
            or type(initial_summary.get('extension_required')) is not bool):
        return result('blocked','Initial opportunity screen must retain the declared first four teachers')
    extended = initial_summary['extension_required']
    if (extension_status != ('completed_eligible' if extended else 'completed_ineligible')
            or final_summary.get('teacher_ids') != list(range(8 if extended else 4))
            or final_summary.get('extension_required') is not False
            or final_summary.get('extension_exhausted') is not extended
            or (not extended and canonical_hash(initial_summary) != canonical_hash(final_summary))):
        return result('blocked','Required extension is incomplete, unauthorized or exhausted inconsistently')
    if not control['sensitive']:
        return result('insensitive_control','Diagnose control sensitivity before interpreting selector efficacy')
    if final_summary['maximum_qualifying_teachers'] < 2:
        return result('retired_regime','No fixed growth/capacity arm exposes the declared opportunity after bounded development')
    gated = final_summary['gated_growth_maximum_qualifying_teachers'] >= 2
    return result('screen_passed','Bounded joint-selector research may proceed with competent ordinary controls',
        advance=True,scope='gated_growth_opportunity' if gated else 'ordinary_capacity_only',gated=gated)
