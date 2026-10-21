"""Canonical practical reporting after actual frozen raw/cost qualification.

Numeric compilers and renderers grant no authority. Protected production input
loaders are separate and must run before output or attestation promotion.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import os

from .deployment_worker_v2 import _source,_digest,_json
from .protected_journal_v2 import _hash,_reference,_canonical,_path,_write_new
from .frozen_practical_analysis_v2 import _compile_comparison_inputs
from treeneuron.analysis.practical_v2 import OBSERVATIONAL,FAMILIES
from treeneuron.analysis.practical_comparisons_v2 import MARGIN
from treeneuron.analysis.report_v2 import render_report,practical_family_report_record

_FALSE=dict(claim_authorized=False,population_superiority_confirmed=False,whole_training_procedure_cost_authenticated=False,
    native_qualification_authenticated=False,protected_access_authorized=False,T_accessed=None)
_LIMITS=[
 'Five named observational holdouts; intervals condition on fixed fitted models and group/day sampling, not retraining or a population of datasets.',
 'CCPP has unverified group exchangeability and no usable temporal metadata. Parkinsons has few held-out subjects; bootstrap draws do not add subjects.',
 'Bike assumes local dependence covered by the declared seven-day calendar blocks; three/fourteen-day sensitivities and leave-one-group checks remain relevant.',
 'Energy is separate small-group simulation transfer. Other outer folds are dependent descriptive robustness results.',
 'Ten/thirty/120-minute prefixes are descriptive and unavailable models are not assigned predictive losses.',
 'P00 versus independently V-tuned P01 is a policy comparison, not fixed-treatment evidence for placement.',
 'The V-selected external champion is a12-family portfolio; its cost includes every P02-P13 search.',
 'Physical frames, standalone procedures and nested diagnostic costs are different views and must not be added.',
 'Dedicated timing and native-float64 versus float32 deployment-quality equivalence remain unqualified; no formal efficiency claim.',
 'This report does not establish a general novel growth principle or universal superiority.']


def _ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _compile_practical_report(payload,reference,metrics,analysis,identity,attempts,costs,*,attempt_resolutions):
    """Supplied numeric metadata only, never actual native/protected evidence."""
    compiled=_compile_comparison_inputs(payload,reference,metrics)
    if (analysis.get('source_sha256')!=payload['source_sha256'] or analysis.get('protocol_reference')!=reference
            or analysis.get('metric_inputs_sha256')!=_hash(metrics) or analysis.get('inputs')!=compiled
            or costs.get('source_sha256')!=payload['source_sha256'] or type(identity) is not dict
            or type(identity.get('pinned_commit')) is not str or len(identity['pinned_commit'])!=40
            or any(c not in '0123456789abcdef' for c in identity['pinned_commit']) or not _digest(identity.get('execution_sha256'))):
        raise ValueError('Exact frozen report metric/analysis/cost and committed execution identities required')
    physical=costs.get('physical_owned_execution_ns');standalone=costs.get('standalone_training_plus_all_frozen_queries_ns')
    if type(physical) is not int or physical<0 or type(standalone) is not dict or any(type(n) is not int or n<0 for n in standalone.values()):raise ValueError('Explicit physical and standalone costs required; never impute unknown')
    folds={(f['dataset'],f['outer_fold']):f for f in metrics['folds']};primary=[];supplement=[];native_jobs=set();averages={}
    for binding in metrics['view_bindings']:
        name=binding['dataset'];fold=binding['outer_fold'];repeat=binding['search_repeat'];qid=binding['query_id'];case=folds[name,fold]
        raw=None if qid is None else copy.deepcopy(case['arithmetic']['methods'][qid]);variance=case['arithmetic']['F_target_variance']
        if raw is not None:
            if any(type(raw.get(k)) not in (int,float) or not math.isfinite(raw[k]) or raw[k]<0 for k in ('primary_MSE','row_RMSE','row_MAE')):raise ValueError('Finite original-unit report metrics required')
        prediction=dict(prediction_id=binding['job_id']+'/'+binding['view'],method_id=binding['family_id'],task_id=f'{name}/outer-{fold}',
            repetition=repeat,status='completed_eligible' if qid is not None else 'completed_ineligible',
            normalization_status='unavailable_prediction' if qid is None else 'undefined_F_variance' if variance==0 else 'defined',
            normalized_risk=None if raw is None else raw['normalized_risk'],primary_MSE=None if raw is None else raw['primary_MSE'],
            row_RMSE=None if raw is None else raw['row_RMSE'],row_MAE=None if raw is None else raw['row_MAE'])
        is_primary=name in OBSERVATIONAL and fold==(4 if name=='bike' else 0) and binding['view']=='native_quality'
        if is_primary:primary.append(prediction)
        supplement.append(dict(**copy.deepcopy(binding),prediction_status=prediction['status'],normalization_status=prediction['normalization_status'],
            metrics=raw,canonical_primary=is_primary,held_out_units=len(case['group_ids']),
            scope='Primary native observational prediction' if is_primary else 'Separate Energy/dependent fold/descriptive prefix; no separate efficacy claim'))
        if binding['view']=='native_quality':
            native_jobs.add(binding['job_id']);averages.setdefault((name,binding['family_id']),[]).append(prediction)
    if len(primary)!=140 or len(supplement)!=3360 or len(native_jobs)!=840:raise ValueError('Exactly140 observational primary predictions and all3360/840 views/jobs required')
    descriptive=[]
    for (name,family),rows in sorted(averages.items()):
        if len(rows)!=10:raise ValueError('All five folds and both search repetitions required for descriptive averages')
        risks=[r['normalized_risk'] for r in rows]
        means={str(r):(None if any(x['normalized_risk'] is None for x in rows if x['repetition']==r)
            else math.fsum(x['normalized_risk'] for x in rows if x['repetition']==r)/5) for r in (0,1)}
        descriptive.append(dict(dataset=name,family_id=family,mean_normalized_risk=None if any(v is None for v in risks) else math.fsum(risks)/10,
            repeat_fold_mean_risks=means,scope='Five dependent folds; equal-fold/R2 descriptive mean only; Energy separate simulation'))
    main=analysis['native_quality']['primary_seven_day'];point=None if main is None else main['champion']['aggregate_point_log_benefit'];interval=None if main is None else main['champion']['aggregate_interval']
    comparisons=[dict(comparison_id='native_quality_conditional_observational_suite',decision='not_tested',
        explanation='Unqualified supplied arithmetic only; actual frozen native/raw/cost provenance is required before any conditional claim.',
        estimate=point,lower=None if interval is None else interval[0],upper=None if interval is None else interval[1],margin=MARGIN,
        independent_unit='Held-out groups/days within five fixed tasks; counts and assumptions reported separately',n_units=None),
        dict(comparison_id='real_world_population_superiority',decision='not_tested',explanation='No dataset-population or retraining inference is admitted by this fixed benchmark.',
            estimate=None,lower=None,upper=None,margin=None,independent_unit='Population of applications',n_units=None)]
    audit=dict(schema='research/v2-canonical-audit',study_id='research-v2-practical-'+reference['sha256'][:12],status='estimation',evidence_scope='software',
        source=dict(commit=identity['pinned_commit'],package_sha256=identity['execution_sha256']),
        provenance=dict(source_authenticated=False,raw_predictions_authenticated=False,protected_protocol_authenticated=False),protocol_sha256=None,
        independent_units='Five fixed observational tasks; held-out groups/days supply conditional sampling units. R2 search repetitions are paired randomizations, not new tasks.',
        planned_prediction_ids=[p['prediction_id'] for p in primary],predictions=primary,attempts=copy.deepcopy(attempts),attempt_resolutions=copy.deepcopy(attempt_resolutions),comparisons=comparisons,
        costs=dict(physical_elapsed_ns=physical,standalone_attributed_ns=sum(standalone.values()),
            scope='Separate physical owned execution and840 standalone training/all-frozen-query totals; shared evaluation/timing overhead and excluded administration remain separate. Never add these views.'),
        limitations=copy.deepcopy(_LIMITS),sources=[copy.deepcopy(reference)])
    if main is not None:audit['secondary_families']=[practical_family_report_record('individual_native_families',main)]
    render_report(audit)
    return dict(schema='research/v2-practical-report-compilation',canonical_audit=audit,supplemental_views=supplement,native_model_count=len(native_jobs),
        secondary_fold_summaries=compiled['secondary_fold_summaries'],descriptive_native_fold_means=descriptive,
        descriptive_prefixes=copy.deepcopy(analysis['descriptive_prefixes']),primary_arithmetic_conditions=None if main is None else copy.deepcopy(main['primary_conditions']),
        bike_sensitivities=copy.deepcopy(analysis['native_quality']['bike_sensitivities']),**_FALSE,
        scope='Unqualified numeric report inputs only; no native/model/raw/protocol/cost or scientific claim authority')


def _read_paid_attempts(models):
    """Called only after actual full model/search qualification; read metadata."""
    attempts=[];resolutions={};references=[]
    from .ledger import TERMINAL
    for row in models.payload['models']:
        path=_path(row['search_directory'])/'events.jsonl';reference=_ref(path);_,content=_reference(reference)
        events=[_json(line) for line in content.splitlines()];starts=[e['payload'] for e in events if e['kind']=='attempt_started'];finishes=[e['payload']['record'] for e in events if e['kind']=='attempt_finished']
        if len(starts)!=len(finishes) or not 1<=len(starts)<=32:raise ValueError('Complete paid attempt history required for report')
        for index,(start,finish) in enumerate(zip(starts,finishes,strict=True)):
            if start['attempt_index']!=index or finish['attempt_index']!=index or finish['status'] not in TERMINAL:raise ValueError('Exact ordered native attempt status required')
            identifier=f"{row['job_id']}/attempt-{index:03}";attempts.append(dict(attempt_id=identifier,status=finish['status']))
            if start['retry_index']==1:
                if (index==0 or finishes[index-1]['status']!='interrupted_recoverable' or start['candidate_id']!=starts[index-1]['candidate_id']
                        or start['config']!=starts[index-1]['config'] or finish['status']=='interrupted_recoverable'):
                    raise ValueError('Historical interruption needs its exact closed same-state retry')
                resolutions[f"{row['job_id']}/attempt-{index-1:03}"]=identifier
        references.append(reference)
    for reference in references:_reference(reference)
    return attempts,resolutions,references


def _actual_report_inputs(reference,kwargs,*,programme_reference,analysis_reference,raw_audit_reference,prediction_stage_reference,costs_reference,directory):
    from .practical_pipeline_costs_v2 import load_practical_pipeline_costs,_actual_inputs
    from .qualified_practical_models_v2 import load_qualified_practical_models
    costs=load_practical_pipeline_costs(costs_reference,protocol_reference=reference,programme_reference=programme_reference,analysis_reference=analysis_reference,
        raw_audit_reference=raw_audit_reference,prediction_stage_reference=prediction_stage_reference,directory=directory,**kwargs)
    inputs=_actual_inputs(reference,kwargs,analysis_reference=analysis_reference,raw_audit_reference=raw_audit_reference,prediction_stage_reference=prediction_stage_reference,directory=directory)
    models=load_qualified_practical_models(kwargs['models_reference'],global_barrier_reference=kwargs['global_barrier_reference'],qualification_reference=kwargs['qualification_reference'])
    try:
        from .production_readiness_v2 import load_production_readiness
    except ImportError as error:raise NotImplementedError('Production source/readiness implementation required before protected report') from error
    ready=load_production_readiness(kwargs['readiness_reference'],qualification_reference=kwargs['qualification_reference'])
    snapshot=ready.payload['execution_snapshot'];identity={k:snapshot[k] for k in ('pinned_commit','execution_sha256')}
    attempts,resolutions,journals=_read_paid_attempts(models)
    protocol,stage,metrics,completion,analysis,raw,pre=inputs
    compiled=_compile_practical_report(protocol.payload,reference,metrics,analysis.payload['analysis'],identity,attempts,costs.payload,attempt_resolutions=resolutions)
    audit=compiled['canonical_audit'];audit.update(status='completed',evidence_scope='confirmation',protocol_sha256=reference['sha256'],
        provenance=dict(source_authenticated=True,raw_predictions_authenticated=True,protected_protocol_authenticated=True))
    audit['sources']=[reference,programme_reference,analysis_reference,raw_audit_reference,prediction_stage_reference,costs_reference,kwargs['readiness_reference'],*journals]
    row=audit['comparisons'][0];conditions=compiled['primary_arithmetic_conditions']
    missing=any(p['status']!='completed_eligible' or p['normalized_risk'] is None for p in audit['predictions'])
    if missing or row['lower'] is None:row.update(decision='blocked',explanation='Incomplete/undefined primary sampling inputs prevent a useful-margin claim.')
    elif conditions['statistical_useful_conditions_pass']:
        row.update(decision='passed',explanation=('Conditional fixed-suite useful improvement' if conditions['robustness_conditions_pass'] else 'Heterogeneous conditional aggregate improvement')+
            '; stated exchangeability/dependence assumptions apply. CCPP exchangeability is unverified and Parkinsons has few clusters; no real-world population superiority is inferred.')
    else:row.update(decision='failed' if row['upper']<MARGIN else 'inconclusive',explanation='The conditional interval '+('excludes the useful2% margin' if row['upper']<MARGIN else 'does not resolve the useful2% margin')+'; this is not universal absence of benefit.')
    compiled.update(native_qualification_authenticated=True,whole_training_procedure_cost_authenticated=True,
        claim_authorized=row['decision']=='passed',T_accessed=True,
        scope='Actual frozen native/model/raw/protocol/complete-cost report inputs; any useful claim is conditional on the named finite tasks and stated sampling assumptions, never population superiority')
    render_report(audit)
    return compiled,costs,inputs


def write_protected_practical_report(reference,*,programme_reference,analysis_reference,raw_audit_reference,prediction_stage_reference,costs_reference,
        directory,output,models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    args=dict(programme_reference=programme_reference,analysis_reference=analysis_reference,raw_audit_reference=raw_audit_reference,
        prediction_stage_reference=prediction_stage_reference,costs_reference=costs_reference,directory=directory)
    compiled,costs,inputs=_actual_report_inputs(reference,kwargs,**args)
    from .protected_prediction_gate_v2 import _publication_location
    from .practical_pipeline_costs_v2 import _cost_output_location
    protocol=inputs[0];_,raw=_reference(programme_reference);programme=_json(raw)
    path=_publication_location(output,protocol=protocol,inspection=inputs[1].payload['inspection']);_cost_output_location(path,programme,programme_reference)
    if path.suffix!='.md':raise ValueError('Canonical practical report requires a fresh Markdown path')
    supplemental=path.with_suffix('.supplement.json');manifest=path.with_suffix('.manifest.json')
    for target in (supplemental,manifest):
        _publication_location(target,protocol=protocol,inspection=inputs[1].payload['inspection']);_cost_output_location(target,programme,programme_reference)
    again,new_costs,new_inputs=_actual_report_inputs(reference,kwargs,**args)
    if again!=compiled or new_costs.payload!=costs.payload or new_inputs[0].payload!=protocol.payload:raise ValueError('Actual report source/evidence changed before publication')
    bundle=dict(schema='research/v2-protected-practical-report-supplement',protocol_reference=reference,source_sha256=protocol.payload['source_sha256'],
        compilation=compiled,costs_reference=costs_reference,population_superiority_confirmed=False,formal_efficiency_claim=False)
    _write_new(supplemental,bundle)
    text=render_report(compiled['canonical_audit'])
    text+='\n## Complete descriptive inventory\n\nAll3,360 views, Energy and dependent-fold/prefix results are retained in ['+supplemental.name+']('+supplemental.name+'). Unavailable prefixes are explicit statuses, never predictive losses.\n'
    text+='\n| Dataset | Family | Five-fold/R2 mean normalized risk (descriptive) |\n| --- | --- | ---: |\n'
    for row in compiled['descriptive_native_fold_means']:
        value='unavailable' if row['mean_normalized_risk'] is None else format(row['mean_normalized_risk'],'.10g')
        text+=f"| {row['dataset']} | {row['family_id']} | {value} |\n"
    content=text.encode();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(content);stream.flush();os.fsync(stream.fileno())
    _write_new(manifest,dict(schema='research/v2-protected-practical-report-publication',protocol_reference=reference,source_sha256=protocol.payload['source_sha256'],
        report_reference=_ref(path),supplement_reference=_ref(supplemental),costs_reference=costs_reference,actual_frozen_evidence_authenticated=True,
        protected_access_authorized=False,population_superiority_confirmed=False,formal_efficiency_claim=False))
    return dict(report_reference=_ref(path),supplement_reference=_ref(supplemental),manifest_reference=_ref(manifest),
        actual_frozen_evidence_authenticated=True,protected_access_authorized=False,population_superiority_confirmed=False,formal_efficiency_claim=False)
