"""Individual M2 jobs; production admission/conditional orchestration is separate."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from treeneuron import train
from treeneuron.checkpoint import load_checkpoint,_descriptor
from treeneuron.config import canonical_hash,resolve_experiment
from treeneuron.evaluation.metrics import regression_metrics
from treeneuron.growth.score import InvalidScoreRun
from .diagnostics import ForkSpec,score_parent,fit_diagnostic_fork
from .head_artifacts import fit_head_artifact,load_head_artifact
from .ledger import AttemptLedger,write_new_json
from .local_calibration import calibrate_local_score
from .opportunity_data import (prepare_data_artifact,load_fitting_data,
    load_evaluation_data,load_control_parameters)
from .opportunity_inputs import save_scores,load_scores,horizon_slots,pair_oracle
from .parent_artifacts import ParentSpec,fit_parent_artifact,load_parent_input
from .seeds import seed64


class OpportunityWorkers:
    def __init__(self,queue_plan,*,guard=lambda:None):
        self.jobs={j['job_id']:copy.deepcopy(j) for j in queue_plan['attempts']}
        self.config=copy.deepcopy(queue_plan['identity']['study_config'])
        self.guard=guard

    def _dependency(self,dependencies,kind):
        found=[record['value'] for name,record in dependencies.items() if self.jobs[name]['kind']==kind]
        if len(found)!=1:raise ValueError(f'Expected exactly one {kind} prerequisite')
        return found[0]

    def _context(self,job,dependencies):
        data=self._dependency(dependencies,'data');prerequisite=self._dependency(dependencies,'parent')
        view=load_fitting_data(data['artifact_path'])
        parent=load_parent_input(prerequisite['artifact_path'],job['payload'].get('snapshot','late_G_best'),view)
        return view,parent

    def __call__(self,job,directory,dependencies,attempt):
        kind=job['kind']
        if kind not in ('data','parent','score','horizon_binding','local_diagnostic','fit','head','pair_oracle','evaluate',
                        'fallback_gate','snapshot_gate','extension_decision'):
            raise NotImplementedError(f'M2 {kind} worker is not implemented; no production admission')
        if job.get('job_id') not in self.jobs or canonical_hash(job)!=canonical_hash(self.jobs[job['job_id']]):
            raise ValueError('Worker job differs from its frozen queue plan')
        self.guard();before=train.capture_rng()
        try:
            value=getattr(self,'_'+kind)(job,Path(directory),dependencies)
            self.guard();return value
        except (np.linalg.LinAlgError,InvalidScoreRun) as error:
            raise FloatingPointError(f'{type(error).__name__}: {error}') from error
        finally:train.restore_rng(before)

    def _fallback_gate(self,job,directory,dependencies):
        from .opportunity_gates import fallback_gate
        result=fallback_gate(job,dependencies)
        write_new_json(directory/'decision.json',result)
        return result

    def _snapshot_gate(self,job,directory,dependencies):
        from .opportunity_gates import snapshot_gate
        result=snapshot_gate(job,dependencies)
        write_new_json(directory/'decision.json',result)
        return result

    def _extension_decision(self,job,directory,dependencies):
        from .opportunity_analysis import opportunity_summary
        from .opportunity_plan import opportunity_decision_contract
        contract=job['payload']['opportunity_decision_contract']
        if contract!=opportunity_decision_contract(self.config):
            raise ValueError('Opportunity decision contract differs from the pre-outcome recipe')
        cases=[j['payload'] for j in self.jobs.values() if j['kind'] in ('fit','head')
            and j['payload']['teacher_index']<4
            and j['payload']['matrix'] in ('M2a','M2e_neural','M2e_head')]
        outcomes={}
        for name,record in dependencies.items():
            source=self.jobs[name]
            if source['kind']!='evaluate':raise ValueError('Extension requires terminal evaluator inputs')
            value=record['value'];row=dict(status=value['status'])
            if value['status']=='completed_eligible':
                row['V_NMSE']=value['metrics']['selected']['V']['normalized_mse']
            outcomes[source['payload']['attempt_id']]=row
        summary=opportunity_summary(cases,outcomes,contract,teachers=range(4))
        status=summary['status']
        if status=='completed_eligible' and not summary['extension_required']:status='completed_ineligible'
        result=dict(status=status,summary=summary,decision_scope='primary M2a/e and required fallback; one extension only')
        write_new_json(directory/'decision.json',result)
        return result

    def _data(self,job,directory,dependencies):
        path=directory/'artifact'
        report=prepare_data_artifact(self.config['base_experiment'],job['payload'],path)
        return dict(status=report['status'],artifact_path=str(path.resolve()),teacher_id=report['teacher_id'])

    def _parent(self,job,directory,dependencies):
        data=self._dependency(dependencies,'data');p=job['payload']
        raw=copy.deepcopy(self.config['base_experiment'])
        raw['seeds'].update({k:p[k+'_seed'] for k in ('teacher','data','reference','optimizer','allocation','contrast')})
        if 'training' in p:raw['training']=copy.deepcopy(p['training'])
        control=None
        if p['parent_recipe']=='raw_positive_control':
            raw['model'].update(d=2,h=8);raw['growth']['m_requested']=2
            raw['data'].update(n_fit=512,n_growth=256,n_validation=256,noise_sigma=.1)
            control=load_control_parameters(data['artifact_path'])
        raw['execution']['artifacts_root']=str(directory.resolve())
        config=resolve_experiment(raw);path=directory/'artifact'
        report=fit_parent_artifact(load_fitting_data(data['artifact_path']),config,path,
            ParentSpec(tuple(p.get('snapshot_updates',())),fixed_budget=not p.get('stop_by_patience',True)),
            control=control,guard=self.guard)
        return dict(status=report['status'],artifact_path=str(path.resolve()),report=report,
            candidate_sample_seed=p['candidate_sample_seed'])

    def _score(self,job,directory,dependencies):
        view,parent=self._context(job,dependencies)
        if parent is None:return dict(status='completed_ineligible',reason='Requested actual parent update was not reached')
        scored=score_parent(parent,view.F);path=directory/'artifact';save_scores(path,scored)
        seed=self._dependency(dependencies,'parent')['candidate_sample_seed']
        return dict(status='completed_eligible',artifact_path=str(path.resolve()),
            candidate_sample_seed=seed,horizon_slots=horizon_slots(scored.table,seed),
            table_sha256=scored.table_sha256,parent_sha256=scored.parent_sha256)

    def _horizon_binding(self,job,directory,dependencies):
        score=self._dependency(dependencies,'score')
        if score['candidate_sample_seed']!=job['payload']['seed']:
            raise ValueError('F-only horizon binding seed differs from its frozen score provenance')
        result=dict(status='completed_eligible',slots=score['horizon_slots'],table_sha256=score['table_sha256'],
            seed=score['candidate_sample_seed'],selection_labels='F_only')
        write_new_json(directory/'binding.json',result);return result

    def _local_diagnostic(self,job,directory,dependencies):
        view,parent=self._context(job,dependencies)
        if parent is None:return dict(status='completed_ineligible',reason='Requested actual parent update was not reached')
        scored=load_scores(self._dependency(dependencies,'score')['artifact_path'],parent)
        seed=seed64(self._dependency(dependencies,'parent')['candidate_sample_seed'],'M2-local-calibration/minibatches')
        calibrated=calibrate_local_score(parent,view.F,scored,seed=seed,minibatches=64,step_size=1e-7)
        path=directory/'arrays.npz'
        with path.open('xb') as stream:np.savez_compressed(stream,**calibrated.arrays)
        result=dict(status=calibrated.report['status'],calibration=calibrated.report,
            arrays_path=str(path.resolve()))
        write_new_json(directory/'calibration.json',result);return result

    def _fit(self,job,directory,dependencies):
        view,parent=self._context(job,dependencies);p=job['payload']
        if parent is None:return dict(status='completed_ineligible',reason='Requested actual parent update was not reached')
        role=p.get('role');ids=p.get('unit_ids',[])
        if p['matrix']=='M2c':
            slots=self._dependency(dependencies,'horizon_binding')['slots']
            slot=next(s for s in slots if s['slot']==p['slot'])
            if slot['status']!='bound':return dict(status='completed_ineligible',reason=slot['reason'])
            role,ids=slot['role'],slot['unit_ids']
        role={'singleton':'subset','pair':'subset','ordinary_keep':'keep'}.get(role,role)
        scored=None
        if any(self.jobs[name]['kind']=='score' for name in dependencies):
            scored=load_scores(self._dependency(dependencies,'score')['artifact_path'],parent)
        spec=ForkSpec(role,p['requested_count'],p['allocation_seed'],p['max_updates'],
            new_stream_seed=p['new_stream_seed'] if p.get('treatment')=='new_stream_retained_adam' else None,
            unit_ids=tuple(ids),fixed_budget=not p['stop_by_patience'],
            selection_prefixes=tuple(p.get('selection_prefixes',())))
        path=directory/'artifact';report=fit_diagnostic_fork(parent,view,path,spec,scored=scored,guard=self.guard)
        return dict(status=report['status'],artifact_path=str(path.resolve()),report=report)

    def _head(self,job,directory,dependencies):
        view,parent=self._context(job,dependencies)
        if parent is None:return dict(status='completed_ineligible',reason='Requested actual parent update was not reached')
        path=directory/'artifact';report=fit_head_artifact(parent,view,path)
        return dict(status=report['status'],artifact_path=str(path.resolve()),report=report)

    def _pair_oracle(self,job,directory,dependencies):
        rows=[]
        for name,record in dependencies.items():
            value=record['value'];row=dict(unit_ids=self.jobs[name]['payload']['unit_ids'],status=value['status'])
            if value['status']=='completed_eligible':row['selected_g_mse']=value['report']['selected_g_mse']
            rows.append(row)
        selected=pair_oracle(rows);result=dict(status=selected['status'],selection=selected)
        if selected['status']=='completed_eligible':
            names=[name for name in dependencies if self.jobs[name]['payload']['unit_ids']==selected['unit_ids']]
            if len(names)!=1:raise ValueError('Oracle winner must name exactly one nominal pair')
            winner=dependencies[names[0]]['value']
            result.update(artifact_path=winner['artifact_path'],report=winner['report'],selected_procedure_job_id=names[0])
        write_new_json(directory/'selection.json',result);return result

    def _evaluate(self,job,directory,dependencies):
        data=self._dependency(dependencies,'data')
        producers=[r['value'] for name,r in dependencies.items() if self.jobs[name]['kind']!='data']
        if len(producers)!=1:raise ValueError('Evaluator requires one retained selected procedure')
        producer=producers[0];report=producer['report'];artifact=Path(producer['artifact_path'])
        models=dict(report['models'])
        for record in report.get('prefixes',{}).get('records',[]):
            path=(artifact/'prefixes'/record['pointer']).resolve()
            if not path.is_relative_to((artifact/'prefixes').resolve()):raise ValueError('Prefix reference escaped artifact')
            pointer,_,_=_descriptor(path)
            if pointer['sha256']!=record['metadata_sha256']:raise ValueError('Prefix commitment differs')
            models['prefix_'+str(record['prefix'])]=dict(path=str(path),loader='training_state',
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        development,E=load_evaluation_data(data['artifact_path'])
        variance=float(np.var(development.F.y,ddof=0))
        arrays={role+'_'+k:getattr(rows,k) for role,rows in (('V',development.V),('E',E))
            for k in ('y','row_id','group_id')}
        metrics={}
        queue_root=directory.parent.parent.parent.resolve()
        for name,reference in models.items():
            path=Path(reference['path']).resolve()
            if not path.is_relative_to(queue_root) or hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']:
                raise ValueError('Retained model pointer commitment/root differs')
            if reference['loader']=='research_frozen_head':model=load_head_artifact(reference)
            elif reference['loader']=='training_state':
                model=load_checkpoint(path,expected_hashes=report['identity_hashes'],device='cpu')
            else:raise ValueError('Unknown retained M2 predictor loader')
            identities=(model.parent_identity_hashes if reference['loader']=='research_frozen_head'
                else model.identity_hashes)
            if (identities['raw']!=development.manifest.dataset_id
                    or identities['split']!=development.manifest.assignment_hash
                    or (reference['loader']=='training_state' and model.data_hashes!=train._fg_hashes(development))):
                raise ValueError('Retained model and evaluator data identity differ')
            metrics[name]={}
            for role,rows in (('V',development.V),('E',E)):
                prediction=model.predict(rows.X);arrays[name+'_'+role]=prediction
                metrics[name][role]=regression_metrics(rows.y,prediction,fit_target_variance=variance,
                    noisefree=rows.noise_free_target)
        target=directory/'predictions.npz'
        with target.open('xb') as stream:np.savez_compressed(stream,**arrays)
        result=dict(status='completed_eligible',metrics=metrics,fit_target_variance=variance,
            predictions_path=str(target.resolve()),V_labels_access='coordinator_after_fitting',E_exploratory=True,T_accessed=False)
        write_new_json(directory/'metrics.json',result);return result


def standalone_costs(plan,ledger):
    """Each procedure pays every ancestor once; the physical queue stays separate."""
    if (plan.get('plan_hash')!=canonical_hash({k:v for k,v in plan.items() if k!='plan_hash'})
            or canonical_hash(plan)!=canonical_hash(ledger.plan)):
        raise ValueError('Cost accounting plan differs from the authenticated attempt ledger')
    ledger=AttemptLedger.resume(ledger.directory,plan)
    latest={event['attempt_id']:event for event in ledger.events};ancestry={};costs={}
    for job in plan['attempts']:
        name=job['job_id'];parents=set(job['depends_on'])
        for parent in job['depends_on']:parents.update(ancestry[parent])
        ancestry[name]=parents;members=parents|{name}
        costs[name]=dict(charged_ns=sum(latest[m]['elapsed_ns'] for m in members if m in latest),
            prerequisite_job_ids=sorted(parents),complete=all(m in latest and latest[m]['status'] in
                ('completed_eligible','completed_ineligible') for m in members),
            cost_boundary='unique prerequisite work plus own job; physical sharing reported separately',
            cost_precision='contains_worker_loss_upper_bound' if any(e.get('cost_precision')==
                'wall_time_upper_bound_after_worker_loss' for e in ledger.events if e['attempt_id'] in members)
                else 'measured_jobs; queue/startup/import overhead remains external')
    return costs
