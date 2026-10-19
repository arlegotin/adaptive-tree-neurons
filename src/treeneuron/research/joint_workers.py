"""M3 native jobs; public admission and complete-cost qualification are separate."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from treeneuron import train
from treeneuron.baselines.causal import ordinary_width_brackets
from treeneuron.config import canonical_hash,resolve_experiment
from treeneuron.growth.allocate import allocate
from treeneuron.growth.score import InvalidScoreRun
from .diagnostics import ForkSpec,fit_diagnostic_fork
from .joint_allocation import SelectionTrace,build_gate_blocks,joint_ridge_select
from .joint_plan import joint_extension_decision
from .joint_selector import JointSelectionConfig
from .ledger import write_new_json
from .local_updates import diagnose_local_updates
from .opportunity_inputs import load_scores
from .opportunity_workers import OpportunityWorkers
from .parent_artifacts import ParentSpec,fit_parent_artifact,load_parent_input


class JointWorkers(OpportunityWorkers):
    def __call__(self,job,directory,dependencies,attempt):
        if job['kind'] not in ('placement','local_model','probe','short_selection','extension_decision'):
            return super().__call__(job,directory,dependencies,attempt)
        if job.get('job_id') not in self.jobs or canonical_hash(job)!=canonical_hash(self.jobs[job['job_id']]):
            raise ValueError('M3 worker job differs from its immutable queue')
        self.guard();before=train.capture_rng()
        try:
            value=getattr(self,'_'+job['kind'])(job,Path(directory),dependencies)
            self.guard();return value
        except (np.linalg.LinAlgError,InvalidScoreRun) as error:
            raise FloatingPointError(f'{type(error).__name__}: {error}') from error
        finally:train.restore_rng(before)

    def _scored(self,job,dependencies):
        data,parent=self._context(job,dependencies)
        if parent is None:raise ValueError('M3 requires the recorded late selected parent')
        scored=load_scores(self._dependency(dependencies,'score')['artifact_path'],parent)
        return data,parent,scored

    def _placement(self,job,directory,dependencies):
        data,parent,scored=self._scored(job,dependencies);p=job['payload']
        cfg=JointSelectionConfig(count=p['requested_count'],parent_ridge=p['parent_ridge'],
            gate_ridge=p['gate_ridge'],condition_on_parent=p['condition_on_parent'])
        trace=joint_ridge_select(parent,data.F,build_gate_blocks(parent,data.F,scored),cfg,
            scored=scored,strategy=p['selection_strategy'])
        path=directory/'selection.json';write_new_json(path,trace.to_dict())
        return dict(status='completed_eligible',trace_sha256=trace.sha256,selection_path=str(path.resolve()),
            selection_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selection_labels='F_only')

    def _trace(self,directory,dependencies):
        value=self._dependency(dependencies,'placement');path=Path(value['selection_path']).resolve()
        queue_root=directory.parent.parent.parent.resolve()
        if (not path.is_relative_to(queue_root) or path.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest()!=value['selection_file_sha256']):
            raise ValueError('Committed placement trace bytes/root differ')
        trace=SelectionTrace(json.dumps(json.loads(path.read_text()),sort_keys=True,separators=(',',':'),allow_nan=False))
        if trace.sha256!=value['trace_sha256']:raise ValueError('Immutable placement trace digest differs')
        return trace

    def _local_model(self,job,directory,dependencies):
        data,parent,scored=self._scored(job,dependencies)
        diagnostic=diagnose_local_updates(parent,data.F,data.G,scored,self._trace(directory,dependencies))
        write_new_json(directory/'diagnostic.json',diagnostic)
        return dict(status='completed_eligible',diagnostic=diagnostic,V_accessed=False,E_accessed=False,T_accessed=False)

    def _probe(self,job,directory,dependencies):return self._fit(job,directory,dependencies)

    def _fit(self,job,directory,dependencies):
        data,parent,scored=self._scored(job,dependencies);p=job['payload'];role=p['role']
        trace=None;ids=tuple(p.get('unit_ids',()));native_role=role
        if role=='fresh_upper':return self._fresh(parent,data,scored,job,directory)
        if role in ('local_singleton','joint','unconditioned','pure_span'):
            native_role='ridge';trace=self._trace(directory,dependencies)
        elif role=='short_singleton':
            selected=self._dependency(dependencies,'short_selection')
            if selected['status']!='completed_eligible':raise ValueError('Complete G probe selection is required')
            if (selected['parent_sha256']!=scored.parent_sha256 or selected['table_sha256']!=scored.table_sha256
                    or selected['requested_count']!=p['requested_count']):
                raise ValueError('G probe selection belongs to a different parent/table/count')
            ids=tuple(selected['selected_ids']);native_role='fixed_subset'
            write_new_json(directory/'structural-selection.json',selected)
        path=directory/'artifact'
        count=p['requested_count']
        spec=ForkSpec(native_role,count,p['allocation_seed'],p['max_updates'],unit_ids=ids,
            fixed_budget=not p['stop_by_patience'])
        report=fit_diagnostic_fork(parent,data,path,spec,scored=scored,selection_trace=trace,guard=self.guard)
        return dict(status=report['status'],artifact_path=str(path.resolve()),report=report,method_role=role)

    def _fresh(self,parent,data,scored,job,directory):
        p=job['payload'];eligible=allocate(scored.table,parent.model.layer.h,policy='adam',seed=0).ids.tolist()
        effective=min(p['requested_count'],len(eligible));d=parent.model.layer.d
        target=sum(v.numel() for v in parent.model.parameters())+2*effective*(d+1)
        lower,upper=ordinary_width_brackets(d,target)
        raw=parent.config.to_dict();raw['model']['h']=upper;raw['seeds']['optimizer']=p['fresh_optimizer_seed']
        raw['training']['base_updates']=p['max_updates'];raw['execution']['artifacts_root']=str(directory.resolve())
        path=directory/'artifact';result=fit_parent_artifact(data,resolve_experiment(raw),path,
            ParentSpec(fixed_budget=True),guard=self.guard)
        if result['status']!='completed_eligible':return dict(status=result['status'],method_role='fresh_upper',report=result)
        selected=path/result['selected']['path'];state=load_parent_input(path,'late_G_best',data)
        report=dict(result,models={'selected':dict(path=str(selected.resolve()),sha256=result['selected']['sha256'],loader='training_state')},
            executed_stage_updates=result['executed_updates'],selected_stage_update=result['selected_update'],
            total_parameters=sum(v.numel() for v in state.model.parameters()),inherit_parent_parameters=False,
            ordinary_width=upper,ordinary_brackets=[lower,upper],gated_parameter_target=target,
            requested_count=p['requested_count'],effective_count=effective,
            cost_scope='Fresh model fit plus common parent/scoring prerequisites needed to resolve actual effective-capacity bracket')
        return dict(status=report['status'],artifact_path=str(path.resolve()),report=report,method_role='fresh_upper')

    def _short_selection(self,job,directory,dependencies):
        essential=[r['value']['status'] for name,r in dependencies.items() if self.jobs[name]['kind']!='probe']
        if any(status!='completed_eligible' for status in essential):
            return dict(status='irrecoverable_missing',selected_ids=None,reason='Missing exact parent/data/score prerequisite')
        data,parent,scored=self._scored(job,dependencies);h=parent.model.layer.h
        eligible=set(allocate(scored.table,h,policy='adam',seed=0).ids.tolist())
        nominal={};keep=None;keep_spec=None;records=[];failed=[]
        def checked_risk(value,spec):
            report=value['report'];g=report['selected_g_mse']
            if (type(g) not in (int,float) or not math.isfinite(g) or g<0
                    or report['executed_stage_updates']!=spec['max_updates']
                    or report['resolved_policy']['max_updates']!=spec['max_updates']
                    or report['resolved_policy']['stop_by_patience'] is not False
                    or report['parent_sha256']!=scored.parent_sha256
                    or report['table_sha256']!=scored.table_sha256
                    or report['request']['new_stream_seed'] is not None):
                raise ValueError('Probe risk/budget/parent differs from its declared fixed horizon')
            return g
        for name,r in dependencies.items():
            if self.jobs[name]['kind']!='probe':continue
            p=self.jobs[name]['payload'];value=r['value'];ids=p['unit_ids']
            if not ids:
                if keep is not None:raise ValueError('Duplicate keep probe')
                keep=value;keep_spec=p;continue
            if len(ids)!=1 or ids[0] in nominal or ids[0] not in range(h):raise ValueError('Invalid nominal singleton inventory')
            unit=ids[0];nominal[unit]=value
            row=dict(unit_id=unit,status=value['status'],eligible=unit in eligible)
            if unit in eligible and value['status']!='completed_eligible' or unit not in eligible and value['status']!='completed_ineligible':
                failed.append(unit)
            if value['status']=='completed_eligible':
                row['selected_g_mse']=checked_risk(value,p)
            records.append(row)
        if set(nominal)!=set(range(h)) or keep is None:raise ValueError('Every nominal unit and keep probe must remain recorded')
        if keep['status']=='completed_eligible':checked_risk(keep,keep_spec)
        if any(self.jobs[name]['payload']['max_updates']!=keep_spec['max_updates']
                for name in dependencies if self.jobs[name]['kind']=='probe'):
            raise ValueError('Singleton and keep probes must share the same executed horizon')
        common=dict(parent_sha256=scored.parent_sha256,table_sha256=scored.table_sha256,
            requested_count=job['payload']['requested_count'],effective_count=min(job['payload']['requested_count'],len(eligible)),
            candidate_records=sorted(records,key=lambda row:row['unit_id']),probe_attempts=h+1,selection_labels='G',
            probe_weights_retained=False,V_accessed=False,E_accessed=False,T_accessed=False)
        if failed or keep['status']!='completed_eligible':
            result=dict(common,status='irrecoverable_missing',selected_ids=None,failed_nominal_ids=failed,
                reason='A required eligible probe or keep failed; no complete-case selection or replacement')
        else:
            keep_g=keep['report']['selected_g_mse'];gains={row['unit_id']:keep_g-row['selected_g_mse'] for row in records if row['eligible']}
            order=[]
            for _ in range(common['effective_count']):
                remaining=eligible-set(order);best=max(gains[j] for j in remaining);tol=1e-12*max(1.,abs(best))
                order.append(min(j for j in remaining if gains[j]>=best-tol))
            for row in records:
                if row['eligible']:row['G_gain_vs_keep']=gains[row['unit_id']]
            result=dict(common,status='completed_eligible',selected_ids=sorted(order),selection_order=order,keep_g_mse=keep_g)
        write_new_json(directory/'selection.json',result);return result

    def _extension_decision(self,job,directory,dependencies):
        if any(r['value']['status']!='completed_eligible' for r in dependencies.values()):
            result=dict(status='irrecoverable_missing',reason='Missing initial final predictions prevent the extension decision')
        else:
            rows=[]
            for name,r in dependencies.items():
                p=self.jobs[name]['payload']
                if p['role'] in ('joint','random'):
                    rows.append(dict(teacher_index=p['teacher_index'],repetition=p['repetition'],method=p['role'],
                        V_NMSE=r['value']['metrics']['selected']['V']['normalized_mse']))
            decision=joint_extension_decision(rows,seed=job['payload']['bootstrap_seed'])
            result=dict(status='completed_eligible' if decision['extend'] else 'completed_ineligible',decision=decision)
        write_new_json(directory/'decision.json',result);return result
