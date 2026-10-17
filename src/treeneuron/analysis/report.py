"""Deterministic rendering of validated audit results, never hand-entered scores."""
from collections.abc import Mapping
from pathlib import Path
import json
import time
import uuid
from ..artifacts import _atomic_json
from ..config import canonical_hash


def render_report(audit_result: Mapping, output: Path) -> None:
    started=time.perf_counter_ns()
    value=dict(audit_result)
    digest=value.pop('audit_sha256',None)
    if digest is None or canonical_hash(value)!=digest:
        raise ValueError('Report requires an intact audit result')
    conclusion=value['conclusion']
    lines=['# Audited research decision', '', f'Status: **{value["status"]}**.', '',
           f'Mode: {value["mode"]}. Protocol: `{value.get("protocol_hash")}`.', '',
           f'Decision: {conclusion["decision"]}. {conclusion["reason"]}', '',
           f'Interpretation: {conclusion["interpretation"]}. Completed-cohort superiority: {conclusion["superiority"]}.', '']
    if value['reasons']:
        lines+=['Evidence problems:', '']+[f'- {reason}' for reason in value['reasons']]+['']
    primary=value.get('primary')
    if primary:
        lines += [f'Declared teachers: {primary["declared_teachers"]}; valid teachers: {primary["valid_teachers"]}; '
                  f'valid runs: {primary["valid_runs"]}/{primary["declared_runs"]}.', '',
                  'All frozen teachers, unfavorable outcomes and zero-growth cases remain in accounting. '
                  'Optimizer repetitions are averaged within each teacher.', '']
        if value['valid']:
            lines += ['| Teacher | Status | D (random − top) | Original-unit D |', '| --- | --- | --- | --- |']
            for teacher in primary['teachers']:
                lines.append(f'| {teacher["teacher_id"]} | {teacher["status"]} | {teacher["D"]} | {teacher["D_original"]} |')
            lines += ['']
        lines += ['Reconstructed uncertainty and precision:', '', '```json',
                  json.dumps(primary['primary'],sort_keys=True,indent=2,allow_nan=False),'```','',
                  'Per-task metrics, descriptive regimes, exclusions/failures and exact family adjustment:', '', '```json',
                  json.dumps({k:primary[k] for k in ('teachers','runs','controls','holm_adjusted','policy')},
                             sort_keys=True,indent=2,allow_nan=False),'```','']
    lines += ['Measured costs and attempt accounting:', '', '```json',
              json.dumps(value.get('accounting'),sort_keys=True,indent=2,allow_nan=False),'```','',
              'Claim and evidence limits:', '']
    lines += [f'- {key}: {text}' for key,text in value['limits'].items()]
    lines += ['', f'Audit digest: `{digest}`.', '']
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines))
    work=output.parent/'report-work'; work.mkdir(exist_ok=True)
    _atomic_json(work/f'{uuid.uuid4().hex}.json',dict(schema_version=1,audit_sha256=digest,
        output=output.name,seconds=(time.perf_counter_ns()-started)/1e9,
        boundary='validate audit digest and render/write report; excludes final work receipt',
        aggregation='separate report invocation; not included in fitting forecast'))
