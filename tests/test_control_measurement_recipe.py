"""Qualification selects a fixed case and keeps nested resource phases separate."""
import importlib.util
from pathlib import Path

import pytest


def recipe():
    path=Path(__file__).resolve().parents[1]/'docs/research/v2/scripts/measure_negative_control.py'
    spec=importlib.util.spec_from_file_location('negative_control_measurement',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_selection_is_only_first_predeclared_affine_case():
    module=recipe()
    attempts=[dict(population='globally_affine',teacher_index=0,repetition=0,role=r)
        for r in ('keep','top','random')]
    plan=dict(config=dict(kind='m1_controls'),attempts=attempts+[
        dict(population='globally_affine',teacher_index=1,repetition=0,role='keep')])
    assert module.select_attempts(plan)==attempts
    plan['attempts']=attempts[:2]
    with pytest.raises(ValueError,match='fixed'):module.select_attempts(plan)


def test_phase_measurements_do_not_sum_nested_clocks_and_include_journal(tmp_path):
    from treeneuron.config import canonical_hash
    module=recipe();files=[]
    for name,n in (('model.pt',10),('model.meta.json',7),('history-one.jsonl',13)):
        p=tmp_path/name;p.write_bytes(b'x'*n);files.append(p)
    plan=dict(source_sha256='a'*64,execution_identity=dict(runtime='b'))
    config=dict(model=dict(d=16,h=64),data=dict(n_fit=2306,n_growth=577,n_validation=577),
        training=dict(batch_size=128))
    report=dict(costs=dict(parent_ns=100),methods={r:dict(status='completed',enclosing_ns=n)
        for r,n in zip(('keep','top','random'),(30,40,50))})
    records=module.measurements(plan,config,report,files,peak_rss_bytes=123)
    assert [r['phase'] for r in records]==['parent','continuation']
    assert [r['elapsed_ns'] for r in records]==[100,50]
    assert all(r['max_checkpoint_transaction_bytes']==4126 for r in records)
    assert all(r['retained_bytes']==30 for r in records)
    assert records[0]['execution_identity_hash']==canonical_hash(plan['execution_identity'])
