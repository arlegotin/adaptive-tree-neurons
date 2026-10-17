"""Independent saved original-unit prediction audit; no treeneuron imports."""
import hashlib,json,os,resource,time,traceback
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BASE=Path('environment/h1-v2/wp09'); OUT=Path('artifacts/h1-v2/wp09/raw-audit')
ORIGINAL=Path('artifacts/h1-v2/scientific-pilot/pilot-fe9efe02afc2444f94bd4aa6511cb76d')
JOIN=Path('artifacts/h1-v2/user-pause-continuation/a6b986ed496b4f22be3a43c65943dc63')
ROLES=('keep','top','random','greedy','widen_upper','ordinary_lower','ordinary_upper','static_partial')
started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=False)
report=dict(status='running',scope='Independent original-unit saved E predictions audit; exploratory only',inputs=[],pairs=[],teachers=[],failures=[])
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(): (OUT/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
inventory={}
for path,expected in [('environment/h1-v2/genuine-pilot/inventory.json','c0a46776b126cd4a30c1f586b25ea3c786155736b362322fccfb428f3c06f4ea'),('environment/h1-v2/user-pause-continuation/inventory.json','d984d8961a59dc5d996862a5361af30f28f5cc9b7c1a45e7be0c8a2fd33ff627')]:
    assert sha(path)==expected
    report['inputs'].append(dict(path=path,sha256=expected))
    inventory.update({x['path']:x['sha256'] for x in json.loads(Path(path).read_text())['files']})
checked=set()
def check(p,expected=None):
    p=Path(p); key=str(p)
    if key in checked:return
    value=sha(p); assert value==(expected or inventory[key]),key
    report['inputs'].append(dict(path=key,sha256=value,bytes=p.stat().st_size));checked.add(key)
def read(p,expected=None):check(p,expected);return json.loads(Path(p).read_text())
def locator(p):
    p=Path(p)
    if p.is_absolute():p=p.relative_to(Path.cwd())
    return p
try:
    members=read(JOIN/'joined-input-members.json','7f42ae967496a7e6a36095175617567454b7384bd74a81121d9be23681f7ab92')['members']
    assert len(members)==32 and {(x['teacher_index'],x['repetition']) for x in members}=={(i,j) for i in range(16) for j in range(2)}
    scalars=read(JOIN/'joined-diagnostics/joined-scalars-and-risks.json','7324942c0ee22b18b3fe76e22ba96c14be2b19d602e096c480ef5bda2a9bbff4')
    summary=read(JOIN/'joined-diagnostics/joined-summary-diagnostic.json','351c2578bde77e225a06478e212ac4a0b2bf60313a7f1daa535b6df5ea578fa1')['summary']
    report['maximum_differences']={k:0. for k in ('h1_rows','static_rows','role_risks','teacher_h1','teacher_static','within_variance','row_variance','summary_means')}
    def diff(key,a,b):
        d=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        report['maximum_differences'][key]=max(report['maximum_differences'][key],d)
        assert d<1e-12,(key,d)
    for i in range(16):
        observation=ORIGINAL/f'teacher-{i}'/'observations.npz';check(observation)
        with np.load(observation,allow_pickle=False) as z: fy=z['F_y'];ey=z['E_y'];er=z['E_rows']
        assert np.isfinite(fy).all() and np.isfinite(ey).all()
        variance=float(np.mean((fy-np.mean(fy))**2));assert variance>0
        hs=[];ss=[];risks=[]
        for j in range(2):
            member=next(m for m in members if (m['teacher_index'],m['repetition'])==(i,j));p=Path(member['path']);record=read(p,member['sha256'])
            assert record['status']=='completed' and not record['failures'] and record['repetition']==j
            assert record['source_sha256']=='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c'
            assert record['recipe_hash']=='9152a59bb3180da35f780b44a78d9eee59f7efa3e1af9677e4acf307a1bbb073'
            assert set(record['methods'])==set(ROLES) and all(x['status']=='completed' for x in record['methods'].values())
            predictions=p.with_name('E-predictions.npz');check(predictions,record['E_predictions_sha256'])
            with np.load(predictions,allow_pickle=False) as z:
                assert set(z.files)==set(ROLES)|{'y','row_id'}
                assert np.array_equal(z['y'],ey) and z['y'].dtype==ey.dtype and np.array_equal(z['row_id'],er) and z['row_id'].dtype==er.dtype
                losses={}
                for role in ROLES:
                    pred=z[role];assert pred.shape==ey.shape and np.isfinite(pred).all()
                    losses[role]=np.square(pred-ey)/variance
                h=losses['random']-losses['top'];s=losses['ordinary_upper']-losses['static_partial']
                role_risks={r:float(losses[r].mean()) for r in ROLES}
            diff('h1_rows',h,record['h1_rows']);diff('static_rows',s,record['static_rows'])
            old=next(x for x in scalars['pair_risks'] if (x['teacher_index'],x['repetition'])==(i,j))
            for r in ROLES:diff('role_risks',role_risks[r],old['normalized_E_risks'][r])
            score=read(p.with_name('score-table.json'));greedy=read(locator(record['greedy_report']))
            eligible=[int(u) for u,e in zip(score['ids'],score['eligible']) if e]
            assert greedy['complete'] and not greedy['not_attempted'] and greedy['probe_updates']==320
            assert len(greedy['trials'])==len(eligible)+1==65
            assert {t['unit_id'] for t in greedy['trials'] if t['unit_id'] is not None}==set(eligible)
            assert all(t['outcome']=='completed' and t['executed_updates']==320 for t in greedy['trials'])
            a=np.asarray([score['adam_scores'][score['ids'].index(u)] for u in eligible]);b=np.asarray([greedy['improvements'][str(u)] for u in eligible])
            defined=len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0
            association=dict(n=len(a),pearson=float(np.corrcoef(a,b)[0,1]) if defined else None,spearman=float(spearmanr(a,b).statistic) if defined else None,
                tied_scores=len(a)-len(np.unique(a)),tied_improvements=len(b)-len(np.unique(b)),nonpositive_improvements=int(np.sum(b<=0)),limitation='320-update finite G probe; no final convergence/generalization claim')
            pair=dict(teacher_index=i,repetition=j,teacher_id=record['teacher_id'],F_population_variance=variance,rows=len(ey),risks=role_risks,
                h1_mean=float(h.mean()),static_mean=float(s.mean()),probe_count=len(greedy['trials']),score_association=association,
                effective_growth=record['methods']['top']['allocation']['effective_count'],near_ties=int(sum(score['near_ties'])),enclosing_ns=record['enclosing_ns'],retained_bytes=record['retained_bytes'])
            report['pairs'].append(pair);hs.append(h);ss.append(s);risks.append(role_risks)
            del record,losses
        h=np.stack(hs);s=np.stack(ss)
        teacher=dict(index=i,teacher_id=report['pairs'][-1]['teacher_id'],h1_mean=float(h.mean()),static_mean=float(s.mean()),within_teacher_H1_variance=float(h.mean(axis=1).var(ddof=1)),
            paired_row_H1_variance=float(h.mean(axis=0).var(ddof=1)),within_teacher_static_mean_variance=float(s.mean(axis=1).var(ddof=1)),method_risks={r:float(np.mean([v[r] for v in risks])) for r in ROLES})
        old=scalars['teachers'][i]
        for k,key in [('h1_mean','teacher_h1'),('static_mean','teacher_static'),('within_teacher_H1_variance','within_variance'),('paired_row_H1_variance','row_variance')]:diff(key,teacher[k],old[k])
        report['teachers'].append(teacher);save();print('AUDITED',i,flush=True)
    report['method_means']={r:float(np.mean([t['method_risks'][r] for t in report['teachers']])) for r in ROLES}
    report['h1_mean']=float(np.mean([t['h1_mean'] for t in report['teachers']]));report['static_mean']=float(np.mean([t['static_mean'] for t in report['teachers']]))
    diff('summary_means',report['h1_mean'],summary['h1_mean']);diff('summary_means',report['static_mean'],summary['static_mean'])
    report.update(status='completed',counts=dict(teachers=16,repetitions=2,pairs=32,roles=256,probes=sum(p['probe_count'] for p in report['pairs'])),script_sha256=sha(__file__),command=[os.sys.executable,__file__])
except BaseException as e:report['status']='failed';report['failures'].append(traceback.format_exc());raise
finally:report['seconds']=time.perf_counter()-started;report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;save()
