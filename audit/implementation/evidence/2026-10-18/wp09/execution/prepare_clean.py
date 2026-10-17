"""Restore only engineering reference and accepted receipt into disposable clone."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path.cwd();CLEAN=Path('/private/tmp/h1-wp09-clean-7a63114');B=ROOT/'environment/h1-v2/wp09'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=Path('environment/h1-v2/backend.json');config=Path('environment/h1-v2/complete-resource/candidate.json')
teacher=Path('artifacts/h1-v2/complete-resource/pilot-5f6c0cc620bd441db53ed126157fbedf/teacher-0')
selected=[receipt,config,teacher/'observations.npz',teacher/'teacher.json',teacher/'splits.json']
selected += [p.relative_to(ROOT) for p in (ROOT/teacher/'repetition-0').rglob('*') if p.is_file()]
inv={x['path']:x['sha256'] for x in json.loads((ROOT/'environment/h1-v2/complete-resource/inventory.json').read_text())['files']}
files=[]
for rel in selected:
    p=ROOT/rel;h=sha(p)
    if str(rel) in inv:assert h==inv[str(rel)]
    q=CLEAN/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==h
    files.append(dict(path=str(rel),sha256=h,bytes=p.stat().st_size,inventory_bound=str(rel) in inv))
egg=CLEAN/'src/treeneuron.egg-info';dest=CLEAN/'environment/install-metadata/wp09/treeneuron.egg-info';metadata=[]
if egg.exists():
    metadata=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in egg.iterdir() if p.is_file()]
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(egg),dest)
report=dict(root=str(ROOT),clean=str(CLEAN),head=subprocess.check_output(['git','-C',str(CLEAN),'rev-parse','HEAD'],text=True).strip(),
    sync_command=['/Users/artemlegotin/.local/bin/uv','sync','--offline','--frozen','--cache-dir',str(ROOT/'.uv-cache'),'--python',str(ROOT/'.uv-python/cpython-3.12.13-macos-aarch64-none/bin/python3.12')],
    sync_exit_code=0,files=files,logical_restored_bytes=sum(f['bytes'] for f in files),physical_du_kib=subprocess.check_output(['du','-sk',str(CLEAN)],text=True).strip(),disk_free_bytes=shutil.disk_usage(CLEAN).free,
    install_metadata=dict(old_path='src/treeneuron.egg-info',new_path=str(dest.relative_to(CLEAN)),files=metadata),script_sha256=sha(Path(__file__)))
assert report['disk_free_bytes']>10*1024**3
(B/'clean-restoration.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('files','install_metadata')},indent=2))
