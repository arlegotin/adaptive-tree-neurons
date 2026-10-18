#!/usr/bin/env python3
"""Network receipts only; no target parsing or modeling. Append-only attempts."""
import argparse, datetime, hashlib, json, os, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent
DATASETS = [
    (291, "airfoil+self+noise", "airfoil_self_noise.dat", "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"),
    (165, "concrete+compressive+strength", "Concrete_Data.xls", "710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4"),
    (294, "combined+cycle+power+plant", "CCPP/Folds5x2_pp.xlsx", "ccd490981db2a2f079963b3d9f0aea30d9d338900a0285428dfc6385396f4651"),
    (189, "parkinsons+telemonitoring", "parkinsons_updrs.data", "f2c7d5025dec4e92e7feae367a5f7ccf58789a10ac6b54bdf15976c599f9dd39"),
    (275, "bike+sharing+dataset", "hour.csv", "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"),
    (242, "energy+efficiency", "ENB2012_data.xlsx", "0089fcffc1415e41e2ff63730cca5280efe54fc43d722dfde1b0aaa808e35dc4"),
]
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def run(dataset_id, slug, kind):
    parent = ROOT / "downloads" / str(dataset_id) / kind
    parent.mkdir(parents=True, exist_ok=True)
    old = list(parent.glob("attempt-*"))
    if old:
        latest = sorted(old)[-1] / "receipt.json"
        if latest.exists() and json.loads(latest.read_text())["returncode"] == 0: return
    attempt = parent / ("attempt-%03d" % (len(old) + 1))
    attempt.mkdir(exist_ok=False)
    url = ("https://archive.ics.uci.edu/static/public/%d/%s.zip" if kind == "archive" else "https://archive.ics.uci.edu/dataset/%d/%s") % (dataset_id, slug)
    body = attempt / ("archive.zip" if kind == "archive" else "official-dataset-page.html")
    headers = attempt / "response-headers.txt"
    argv = ["curl", "--location", "--fail", "--silent", "--show-error", "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "15", "--max-time", "60", "--max-filesize", "33554432", "--dump-header", str(headers), "--output", str(body), "--write-out", "%{json}", url]
    start = now()
    proc = subprocess.run(argv, capture_output=True, check=False)
    (attempt / "stdout.txt").write_bytes(proc.stdout)
    (attempt / "stderr.txt").write_bytes(proc.stderr)
    try: transfer = json.loads(proc.stdout)
    except Exception: transfer = None
    receipt = {"dataset_id": dataset_id, "kind": kind, "url": url, "argv": argv, "cwd": os.getcwd(), "start_utc": start, "end_utc": now(), "returncode": proc.returncode, "body_relative_path": str(body.relative_to(ROOT)) if body.exists() else None, "body_bytes": body.stat().st_size if body.exists() else None, "body_sha256": sha(body) if body.exists() else None, "transfer": transfer, "stderr": proc.stderr.decode("utf-8", "replace"), "status": "downloaded" if proc.returncode == 0 else "download_failed", "requested_scope": "official UCI bytes and license source page; acquisition only", "execution_argv": sys.argv, "script_sha256": sha(pathlib.Path(__file__))}
    with (attempt / "receipt.json").open("x") as f: json.dump(receipt, f, indent=2, sort_keys=True); f.write("\n")
    print(json.dumps({"dataset_id": dataset_id, "kind": kind, "returncode": proc.returncode, "bytes": receipt["body_bytes"], "receipt": str((attempt / "receipt.json").relative_to(ROOT))}), flush=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-context", required=True)
    parser.parse_args()
    for dataset_id, slug, raw, expected in DATASETS:
        run(dataset_id, slug, "archive")
        run(dataset_id, slug, "license-source")
