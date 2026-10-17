"""Copy bounded original continuation evidence without modifying bulk records."""
import hashlib
import json
import shutil
from pathlib import Path

target = Path("audit/implementation/evidence/2026-10-17/complete-linked-pilot")
target.mkdir(parents=True, exist_ok=False)
bundle = Path("artifacts/h1-v2/user-pause-continuation/a6b986ed496b4f22be3a43c65943dc63")
env = Path("environment/h1-v2/user-pause-continuation")
entries = [
    (Path(".superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/scientific-user-pause-continuation-report.md"), "report.md"),
    (env / "inventory.json", "bulk-inventory.json"),
    (bundle / "continuation-manifest.json", "continuation-manifest.json"),
    (bundle / "joined-input-members.json", "joined-input-members.json"),
    (bundle / "joined-diagnostics/analysis-manifest.json", "joined-analysis-manifest.json"),
    (bundle / "joined-diagnostics/joined-summary-diagnostic.json", "joined-summary-diagnostic.json"),
    (bundle / "joined-diagnostics/joined-design-diagnostic.json", "joined-design-diagnostic.json"),
    (bundle / "joined-diagnostics/joined-scalars-and-risks.json", "joined-scalars-and-risks.json"),
    (env / "terminal-summary.json", "terminal-summary.json"),
    (env / "invocation.json", "first-failed-fit-invocation.json"),
    (env / "run.py", "first-failed-fit-wrapper.py"),
    (env / "run.stderr", "first-failed-fit.stderr"),
    (env / "attempt-2/run.py", "successful-fit-wrapper.py"),
    (env / "attempt-2/invocation.json", "successful-fit-invocation.json"),
    (env / "attempt-2/readout.json", "successful-fit-readout.json"),
    (env / "join.py", "first-failed-join.py"),
    (env / "join.stderr", "first-failed-join.stderr"),
    (env / "join-invocation.json", "first-failed-join-invocation.json"),
    (env / "join-v2.py", "successful-join.py"),
    (env / "join-v2-invocation.json", "successful-join-invocation.json"),
    (Path("environment/h1-v2/user-resume-root/continuation-verification.json"), "root-verification.json"),
    (Path("environment/h1-v2/user-resume-root/continuation_verification.py"), "root-verification.py"),
    (Path(__file__), "curate.py"),
]
items = []
for source, name in entries:
    content = source.read_bytes()
    destination = target / name
    assert not destination.exists()
    shutil.copyfile(source, destination)
    assert content == destination.read_bytes()
    items.append({"original_path": str(source), "archive_path": str(destination), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
value = {
    "scope": "Complete linked exploratory E pilot; original timeout preserved; no canonical freeze or confirmation",
    "teachers": 16, "repetitions": 2, "pairs": 32, "roles": 256, "B6_probes": 2080,
    "files": items, "bytes": sum(item["bytes"] for item in items),
    "original_inventory_sha256": "c0a46776b126cd4a30c1f586b25ea3c786155736b362322fccfb428f3c06f4ea",
    "new_inventory_sha256": "d984d8961a59dc5d996862a5361af30f28f5cc9b7c1a45e7be0c8a2fd33ff627",
    "canonical_freeze_eligible": False,
}
(target / "index.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps({"files": len(items), "bytes": value["bytes"], "index": str(target / "index.json")}))
