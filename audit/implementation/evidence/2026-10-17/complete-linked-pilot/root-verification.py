"""Independent root binding check; raw-prediction recomputation is WP09."""
import hashlib
import json
import math
import time
from pathlib import Path

started = time.perf_counter()
root = Path.cwd()
bundle = root / "artifacts/h1-v2/user-pause-continuation/a6b986ed496b4f22be3a43c65943dc63"
errors = []


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(condition, description):
    if not condition:
        errors.append(description)


inventory_path = root / "environment/h1-v2/user-pause-continuation/inventory.json"
check(sha(inventory_path) == "d984d8961a59dc5d996862a5361af30f28f5cc9b7c1a45e7be0c8a2fd33ff627", "new inventory identity")
inventory = json.loads(inventory_path.read_text())
for item in inventory["files"]:
    p = root / item["path"]
    check(p.is_file() and p.stat().st_size == item["bytes"] and sha(p) == item["sha256"], item["path"])
check(len(inventory["files"]) == inventory["count"] == 1995, "inventory count")
check(sum(i["bytes"] for i in inventory["files"]) == inventory["bytes"] == 741223772, "inventory bytes")
original = root / "artifacts/h1-v2/scientific-pilot/pilot-fe9efe02afc2444f94bd4aa6511cb76d"
for p, digest in [
    (original / "raw.json", "7478e09a4369292642bf4de55c9fe8634558067e89b8ab87f7b5ea354123d716"),
    (original / "raw-seal.json", "48fbb78a29c4caa1ffb776f929326481d4a911554e64bf6c83ae8304a78f9842"),
    (root / "environment/h1-v2/genuine-pilot/inventory.json", "c0a46776b126cd4a30c1f586b25ea3c786155736b362322fccfb428f3c06f4ea"),
]:
    check(sha(p) == digest, "original unchanged: " + str(p))

index = json.loads((bundle / "joined-input-members.json").read_text())
scalars = json.loads((bundle / "joined-diagnostics/joined-scalars-and-risks.json").read_text())
summary = json.loads((bundle / "joined-diagnostics/joined-summary-diagnostic.json").read_text())
check(not index["canonical_freeze_eligible"] and not summary["canonical_freeze_eligible"], "noncanonical scope")
check([(i["teacher_index"], i["repetition"]) for i in index["members"]] == [(t, r) for t in range(16) for r in range(2)], "exact prescribed ordered membership")
check(len(scalars["teachers"]) == 16 and len(scalars["pair_risks"]) == 32, "scalar counts")
roles = {"keep", "top", "random", "greedy", "widen_upper", "ordinary_lower", "ordinary_upper", "static_partial"}
max_mean_error = 0.0
for item in index["members"]:
    t, r = item["teacher_index"], item["repetition"]
    p = root / item["path"]
    check(sha(p) == item["sha256"], "pair hash: " + str((t, r)))
    check((str(p).startswith(str(original))) == (t < 15), "old/new boundary: " + str((t, r)))
    pair = json.loads(p.read_text())
    saved = scalars["teachers"][t]["exact_pair_means"][r]
    check(saved["path"] == item["path"] and saved["sha256"] == item["sha256"], "scalar source: " + str((t, r)))
    check(pair["repetition"] == r and pair["teacher_id"] == saved["teacher_id"] and pair["optimizer_seed"] == saved["optimizer_seed"], "pair identity: " + str((t, r)))
    check(pair["status"] == "completed" and not pair["failures"] and not pair["T_accessed"] and not pair["V_accessed"], "pair status: " + str((t, r)))
    check(pair["source_sha256"] == "d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c" and pair["recipe_hash"] == "9152a59bb3180da35f780b44a78d9eee59f7efa3e1af9677e4acf307a1bbb073", "source/config: " + str((t, r)))
    check(set(pair["methods"]) == roles, "role set: " + str((t, r)))
    for rows_key, mean_key in [("h1_rows", "h1_mean"), ("static_rows", "static_mean")]:
        rows = pair[rows_key]
        check(len(rows) == saved["evaluation_rows"] == 262144 and all(math.isfinite(v) for v in rows), "finite row counts: " + str((t, r, rows_key)))
        delta = abs(math.fsum(rows) / len(rows) - saved[mean_key])
        max_mean_error = max(max_mean_error, delta)
        check(delta <= 1e-12, "saved row mean: " + str((t, r, rows_key)))
    del pair
check(abs(math.fsum(t["h1_mean"] for t in scalars["teachers"]) / 16 - summary["summary"]["h1_mean"]) <= 1e-12, "joined H1 mean")
result = {
    "scope": "Independent root rehash and source/membership/saved-row-mean check; not raw prediction audit",
    "new_inventory_count": inventory["count"], "new_inventory_bytes": inventory["bytes"],
    "original_raw_seal_inventory_unchanged": True if not errors else None,
    "exact_pairs": 32, "max_saved_row_mean_abs_error": max_mean_error,
    "errors": errors, "elapsed_seconds": time.perf_counter() - started,
}
out = root / "environment/h1-v2/user-resume-root/continuation-verification.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
assert not errors
