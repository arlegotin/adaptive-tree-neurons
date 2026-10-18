#!/usr/bin/env python3
"""Opaque-byte ZIP validation/extraction only; no tables, targets, roles or models."""
import datetime, hashlib, io, json, ntpath, os, pathlib, re, stat, sys, zipfile
sys.dont_write_bytecode = True
from download import DATASETS, ROOT, sha
MAX_ENTRIES = 10000
MAX_UNCOMPRESSED = 268435456
MAX_MEMBER = 67108864
MAX_DEPTH = 3

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rel(path): return str(path.relative_to(ROOT))
def successful_receipt(dataset_id, kind):
    attempts = []
    for path in sorted((ROOT / "downloads" / str(dataset_id) / kind).glob("attempt-*/receipt.json")):
        item = json.loads(path.read_text())
        attempts.append({"receipt": rel(path), "status": item["status"], "returncode": item["returncode"], "url": item["url"], "body_relative_path": item["body_relative_path"], "body_bytes": item["body_bytes"], "body_sha256": item["body_sha256"]})
    good = [a for a in attempts if a["returncode"] == 0]
    return attempts, good[-1] if good else None

def safety(info):
    name = info.filename
    errors = []
    if not name or "\\" in name or "\x00" in name or name.startswith("/") or ntpath.splitdrive(name)[0]: errors.append("absolute/ambiguous/NUL/backslash member path")
    parts = name.rstrip("/").split("/")
    if any(p in ("", ".", "..") for p in parts): errors.append("empty/dot/traversal path component")
    mode = info.external_attr >> 16
    ftype = stat.S_IFMT(mode)
    if ftype not in (0, stat.S_IFREG, stat.S_IFDIR): errors.append("link or special file")
    if info.flag_bits & 1: errors.append("encrypted member")
    if info.file_size > MAX_MEMBER: errors.append("oversized member")
    return errors

def inventory(zf, chain, expected, inventory_rows, matches, security_errors, budget, depth=0):
    infos = zf.infolist()
    budget["entries"] += len(infos)
    budget["uncompressed_bytes"] += sum(info.file_size for info in infos)
    if budget["entries"] > MAX_ENTRIES or budget["uncompressed_bytes"] > MAX_UNCOMPRESSED: raise ValueError("ZIP inspection bound exceeded")
    names = set()
    for info in infos:
        errs = safety(info)
        if info.filename in names: errs.append("duplicate member name")
        names.add(info.filename)
        address = chain + [info.filename]
        row = {"member_chain": address, "member": info.filename, "directory": info.is_dir(), "compressed_bytes": info.compress_size, "uncompressed_bytes": info.file_size, "crc32": "%08x" % info.CRC, "validation_errors": errs, "disposition": "directory" if info.is_dir() else "unmatched_retained_in_archive"}
        if errs: security_errors.append({"member_chain": address, "errors": errs})
        inventory_rows.append(row)
        if info.filename == expected and not info.is_dir():
            row["disposition"] = "matched_needed_raw_candidate"
            matches.append((list(address), zf.read(info)))
        if info.filename.lower().endswith(".zip") and not info.is_dir() and not errs:
            if depth >= MAX_DEPTH: raise ValueError("nested ZIP depth bound exceeded")
            data = zf.read(info)
            row["disposition"] = "nested_zip_inspected_retained_in_archive"
            row["sha256"] = hashlib.sha256(data).hexdigest()
            with zipfile.ZipFile(io.BytesIO(data)) as child:
                inventory(child, address, expected, inventory_rows, matches, security_errors, budget, depth + 1)

def process(dataset_id, slug, expected, expected_sha):
    page_url = "https://archive.ics.uci.edu/dataset/%d/%s" % (dataset_id, slug)
    item = {"dataset_id": dataset_id, "slug": slug, "official_archive_url": "https://archive.ics.uci.edu/static/public/%d/%s.zip" % (dataset_id, slug), "expected_raw_member": expected, "expected_raw_sha256": expected_sha, "scope": "archive acquisition and opaque raw-byte hash verification only", "raw_values_parsed": False, "splits_created_or_sealed": False, "models_fitted": False, "zip_inventory": [], "security_errors": [], "matched_members": [], "unmatched_members": [], "missing_members": []}
    archive_attempts, archive = successful_receipt(dataset_id, "archive")
    page_attempts, page = successful_receipt(dataset_id, "license-source")
    item["archive_attempts"] = archive_attempts
    item["license_source_attempts"] = page_attempts
    item["license"] = {"license_id": "CC-BY-4.0", "source_url": page_url, "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode", "verification": "official UCI dataset page via browser and retained HTML", "source_receipt": page["receipt"] if page else None}
    if page:
        html = (ROOT / page["body_relative_path"]).read_text(errors="replace")
        item["license"]["retained_html_contains_license_url"] = "creativecommons.org/licenses/by/4.0" in html
        item["license"]["retained_html_sha256"] = sha(ROOT / page["body_relative_path"])
    if not archive:
        item["status"] = "archive_download_failed"
        item["missing_members"] = [expected]
        return item
    ap = ROOT / archive["body_relative_path"]
    item["archive"] = {"path": rel(ap), "bytes": ap.stat().st_size, "sha256": sha(ap)}
    matches = []
    budget = {"entries": 0, "uncompressed_bytes": 0}
    try:
        with zipfile.ZipFile(ap) as zf:
            inventory(zf, [], expected, item["zip_inventory"], matches, item["security_errors"], budget)
        item["inspection_totals"] = budget
        item["unmatched_members"] = [r["member_chain"] for r in item["zip_inventory"] if r["disposition"] == "unmatched_retained_in_archive"]
        if item["security_errors"]: raise ValueError("unsafe ZIP member(s); extraction refused")
        if len(matches) != 1:
            item["missing_members"] = [expected] if not matches else []
            item["status"] = "missing_expected_raw" if not matches else "ambiguous_expected_raw"
            return item
        chain, raw = matches[0]
        actual_sha = hashlib.sha256(raw).hexdigest()
        matched = actual_sha == expected_sha
        # Preserve changed bytes separately; never silently use bytes that miss the pin.
        destination = ROOT / ("raw" if matched else "changed-raw") / str(dataset_id) / expected
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as f: f.write(raw)
        if sha(destination) != actual_sha: raise ValueError("written raw hash changed")
        item["matched_members"] = [{"member_chain": chain, "path": rel(destination), "bytes": len(raw), "sha256": actual_sha, "expected_sha256": expected_sha, "hash_matches": matched, "crc_checked_by_zipfile_read": True}]
        for row in item["zip_inventory"]:
            if row["member_chain"] == chain: row["disposition"] = "extracted_raw_hash_matched" if matched else "changed_raw_bytes_preserved"
        item["status"] = "raw_hash_matched" if matched else "raw_hash_mismatch"
        item["processing_constraint"] = "CCPP Sheet1 selection is deferred; workbook bytes alone were hashed" if dataset_id == 294 else "schema, rows, group/split assignments and targets remain unparsed"
    except Exception as exc:
        item["status"] = "zip_validation_or_extraction_failed"
        item["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return item

if __name__ == "__main__":
    manifest_path = ROOT / "acquisition-manifest.json"
    if manifest_path.exists(): raise FileExistsError(manifest_path)
    start = now()
    datasets = [process(*row) for row in DATASETS]
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file())
    inventory_files = [{"path": rel(p), "bytes": p.stat().st_size, "sha256": sha(p), "allocated_bytes": p.stat().st_blocks * 512} for p in paths]
    total_before = sum(p.stat().st_size for p in paths)
    manifest = {"schema_version": 1, "purpose": "WP07 official UCI archive acquisition, secure exact-member extraction and pinned raw-byte verification only", "start_utc": start, "end_utc": now(), "argv": sys.argv, "cwd": os.getcwd(), "script_sha256": sha(pathlib.Path(__file__)), "design_source": "docs/superpowers/specs/2026-10-18-research-v2-design.md", "design_source_sha256_at_acquisition": sha(ROOT.parents[3] / "docs/superpowers/specs/2026-10-18-research-v2-design.md"), "prohibited_work_performed": {"target_value_parsing": False, "model_fitting": False, "prediction_inspection": False, "split_sealing": False, "WP07_full_qualification": False}, "zip_limits": {"max_entries": MAX_ENTRIES, "max_uncompressed_bytes": MAX_UNCOMPRESSED, "max_member_bytes": MAX_MEMBER, "max_nested_zip_depth": MAX_DEPTH}, "datasets": datasets, "summary": {"requested_datasets": len(DATASETS), "raw_hash_matches": sum(d["status"] == "raw_hash_matched" for d in datasets), "raw_hash_mismatches": sum(d["status"] == "raw_hash_mismatch" for d in datasets), "missing_expected_members": sum(len(d["missing_members"]) for d in datasets), "archive_bytes": sum(d.get("archive", {}).get("bytes", 0) for d in datasets), "extracted_raw_bytes": sum(r["bytes"] for d in datasets for r in d["matched_members"]), "unmatched_files_retained_in_archives": sum(len(d["unmatched_members"]) for d in datasets)}, "artifact_inventory_excluding_this_manifest": inventory_files, "disk_usage": {"existing_regular_file_count": len(paths), "existing_regular_file_apparent_bytes": total_before, "existing_regular_file_allocated_bytes": sum(p.stat().st_blocks * 512 for p in paths), "manifest_bytes": 0, "total_regular_file_count_including_manifest": len(paths) + 1, "total_regular_file_apparent_bytes_including_manifest": 0, "note": "File byte total includes this manifest exactly; allocated bytes exclude manifest and directories at pre-manifest snapshot. Whole-tree allocated disk usage verified separately after writing."}}
    for _ in range(20):
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        size = len(payload)
        if manifest["disk_usage"]["manifest_bytes"] == size and manifest["disk_usage"]["total_regular_file_apparent_bytes_including_manifest"] == total_before + size: break
        manifest["disk_usage"]["manifest_bytes"] = size
        manifest["disk_usage"]["total_regular_file_apparent_bytes_including_manifest"] = total_before + size
    else: raise RuntimeError("manifest byte accounting did not converge")
    with manifest_path.open("xb") as f: f.write(payload)
    actual_total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    if actual_total != manifest["disk_usage"]["total_regular_file_apparent_bytes_including_manifest"]: raise AssertionError("disk byte accounting mismatch")
    print(json.dumps({"manifest": str(manifest_path), "summary": manifest["summary"], "statuses": [{"id": d["dataset_id"], "status": d["status"], "matched": d["matched_members"], "unmatched": d["unmatched_members"], "missing": d["missing_members"]} for d in datasets], "total_regular_file_bytes": actual_total}, indent=2))
