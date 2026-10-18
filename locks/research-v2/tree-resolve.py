"""Verify/download the resolved macOS tree wheels without installing any package.

Run with the qualified core interpreter and a fresh --output scratch directory.
The uv resolution is a separate retained input; this script selects compatible
wheels, verifies their actual metadata/dependency closure, and writes a narrow
overlay lock. It never extracts, imports, or installs a downloaded package.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import email.parser
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import time
import urllib.request
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]
LOCKS = ROOT / 'locks/research-v2'
SOURCE_PINS = {
    'catboost': ('catboost/catboost', 'v1.2.10', 'b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084'),
    'xgboost': ('dmlc/xgboost', 'v3.4.1', '6fe8c547bdd21c73e4555d85b087d9260595d30d'),
    'lightgbm': ('lightgbm-org/LightGBM', 'v4.7.0', '8f7036f03627054d5a54a6f965b13f4b9ff2cb63'),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def core_snapshot():
    values = []
    for dist in importlib.metadata.distributions():
        tops = (dist.read_text('top_level.txt') or '').splitlines()
        origins = {}
        for top in tops:
            if top.isidentifier():
                spec = importlib.util.find_spec(top)
                origins[top] = None if spec is None else spec.origin
        values.append(dict(name=dist.metadata['Name'], version=dist.version,
            path=str(dist._path), record_sha256=sha((dist.read_text('RECORD') or '').encode()),
            metadata_sha256=sha((dist.read_text('METADATA') or '').encode()),
            top_level_module_origins=origins))
    return sorted(values, key=lambda x: canonicalize_name(x['name']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise SystemExit('This lock is qualified for CPython 3.12 on macOS arm64 only')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'metadata').mkdir()
    wheelhouse = output / 'wheels'
    wheelhouse.mkdir()
    events = output / 'acquisition-events.jsonl'
    before = core_snapshot()
    frozen = json.loads((LOCKS / 'core.json').read_text())
    frozen_names = {canonicalize_name(x['name']): x for x in frozen['distributions']}
    actual_names = {canonicalize_name(x['name']): x for x in before}
    if len(actual_names) != len(before) or set(actual_names) != set(frozen_names):
        raise ValueError('Core distribution inventory differs from WP00; no overlay resolution admitted')
    for name, expected in frozen_names.items():
        actual = actual_names[name]
        if actual['version'] != expected['version'] or actual['record_sha256'] != expected['record_sha256']:
            raise ValueError(f'Core identity changed: {name}')
    dump(output / 'core-before.json', before)

    def event(value):
        with events.open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def fetch(url, destination):
        if not url.startswith(('https://pypi.org/', 'https://files.pythonhosted.org/', 'https://api.github.com/')):
            raise ValueError(f'Unexpected asset host: {url}')
        if destination.exists():
            raise ValueError('Acquisition destination exists')
        for attempt in range(3):
            started = time.time_ns()
            temporary = destination.with_name(destination.name + f'.attempt-{attempt}.partial')
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'adaptive-tree-neurons-WP07-lock/2'})
                with urllib.request.urlopen(request, timeout=30) as response, temporary.open('xb') as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                    status = response.status
                temporary.rename(destination)
                event(dict(url=url, destination=str(destination), attempt=attempt,
                    started_utc_ns=started, ended_utc_ns=time.time_ns(), status=status,
                    bytes=destination.stat().st_size, sha256=sha(destination.read_bytes())))
                return destination.read_bytes()
            except Exception as error:
                event(dict(url=url, destination=str(destination), attempt=attempt,
                    started_utc_ns=started, ended_utc_ns=time.time_ns(),
                    status='input_or_dependency_failure', error=repr(error),
                    retained_partial=str(temporary) if temporary.exists() else None))
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    resolved_path = LOCKS / 'tree-resolved-all.txt'
    pins = {}
    for match in re.finditer(r'^([A-Za-z][A-Za-z0-9_.-]*)==([^\s\\]+)', resolved_path.read_text(), re.M):
        name, version = canonicalize_name(match[1]), match[2]
        if name in pins:
            raise ValueError('Duplicate resolved package')
        pins[name] = version
    if not pins or any(pins.get(name) != tag[1:] for name, (_, tag, _) in SOURCE_PINS.items()):
        raise ValueError('Resolved direct package pins changed')
    for name in set(pins) & set(frozen_names):
        if pins[name] != frozen_names[name]['version']:
            raise ValueError(f'Resolution would replace frozen core {name}')

    source_records = {}
    for name, (repo, tag, expected_commit) in SOURCE_PINS.items():
        url = f'https://api.github.com/repos/{repo}/git/ref/tags/{tag}'
        ref = json.loads(fetch(url, output / 'metadata' / f'{name}-source-ref.json'))
        obj = ref['object']
        if obj['type'] == 'tag':
            annotated = json.loads(fetch(obj['url'], output / 'metadata' / f'{name}-source-tag.json'))
            obj = annotated['object']
        if obj['type'] != 'commit' or obj['sha'] != expected_commit:
            raise ValueError(f'Official source tag moved for {name}')
        source_records[name] = dict(repository=f'https://github.com/{repo}', tag=tag,
            commit=expected_commit, commit_url=f'https://github.com/{repo}/tree/{expected_commit}')

    tag_priority = {tag: i for i, tag in enumerate(sys_tags())}
    wheels, metadata = [], {}
    for name, version in sorted(pins.items()):
        if name in frozen_names:
            dist = importlib.metadata.distribution(name)
            metadata[name] = email.parser.Parser().parsestr(dist.read_text('METADATA'))
            continue
        release_url = f'https://pypi.org/pypi/{name}/{version}/json'
        release_bytes = fetch(release_url, output / 'metadata' / f'{name}-{version}-pypi.json')
        release = json.loads(release_bytes)
        candidates = []
        for item in release['urls']:
            if item['packagetype'] != 'bdist_wheel' or item['yanked']:
                continue
            parsed_name, parsed_version, _, tags = parse_wheel_filename(item['filename'])
            ranks = [tag_priority[tag] for tag in tags if tag in tag_priority]
            if parsed_name == name and parsed_version == Version(version) and ranks:
                candidates.append((min(ranks), item['filename'], item))
        if not candidates:
            raise ValueError(f'No non-yanked CPython 3.12 macOS arm64 wheel: {name}=={version}')
        item = min(candidates)[2]
        path = wheelhouse / item['filename']
        data = fetch(item['url'], path)
        if sha(data) != item['digests']['sha256'] or len(data) != item['size']:
            raise ValueError(f'Official wheel identity differs: {path.name}')
        with zipfile.ZipFile(path) as archive:
            meta_paths = [p for p in archive.namelist() if p.endswith('.dist-info/METADATA')]
            if len(meta_paths) != 1:
                raise ValueError('Wheel has ambiguous METADATA')
            raw_metadata = archive.read(meta_paths[0])
            meta = email.parser.BytesParser().parsebytes(raw_metadata)
            if canonicalize_name(meta['Name']) != name or meta['Version'] != version:
                raise ValueError('Wheel METADATA differs from selected identity')
            metadata[name] = meta
            license_files = [dict(path=p, bytes=len(archive.read(p)), sha256=sha(archive.read(p)))
                for p in archive.namelist() if not p.endswith('/') and
                any(Path(p).name.upper().startswith(prefix) for prefix in ('LICENSE', 'COPYING', 'NOTICE'))]
            uncompressed = sum(info.file_size for info in archive.infolist())
        wheels.append(dict(name=name, version=version, filename=path.name, url=item['url'],
            sha256=sha(data), bytes=len(data), uncompressed_bytes=uncompressed,
            uploaded_utc=item['upload_time_iso_8601'], pypi_metadata_url=release_url,
            pypi_metadata_sha256=sha(release_bytes), wheel_metadata_sha256=sha(raw_metadata),
            requires_python=meta.get('Requires-Python'), requires_dist=meta.get_all('Requires-Dist', []),
            license_expression=None if meta.get('License-Expression') is None else str(meta.get('License-Expression')),
            license=None if meta.get('License') is None else str(meta.get('License')),
            license_classifiers=[x for x in meta.get_all('Classifier', []) if x.startswith('License ::')],
            license_files=license_files, project_urls=release['info']['project_urls'],
            source=source_records.get(name)))

    available = {name: entry['version'] for name, entry in frozen_names.items()}
    available.update({entry['name']: entry['version'] for entry in wheels})
    environment = default_environment()
    environment['extra'] = ''
    edges = []
    for name, meta in metadata.items():
        python_req = meta.get('Requires-Python')
        if python_req:
            from packaging.specifiers import SpecifierSet
            if not SpecifierSet(python_req).contains(environment['python_full_version']):
                raise ValueError(f'{name}: interpreter not accepted by actual wheel metadata')
        for raw in meta.get_all('Requires-Dist', []):
            req = Requirement(raw)
            admitted = req.marker is None or req.marker.evaluate(environment)
            target = canonicalize_name(req.name)
            if admitted:
                if req.extras or req.url:
                    raise ValueError(f'Unexpected active extra/direct URL dependency: {name}: {raw}')
                if target not in available or not req.specifier.contains(available[target]):
                    raise ValueError(f'Incomplete/incompatible dependency: {name}: {raw}')
                if target not in pins:
                    raise ValueError(f'Active dependency absent from uv closure: {name}: {raw}')
            edges.append(dict(package=name, requirement=raw, marker_active=admitted,
                satisfied_by=('frozen_core' if target in frozen_names else 'tree_overlay') if admitted else None,
                resolved_version=available.get(target) if admitted else None))

    after = core_snapshot()
    dump(output / 'core-after.json', after)
    if before != after:
        raise ValueError('Core import/distribution identities changed during lock acquisition')
    narrow = ['# CPython 3.12 / macOS arm64 optional tree overlay only.',
        '# Install with --no-deps --require-hashes; core requirements are separate and validated.',
        '# Each hash admits only the actually downloaded compatible wheel in tree-resolution.json.']
    for item in wheels:
        narrow.extend([f"# {item['filename']}", f"# {item['url']}",
            f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}"])
    narrow_text = '\n'.join(narrow) + '\n'
    (LOCKS / 'tree-requirements.txt').write_text(narrow_text)
    libomp = Path('/opt/homebrew/opt/libomp/lib/libomp.dylib')
    libomp_record = None if not libomp.exists() else dict(path=str(libomp.resolve()),
        sha256=sha(libomp.read_bytes()), bytes=libomp.stat().st_size,
        status='existing_system_library_not_loaded_or_installed_by_this_task')
    download = sum(x['bytes'] for x in wheels)
    unpacked = sum(x['uncompressed_bytes'] for x in wheels)
    record = dict(schema='research/v2-tree-lock-resolution',
        status='resolved_downloaded_verified_not_installed_not_qualified',
        argv=sys.argv, cwd=str(Path.cwd()), environment_overrides={'PYTHONDONTWRITEBYTECODE': '1'},
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        root=str(ROOT), scratch=str(output), wheelhouse=str(wheelhouse),
        uv_resolution_sha256=sha(resolved_path.read_bytes()),
        core_lock_sha256=sha((ROOT / 'uv.lock').read_bytes()),
        core_receipt_sha256=sha((LOCKS / 'core.json').read_bytes()),
        core_constraints_sha256=sha((LOCKS / 'tree-core-constraints.txt').read_bytes()),
        narrow_requirements_sha256=sha(narrow_text.encode()),
        source_pins=source_records, resolved_versions=pins,
        core_reused=[actual_names[name] for name in sorted(set(pins) & set(frozen_names))],
        core_snapshot_before=before, core_snapshot_after=after,
        core_unchanged=before == after, wheels=wheels, dependencies=edges,
        downloads_bytes=download, wheel_uncompressed_bytes=unpacked,
        projected_overlay_bytes_with_15pct_and_16MiB_margin=int(unpacked * 1.15) + 16 * 1024**2,
        projected_extra_install_transaction_bytes_excluding_retained_wheels=int(unpacked * 2.3) + 32 * 1024**2,
        projection_limitations='Wheel member sizes plus explicit planning margins; no installed-size/RSS/runtime measurement; no .pyc compilation requested',
        libomp=libomp_record, acquisition_events_sha256=sha(events.read_bytes()),
        acquisition_events=str(events), protected_data_accessed=False,
        packages_installed=False, fits_executed=False, completed_utc_ns=time.time_ns())
    dump(output / 'resolution.json', record)
    dump(LOCKS / 'tree-resolution.json', record)
    print(json.dumps(dict(status=record['status'], overlay_packages=len(wheels),
        reused_core_packages=len(record['core_reused']), downloads_bytes=download,
        uncompressed_bytes=unpacked, wheelhouse=str(wheelhouse),
        narrow_lock_sha256=record['narrow_requirements_sha256'])))


if __name__ == '__main__':
    main()
