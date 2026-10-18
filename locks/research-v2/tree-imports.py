"""Import-only qualification of the pinned optional native tree overlay.

Each baseline imports in a fresh process. Downloaded packages are never fitted;
this script does not load datasets or import treeneuron. The output directory
must be fresh. RECORD and core identity checks bracket all child processes.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import base64
import csv
import ctypes
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[2]
LOCKS = ROOT / 'locks/research-v2'
THREADS = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    PYTORCH_ENABLE_MPS_FALLBACK='0', PYTHONDONTWRITEBYTECODE='1')
CORE_MODULES = ('numpy', 'scipy', 'pandas', 'torch', 'sklearn', 'yaml',
                'packaging', 'narwhals', 'dateutil', 'six')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def inventory():
    return sorted([dict(name=d.metadata['Name'], version=d.version,
        path=str(d._path), record_sha256=hashlib.sha256((d.read_text('RECORD') or '').encode()).hexdigest(),
        metadata_sha256=sha(Path(d._path) / 'METADATA'))
        for d in importlib.metadata.distributions()], key=lambda x: canonicalize_name(x['name']))


def origins():
    return {name: importlib.util.find_spec(name).origin for name in CORE_MODULES}


def validate_environment(overlay):
    frozen = json.loads((LOCKS / 'core.json').read_text())
    resolution = json.loads((LOCKS / 'tree-resolution.json').read_text())
    expected_core = {canonicalize_name(x['name']): x for x in frozen['distributions']}
    expected_overlay = {x['name']: x for x in resolution['wheels']}
    installed = inventory()
    groups = {}
    for item in installed:
        groups.setdefault(canonicalize_name(item['name']), []).append(item)
    duplicates = {name: values for name, values in groups.items() if len(values) != 1}
    if duplicates:
        raise ValueError(f'Duplicate distributions: {sorted(duplicates)}')
    if set(groups) != set(expected_core) | set(expected_overlay):
        raise ValueError('Import environment differs from the exact core plus tree overlay')
    for name, expected in expected_core.items():
        actual = groups[name][0]
        if (actual['version'] != expected['version']
                or actual['record_sha256'] != expected['record_sha256']
                or actual['path'] != expected['path']):
            raise ValueError(f'Frozen core identity changed: {name}')
    checked = []
    claimed_paths = set()
    for name, expected in sorted(expected_overlay.items()):
        dist = importlib.metadata.distribution(name)
        actual = groups[name][0]
        if (actual['version'] != expected['version']
                or not Path(actual['path']).resolve().is_relative_to(overlay)):
            raise ValueError(f'Overlay import/distribution identity differs: {name}')
        if actual['metadata_sha256'] != expected['wheel_metadata_sha256']:
            raise ValueError(f'Installed METADATA differs from verified wheel: {name}')
        record = dist.read_text('RECORD')
        if record is None:
            raise ValueError(f'Installed RECORD absent: {name}')
        records = []
        for relative, checksum, size in csv.reader(io.StringIO(record)):
            path = Path(dist.locate_file(relative)).resolve()
            if not path.is_relative_to(overlay) or not path.is_file():
                raise ValueError(f'Installed RECORD path absent or escaped overlay: {name}: {relative}')
            claimed_paths.add(path)
            if not checksum or not size:
                if not relative.endswith('.dist-info/RECORD') or checksum or size:
                    raise ValueError(f'Unexpected unhashed installed file: {name}: {relative}')
                records.append(dict(path=relative, bytes=path.stat().st_size,
                    sha256=sha(path), record_self_hash_not_declared=True))
                continue
            algorithm, encoded = checksum.split('=', 1)
            if algorithm != 'sha256':
                raise ValueError('Unexpected installed digest algorithm')
            actual_digest = sha(path)
            wanted_digest = base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)).hex()
            if actual_digest != wanted_digest or path.stat().st_size != int(size):
                raise ValueError(f'Installed RECORD digest/size mismatch: {name}: {relative}')
            records.append(dict(path=relative, bytes=int(size), sha256=actual_digest))
        checked.append(dict(name=name, version=dist.version, distribution_path=actual['path'],
            record_sha256=actual['record_sha256'], records=records,
            records_checked=len(records), installed_file_bytes=sum(x['bytes'] for x in records)))
    unowned = [str(path.relative_to(overlay)) for path in overlay.rglob('*')
        if path.is_file() and path.resolve() not in claimed_paths and path.name != '.lock']
    if unowned:
        raise ValueError(f'Unowned overlay files: {unowned[:10]}')
    return dict(inventory=installed, duplicate_distributions=duplicates,
        core_count=len(expected_core), overlay_count=len(expected_overlay), core_import_origins=origins(),
        overlay_records=checked, unowned_files=unowned,
        tree_resolution_sha256=sha(LOCKS / 'tree-resolution.json'),
        requirements_sha256=sha(LOCKS / 'tree-requirements.txt'),
        core_lock_sha256=sha(ROOT / 'uv.lock'))


def dyld_images():
    process = ctypes.CDLL(None)
    count = process._dyld_image_count
    count.restype = ctypes.c_uint32
    name = process._dyld_get_image_name
    name.argtypes = [ctypes.c_uint32]
    name.restype = ctypes.c_char_p
    return [name(i).decode('utf-8') for i in range(count())]


def worker(method, output, overlay):
    before = inventory()
    image_names_before = set(dyld_images())
    torch_before = any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
    started = time.perf_counter_ns()
    module = importlib.import_module(method)
    import_ns = time.perf_counter_ns() - started
    torch_after = any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
    # threadpoolctl inspects the loaded libraries; it does not import Torch or fit a model.
    from threadpoolctl import threadpool_info
    pools = threadpool_info()
    images = []
    relevant = []
    baseline_root = overlay / method
    pool_paths = {str(Path(item['filepath']).resolve()) for item in pools if item.get('filepath')}
    for raw in dyld_images():
        path = Path(raw)
        system = raw.startswith(('/System/Library/', '/usr/lib/'))
        entry = dict(dyld_path=raw, loaded_during_import=raw not in image_names_before,
            system_shared_cache_candidate=system, exists_on_filesystem=path.is_file())
        if path.is_file() and not system:
            entry.update(resolved_path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))
        else:
            entry['hash_unavailable_reason'] = 'Apple system/shared-cache image' if system else 'No readable file at dyld path'
        images.append(entry)
        is_omp = any(token in path.name.lower() for token in ('libomp', 'libiomp', 'libgomp'))
        if path.is_file() and (path.resolve().is_relative_to(baseline_root)
                or is_omp or str(path.resolve()) in pool_paths):
            argv = ['/usr/bin/otool', '-L', str(path.resolve())]
            linked = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            relevant.append(dict(**entry, openmp_runtime_name=is_omp,
                otool=dict(argv=argv, exit_code=linked.returncode, output=linked.stdout)))
            if linked.returncode != 0:
                raise ValueError(f'otool failed for loaded native library: {path}')
    omp = {item['resolved_path']: item for item in relevant if item['openmp_runtime_name']}
    modules = {}
    for name in (method, *CORE_MODULES, 'matplotlib', 'plotly', 'graphviz'):
        imported = sys.modules.get(name)
        if imported is not None:
            modules[name] = dict(origin=getattr(imported, '__file__', None),
                version=getattr(imported, '__version__', None))
    after = inventory()
    expected_version = importlib.metadata.version(method)
    if getattr(module, '__version__', None) != expected_version:
        raise ValueError('Method version disagrees with distribution metadata')
    if not Path(module.__file__).resolve().is_relative_to(baseline_root):
        raise ValueError('Method import escaped pinned overlay')
    if before != after:
        raise ValueError('Distribution metadata changed during import')
    record = dict(schema='research/v2-native-tree-import', method=method,
        status='imported_pending_model_fixture_and_anchor_qualification',
        argv=sys.argv, cwd=str(Path.cwd()), executable=sys.executable, python=sys.version,
        platform=platform.platform(), environment_overrides={key: os.environ.get(key)
            for key in (*THREADS, 'PYTHONPATH', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'MPLBACKEND')},
        module_origin=module.__file__, module_version=module.__version__,
        imported_modules=modules, imported_torch_before=torch_before, imported_torch_after=torch_after,
        import_body_ns=import_ns, peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_units='macOS ru_maxrss bytes, process peak including interpreter and imports',
        threadpools=pools, dyld_images=images, relevant_native_libraries=relevant,
        distinct_loaded_openmp_paths=sorted(omp),
        multiple_openmp_libraries_require_review=len(omp) > 1,
        distribution_inventory_unchanged=before == after, core_import_origins=origins(),
        protected_data_accessed=False, models_constructed=False, fits_executed=False,
        completed_utc_ns=time.time_ns())
    write(output / 'import.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--worker', choices=('catboost', 'xgboost', 'lightgbm'))
    args = parser.parse_args()
    output, overlay = args.output.resolve(), args.overlay.resolve()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('This import receipt requires the qualified macOS arm64 runtime')
    if args.worker:
        worker(args.worker, output, overlay)
        return
    output.mkdir(parents=True, exist_ok=False)
    started = time.time_ns()
    before = validate_environment(overlay)
    write(output / 'environment-before.json', before)
    attempts = []
    for method in ('catboost', 'xgboost', 'lightgbm'):
        method_output = output / method
        method_output.mkdir()
        env = dict(os.environ)
        # Never inherit the unsupported OpenMP conflict-suppression workaround.
        env.pop('KMP_DUPLICATE_LIB_OK', None)
        env.update(THREADS)
        env.update(PYTHONPATH=str(overlay), MPLCONFIGDIR=str(method_output / 'mpl-config'),
            XDG_CACHE_HOME=str(method_output / 'cache'), MPLBACKEND='Agg')
        argv = [sys.executable, str(Path(__file__).resolve()), '--worker', method,
            '--output', str(method_output), '--overlay', str(overlay)]
        begin = time.perf_counter_ns()
        with (method_output / 'process.log').open('x') as log:
            process = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt = dict(method=method, argv=argv, cwd=str(ROOT),
            environment_overrides={**{key: env[key] for key in (*THREADS, 'PYTHONPATH', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'MPLBACKEND')},
                'KMP_DUPLICATE_LIB_OK': None},
            complete_process_wall_ns=time.perf_counter_ns() - begin, exit_code=process.returncode,
            log_sha256=sha(method_output / 'process.log'),
            status='completed_import' if process.returncode == 0 else 'input_or_dependency_failure')
        if process.returncode == 0:
            receipt['import_record_sha256'] = sha(method_output / 'import.json')
        write(method_output / 'command.json', receipt)
        attempts.append(receipt)
    after = validate_environment(overlay)
    write(output / 'environment-after.json', after)
    unchanged = before == after
    if not unchanged:
        raise ValueError('Installed overlay/core changed across imports')
    failures = [x for x in attempts if x['exit_code'] != 0]
    reviews = [x['method'] for x in attempts if x['exit_code'] == 0
        and json.loads((output / x['method'] / 'import.json').read_text())['multiple_openmp_libraries_require_review']]
    result = dict(schema='research/v2-native-tree-import-suite',
        status='dependency_and_import_qualified_only' if not failures and not reviews else 'qualification_incomplete',
        argv=sys.argv, cwd=str(Path.cwd()), interpreter=sys.executable, source_script_sha256=sha(__file__),
        overlay=str(overlay), started_utc_ns=started, completed_utc_ns=time.time_ns(),
        environment_before_sha256=sha(output / 'environment-before.json'),
        environment_after_sha256=sha(output / 'environment-after.json'),
        installed_environment_unchanged=unchanged, core_count=before['core_count'],
        overlay_count=before['overlay_count'], attempts=attempts,
        import_failures=[x['method'] for x in failures], multiple_openmp_review_methods=reviews,
        protected_data_accessed=False, models_constructed=False, fits_executed=False,
        remaining='Native model/API/categorical/stopping/save-load/recovery fixtures and eligible full anchor/workload qualification; Gate D pending')
    write(output / 'suite.json', result)
    print(json.dumps(result, indent=2))
    if failures or reviews:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
