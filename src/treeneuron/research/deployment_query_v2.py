"""Deterministic target-free F workload projection; no native-role authority."""
import copy
import hashlib
import io
import os
from pathlib import Path
import zipfile

from .deployment_worker_v2 import _source,_digest,_json
from .protected_journal_v2 import _path,_reference,_hash,_write_new

_FALSE=dict(label_values_decoded=False,original_F_role_authenticated=False,native_models_authenticated=False,
    quality_precision_equivalence_authenticated=False,cost_claim_authorized=False,T_accessed=False)


def _metadata(value):
    from .protected_query_v2 import _feature_schema
    value=copy.deepcopy(value)
    fields={'schema','source_sha256','role','dataset','outer_fold','manifest_hash','split_hash','F_reference','F_bytes','feature_schema','temporal'}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-F-deployment-input'
            or value['source_sha256']!=_source() or value['role']!='F' or type(value['dataset']) is not str or not value['dataset']
            or type(value['outer_fold']) is not int or not 0<=value['outer_fold']<5
            or any(not _digest(value[k]) for k in ('manifest_hash','split_hash')) or type(value['temporal']) is not bool
            or type(value['F_bytes']) is not int or not 0<value['F_bytes']<=64*1024**2
            or type(value['F_reference']) is not dict or set(value['F_reference'])!={'path','sha256'}
            or not _digest(value['F_reference']['sha256'])):raise ValueError('Strict current-source F-only deployment input required')
    _feature_schema(value['feature_schema']);path=_path(value['F_reference']['path'])
    if path.name!='F.npz' or path.parent.name!=f"outer-{value['outer_fold']}":raise ValueError('Original explicit F archive required')
    return value


def _read_F_features(metadata):
    """Hash F archive bytes but decode only features and row IDs, never targets."""
    import numpy as np
    _,raw=_reference(metadata['F_reference'])
    if len(raw)!=metadata['F_bytes']:raise ValueError('F archive byte count changed')
    fields={'X.npy','y.npy','row_id.npy','group_id.npy'}|({'timestamps.npy'} if metadata['temporal'] else set())
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members=archive.infolist()
            if len(members)!=len(fields) or {m.filename for m in members}!=fields or sum(m.file_size for m in members)>64*1024**2:
                raise ValueError('Exact bounded F archive fields required; extra fields forbidden')
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:X=archive['X'].copy();rows=archive['row_id'].copy()
    except zipfile.BadZipFile as error:raise ValueError('Invalid F archive') from error
    if (X.dtype!=np.float64 or X.ndim!=2 or not 2<=len(X)<=20000 or X.shape[1]!=len(metadata['feature_schema']['columns'])
            or not np.isfinite(X).all() or rows.dtype!=np.int64 or rows.shape!=(len(X),) or len(np.unique(rows))!=len(rows)):
        raise ValueError('Finite native float64 F features and unique int64 rows required')
    return X,rows


def _ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _provenance(meta,query,source_rows):
    return dict(schema='research/v2-F-derived-deployment-query',source_sha256=meta['source_sha256'],source_F_reference=meta['F_reference'],
        query_reference=query,row_policy='cycle_original_F_order_to_1024',source_rows=source_rows,workload_rows=1024,
        source_role='F',original_role_verified_by_native_controller=False)


def build_F_workload_query(metadata,*,output):
    """Private array construction only; an actual frozen-role gate must wrap it."""
    import numpy as np
    from treeneuron.data.types import array_hash
    from .primary_program_v2 import _check_program_location
    meta=_metadata(metadata);root=_path(output);source=_path(meta['F_reference']['path'])
    _check_program_location(root,working_roots=[str(source.parent.parent)],proof_roots=[],input_files=[str(source)])
    if root.exists():raise ValueError('Fresh immutable F projection output required')
    X,rows=_read_F_features(meta);indices=np.arange(1024,dtype=np.int64)%len(X)
    try:
        with np.errstate(over='raise',invalid='raise'):values=X[indices].astype(np.float32)
    except FloatingPointError as error:raise ValueError('F features cannot be represented as finite float32 without clipping') from error
    if not np.isfinite(values).all():raise ValueError('Nonfinite float32 F workload')
    root.mkdir(parents=True,exist_ok=False);target=root/'F-deployment-features.npz';pending=root/'.F-deployment-features.npz.pending'
    with pending.open('xb') as stream:
        np.savez(stream,X=values,row_id=np.arange(1024,dtype=np.int64));stream.flush();os.fsync(stream.fileno())
    os.link(pending,target);pending.unlink()
    query=_ref(target);provenance=_provenance(meta,query,len(X));_write_new(root/'query-provenance.json',provenance)
    _reference(meta['F_reference'])
    if _source()!=meta['source_sha256']:raise ValueError('Source changed during F projection')
    manifest=dict(schema='research/v2-unqualified-F-deployment-projection',metadata=meta,query_reference=query,
        provenance_reference=_ref(root/'query-provenance.json'),source_shape=list(X.shape),workload_shape=list(values.shape),
        source_feature_array_sha256=array_hash(X),source_row_ids_sha256=array_hash(rows),
        workload_origin_row_ids_sha256=array_hash(rows[indices]),F_archive_bytes_hashed=True,**_FALSE,
        scope='Numeric F-derived feature projection only; actual native/controller role and quality/cost qualification remain required')
    _write_new(root/'projection.json',manifest)
    return dict(manifest_reference=_ref(root/'projection.json'),query_reference=query,provenance_reference=manifest['provenance_reference'],**_FALSE)


def inspect_F_workload_query(reference,*,expected_metadata):
    """Rederive from F without decoding labels or trusting projection flags."""
    import numpy as np
    from treeneuron.data.types import array_hash
    from .protected_label_evaluator_v2 import _npz
    meta=_metadata(expected_metadata);path,raw=_reference(reference);manifest=_json(raw)
    fields={'schema','metadata','query_reference','provenance_reference','source_shape','workload_shape','source_feature_array_sha256',
        'source_row_ids_sha256','workload_origin_row_ids_sha256','F_archive_bytes_hashed','scope',*_FALSE}
    if (path.name!='projection.json' or type(manifest) is not dict or set(manifest)!=fields
            or manifest['schema']!='research/v2-unqualified-F-deployment-projection' or manifest['metadata']!=meta
            or manifest['query_reference'].get('path')!=str(path.parent/'F-deployment-features.npz')
            or manifest['provenance_reference'].get('path')!=str(path.parent/'query-provenance.json')
            or manifest['F_archive_bytes_hashed'] is not True or any(manifest[k] is not False for k in _FALSE)):
        raise ValueError('Exact current-source F projection/role/ref metadata required')
    X,rows=_read_F_features(meta)
    # A separate construction checks the saved indexing/cast result.
    copies=(1024+len(X)-1)//len(X)
    try:
        with np.errstate(over='raise',invalid='raise'):expected=np.tile(X,(copies,1))[:1024].astype(np.float32)
    except FloatingPointError as error:raise ValueError('Unrepresentable float32 F verification workload') from error
    origin=np.tile(rows,copies)[:1024]
    _,raw=_reference(manifest['query_reference']);arrays=_npz(raw,['X','row_id'])
    if (arrays['X'].dtype!=np.float32 or arrays['X'].shape!=expected.shape or arrays['X'].tobytes()!=expected.tobytes()
            or arrays['row_id'].dtype!=np.int64 or not np.array_equal(arrays['row_id'],np.arange(1024,dtype=np.int64))
            or manifest['source_shape']!=list(X.shape) or manifest['workload_shape']!=list(expected.shape)
            or manifest['source_feature_array_sha256']!=array_hash(X) or manifest['source_row_ids_sha256']!=array_hash(rows)
            or manifest['workload_origin_row_ids_sha256']!=array_hash(origin)):
        raise ValueError('Saved F workload differs from exact deterministic source projection')
    _,raw=_reference(manifest['provenance_reference'])
    if _json(raw)!=_provenance(meta,manifest['query_reference'],len(X)):raise ValueError('F projection provenance changed')
    _reference(reference);_reference(meta['F_reference'])
    if _source()!=meta['source_sha256']:raise ValueError('Source changed during F projection verification')
    return dict(schema='research/v2-F-deployment-projection-readback',manifest_reference=reference,metadata=meta,
        query_reference=manifest['query_reference'],provenance_reference=manifest['provenance_reference'],numeric_projection_checked=True,**_FALSE,
        scope='Exact numeric/byte derivation only; actual frozen native F-role identity and timing/precision/quality/cost gates remain separate')
