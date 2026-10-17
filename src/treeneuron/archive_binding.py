"""Immutable terminal eligibility and prospective portable archive bindings.

These receipts grant no evaluation or execution authority. Archive inventories
are validated directly, without recursively invoking a live audit.
"""
from pathlib import Path
from .artifacts import _atomic_json
from .config import canonical_hash
from .protocol import _read, _hash, _keys, _relative


def completion_record(directory, protocol_hash):
    directory=Path(directory); ledger=directory/'confirmation.json'
    value=_read(ledger)
    if value.get('status') not in ('completed','incomplete','resource_limited'):
        if (directory/'archive-completion.json').exists(): raise ValueError('Archive completion is no longer terminal')
        return
    if value['protocol_hash']!=protocol_hash: raise ValueError('Archive completion protocol differs')
    record=dict(schema_version=1,protocol_hash=protocol_hash,confirmation_sha256=_hash(ledger),
                terminal_status=value['status'],status='unsealed',
                reason='No verified prospective archive supplied at completion')
    path=directory/'archive-completion.json'
    if path.exists():
        if _read(path)!=record: raise ValueError('Archive completion identity differs')
    else: _atomic_json(path,record)


def _verified_binding(directory, root, protocol_hash, value):
    from .analysis.audit import _read_archive
    _keys(value,('schema_version','protocol_hash','confirmation_sha256','archive_sha256','scope','path','reason'),
          'archive binding')
    if (value['schema_version']!=1 or value['protocol_hash']!=protocol_hash
            or value['confirmation_sha256']!=_hash(directory/'confirmation.json')):
        raise ValueError('Archive binding completion identity differs')
    if value['scope']=='repo_local':
        if value['reason'] is not None: raise ValueError('Portable archive binding reason differs')
        _,manifest,digest,copied,_=_read_archive(_relative(root,value['path']))
        if (digest!=value['archive_sha256'] or manifest['protocol_hash']!=protocol_hash
                or manifest['study_path']!=directory.relative_to(root).as_posix()
                or _hash(copied/manifest['study_path']/'confirmation.json')!=value['confirmation_sha256']):
            raise ValueError('Archive binding inventory identity differs')
    elif value['scope']=='external_unbound':
        if value['path'] is not None or value['reason']!='External archive requires explicit saved-evidence selection':
            raise ValueError('External archive eligibility differs')
    else: raise ValueError('Unknown archive binding scope')


def archive_eligibility(directory, root, protocol_hash):
    directory=Path(directory);root=Path(root)
    path=directory/'archive-completion.json'
    bindings=[]
    if path.exists():
        completion_record(directory,protocol_hash)
    for path in sorted((directory/'archive-bindings').glob('*.json')):
        value=_read(path)
        if path.stem!=canonical_hash(value): raise ValueError('Archive binding receipt digest differs')
        _verified_binding(directory,root,protocol_hash,value);bindings.append(value)
    if bindings and not (directory/'archive-completion.json').is_file():
        raise ValueError('Archive bindings lack completion receipt')
    bound=any(value['scope']=='repo_local' for value in bindings)
    status='bound' if bound else 'external_unbound' if bindings else 'unsealed'
    reason=None if bound else ('External archive requires explicit saved-evidence selection' if bindings
                              else 'No verified prospective archive supplied at completion')
    return dict(schema_version=1,status=status,reason=reason,bindings=bindings)


def bind_archive(directory, root, protocol_hash, bundle):
    """Publish only after exact copied inventory verification; never edit history."""
    from .analysis.audit import _read_archive
    directory=Path(directory).resolve();root=Path(root).resolve();bundle=Path(bundle).resolve()
    _,manifest,digest,copied,_=_read_archive(bundle)
    if (manifest['protocol_hash']!=protocol_hash
            or manifest['study_path']!=directory.relative_to(root).as_posix()
            or _hash(copied/manifest['study_path']/'confirmation.json')!=_hash(directory/'confirmation.json')):
        raise ValueError('Archive binding original completion differs')
    completion_record(directory,protocol_hash)
    # Running/interrupted evidence may be archived, but has no terminal binding.
    if not (directory/'archive-completion.json').is_file(): return
    local=bundle.is_relative_to(root)
    value=dict(schema_version=1,protocol_hash=protocol_hash,confirmation_sha256=_hash(directory/'confirmation.json'),
               archive_sha256=digest,scope='repo_local' if local else 'external_unbound',
               path=bundle.relative_to(root).as_posix() if local else None,
               reason=None if local else 'External archive requires explicit saved-evidence selection')
    _verified_binding(directory,root,protocol_hash,value)
    destination=directory/'archive-bindings'/f'{canonical_hash(value)}.json'
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists(): _atomic_json(destination,value)
