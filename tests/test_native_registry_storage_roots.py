"""Metadata path guards protect the registry's own immutable directory too."""
import importlib.util
from pathlib import Path
import sys

import pytest


def api():
    from treeneuron.research import native_qualification_registry_v2
    assert hasattr(native_qualification_registry_v2,'native_registry_storage_roots'), 'Missing complete registry storage boundaries'
    return native_qualification_registry_v2


@pytest.fixture(scope='module')
def declared_roots(tmp_path_factory):
    from test_native_qualification_registry_v2 import sentinel_bundle
    root=tmp_path_factory.mktemp('declared-registry-storage')
    inputs=sentinel_bundle(root)
    reference=dict(root=str(root/'registry-itself'),inventory_sha256='a'*64)
    return root,inputs,reference


def test_storage_metadata_includes_registry_itself_without_authenticating_it(declared_roots):
    root,inputs,reference=declared_roots
    result=api().native_registry_storage_roots(inputs=inputs,registry_reference=reference)
    assert reference['root'] in result['proof_roots']
    assert set(result['working_roots'])==set(inputs['prepared_roots'].values())
    assert result['native_qualification_authenticated'] is False
    assert not (root/'registry-itself').exists()


@pytest.mark.parametrize('kind',['registry','working','proof','executable','interpreter'])
def test_qualified_callback_search_location_preserves_every_declared_boundary(declared_roots,kind):
    from treeneuron.research import qualified_native_trial_v2
    root,inputs,reference=declared_roots
    api()
    location=(Path(reference['root']) if kind=='registry' else Path(inputs['prepared_roots']['bike'])
        if kind=='working' else Path(inputs['preparation_reference']['root']) if kind=='proof'
        else Path(sys.prefix) if kind=='interpreter' else Path(__file__).resolve().parents[1]/'src')/'new-search'
    with pytest.raises(ValueError):
        qualified_native_trial_v2._check_search_location(dict(inputs=inputs),reference,location)
    assert not location.exists()


def test_disjoint_search_location_is_only_a_path_check_not_admission(declared_roots):
    from treeneuron.research import qualified_native_trial_v2
    root,inputs,reference=declared_roots;api()
    assert qualified_native_trial_v2._check_search_location(dict(inputs=inputs),reference,root/'separate-search') is None
    assert not (root/'separate-search').exists()


@pytest.mark.parametrize('kind',['working','interpreter'])
def test_new_registry_publication_cannot_pollute_working_data_or_interpreter(declared_roots,monkeypatch,kind):
    root,inputs,reference=declared_roots;module=api()
    output=(Path(inputs['prepared_roots']['energy']) if kind=='working' else Path(sys.prefix))/'new-registry'
    def forbidden(*args,**kwargs):raise AssertionError('Unsafe output must reject before component reads')
    monkeypatch.setattr(module,'_inspect',forbidden)
    with pytest.raises(ValueError):module.seal_native_qualification_registry(**inputs,output=output)
    assert not output.exists()
