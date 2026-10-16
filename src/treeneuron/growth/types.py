"""Serializable named migration records, independent of optimizer parameter IDs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class ParameterMapping:
    """Map an old named tensor region onto a new one (half-open axis slices).

    Selective growth uses full-tensor identity maps. Explicit shapes and slices
    also leave the migration representation usable for later widening adapters.
    """

    old_name: str
    new_name: str
    old_shape: tuple[int, ...]
    new_shape: tuple[int, ...]
    old_slice: tuple[tuple[int, int], ...]
    new_slice: tuple[tuple[int, int], ...]
    kind: Literal['identity', 'slice'] = 'identity'


@dataclass(frozen=True)
class GrowthManifest:
    """One producer-scoped event; new_parameter_names lists only added tensors.

    SelectiveAffineLayer emits layer-relative names/counts excluding the caller's
    head and retains old Parameter objects. widen_parent emits model-qualified
    names/counts including the head, explicitly records that scope in its state
    metadata, and maps copied old values/moments by full slices. Added tensors
    have no old mapping and require fresh optimizer state in either producer.
    """

    schema_version: int
    d: int
    h: int
    grown_ids: tuple[int, ...]
    growth_applied: bool
    old_parameter_count: int
    new_parameter_count: int
    added_parameter_count: int
    old_parameter_names: tuple[str, ...]
    new_parameter_names: tuple[str, ...]
    parameter_mappings: tuple[ParameterMapping, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record (tuples are accepted by JSON)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> GrowthManifest:
        """Restore a version-1 record after JSON decoding."""
        if record.get('schema_version') != 1:
            raise ValueError('Unsupported growth manifest schema_version')
        fields = dict(record)
        for key in ('grown_ids', 'old_parameter_names', 'new_parameter_names'):
            fields[key] = tuple(fields[key])
        mappings = []
        for value in fields['parameter_mappings']:
            value = dict(value)
            for key in ('old_shape', 'new_shape'):
                value[key] = tuple(value[key])
            for key in ('old_slice', 'new_slice'):
                value[key] = tuple(tuple(axis) for axis in value[key])
            mappings.append(ParameterMapping(**value))
        fields['parameter_mappings'] = tuple(mappings)
        return cls(**fields)
