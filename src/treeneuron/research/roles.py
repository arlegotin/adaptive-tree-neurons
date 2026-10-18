"""F/G-only worker view; validation labels remain in the coordinator."""
from treeneuron.data.types import DevelopmentData, ObservationSet


def _without_oracle(rows):
    if rows.noise_free_target is None:
        return rows
    return ObservationSet(rows.X, rows.y, rows.row_id, rows.group_id,
        dates=rows.dates, stream_id=rows.stream_id, feature_names=rows.feature_names)


class FittingData(DevelopmentData):
    def __init__(self, data):
        for name, value in dict(F=_without_oracle(data.F), G=_without_oracle(data.G),
                teacher_id=data.teacher_id, manifest=data.manifest,
                validation_features=data.V.X).items():
            object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        if name in ('V', 'T'):
            raise PermissionError(f'{name} labels are unavailable to fitting workers')
        return super().__getattribute__(name)


def fitting_view(data):
    return FittingData(data)
