"""Predeclared finite-signal allocation controls in raw coordinates."""
from dataclasses import dataclass

import numpy as np
import torch

from treeneuron.config import canonical_hash
from treeneuron.data.types import DevelopmentData, ObservationSet, _readonly
from treeneuron.train import SelectiveRegressor
from .seeds import seed64

BASE_ROWS = ((1, 0, 2), (1, 0, 2), (-1, 0, 0), (0, 1, 2),
             (0, -1, 2), (1, 1, 3), (-1, 1, 3), (0, 0, -2))


@dataclass(frozen=True)
class ControlParameters:
    """Predeclared parent/oracle coefficients; contains no observations."""
    base_rows: np.ndarray
    head: np.ndarray
    gamma: float
    generating_ids: tuple[int, ...] = (0, 2)

    @property
    def contrasts(self):
        return np.asarray(((0., .5, 0.), (.5, 0., 0.)))

    @property
    def gates(self):
        return np.asarray(((self.gamma, 0., 0.), (0., self.gamma, 0.)))


@dataclass(frozen=True)
class PositiveControl:
    base_rows: np.ndarray
    head: np.ndarray
    gamma: float
    sigma: float
    instance_seed: int
    data_seed: int
    development: DevelopmentData
    evaluation: ObservationSet
    generating_ids: tuple[int, ...] = (0, 2)

    @property
    def parameters(self):
        return ControlParameters(self.base_rows, self.head, self.gamma)

    @property
    def contrasts(self):
        return np.asarray(((0., .5, 0.), (.5, 0., 0.)))

    @property
    def gates(self):
        return np.asarray(((self.gamma, 0., 0.), (0., self.gamma, 0.)))

    def raw_predict(self, X, *, teacher):
        return _raw_predict(X, self.base_rows, self.head, self.gamma, teacher)


def _raw_predict(X, base, head, gamma, teacher):
    X = np.asarray(X, dtype=np.float64)
    aug = np.column_stack((X, np.ones(len(X))))
    z = aug @ base.T
    if teacher:
        z[:, 0] += np.tanh(gamma*X[:, 0]/2)*(.5*X[:, 1])
        z[:, 2] += np.tanh(gamma*X[:, 1]/2)*(.5*X[:, 0])
    return np.maximum(z, 0) @ head


def make_positive_control(*, instance_seed, data_seed, gamma, sigma):
    seed64(instance_seed, 'validate')
    seed64(data_seed, 'validate')
    if type(gamma) not in (int, float) or gamma not in (.5, 2.) or sigma not in (0., .1):
        raise ValueError('Positive control requires declared gamma/noise coordinates')
    base = _readonly(BASE_ROWS, np.dtype('float64'))
    factors = np.random.default_rng(instance_seed).uniform(.95, 1.05, 8)
    factors[1] = factors[0]
    head = _readonly(np.asarray((.5, .5, 1, .2, -.2, .1, -.1, .1))*factors, np.dtype('float64'))
    teacher_id = canonical_hash(dict(version='research/v2-planted-stump', base=base.tolist(),
        head=head.tolist(), gamma=gamma, sigma=sigma, instance_seed=instance_seed))
    rows, start = {}, 0
    for role, n in (('F', 512), ('G', 256), ('V', 256), ('E', 16384)):
        rng = np.random.default_rng(seed64(data_seed, 'positive', role))
        x = rng.uniform(-1, 1, (n, 2))
        target = _raw_predict(x, base, head, gamma, True)
        y = target + sigma*rng.standard_normal(n)
        ids = np.arange(start, start+n, dtype=np.int64)
        rows[role] = ObservationSet(x, y, ids, ids, noise_free_target=target,
            stream_id=f'research/v2/positive/{data_seed}/{role}')
        start += n
    data = DevelopmentData(rows['F'], rows['G'], rows['V'], teacher_id)
    return PositiveControl(base, head, float(gamma), float(sigma), instance_seed,
        data_seed, data, rows['E'])


def transport_parent(control, transform, *, teacher):
    """Exact raw-to-F-standardized transport; do not fit teacher coefficients."""
    state = transform.to_state()
    if (len(state['columns']) != 2 or any(c['kind'] != 'numeric' or c['indicator']
                                       for c in state['columns'])):
        raise ValueError('Control transport requires two complete numerical features')
    mu = np.asarray([c['mean'] for c in state['columns']])
    scale = np.asarray([c['scale'] for c in state['columns']])

    def transported(raw):
        value = raw.copy()
        value[:, :2] *= scale
        value[:, 2] += raw[:, :2] @ mu
        return value

    with torch.random.fork_rng(devices=[]):
        model = SelectiveRegressor(2, 8, 1.).double().eval()
        base = transported(control.base_rows)
        with torch.no_grad():
            model.layer.base.weight.copy_(torch.from_numpy(base[:, :2]))
            model.layer.base.bias.copy_(torch.from_numpy(base[:, 2]))
            model.head.weight.copy_(torch.from_numpy((control.head/state['target_scale'])[None, :]))
            model.head.bias.fill_(-state['target_mean']/state['target_scale'])
        if teacher:
            contrasts = transported(control.contrasts)
            model.layer.grow(torch.tensor(control.generating_ids, dtype=torch.int64),
                             torch.from_numpy(contrasts))
            gates = transported(control.gates)
            with torch.no_grad():
                model.layer.gate.weight.copy_(torch.from_numpy(gates[:, :2]))
                model.layer.gate.bias.copy_(torch.from_numpy(gates[:, 2]))
    return model


def initialize_control_state(control, config, paths, *, data):
    """Supplied ungrown parent at update0; no ordinary-parent training stage."""
    import random
    from treeneuron import train
    from treeneuron.data.transforms import FittedTransform
    from treeneuron.evaluation.metrics import regression_metrics
    if not isinstance(control, ControlParameters):
        raise ValueError('Worker requires observation-free control parameters')
    train._validate_inputs(data, config, paths)
    random.seed(config.seeds.optimizer)
    np.random.seed(config.seeds.optimizer)
    torch.manual_seed(config.seeds.optimizer)
    transform = FittedTransform.fit(data.F.X, data.F.y)
    model = transport_parent(control, transform, teacher=False).float()
    optimizer = train._optimizer(model, config)
    schedule = train.UpdateSchedule(optimizer, kind=config.training.scheduler,
        horizon=config.training.base_updates+config.training.continuation_updates,
        min_lr=config.training.min_learning_rate, base_lrs=[config.training.learning_rate])
    generator = torch.Generator().manual_seed(config.seeds.optimizer)
    state = train.TrainingState(model, optimizer, schedule, 0, 0,
        torch.randperm(len(data.F.y), generator=generator), 0, generator,
        train.capture_rng(), transform, train._identities(data, config, transform),
        config, train._backend_evidence(config, data), float(np.var(data.F.y)), train._fg_hashes(data))
    train._new_stage(state, train.TrainingPolicy.from_config(config), 'base')
    state.stage.update(complete=True, best_mse=regression_metrics(data.G.y,
        state.predict(data.G.X), fit_target_variance=state.fit_target_variance)['mse'])
    train._observe_initial(state, data, paths)
    train.pack_state(state)  # Establish original birth0 contract before any fork.
    return state
