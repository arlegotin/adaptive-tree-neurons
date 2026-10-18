"""Strict, immutable v2 study configurations and finite attempt expansion."""
from dataclasses import dataclass
import json
import re

from treeneuron.config import canonical_hash, canonical_json, resolve_experiment
from .seeds import seed64, seed32

KINDS = ('m0_reference', 'm1_controls', 'm2_opportunity', 'm3_joint',
         'm3_regularization', 'm4_directions', 'm5_newborn', 'm6_overgrow_prune',
         'm6_residual', 'practical_search', 'calibration')
ROLES = ('keep', 'top', 'random', 'greedy', 'widen_upper', 'ordinary_lower',
         'ordinary_upper', 'static_partial')


@dataclass(frozen=True)
class ResearchConfig:
    resolved_json: str

    @classmethod
    def from_dict(cls, raw):
        keys = {'schema', 'study_id', 'kind', 'seed_root', 'teachers', 'repetitions',
                'evaluation_rows', 'base_experiment'}
        if (not isinstance(raw, dict) or set(raw)-{'resource_admission'} != keys
                or raw['schema'] != 'research/v2'):
            raise ValueError('Invalid research/v2 fields/schema')
        if (type(raw['study_id']) is not str or not re.fullmatch(r'[A-Za-z0-9_-]+', raw['study_id'])
                or raw['kind'] not in KINDS):
            raise ValueError('Invalid study ID or unregistered experiment kind')
        for field in ('teachers', 'repetitions', 'evaluation_rows'):
            if type(raw[field]) is not int or raw[field] <= 0:
                raise ValueError(f'{field} must be a positive integer')
        seed64(raw['seed_root'], 'validation')
        if 'resource_admission' in raw and (type(raw['resource_admission']) is not str or not raw['resource_admission']):
            raise ValueError('resource_admission must name a measured local record')
        experiment = resolve_experiment(raw['base_experiment'])
        if experiment.study_id != raw['study_id']:
            raise ValueError('Study identity must match base experiment')
        if experiment.data.n_test is not None:
            raise ValueError('Development studies cannot request T')
        from treeneuron.train import TrainingPolicy
        TrainingPolicy.from_config(experiment)
        TrainingPolicy.from_config(experiment, continuation=True)
        resolved = {**raw, 'base_experiment': experiment.to_dict()}
        return cls(canonical_json(resolved))

    def to_dict(self):
        return json.loads(self.resolved_json)


def load_research_config(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError(f'Duplicate config key: {k}')
            result[k] = v
        return result
    return ResearchConfig.from_dict(json.loads(path.read_text(), object_pairs_hook=unique))


@dataclass(frozen=True)
class StudyPlan:
    resolved_json: str

    def to_dict(self):
        return json.loads(self.resolved_json)

    @property
    def attempts(self):
        return tuple(self.to_dict()['attempts'])

    @property
    def probe_trials(self):
        return self.to_dict()['probe_trials']


def expand(config: ResearchConfig) -> StudyPlan:
    from treeneuron.train import source_hash
    c = config.to_dict()
    if c['kind'] == 'm2_opportunity':
        from .opportunity_plan import expand_opportunity
        from .identity import execution_identity
        value = dict(schema='research/v2-plan',config=c,source_sha256=source_hash(),
            **expand_opportunity(c),protected_labels_accessed=False,
            fit_worker_roles=['F','G'],coordinator_roles=['V','E'],
            recovery_policy={'infrastructure_same_seed_pair_reruns':1,
                             'numerical_failure_reruns':0,'retain_all_costs':True},
            execution_identity=execution_identity(c))
        value['plan_hash'] = canonical_hash(value)
        return StudyPlan(canonical_json(value))
    if c['kind'] not in ('m0_reference', 'm1_controls'):
        raise NotImplementedError(f"{c['kind']} requires its owning WP03–WP09 implementation")
    attempts, used32 = [], set()
    cells = [('primary', {}, ROLES)] if c['kind'] == 'm0_reference' else (
        [('positive', {'gamma': g, 'sigma': s}, ('keep', 'top', 'random', 'oracle_current', 'oracle_true'))
         for g in (.5, 2.) for s in (0., .1)] +
        [(p, {}, ('keep', 'top', 'random')) for p in ('globally_affine',
          'additive_sine_quadratic', 'pure_noise', 'random_label_permutation')])
    seed_records = {}
    for population, coordinate, roles in cells:
        cell = population + ''.join(f'-{k}-{v}' for k, v in coordinate.items())
        for teacher in range(c['teachers']):
            labels = (c['study_id'], population, 'teacher', str(teacher))
            shared = {name+'_seed': seed64(c['seed_root'], *labels, name)
                      for name in ('teacher', 'data', 'reference', 'evaluation')}
            for repeat in range(c['repetitions']):
                pair = (*labels, 'repeat', str(repeat))
                if pair not in seed_records:
                    optimizer, counter = seed32(c['seed_root'], *pair, 'optimizer', used=used32)
                    seed_records[pair] = dict(optimizer_seed=optimizer, optimizer_collision_counter=counter,
                        allocation_seed=seed64(c['seed_root'], *pair, 'allocation'),
                        contrast_seed=seed64(c['seed_root'], *pair, 'contrast'))
                for role in roles:
                    attempts.append(dict(attempt_id=f'{cell}/teacher-{teacher}/repeat-{repeat}/{role}',
                        population=population, cell=cell, teacher_index=teacher, repetition=repeat,
                        role=role, **coordinate, **shared, **seed_records[pair]))
    # Nominal trial ceiling; actual eligibility and every skipped nominal unit
    # are recorded by the worker, never represented as completed fits.
    trials = c['teachers']*c['repetitions']*(c['base_experiment']['model']['h']+1) if c['kind']=='m0_reference' else 0
    value = dict(schema='research/v2-plan', config=c, source_sha256=source_hash(),
        attempts=attempts, probe_trials=trials, protected_labels_accessed=False,
        fit_worker_roles=['F', 'G'], coordinator_roles=['V', 'E'],
        recovery_policy={'infrastructure_same_seed_pair_reruns': 1,
                         'numerical_failure_reruns': 0, 'retain_all_costs': True})
    from .identity import execution_identity
    value['execution_identity'] = execution_identity(c)
    value['plan_hash'] = canonical_hash(value)
    return StudyPlan(canonical_json(value))
