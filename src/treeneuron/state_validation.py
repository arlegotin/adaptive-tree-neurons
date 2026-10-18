"""Read-only semantic validation of serialized model and named Adam tensors.

Shapes follow the declared bounded topology, without constructing a model or
consuming RNG. Legacy unfrozen incumbents receive one Adam update per global
update; declared insertion times additionally identify newborn ages. Applied
update schedules with freezing require a separate contract, not this policy.
"""
from collections.abc import Mapping

import torch


def model_tensor_spec(topology):
    layer = topology['layer']
    d, h = layer['d'], layer['h']
    result = {}

    def affine(prefix, width, inputs=d, bias=True):
        result[prefix + '.weight'] = ((width, inputs), torch.float32)
        if bias:
            result[prefix + '.bias'] = ((width,), torch.float32)

    kind = topology['kind']
    if kind == 'selective_relu_scalar':
        affine('layer.base', h)
        affine('head', 1, h)
        ids = layer['grown_ids']
        result['layer.grown_ids'] = ((len(ids),), torch.int64)
        if ids:
            affine('layer.gate', len(ids))
            affine('layer.contrast', len(ids))
    elif kind == 'widened_relu_scalar':
        parent = layer['parent_width']
        affine('layer.base', parent)
        affine('head.base', 1, parent)
        if h > parent:
            affine('layer.added', h-parent)
            affine('head.added', 1, h-parent, bias=False)
    elif kind == 'static_gated_relu_scalar':
        count = layer['count']
        if count < h:
            affine('layer.ordinary', h-count)
        if count:
            for part in (('gate', 'center', 'contrast') if layer['coordinates'] == 'centered'
                         else ('gate', 'left', 'right')):
                affine('layer.' + part, count)
        affine('head', 1, h)
    else:
        raise ValueError('Unsupported model tensor topology')
    return result


def validate_model_tensors(meta, tensors):
    saved = tensors['model']
    spec = model_tensor_spec(meta['model'])
    if not isinstance(saved, Mapping) or set(saved) != set(spec):
        raise ValueError('Model tensor names differ from topology')
    # Checkpoint payloads are always copied to CPU, including MPS producers.
    for name, (shape, dtype) in spec.items():
        value = saved[name]
        if (not isinstance(value, torch.Tensor) or value.shape != torch.Size(shape)
                or value.dtype != dtype or value.device.type != 'cpu'):
            raise ValueError(f'Model tensor shape/dtype/device mismatch: {name}')
    if meta['model']['kind'] == 'selective_relu_scalar':
        if saved['layer.grown_ids'].tolist() != meta['model']['layer']['grown_ids']:
            raise ValueError('Model tensor grown IDs differ from topology')
    return spec


def _added_prefixes(kind):
    return {
        'selective_relu_scalar': ('layer.gate.', 'layer.contrast.'),
        'widened_relu_scalar': ('layer.added.', 'head.added.'),
        'static_gated_relu_scalar': (),
    }[kind]


def establish_birth_updates(state):
    """Register live births once; external snapshots must already declare them.

    A manually assembled local added block can be registered only before its
    first update, with complete zero moments. Never infer a past birth from an
    observed nonzero age. Managed states retain their map across selection/load.
    """
    births = dict(state.parameter_birth_updates)
    prefixes = _added_prefixes(state.model.export_config()['kind'])
    named = dict(state.model.named_parameters())
    if set(births) - set(named):
        raise ValueError('Unmapped Adam parameter birth metadata')
    for name, parameter in named.items():
        if name in births:
            continue
        if not name.startswith(prefixes):
            births[name] = 0
            continue
        record = state.optimizer.state.get(parameter, {})
        if (not record or record['step'].item() != 0
                or bool((record['exp_avg'] != 0).any())
                or bool((record['exp_avg_sq'] != 0).any())):
            raise ValueError(f'Cannot establish unknown Adam birth: {name}')
        births[name] = state.update
    state.parameter_birth_updates = births
    return dict(births)


def validate_adam_tensors(meta, records, spec):
    added_prefixes = _added_prefixes(meta['model']['kind'])
    update = meta['update']
    birth = meta['growth_metadata'].get('insertion_update')
    if birth is not None and (type(birth) is not int or not 0 <= birth <= update):
        raise ValueError('Invalid Adam insertion birth update')
    for name, record in records.items():
        if not record:
            continue  # The complete-history validator permits lazy originals only at0.
        shape, dtype = spec[name]
        for key, value in record.items():
            expected_shape = torch.Size([]) if key == 'step' else torch.Size(shape)
            if (not isinstance(value, torch.Tensor) or value.shape != expected_shape
                    or value.dtype != dtype or value.device.type != 'cpu'
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f'Adam tensor shape/dtype/device mismatch: {name}.{key}')
        age = record['step'].item()
        if age < 0 or age != int(age) or age > update:
            raise ValueError(f'Invalid Adam age: {name}')
        declared = meta['parameter_birth_updates'][name]
        if not name.startswith(added_prefixes) and declared != 0:
            raise ValueError(f'Invalid incumbent Adam birth: {name}')
        if name.startswith(added_prefixes) and birth is not None and declared != birth:
            raise ValueError(f'Inconsistent added Adam birth: {name}')
        if age != update-declared:
            raise ValueError(f'Adam age differs from applied update contract: {name}')
        if bool((record['exp_avg_sq'] < 0).any()):
            raise ValueError(f'Invalid second Adam moment: {name}')
