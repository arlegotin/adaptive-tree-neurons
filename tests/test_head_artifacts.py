"""Retained frozen-head predictions are reconstructible without refitting."""
import json
import hashlib
from pathlib import Path
import copy

import numpy as np
import pytest

from test_train import training_case,cpu_training_evidence


def test_frozen_head_artifact_roundtrip_exact_and_bound_to_parent(training_case,tmp_path):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.headroom import refit_frozen_head
    from treeneuron.research.head_artifacts import fit_head_artifact,load_head_artifact
    from test_checkpoint import assert_nested_equal
    data,config,paths=training_case;view=fitting_view(data)
    parent=train.fit_parent(view,config,paths);rng=train.capture_rng();packed=train.pack_state(parent)
    expected=refit_frozen_head(parent,view).predict(data.V.X)
    result=fit_head_artifact(parent,view,tmp_path/'head')
    assert result['status']=='completed_eligible' and result['ridge_solve_count']==6
    assert len(result['trials'])==6 and result['costs']['standalone_total_ns'] is None
    reference=result['models']['selected']
    predictor=load_head_artifact(reference)
    np.testing.assert_array_equal(predictor.predict(data.V.X),expected)
    assert_nested_equal(train.pack_state(parent),packed)
    assert_nested_equal(train.capture_rng(),rng)
    changed={**reference,'parent_sha256':'0'*64}
    with pytest.raises(ValueError,match='parent'):
        load_head_artifact(changed)
    path=Path(reference['path'])
    record=json.loads(path.read_text());record['parent_identity_hashes']['runtime']='0'*64
    incompatible=tmp_path/'wrong-runtime.json';incompatible.write_text(json.dumps(record))
    changed={**reference,'path':str(incompatible),'sha256':hashlib.sha256(incompatible.read_bytes()).hexdigest()}
    with pytest.raises(ValueError,match='runtime'):
        load_head_artifact(changed)
    valid=json.loads(path.read_text())
    for label in ('missing_trials','wrong_alpha','wrong_coefficients'):
        bad=copy.deepcopy(valid)
        if label=='missing_trials':bad['records_json']=json.dumps({'trials':[]})
        elif label=='wrong_alpha':bad['alpha']=.3
        else:bad['arrays']['coefficients']['values'][0]+=1.
        broken=tmp_path/(label+'.json');broken.write_text(json.dumps(bad))
        changed={**reference,'path':str(broken),'sha256':hashlib.sha256(broken.read_bytes()).hexdigest()}
        with pytest.raises(ValueError,match='selection'):
            load_head_artifact(changed)
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='checksum'):
        load_head_artifact(reference)
