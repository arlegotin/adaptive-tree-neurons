# Task 1 implementation report

Status: DONE_WITH_CONCERNS (independent root review remains required before numerical execution).

Commit: `b2a9937` — `feat: add opt-in long-probe allocation diagnostic`.
Source base supplied by controller: `5b0483e`. Work stayed on current `main`.

## Delivered behavior

- `run_greedy_probes` admits exactly `fixed_updates` and `patience`. Default fixed allowance rejection remains. Patience accepts only a finite completed stage explained by its declared ceiling or configured patience. Each trial records allowance, executed/selected updates, reason and ceiling status, with stopping metadata retained in its selected checkpoint. No trainer stopping/ranking/count semantics changed. Parent state and global RNG are preserved.
- The existing eight-role graph remains the default. Positive keyword-only `reference_probe_updates` adds independent `greedy_long` ranking and a ninth final role, reloading the same original parent. The long fork explicitly checks original parameters and Adam state, sampler/RNG/policy and common executed row prefix, original table/count/seed/contrasts, and complete-state identity. Zero growth remains a retained complete comparison.
- The new standalone module exposes `load_spec`, `measure`, `run`, `analyze`, `report` and all specified CLI commands. It does not call canonical calibration/freeze/readiness, generate T, fit on E, or use E accuracy to select resources or members.
- Separate cost/study root seed namespaces use the exact approved values in the shipped protocol. The published base config has d16/h64/m16, F2306/G577/V577/E262144, lr0.005, batch128, ceilings8000, short320, interval25, patience20, epsilon0.05, tau1, noise0.1 and CPU float32/CPU64 scoring. Actual ordinary brackets resolve to 1693/1711 around1697. Default paths are `environment/long-probes-20261018/cpu128-receipt.json` and `artifacts/long-probes-20261018`.
- Cost/resource decisions, study locks, original observations and per-pair artifact inventories are hash bound. The cumulative ledger holds references, statuses, costs and hashes, not full pair reports or row arrays. Failed members and artifacts remain; the fixed cohort never retries, shrinks or expands automatically. Source/config/lock/runtime/receipt drift fails, while a later docs-only HEAD is recorded and allowed when all execution bytes remain identical.
- Resource checks run between generation, fitting/scoring stages and individual probe attempts, with CPU1/inter-op1 preflight, all five BLAS variables1, fallback0, process-scoped caffeinate, 6GiB RSS cap, <=10% availability plus new swapout pressure gate, and projected remaining disk plus10GiB reserve. There is no elapsed-forecast time kill.
- Analysis independently reconstructs original data from locked seeds, checks saved F/E values and row IDs, recomputes F variance and original-unit MSE/RMSE/MAE/normalized MSE, checks source/recipe/data/optimizer identities, validates all retained artifact hashes and original-parent/contrast/row-stream bindings, then averages repetitions within teacher and teachers equally. All zero/negative effects stay. It uses one fixed512000 paired whole-teacher bootstrap with seed261031106 and descriptive percentile95%/Student-t intervals. Report rendering is deterministic and human-readable.

## Changed files

1. `src/treeneuron/baselines/greedy.py`
2. `src/treeneuron/pilot/calibrate.py`
3. `src/treeneuron/pilot/probe_diagnostic.py`
4. `tests/test_greedy.py`
5. `tests/test_pilot_calibrate.py`
6. `tests/test_probe_diagnostic.py`
7. `configs/diagnostics/long-probes-base.json`
8. `protocol/diagnostics/2026-10-18-long-probes.json`

No controller ledger, README, reports, DECISIONS, historical artifacts, dependency manifests or unrelated source were edited.

## Verification and self-review

The final affected-file run passed **55 tests in36.45s**. `git diff --check` returned exit0 with empty output. No broad historical/full suite, production cost problem, cohort, backend probe/receipt producer, numerical study or old receipt renewal was run. Numerical work was limited to the authorized bounded test fixtures and the existing test receipt fixture. No subagents or reviewers were spawned by this implementer.

Initial direct `.venv/bin/pytest -q tests/test_greedy.py -k 'stopping_mode or patience_mode or short_and_long'` produced7 setup errors, before the intended behavior ran: `ValueError: Live interop threads differ from backend evidence`. This was a test-runtime prerequisite failure, not behavioral RED. Fresh processes with the documented CPU1/inter-op1/BLAS1/fallback0 harness resolved it.

Self-review corrections were driven by the retained failing checks:

- Independent checkpoint saves include UUID-named descriptor/payload locators; they cannot have equal pointer/metadata byte hashes. Added a canonical complete-state digest while retaining both actual distinct checkpoint hashes and locators.
- The existing predictor loader intentionally requires exactly eight registered methods. Kept it unchanged; the opt-in wrapper validates and loads the extra checkpoint separately.
- Removed legacy H1/static per-E-row JSON arrays only on the opt-in path; the old default output remains unchanged.
- Reduced the disk projection only after a completed member, preventing the guard from reserving the entire cohort again midway through the run.
- New diagnostic observation retention never accesses V arrays/labels. It retains F/G and fresh E plus the development manifest.
- A new selected-checkpoint growth-metadata assertion detected a stale manifest growth hash. The initial setup hash is now logged as an event; the immutable manifest growth identity is published once after final stopping metadata is complete. Weight hash/retention are updated consistently with the saved checkpoint.
- The explicit full-state audit initially indexed Adam's defaultdict at an update-zero parent, adding empty state entries. The real tiny diagnostic fixture caught this (`54 passed,1 failed` in an intermediate run). The audit now uses nonmutating `.get(...,{})`; the final run verifies update-zero and nonzero paths.

The intermediate complete-file runs and all failure output are retained under `task-1-evidence/`. These include `52 passed in36.18s`, `55 passed in38.96s`, the intentional RED failures, fixture-expectation corrections, and the final `55 passed in36.45s`; no failure was hidden as a passing run.

## Concerns and remaining controller work

1. Root's independent whole-change spec/quality review is still required before any numerical execution. No independent-review completion is claimed here.
2. Runtime/retained-byte forecasts remain explicitly one-sample engineering estimates with1.5 safety factor; they do not promise runtime, variance coverage or effect signs. Final member-ledger/seal accounting publication is outside the sampled enclosing clock; all generation, fitting, scoring, searches, predictions, checkpoint/raw-event publication and per-member artifact hashing/publication are inside it. The controller may measure a still broader outer wall clock when publishing its cost evidence.
3. Representative ranking replay, long-checkpoint restoration and exact state/prediction parity under the final committed source are the controller's later evidence task. The necessary checkpoints, events, score tables, actual checkpoint hashes and canonical original-parent bindings are retained by the implementation; this report does not claim to have run a production replay.
4. A missing native resource counter is retained with an explicit reason; the severe-pressure conjunction can only be evaluated when availability and swapout counters are observable. The real launch should inspect those observations.

## Exact RED/GREEN commands and outputs

The blocks below give the argument-equivalent commands used; each process was fresh. Complete byte-for-byte captured output is copied into the named evidence file. Shell output redirection does not change the executed pytest arguments.

### Greedy API RED

Command:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -c "import torch,pytest; torch.set_num_interop_threads(1); raise SystemExit(pytest.main(['-q', 'tests/test_greedy.py', '-k', 'stopping_mode or patience_mode or short_and_long']))"
```

Output (`task-1-evidence/long-probe-greedy-red.txt`):

```text
FFFFFFF                                                                  [100%]
=================================== FAILURES ===================================
_______ test_probe_rejects_unknown_stopping_mode_before_trials[unknown] ________

probe_case = (DevelopmentData(F=ObservationSet(X=array([[-1.46692975, -0.4196381 ],
       [-0.04056875,  0.39279744],
       [-1.0...gotin/pytest-4244/test_probe_rejects_unknown_sto0/74820691032c4e53ac8af38e27c3686a/9e95ec2623ce4060bbe77fa504857c89')))
mode = 'unknown'

    @pytest.mark.parametrize('mode', ['unknown', None, True])
    def test_probe_rejects_unknown_stopping_mode_before_trials(probe_case, mode):
        from treeneuron.baselines.greedy import run_greedy_probes
        data,parent,table,paths=probe_case
        with pytest.raises(ValueError,match='stopping mode'):
>           run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=paths,stopping_mode=mode)
E           TypeError: run_greedy_probes() got an unexpected keyword argument 'stopping_mode'

tests/test_greedy.py:258: TypeError
_________ test_probe_rejects_unknown_stopping_mode_before_trials[None] _________

probe_case = (DevelopmentData(F=ObservationSet(X=array([[-1.46692975, -0.4196381 ],
       [-0.04056875,  0.39279744],
       [-1.0...gotin/pytest-4244/test_probe_rejects_unknown_sto1/42d9b8b53c55404b8a46e39218510f51/90bd2e2a1df24ff3b288c3fe70cceff9')))
mode = None

    @pytest.mark.parametrize('mode', ['unknown', None, True])
    def test_probe_rejects_unknown_stopping_mode_before_trials(probe_case, mode):
        from treeneuron.baselines.greedy import run_greedy_probes
        data,parent,table,paths=probe_case
        with pytest.raises(ValueError,match='stopping mode'):
>           run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=paths,stopping_mode=mode)
E           TypeError: run_greedy_probes() got an unexpected keyword argument 'stopping_mode'

tests/test_greedy.py:258: TypeError
_________ test_probe_rejects_unknown_stopping_mode_before_trials[True] _________

probe_case = (DevelopmentData(F=ObservationSet(X=array([[-1.46692975, -0.4196381 ],
       [-0.04056875,  0.39279744],
       [-1.0...gotin/pytest-4244/test_probe_rejects_unknown_sto2/8e2de2a1f8364b05a54af4eeffd38252/a893bbaf43e442d2b7f64f978aa0f756')))
mode = True

    @pytest.mark.parametrize('mode', ['unknown', None, True])
    def test_probe_rejects_unknown_stopping_mode_before_trials(probe_case, mode):
        from treeneuron.baselines.greedy import run_greedy_probes
        data,parent,table,paths=probe_case
        with pytest.raises(ValueError,match='stopping mode'):
>           run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=paths,stopping_mode=mode)
E           TypeError: run_greedy_probes() got an unexpected keyword argument 'stopping_mode'

tests/test_greedy.py:258: TypeError
__________ test_patience_mode_accepts_genuine_completed_early_stop[1] __________

probe_case = (DevelopmentData(F=ObservationSet(X=array([[-1.46692975, -0.4196381 ],
       [-0.04056875,  0.39279744],
       [-1.0...gotin/pytest-4244/test_patience_mode_accepts_gen0/bbcf372212794703951a42f5dc8c6293/80c290f6877b440a8f64963e8a5ee30f')))
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x115b8ca40>

    @pytest.mark.parametrize('probe_case',[1],indirect=True)
    def test_patience_mode_accepts_genuine_completed_early_stop(probe_case,monkeypatch):
        from treeneuron.baselines import greedy
        from treeneuron import train
        data,parent,table,paths=probe_case
        monkeypatch.setattr(train,'regression_metrics',lambda *a,**k: {'mse':1.})
        before=copy.deepcopy(parent)
>       greedy.run_greedy_probes(parent,table,data,probe_updates=6,seed=7,paths=paths,stopping_mode='patience')
E       TypeError: run_greedy_probes() got an unexpected keyword argument 'stopping_mode'

tests/test_greedy.py:270: TypeError
___ test_patience_mode_rejects_unfinished_or_unexplained_return[unfinished] ____

probe_case = (DevelopmentData(F=ObservationSet(X=array([[-1.46692975, -0.4196381 ],
       [-0.04056875,  0.39279744],
       [-1.0...gotin/pytest-4244/test_patience_mode_rejects_unf0/b13d823c297844228e0932fae1f33792/dba5264750514b54b63fcbc4d5fc9882')))
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x115b68410>
mutation = 'unfinished'

    @pytest.mark.parametrize('mutation',['unfinished','unexplained'])
    def test_patience_mode_rejects_unfinished_or_unexplained_return(probe_case,monkeypatch,mutation):
        from treeneuron.baselines import greedy
        data,parent,table,paths=probe_case
        real=greedy.continue_fit
        def corrupt(*a,**kw):
            state=real(*a,**kw)
            if mutation=='unfinished': state.stage['complete']=False
            else: state.stage['updates_completed']=1
            return state
        monkeypatch.setattr(greedy,'continue_fit',corrupt)
        with pytest.raises(ValueError,match='completed|unexplained'):
>           greedy.run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=paths,stopping_mode='patience')
E           TypeError: run_greedy_probes() got an unexpected keyword argument 'stopping_mode'

tests/test_greedy.py:292: TypeError
___ test_patience_mode_rejects_unfinished_or_unexplained_return[unexplained] ___

probe_case = (DevelopmentData(F=ObservationSet(X=array([[-1.46692975, -0.4196381 ],
       [-0.04056875,  0.39279744],
       [-1.0...gotin/pytest-4244/test_patience_mode_rejects_unf1/5153ca69c94240219577d479a9c9537b/96f2f6141f9348f9a24488dcb642c2eb')))
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x115ab7110>
mutation = 'unexplained'

    @pytest.mark.parametrize('mutation',['unfinished','unexplained'])
    def test_patience_mode_rejects_unfinished_or_unexplained_return(probe_case,monkeypatch,mutation):
        from treeneuron.baselines import greedy
        data,parent,table,paths=probe_case
        real=greedy.continue_fit
        def corrupt(*a,**kw):
            state=real(*a,**kw)
            if mutation=='unfinished': state.stage['complete']=False
            else: state.stage['updates_completed']=1
            return state
        monkeypatch.setattr(greedy,'continue_fit',corrupt)
        with pytest.raises(ValueError,match='completed|unexplained'):
>           greedy.run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=paths,stopping_mode='patience')
E           TypeError: run_greedy_probes() got an unexpected keyword argument 'stopping_mode'

tests/test_greedy.py:292: TypeError
_________ test_short_and_long_have_independent_identical_parent_starts _________

probe_case = (DevelopmentData(F=ObservationSet(X=array([[-1.46692975, -0.4196381 ],
       [-0.04056875,  0.39279744],
       [-1.0...gotin/pytest-4244/test_short_and_long_have_indep0/3151e773c34d4990b4d4b918b7883d4c/35782ff622194495944643d3e8db304b')))
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x115b8cd10>

    def test_short_and_long_have_independent_identical_parent_starts(probe_case,monkeypatch):
        from treeneuron.baselines import greedy
        data,parent,table,short=probe_case
        long=create_run(short.directory/'long',parent.config.to_dict())
        starts=[]
        real=greedy.continue_fit
        def observe(state,*a,**kw):
            starts.append(copy.deepcopy(state))
            return real(state,*a,**kw)
        monkeypatch.setattr(greedy,'continue_fit',observe)
        before=copy.copy(parent); before.rng_states=capture_rng()
        greedy.run_greedy_probes(parent,table,data,probe_updates=2,seed=7,paths=short)
>       greedy.run_greedy_probes(parent,table,data,probe_updates=6,seed=7,paths=long,stopping_mode='patience')
E       TypeError: run_greedy_probes() got an unexpected keyword argument 'stopping_mode'

tests/test_greedy.py:310: TypeError
=========================== short test summary info ============================
FAILED tests/test_greedy.py::test_probe_rejects_unknown_stopping_mode_before_trials[unknown]
FAILED tests/test_greedy.py::test_probe_rejects_unknown_stopping_mode_before_trials[None]
FAILED tests/test_greedy.py::test_probe_rejects_unknown_stopping_mode_before_trials[True]
FAILED tests/test_greedy.py::test_patience_mode_accepts_genuine_completed_early_stop[1]
FAILED tests/test_greedy.py::test_patience_mode_rejects_unfinished_or_unexplained_return[unfinished]
FAILED tests/test_greedy.py::test_patience_mode_rejects_unfinished_or_unexplained_return[unexplained]
FAILED tests/test_greedy.py::test_short_and_long_have_independent_identical_parent_starts
7 failed, 16 deselected in 4.56s
```

### Opt-in fitting graph RED

Command:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -c "import torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main(['-q', 'tests/test_pilot_calibrate.py', '-k', 'long_probe']))"
```

Output (`task-1-evidence/long-probe-fit-red.txt`):

```text
FFFFFF                                                                   [100%]
=================================== FAILURES ===================================
_____ test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[0] _____

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4246/test_long_probe_opt_in_nine_ro0')
cpu_training_evidence = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4246/training-evidence0/report.json')
growth_count = 0

    @pytest.mark.parametrize('growth_count',[0,1])
    def test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts(tmp_path,cpu_training_evidence,growth_count):
        import torch
        from treeneuron.pilot.calibrate import teacher_inputs,repetition_config,run_repetition
        raw=config(tmp_path).to_dict()
        raw['execution']['backend_report']=str(cpu_training_evidence)
        raw['backend']['interop_threads']=torch.get_num_interop_threads()
        raw['growth']['m_requested']=growth_count
        c=resolve_experiment(raw)
        _,data,e,tc=teacher_inputs(c,0)
>       result=run_repetition(repetition_config(tc,0),data,e,tmp_path/'long-pair','diagnostic',reference_probe_updates=1)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       TypeError: run_repetition() got an unexpected keyword argument 'reference_probe_updates'

tests/test_pilot_calibrate.py:220: TypeError
_____ test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[1] _____

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4246/test_long_probe_opt_in_nine_ro1')
cpu_training_evidence = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4246/training-evidence0/report.json')
growth_count = 1

    @pytest.mark.parametrize('growth_count',[0,1])
    def test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts(tmp_path,cpu_training_evidence,growth_count):
        import torch
        from treeneuron.pilot.calibrate import teacher_inputs,repetition_config,run_repetition
        raw=config(tmp_path).to_dict()
        raw['execution']['backend_report']=str(cpu_training_evidence)
        raw['backend']['interop_threads']=torch.get_num_interop_threads()
        raw['growth']['m_requested']=growth_count
        c=resolve_experiment(raw)
        _,data,e,tc=teacher_inputs(c,0)
>       result=run_repetition(repetition_config(tc,0),data,e,tmp_path/'long-pair','diagnostic',reference_probe_updates=1)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       TypeError: run_repetition() got an unexpected keyword argument 'reference_probe_updates'

tests/test_pilot_calibrate.py:220: TypeError
___________ test_long_probe_allowance_validated_before_parent_fit[0] ___________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4246/test_long_probe_allowance_vali0')
allowance = 0

    @pytest.mark.parametrize('allowance',[0,-1,True,2])
    def test_long_probe_allowance_validated_before_parent_fit(tmp_path,allowance):
        from treeneuron.pilot.calibrate import teacher_inputs,fit_repetition
        c=config(tmp_path)
        _,data,_,_=teacher_inputs(c,0)
        with pytest.raises(ValueError,match='reference probe'):
>           fit_repetition(c,data,tmp_path/'invalid','recipe',reference_probe_updates=allowance)
E           TypeError: fit_repetition() got an unexpected keyword argument 'reference_probe_updates'

tests/test_pilot_calibrate.py:246: TypeError
__________ test_long_probe_allowance_validated_before_parent_fit[-1] ___________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4246/test_long_probe_allowance_vali1')
allowance = -1

    @pytest.mark.parametrize('allowance',[0,-1,True,2])
    def test_long_probe_allowance_validated_before_parent_fit(tmp_path,allowance):
        from treeneuron.pilot.calibrate import teacher_inputs,fit_repetition
        c=config(tmp_path)
        _,data,_,_=teacher_inputs(c,0)
        with pytest.raises(ValueError,match='reference probe'):
>           fit_repetition(c,data,tmp_path/'invalid','recipe',reference_probe_updates=allowance)
E           TypeError: fit_repetition() got an unexpected keyword argument 'reference_probe_updates'

tests/test_pilot_calibrate.py:246: TypeError
_________ test_long_probe_allowance_validated_before_parent_fit[True] __________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4246/test_long_probe_allowance_vali2')
allowance = True

    @pytest.mark.parametrize('allowance',[0,-1,True,2])
    def test_long_probe_allowance_validated_before_parent_fit(tmp_path,allowance):
        from treeneuron.pilot.calibrate import teacher_inputs,fit_repetition
        c=config(tmp_path)
        _,data,_,_=teacher_inputs(c,0)
        with pytest.raises(ValueError,match='reference probe'):
>           fit_repetition(c,data,tmp_path/'invalid','recipe',reference_probe_updates=allowance)
E           TypeError: fit_repetition() got an unexpected keyword argument 'reference_probe_updates'

tests/test_pilot_calibrate.py:246: TypeError
___________ test_long_probe_allowance_validated_before_parent_fit[2] ___________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4246/test_long_probe_allowance_vali3')
allowance = 2

    @pytest.mark.parametrize('allowance',[0,-1,True,2])
    def test_long_probe_allowance_validated_before_parent_fit(tmp_path,allowance):
        from treeneuron.pilot.calibrate import teacher_inputs,fit_repetition
        c=config(tmp_path)
        _,data,_,_=teacher_inputs(c,0)
        with pytest.raises(ValueError,match='reference probe'):
>           fit_repetition(c,data,tmp_path/'invalid','recipe',reference_probe_updates=allowance)
E           TypeError: fit_repetition() got an unexpected keyword argument 'reference_probe_updates'

tests/test_pilot_calibrate.py:246: TypeError
=========================== short test summary info ============================
FAILED tests/test_pilot_calibrate.py::test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[0]
FAILED tests/test_pilot_calibrate.py::test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[1]
FAILED tests/test_pilot_calibrate.py::test_long_probe_allowance_validated_before_parent_fit[0]
FAILED tests/test_pilot_calibrate.py::test_long_probe_allowance_validated_before_parent_fit[-1]
FAILED tests/test_pilot_calibrate.py::test_long_probe_allowance_validated_before_parent_fit[True]
FAILED tests/test_pilot_calibrate.py::test_long_probe_allowance_validated_before_parent_fit[2]
6 failed, 10 deselected in 3.48s
```

### Diagnostic module RED

Command:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -c "import torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main(['-q', 'tests/test_probe_diagnostic.py']))"
```

Output (`task-1-evidence/long-probe-diagnostic-red.txt`):

```text
FFFFFFFFFFF                                                              [100%]
=================================== FAILURES ===================================
_______________ test_spec_strict_and_cost_study_streams_disjoint _______________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4248/test_spec_strict_and_cost_stud0')

    def test_spec_strict_and_cost_study_streams_disjoint(tmp_path):
>       m=module(); value=spec(tmp_path)
          ^^^^^^^^

tests/test_probe_diagnostic.py:33: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_probe_diagnostic.py:16: in module
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.uv-python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'treeneuron.pilot.probe_diagnostic'
import_ = <function _gcd_import at 0x105e1c0e0>

>   ???
E   ModuleNotFoundError: No module named 'treeneuron.pilot.probe_diagnostic'

<frozen importlib._bootstrap>:1324: ModuleNotFoundError
_____________ test_measurement_lock_and_ordered_failure_membership _____________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4248/test_measurement_lock_and_orde0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x11341a7b0>

    def test_measurement_lock_and_ordered_failure_membership(tmp_path,monkeypatch):
>       m,calls=fake_execution(monkeypatch,tmp_path)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_probe_diagnostic.py:70: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_probe_diagnostic.py:50: in fake_execution
    m=module(); calls=[]
      ^^^^^^^^
tests/test_probe_diagnostic.py:16: in module
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.uv-python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'treeneuron.pilot.probe_diagnostic'
import_ = <function _gcd_import at 0x105e1c0e0>

>   ???
E   ModuleNotFoundError: No module named 'treeneuron.pilot.probe_diagnostic'

<frozen importlib._bootstrap>:1324: ModuleNotFoundError
__________ test_resource_failure_retains_attempt_without_silent_retry __________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4248/test_resource_failure_retains_0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x113419130>

    def test_resource_failure_retains_attempt_without_silent_retry(tmp_path,monkeypatch):
>       m,calls=fake_execution(monkeypatch,tmp_path)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_probe_diagnostic.py:96: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_probe_diagnostic.py:50: in fake_execution
    m=module(); calls=[]
      ^^^^^^^^
tests/test_probe_diagnostic.py:16: in module
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.uv-python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'treeneuron.pilot.probe_diagnostic'
import_ = <function _gcd_import at 0x105e1c0e0>

>   ???
E   ModuleNotFoundError: No module named 'treeneuron.pilot.probe_diagnostic'

<frozen importlib._bootstrap>:1324: ModuleNotFoundError
_____ test_severe_pressure_requires_both_low_availability_and_new_swapouts _____

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4248/test_severe_pressure_requires_0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x11341a060>

    def test_severe_pressure_requires_both_low_availability_and_new_swapouts(tmp_path,monkeypatch):
>       m=module(); policy=spec(tmp_path)['resources']
          ^^^^^^^^

tests/test_probe_diagnostic.py:106: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_probe_diagnostic.py:16: in module
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.uv-python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'treeneuron.pilot.probe_diagnostic'
import_ = <function _gcd_import at 0x105e1c0e0>

>   ???
E   ModuleNotFoundError: No module named 'treeneuron.pilot.probe_diagnostic'

<frozen importlib._bootstrap>:1324: ModuleNotFoundError
_____ test_analysis_teacher_weighting_original_units_and_retained_effects ______

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4248/test_analysis_teacher_weightin0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x11341ade0>

    def test_analysis_teacher_weighting_original_units_and_retained_effects(tmp_path,monkeypatch):
>       m,root=analysis_fixture(tmp_path,monkeypatch)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_probe_diagnostic.py:146: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_probe_diagnostic.py:118: in analysis_fixture
    m,calls=fake_execution(monkeypatch,tmp_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_probe_diagnostic.py:50: in fake_execution
    m=module(); calls=[]
      ^^^^^^^^
tests/test_probe_diagnostic.py:16: in module
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.uv-python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'treeneuron.pilot.probe_diagnostic'
import_ = <function _gcd_import at 0x105e1c0e0>

>   ???
E   ModuleNotFoundError: No module named 'treeneuron.pilot.probe_diagnostic'

<frozen importlib._bootstrap>:1324: ModuleNotFoundError
________ test_analysis_rejects_missing_or_altered_evidence[prediction] _________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4248/test_analysis_rejects_missing_0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x1134c0080>
tamper = 'prediction'

    @pytest.mark.parametrize('tamper',['prediction','targets','rows','fit_variance','pair','source'])
    def test_analysis_rejects_missing_or_altered_evidence(tmp_path,monkeypatch,tamper):
>       m,root=analysis_fixture(tmp_path,monkeypatch)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_probe_diagnostic.py:164: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_probe_diagnostic.py:118: in analysis_fixture
    m,calls=fake_execution(monkeypatch,tmp_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_probe_diagnostic.py:50: in fake_execution
    m=module(); calls=[]
      ^^^^^^^^
tests/test_probe_diagnostic.py:16: in module
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.uv-python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'treeneuron.pilot.probe_diagnostic'
import_ = <function _gcd_import at 0x105e1c0e0>

>   ???
E   ModuleNotFoundError: No module named 'treeneuron.pilot.probe_diagnostic'

<frozen importlib._bootstrap>:1324: ModuleNotFoundError
__________ test_analysis_rejects_missing_or_altered_evidence[targets] __________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4248/test_analysis_rejects_missing_1')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x1134c34d0>
tamper = 'targets'

    @pytest.mark.parametrize('tamper',['prediction','targets','rows','fit_variance','pair','source'])
    def test_analysis_rejects_missing_or_altered_evidence(tmp_path,monkeypatch,tamper):
>       m,root=analysis_fixture(tmp_path,monkeypatch)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_probe_diagnostic.py:164: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_probe_diagnostic.py:118: in analysis_fixture
    m,calls=fake_execution(monkeypatch,tmp_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_probe_diagnostic.py:50: in fake_execution
    m=module(); calls=[]
      ^^^^^^^^
tests/test_probe_diagnostic.py:16: in module
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.uv-python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'treeneuron.pilot.probe_diagnostic'
import_ = <function _gcd_import at 0x105e1c0e0>

>   ???
E   ModuleNotFoundError: No module named 'treeneuron.pilot.probe_diagnostic'

<frozen importlib._bootstrap>:1324: ModuleNotFoundError
___________ test_analysis_rejects_missing_or_altered_evidence[rows] ____________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4248/test_analysis_rejects_missing_2')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x11341ad80>
tamper = 'rows'

    @pytest.mark.parametrize('tamper',['prediction','targets','rows','fit_variance','pair','source'])
    def test_analysis_rejects_missing_or_altered_evidence(tmp_path,monkeypatch,tamper):
>       m,root=analysis_fixture(tmp_path,monkeypatch)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_probe_diagnostic.py:164: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_probe_diagnostic.py:118: in analysis_fixture
    m,calls=fake_execution(monkeypatch,tmp_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_probe_diagnostic.py:50: in fake_execution
    m=module(); calls=[]
      ^^^^^^^^
tests/test_probe_diagnostic.py:16: in module
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.uv-python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'treeneuron.pilot.probe_diagnostic'
import_ = <function _gcd_import at 0x105e1c0e0>

>   ???
E   ModuleNotFoundError: No module named 'treeneuron.pilot.probe_diagnostic'

<frozen importlib._bootstrap>:1324: ModuleNotFoundError
_______ test_analysis_rejects_missing_or_altered_evidence[fit_variance] ________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4248/test_analysis_rejects_missing_3')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x113419130>
tamper = 'fit_variance'

    @pytest.mark.parametrize('tamper',['prediction','targets','rows','fit_variance','pair','source'])
    def test_analysis_rejects_missing_or_altered_evidence(tmp_path,monkeypatch,tamper):
>       m,root=analysis_fixture(tmp_path,monkeypatch)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_probe_diagnostic.py:164: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_probe_diagnostic.py:118: in analysis_fixture
    m,calls=fake_execution(monkeypatch,tmp_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_probe_diagnostic.py:50: in fake_execution
    m=module(); calls=[]
      ^^^^^^^^
tests/test_probe_diagnostic.py:16: in module
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.uv-python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'treeneuron.pilot.probe_diagnostic'
import_ = <function _gcd_import at 0x105e1c0e0>

>   ???
E   ModuleNotFoundError: No module named 'treeneuron.pilot.probe_diagnostic'

<frozen importlib._bootstrap>:1324: ModuleNotFoundError
___________ test_analysis_rejects_missing_or_altered_evidence[pair] ____________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4248/test_analysis_rejects_missing_4')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x1134c3710>
tamper = 'pair'

    @pytest.mark.parametrize('tamper',['prediction','targets','rows','fit_variance','pair','source'])
    def test_analysis_rejects_missing_or_altered_evidence(tmp_path,monkeypatch,tamper):
>       m,root=analysis_fixture(tmp_path,monkeypatch)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_probe_diagnostic.py:164: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_probe_diagnostic.py:118: in analysis_fixture
    m,calls=fake_execution(monkeypatch,tmp_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_probe_diagnostic.py:50: in fake_execution
    m=module(); calls=[]
      ^^^^^^^^
tests/test_probe_diagnostic.py:16: in module
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.uv-python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'treeneuron.pilot.probe_diagnostic'
import_ = <function _gcd_import at 0x105e1c0e0>

>   ???
E   ModuleNotFoundError: No module named 'treeneuron.pilot.probe_diagnostic'

<frozen importlib._bootstrap>:1324: ModuleNotFoundError
__________ test_analysis_rejects_missing_or_altered_evidence[source] ___________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4248/test_analysis_rejects_missing_5')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x1134c1ac0>
tamper = 'source'

    @pytest.mark.parametrize('tamper',['prediction','targets','rows','fit_variance','pair','source'])
    def test_analysis_rejects_missing_or_altered_evidence(tmp_path,monkeypatch,tamper):
>       m,root=analysis_fixture(tmp_path,monkeypatch)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_probe_diagnostic.py:164: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_probe_diagnostic.py:118: in analysis_fixture
    m,calls=fake_execution(monkeypatch,tmp_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_probe_diagnostic.py:50: in fake_execution
    m=module(); calls=[]
      ^^^^^^^^
tests/test_probe_diagnostic.py:16: in module
    return importlib.import_module('treeneuron.pilot.probe_diagnostic')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.uv-python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'treeneuron.pilot.probe_diagnostic'
import_ = <function _gcd_import at 0x105e1c0e0>

>   ???
E   ModuleNotFoundError: No module named 'treeneuron.pilot.probe_diagnostic'

<frozen importlib._bootstrap>:1324: ModuleNotFoundError
=========================== short test summary info ============================
FAILED tests/test_probe_diagnostic.py::test_spec_strict_and_cost_study_streams_disjoint
FAILED tests/test_probe_diagnostic.py::test_measurement_lock_and_ordered_failure_membership
FAILED tests/test_probe_diagnostic.py::test_resource_failure_retains_attempt_without_silent_retry
FAILED tests/test_probe_diagnostic.py::test_severe_pressure_requires_both_low_availability_and_new_swapouts
FAILED tests/test_probe_diagnostic.py::test_analysis_teacher_weighting_original_units_and_retained_effects
FAILED tests/test_probe_diagnostic.py::test_analysis_rejects_missing_or_altered_evidence[prediction]
FAILED tests/test_probe_diagnostic.py::test_analysis_rejects_missing_or_altered_evidence[targets]
FAILED tests/test_probe_diagnostic.py::test_analysis_rejects_missing_or_altered_evidence[rows]
FAILED tests/test_probe_diagnostic.py::test_analysis_rejects_missing_or_altered_evidence[fit_variance]
FAILED tests/test_probe_diagnostic.py::test_analysis_rejects_missing_or_altered_evidence[pair]
FAILED tests/test_probe_diagnostic.py::test_analysis_rejects_missing_or_altered_evidence[source]
11 failed in 0.44s
```

### Reference-only output and remaining-storage RED

Command:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -c "import torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main(['-q', 'tests/test_probe_diagnostic.py', 'tests/test_pilot_calibrate.py', '-k', 'projection_decreases or long_probe_opt_in']))"
```

Output (`task-1-evidence/long-probe-retention-red.txt`):

```text
FFF                                                                      [100%]
=================================== FAILURES ===================================
________ test_resource_projection_decreases_only_for_completed_members _________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4252/test_resource_projection_decre0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x111640320>

    def test_resource_projection_decreases_only_for_completed_members(tmp_path,monkeypatch):
        m=module(); policy=spec(tmp_path)['resources']; reserve=policy['disk_reserve_bytes']
        ledger=dict(resource_observations=[],expected_members=[{},{}],members=[])
        snapshots=iter([dict(free_disk_bytes=reserve+100,peak_rss_bytes=1,available_percent=80,swapouts=0),
            dict(free_disk_bytes=reserve+50,peak_rss_bytes=1,available_percent=80,swapouts=0)])
        monkeypatch.setattr(m,'_resource_snapshot',lambda p:next(snapshots))
        guard=m._resource_guard(tmp_path,policy,ledger,100)
>       guard(); ledger['members'].append({'status':'completed'}); guard()
                                                                   ^^^^^^^

tests/test_probe_diagnostic.py:218: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def check():
        nonlocal previous
        observation=_resource_snapshot(output)
        observation['elapsed_ns']=time.perf_counter_ns()-ledger.get('started_ns',time.perf_counter_ns())
        ledger['resource_observations'].append(observation)
        before=previous; previous=observation['swapouts']
        if observation['free_disk_bytes']<projected_bytes+policy['disk_reserve_bytes']:
>           raise RuntimeError('Diagnostic disk reserve/projection guard failed')
E           RuntimeError: Diagnostic disk reserve/projection guard failed

src/treeneuron/pilot/probe_diagnostic.py:178: RuntimeError
_____ test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[0] _____

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4252/test_long_probe_opt_in_nine_ro0')
cpu_training_evidence = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4252/training-evidence0/report.json')
growth_count = 0

    @pytest.mark.parametrize('growth_count',[0,1])
    def test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts(tmp_path,cpu_training_evidence,growth_count):
        import torch
        from treeneuron.pilot.calibrate import teacher_inputs,repetition_config,run_repetition
        raw=config(tmp_path).to_dict()
        raw['execution']['backend_report']=str(cpu_training_evidence)
        raw['backend']['interop_threads']=torch.get_num_interop_threads()
        raw['growth']['m_requested']=growth_count
        c=resolve_experiment(raw)
        _,data,e,tc=teacher_inputs(c,0)
        result=run_repetition(repetition_config(tc,0),data,e,tmp_path/'long-pair','diagnostic',reference_probe_updates=1)
        assert set(result['methods'])=={'keep','top','random','greedy','greedy_long','widen_upper','ordinary_lower','ordinary_upper','static_partial'}
        assert result['pairing_audit']['passed'] and result['pairing_audit']['warm_forks']==6
        assert result['long_probe_audit']['passed']
>       assert 'h1_rows' not in result and 'static_rows' not in result
E       AssertionError: assert ('h1_rows' not in {'status': 'completed', 'teacher_id': 'bc8e4310b841f8bfc394626d05db20df5358e248d33f790bf336e74f967b5310', 'optimizer_seed': 1516600993, 'data_hash': '373195c9e64e589c7adf60e3ab430d020a44ed3d6ce39ce9f6de49ced2c33fcf', ...})

tests/test_pilot_calibrate.py:224: AssertionError
_____ test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[1] _____

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4252/test_long_probe_opt_in_nine_ro1')
cpu_training_evidence = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4252/training-evidence0/report.json')
growth_count = 1

    @pytest.mark.parametrize('growth_count',[0,1])
    def test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts(tmp_path,cpu_training_evidence,growth_count):
        import torch
        from treeneuron.pilot.calibrate import teacher_inputs,repetition_config,run_repetition
        raw=config(tmp_path).to_dict()
        raw['execution']['backend_report']=str(cpu_training_evidence)
        raw['backend']['interop_threads']=torch.get_num_interop_threads()
        raw['growth']['m_requested']=growth_count
        c=resolve_experiment(raw)
        _,data,e,tc=teacher_inputs(c,0)
        result=run_repetition(repetition_config(tc,0),data,e,tmp_path/'long-pair','diagnostic',reference_probe_updates=1)
        assert set(result['methods'])=={'keep','top','random','greedy','greedy_long','widen_upper','ordinary_lower','ordinary_upper','static_partial'}
        assert result['pairing_audit']['passed'] and result['pairing_audit']['warm_forks']==6
        assert result['long_probe_audit']['passed']
>       assert 'h1_rows' not in result and 'static_rows' not in result
E       AssertionError: assert ('h1_rows' not in {'status': 'completed', 'teacher_id': 'bc8e4310b841f8bfc394626d05db20df5358e248d33f790bf336e74f967b5310', 'optimizer_seed': 1516600993, 'data_hash': '373195c9e64e589c7adf60e3ab430d020a44ed3d6ce39ce9f6de49ced2c33fcf', ...})

tests/test_pilot_calibrate.py:224: AssertionError
=========================== short test summary info ============================
FAILED tests/test_probe_diagnostic.py::test_resource_projection_decreases_only_for_completed_members
FAILED tests/test_pilot_calibrate.py::test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[0]
FAILED tests/test_pilot_calibrate.py::test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[1]
3 failed, 28 deselected in 8.55s
```

### Unused V retention RED

Command:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -c "import torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main(['-q', 'tests/test_probe_diagnostic.py', '-k', 'does_not_access_V']))"
```

Output (`task-1-evidence/long-probe-v-red.txt`):

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_____________ test_observation_retention_does_not_access_V_labels ______________

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4253/test_observation_retention_doe0')

    def test_observation_retention_does_not_access_V_labels(tmp_path):
        m=module()
        from treeneuron.pilot.calibrate import teacher_inputs
        teacher,data,e,_=teacher_inputs(m._config(spec(tmp_path),'cost'),0)
        class Forbidden:
            def __getattribute__(self,name): raise AssertionError('V must remain unused')
        object.__setattr__(data,'V',Forbidden())
>       path=m._observations(tmp_path,0,teacher,data,e)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_probe_diagnostic.py:240: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
src/treeneuron/pilot/probe_diagnostic.py:211: in _observations
    **{f'{part}_{key}':getattr(getattr(data,part),key) for part in 'FGV' for key in ('X','y','row_id')})
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <test_probe_diagnostic.test_observation_retention_does_not_access_V_labels.<locals>.Forbidden object at 0x1137d2180>
name = 'X'

>   def __getattribute__(self,name): raise AssertionError('V must remain unused')
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   AssertionError: V must remain unused

tests/test_probe_diagnostic.py:238: AssertionError
=========================== short test summary info ============================
FAILED tests/test_probe_diagnostic.py::test_observation_retention_does_not_access_V_labels
1 failed, 15 deselected in 0.77s
```

### Saved checkpoint/manifest metadata RED

Command:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -c "import torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main(['-q', 'tests/test_greedy.py', '-k', 'genuine_completed']))"
```

Output (`task-1-evidence/long-probe-metadata-red.txt`):

```text
F                                                                        [100%]
=================================== FAILURES ===================================
__________ test_patience_mode_accepts_genuine_completed_early_stop[1] __________

probe_case = (DevelopmentData(F=ObservationSet(X=array([[-1.46692975, -0.4196381 ],
       [-0.04056875,  0.39279744],
       [-1.0...gotin/pytest-4255/test_patience_mode_accepts_gen0/09effeac767242f5967ea3e29809f89c/a01c700c15ed4c02848cb001345f3627')))
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x111d089e0>

    @pytest.mark.parametrize('probe_case',[1],indirect=True)
    def test_patience_mode_accepts_genuine_completed_early_stop(probe_case,monkeypatch):
        from treeneuron.baselines import greedy
        from treeneuron import train
        data,parent,table,paths=probe_case
        monkeypatch.setattr(train,'regression_metrics',lambda *a,**k: {'mse':1.})
        before=copy.deepcopy(parent)
        greedy.run_greedy_probes(parent,table,data,probe_updates=6,seed=7,paths=paths,stopping_mode='patience')
        report=json.loads((paths.directory/'greedy.json').read_text())
        assert report['complete'] and report['stopping_mode']=='patience'
        for trial in report['trials']:
            assert trial['executed_updates']==2 and trial['selected_stage_updates']==0
            assert trial['stopping_reason']=='patience' and not trial['hit_ceiling']
            assert trial['allowance']==6
            from treeneuron.checkpoint import load_checkpoint
            from treeneuron.config import canonical_hash
            child=paths.directory/trial['path']
            restored=load_checkpoint(child/'checkpoints'/'best.json',expected_hashes=parent.identity_hashes,device='cpu')
            manifest=json.loads((child/'manifest.json').read_text())
>           assert manifest['identity']['growth_hash']==canonical_hash(restored.growth_metadata)
E           AssertionError: assert '376f02323d10...4e5d74a3069cb' == 'ccf1c937e27a...4e60d61c6de28'
E             
E             - ccf1c937e27aa024e0baf5fb39883b48e67bd63ba4121a77a9a4e60d61c6de28
E             + 376f02323d1087e54e5cb7fb7d0d1915d83a4d63e6282b902434e5d74a3069cb

tests/test_greedy.py:282: AssertionError
=========================== short test summary info ============================
FAILED tests/test_greedy.py::test_patience_mode_accepts_genuine_completed_early_stop[1]
1 failed, 22 deselected in 3.78s
```

### Explicit final-fork state-audit RED

Command:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -c "import torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main(['-q', 'tests/test_pilot_calibrate.py', '-k', 'long_probe_opt_in']))"
```

Output (`task-1-evidence/long-probe-joint-state-red2.txt`):

```text
FF                                                                       [100%]
=================================== FAILURES ===================================
_____ test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[0] _____

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4257/test_long_probe_opt_in_nine_ro0')
cpu_training_evidence = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4257/training-evidence0/report.json')
growth_count = 0

    @pytest.mark.parametrize('growth_count',[0,1])
    def test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts(tmp_path,cpu_training_evidence,growth_count):
        import torch
        from treeneuron.pilot.calibrate import teacher_inputs,repetition_config,run_repetition
        raw=config(tmp_path).to_dict()
        raw['execution']['backend_report']=str(cpu_training_evidence)
        raw['backend']['interop_threads']=torch.get_num_interop_threads()
        raw['growth']['m_requested']=growth_count
        c=resolve_experiment(raw)
        _,data,e,tc=teacher_inputs(c,0)
        result=run_repetition(repetition_config(tc,0),data,e,tmp_path/'long-pair','diagnostic',reference_probe_updates=1)
        assert set(result['methods'])=={'keep','top','random','greedy','greedy_long','widen_upper','ordinary_lower','ordinary_upper','static_partial'}
        assert result['pairing_audit']['passed'] and result['pairing_audit']['warm_forks']==6
        assert result['long_probe_audit']['passed']
>       assert result['methods']['greedy_long']['original_parent_audit']['passed']
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       KeyError: 'original_parent_audit'

tests/test_pilot_calibrate.py:224: KeyError
_____ test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[1] _____

tmp_path = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4257/test_long_probe_opt_in_nine_ro1')
cpu_training_evidence = PosixPath('/private/var/folders/q3/tm7kkng56417wgzydc4q8p8r0000gn/T/pytest-of-artemlegotin/pytest-4257/training-evidence0/report.json')
growth_count = 1

    @pytest.mark.parametrize('growth_count',[0,1])
    def test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts(tmp_path,cpu_training_evidence,growth_count):
        import torch
        from treeneuron.pilot.calibrate import teacher_inputs,repetition_config,run_repetition
        raw=config(tmp_path).to_dict()
        raw['execution']['backend_report']=str(cpu_training_evidence)
        raw['backend']['interop_threads']=torch.get_num_interop_threads()
        raw['growth']['m_requested']=growth_count
        c=resolve_experiment(raw)
        _,data,e,tc=teacher_inputs(c,0)
        result=run_repetition(repetition_config(tc,0),data,e,tmp_path/'long-pair','diagnostic',reference_probe_updates=1)
        assert set(result['methods'])=={'keep','top','random','greedy','greedy_long','widen_upper','ordinary_lower','ordinary_upper','static_partial'}
        assert result['pairing_audit']['passed'] and result['pairing_audit']['warm_forks']==6
        assert result['long_probe_audit']['passed']
>       assert result['methods']['greedy_long']['original_parent_audit']['passed']
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       KeyError: 'original_parent_audit'

tests/test_pilot_calibrate.py:224: KeyError
=========================== short test summary info ============================
FAILED tests/test_pilot_calibrate.py::test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[0]
FAILED tests/test_pilot_calibrate.py::test_long_probe_opt_in_nine_roles_same_parent_score_and_contrasts[1]
2 failed, 14 deselected in 8.61s
```

### Final affected-file GREEN

Command:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -c "import torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main(['-q', 'tests/test_greedy.py', 'tests/test_pilot_calibrate.py', 'tests/test_probe_diagnostic.py']))"
```

Output (`task-1-evidence/long-probe-all-green4.txt`):

```text
.......................................................                  [100%]
55 passed in 36.45s
```

