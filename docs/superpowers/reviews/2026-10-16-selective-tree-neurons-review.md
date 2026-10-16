# Planning review record

Date: 16 October 2026. This is an author-side documentation check, not independent scientific review or an experimental audit.

## Spec builder check

The Superpowers brainstorming workflow requested a fresh builder review. A separate research-software reviewer read the design and linked evidence record without the conversation and was asked only for consequential questions that blocked an implementation plan.

The reviewer found one missing definition: the mandatory “original 2-D conditional-slope target” lacked an equation, distribution and noise model. The supplied brief already answered the equation: `x2` for `x1<0`, otherwise `1-x2`. The design now includes it. The author chose independent `x1~Uniform(-1,1)`, `x2~Uniform(0,1)`, and zero noise as a visible implementation choice because those details were unspecified. The control is separate from the primary Gaussian population, and exact soft fitting of the discontinuity is not required.

The reviewer raised no other blocking questions. This check did not verify the research bibliography, novelty, package compatibility, algorithm correctness, or predictive value.

## Plan self-review

The author reviewed both plans against the specification using the writing-plans checklist:

- Coverage: every design section has an owning task in the evidence plan’s coverage table. The core implements WP00–WP04 and stages raw-data/practical adapters into the evidence plan; no broad infrastructure or extension is required before the mechanism gate.
- Step specificity: tasks name files, producer/consumer APIs, meaningful test assertions, verification commands and commit boundaries. Numeric runtime/sample choices are measured gate outputs with deterministic selection rules, not unspecified implementation discretion. All execution boxes remain unchecked.
- Interface consistency: shared `ObservationSet`/`SplitManifest` live in the core, including optional dates, so synthetic confirmation does not depend on running transfer acquisition. Allocation count fields and pilot planning-effect fields are explicit. Adapter contracts return finite original-unit `[N]` predictions and isolated workers preserve incompatible environments.
- Review Focus: zero growth, batch-mean scaling, optimizer/resume state, missing/constant data, drift/access, few groups, vendor semantics, degenerate variance and incomplete cohorts have task-owned assertions.
- Proportion: two phased plans carry task decisions and representative assertions rather than implementations. They defer extension algorithms until evidence identifies a question; conditional assets and unavailable methods stay visible.

The self-review also resolved two consequential lifecycle ambiguities. Complete G-best optimizer/sampler/RNG state must accompany the selected parent, and continuation update zero remains eligible. Source identity uses a pinned commit plus execution/input digests, with the actual checkout commit separately recorded, so merely committing a frozen artifact does not invalidate its own protocol. Separate exploratory E observations can calibrate variance without protected T access or future teacher reuse.

Documentation validation checks relative links, Markdown anchors/fences, task/step structure, assertion-snippet Python syntax, key parameter/sample calculations, absence of a fake frozen protocol, and whitespace. These checks validate the planning artifacts only. No model, reference algebra, package adapter, latency or scientific hypothesis was tested in this documentation task.

The validation run passed for 7 Markdown documents, 18 relative links/anchors, 27 Python snippets parsed without execution, 25 task contracts and the R01–R39 ledger. Parameter/sample checks confirmed 1,153→1,697 parameters, ordinary widths 94/95 at 1,693/1,711 parameters, F=2,306 and G=V=577, and the 36 GiB/9 GiB reservation calculations. `git diff --check` passed. No `src/`, `pyproject.toml` or `protocol/frozen.yaml` existed at validation time.
