# Independent publication review — 2026-10-18

Reviewed base `7af00e8` through preparation commit `a7e692c`, the supplied
`review-package.diff`, publication design, implementation plan and progress ledger.
Scope: release claims, readability, commands/API, citation, packaging/licensing,
and publication/privacy boundaries. No tests, model fits, source edits, Git
mutations or remote changes were performed. This report is the only review edit.

## Verdict

**Spec: substantially satisfied, with one important package-metadata correction
required. Quality: approve after that correction and the already-planned final
publication gates.** No scientific-claim, attribution-text, API-example or
privacy-boundary blocker was found within this review's scope. The owner's
existing authorization covers push/public visibility once preparation is complete.

## Important correction

**P2 — Include MIT in the distribution license expression.**
`pyproject.toml:10` declares `AGPL-3.0-only AND CC-BY-4.0`; the actual built wheel's
`License-Expression` and `provenance/publication/2026-10-18/package-verification.json`
repeat it. The wheel also contains the MIT-licensed NODE, entmax and QHOptim
runtime copies. Those notices and licenses are correctly retained, but automated
consumers of the distribution metadata receive an incomplete list of applicable
licenses. The [PyPA guidance for included third-party code](https://packaging.python.org/en/latest/guides/licensing-examples-and-user-scenarios/#my-package-includes-other-code-under-different-licenses)
uses `AND` to describe included files under distinct licenses.

Use `AGPL-3.0-only AND CC-BY-4.0 AND MIT` for this distribution, update the
explanation at `LICENSING.md:40`, and rebuild/reinspect the distribution so the
retained verification metadata and hashes describe the corrected package.
This does not change the owner's AGPL-only software grant or the original prose
grant, and does not require relicensing vendor files. CFF can continue describing
the original project licenses; retain its explicit third-party boundaries in the
linked licensing documentation. No canonical license-text changes are requested.

## Evidence checked

- All nine README normalized-MSE rows agree with saved `analysis.json` and
  `independent.json` at the displayed precision. Long-search benefit, bootstrap
  endpoints, three wins/five losses, 144 final instances and 262,144 E rows agree
  with the saved analysis and independent audit. The 2h10m cohort and approximately
  70m additional search are consistent with the recorded enclosing/nested clocks.
  The 130 trials, 53,300 updates and 133 saved-state comparisons match the replay
  account. These were evidence comparisons, not a fresh raw-data reconstruction.
- The separate 16-problem pilot numbers and the larger saved-bootstrap interval
  match its research account. The two studies are not pooled or presented as a
  before/after improvement. Fresh ordinary lower/upper brackets and inherited
  ordinary widening have separate labels. No demonstrated selector superiority,
  model-family novelty, independent confirmation, practical tabular superiority
  or hard-routing saving is asserted. Internal reproducibility is distinguished
  from independent laboratory replication.
- The README explains the centered gate and nonzero contrast correctly. Its
  exact code block equals the retained executed example. The current layer's
  constructor, `grow` contract, 544 added parameters and preactivation output
  agree with that example; the retained output is consistent. Setup, focused
  checks and archive requirements are clearly separated. Local links in README,
  REPRODUCE, LICENSING, THIRD_PARTY_NOTICES and PUBLICATION resolve.
- Source, tests, configs, `uv.lock` and historical reports have no diff across
  the review range. The generated diagnostic report still hashes to
  `9baa18c96a55a0023adf469984b8f5bebf948d1985a73675fbde115dc5e5335b`.
- The CFF identifies Artem Legotin, software version 0.1.0, the requested release
  date and repository. It invents no DOI/publication. Its recorded official
  CFF 1.2.0 schema validation passed; fields and scientific abstract were reviewed.
  Citation requests are explicitly separate from license obligations.
- Original software/prose scope, executable examples, original synthetic
  fixtures, historical original software, facts/methods and third-party
  exceptions are described explicitly. NODE/entmax/QHOptim notices remain MIT.
  The AGPL and CC BY texts have the exact documented SHA-256 hashes, including
  the preserved upstream CC BY final blank line.
- Inspected the actual retained wheel and sdist without executing their contents.
  Their SHA-256 values match `package-verification.json`. All 66 wheel package
  members match `a7e692c` source bytes. All eight legal/notice files in the wheel
  equal their corresponding repository bytes. The sdist has the recorded 134
  members and excludes ignored environments and experimental artifact roots.
  The issue above concerns the aggregate metadata expression, not missing notices.
- The redacted scanner report contains 71 candidates. Classification indices
  cover every candidate and match its commit, file and fingerprint. The ignore
  file equals exactly the 59 distinct immutable historical fingerprints: no
  blanket directory, rule, hexadecimal-pattern or new-value suppression exists.
  The earlier independent audit records 16 recomputed configuration hashes,
  52 recomputed source hashes and three contextually identified external-ledger
  hashes whose original payload was unavailable. This review checked the audit
  consistency and narrow ignore scope, rather than repeating its whole-history
  credential investigation.
- Independently enumerated 26 tracked `.pt`/`.npz` paths totaling 6,424,413 bytes;
  the path set equals the inspected binary inventory. Documentation honestly
  includes these small synthetic engineering fixtures while excluding the full
  scientific archive and third-party pretrained weights. Owner identity, local
  paths and coarse environment information are disclosed as retained provenance.

## Remaining gates and limits

The root's clean-clone full-suite run was ongoing during this review. I did not
run tests or infer its outcome. Current prose accurately labels historical test
counts; replace/add publication status using the actual clone result before
publication. Likewise, the fresh final history/publication-file scan and remote
visibility/anonymous-access/HEAD checks remain publication steps, not successes
claimed by this review.

Full raw-result authentication cannot be established from the public Git tree:
original observations, predictions and checkpoint payloads remain external.
The README and reproduction guide state this limit. I did not repeat the full
scientific source review, regenerate synthetic fixtures, independently validate
other platforms, or inspect unreachable Git objects. The earlier history audit's
bounded scope and three unrecomputed ledger hashes remain explicit.
