# Fresh confirmation seed commitment

This fixture-tested component records a fresh random root after a chosen-method
identity exists and before any confirmation observations are generated. It is a
seed commitment, not a protocol freeze or a permission to create/read T.

`commit_confirmation_seeds` loads an externally committed provisional method,
authenticates its source and a sorted unique registry of excluded teacher seeds,
and draws one63-bit root from the operating system. It derives exactly the
declared K teacher seeds, rejects overlap/duplicates, and rechecks input
commitments after the draw. C1 placement/pruning and C2 residual capacity use
separate matrix namespaces. Within a matrix, the canonical labels are
`matrix,'primary','teacher-<index>','teacher'`; the seed64 algorithm is unchanged.

The owner-only private file is written/fsynced first. The public file commits its
bytes, the root, ordered teacher seeds, chosen method, exclusions and source,
without publishing the root or individual seeds. External file commitments are
required when reopening both this record and its method identity. A public-write
failure retains the private file for diagnosis; the creation API refuses to
regenerate over it. `resume_confirmation_seed_publication` requires that exact
private-file commitment, the same method/source/exclusion commitments and an
absent public output. It publishes the retained root without drawing entropy or
changing private bytes. No roots are selected from outcomes.

The registry's bytes and known seed exclusions are verified. Registry completeness
and actual generated teacher IDs are explicitly unqualified: the generator must
check those IDs against all historical/development identities before a final
freeze admits a teacher. F/G/V/T observation namespaces, the protected attempt
list, power/resource/readiness validation, final root reveal and protected
evaluation are separate pending requirements. The returned plan grants no
protected access. The private file is a procedural separation for cooperative
workers, not encryption against the machine's owner.

All evidence so far is temporary Git/metadata fixtures with fixed entropy where
needed to exercise collision and replay paths.17 initial failures for the missing
component became17 passing checks. Three subsequently failing input-drift
checks led to revalidation before publication;87 affected seed/protocol/design/
teacher-analysis checks then passed. No real confirmation root, calibration
teacher, observation arrays, model fits or protected outcomes were created.

Same-root publication recovery subsequently added two failing regression checks,
then89 affected checks passed. The proof forces a public-write failure, retains
the exact private bytes, prohibits another entropy draw, and reopens the recovered
commitment with the original teacher seeds. Changed private commitments and
overwriting an existing public record fail. This qualifies root publication
recovery only; native model/optimizer and complete search recovery are separate.
