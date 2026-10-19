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

## Observation and repeat namespace contract

The fixture-tested component, `research/confirmation_namespaces.py`, derives only
seed identities from the already committed private root. It creates no teachers,
observations or protected-access capability. It authenticates the public/private
commitment, method/exclusion/source bindings and its own committed module bytes.
The existing root and teacher-seed derivation remain unchanged; no new entropy is
drawn and a collision never silently replaces a teacher or root.

For matrix C1 or C2 and index i, every base namespace is
`seed64(root,matrix,'primary','teacher-<i>',role)`. Roles are `teacher`,
`reference`, F, G, V and T. These do not include a method or training repetition.
Optimizer, continuation-stream, allocation and contrast namespaces append
`'repeat','<r>',role` for r=0,1; paired arms reuse the same records. A separate
library32 optimizer seed uses the existing collision-counted `seed32` law, never
an implicit truncation. Independent 63-bit namespaces must not collide within
the expanded plan. The two repeats share F/G/V/T and teacher/reference seeds.

The returned immutable plan hides individual seeds from its representation and
returns copies to the trusted coordinator. It still grants no observation
generation, calibration or T-label access, and does not certify historical
registry completeness or actual generated-teacher ID exclusions. Those remain
mandatory final freeze/evaluator checks. This is procedural separation for
cooperating workers, not encryption or permission to hand the complete private
plan to an ordinary fitting worker.

A retained public-field/development-plan inventory now supplies a durable
[conservative known-seed registry](teacher-exclusion-inventory.md):10,171seed
exclusions and74ID candidates. It does not promote completeness or actual
generated-teacher independence; those final semantic/coverage checks remain
pending. No real confirmation root was drawn for this inventory.

Nine missing-component RED checks preceded the implementation; 50 affected
namespace/commitment/protocol checks then passed in 27.25 s. Fixed-entropy
temporary Git fixtures exercise all three claim kinds, 448 distinct 63-bit
namespaces and 64 collision-managed library32 optimizer seeds for 32 teacher
identities. Generator/entropy sentinels prove expansion creates no teachers or
observations. Forced collisions fail without replacing private root bytes;
commitment/source changes reject. These are software fixtures using root 123,
not a real confirmation root or new scientific teachers. Historical registry
completeness and generated teacher-ID exclusions remain unqualified. See
[the receipt](WP09-confirmation-namespaces-receipt.json).
