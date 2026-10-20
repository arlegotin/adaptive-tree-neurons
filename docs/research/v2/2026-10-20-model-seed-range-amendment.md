# Prospective paired model-seed compatibility repair

Before the first native-quality production search, code review found that the
planner assigned unsigned 32-bit model seeds, while the unchanged native tree
parameter validator requires `0 <= seed < 2**31`. Of the 60 default paired
dataset/fold/repetition labels, 32 exceeded that range. This is a prospective
execution defect; it explains no observed predictive result.

All 14 families continue sharing one numerical model seed per paired label.
Native-quality plans now declare `paired-signed31-v1`. Derive that common value
with `seed64(root, *labels, 'library31', str(counter)) & ((1 << 31) - 1)`,
excluding already assigned HPO32 and model values and incrementing the collision
counter deterministically. Record `fit_seed_bits=31` and
`fit_seed31_collision_counter`. This uses a separate namespace rather than
truncating an already assigned library32 seed. Keep `fit_seed64`, HPO64 and HPO32
values unchanged. The primary inventory rejects an unknown policy or changed
job seed even after a caller recomputes its outer hash.

The prospective initialization numbers intentionally change. The engineering
native anchor runs use seed17; completed M1/M2/M3/B0 runs retain their own
source and seed identities. None is rerun, reseeded or migrated by this repair.
Legacy bounded plans retain their old behavior. No protected data or production
native-quality outcome was available for choosing this policy.

The range regression failed before the repair. All 240 planned tree coordinates
then passed the actual native parameter guard; all-14-family pairing and
collision checks passed. An independently fixed digest verifies unchanged
HPO64/HPO32/fit-root64 vectors. The final affected suite passed 78 checks; the
[receipt](WP08-primary-search-declaration-receipt.json) preserves the old vectors'
diagnostic hash, initial failures, source identities and exact commands.
