# Target-free deployment feature inputs

The projection takes the original F archive and semantic feature schema. It
hashes the archive bytes but decodes only X and row IDs. It neither decodes
target values nor fits preprocessing or reads G/V/T. It selects1,024 rows by
cycling the original F order, converts features to float32 without clipping,
and assigns unique workload IDs. Original row-map and feature hashes remain
in the manifest.

The source must have finite float64 features, unique int64 row IDs and the
exact declared F archive fields. Unknown fields, wrong roles/hashes, invalid
shapes/dtypes, nonfinite values and float32 overflow fail before output. Fresh
outputs preserve original data/source/runtime. Complete archives publish
atomically; a failed disk transaction retains its pending file without exposing
a partial query or overwriting source data.

The independent reader rederives the row sequence and float32 values from F,
then checks the saved arrays, origin map, provenance and all byte references.
Rehashing a wrong projection cannot make it agree with the original features.
Changing F targets leaves the feature projection unchanged.

The provenance matches the original-snapshot measurement worker's input
contract. Both components keep native/F-role, precision and cost authority
false. The actual production controller must compare the supplied F descriptor
with the genuine frozen protocol, validate model/request identities, freeze
the complete measurement schedule before T and account for all attempts.
This private projection is no execution or qualification capability.

The quality procedure and protected query inventory remain unchanged. Formal
efficiency still requires matched precision/quality and qualified timing;
descriptive float32 feature timing does not supply those conditions.

The final three-module command passed **59 checks in31.53seconds**, including
the16 projection checks. The [receipt](WP10-F-deployment-projection-receipt.json)
retains the initial14 RED failures and their passing verification. Counts
overlap older commands, not a whole-suite pass. The known-data capture records
only X/row-ID decoding, the independently rejected wrong projection and
compatibility with one known saved ridge model's request validator. No
benchmark models, targets or actual owned measurement were run in that capture.
