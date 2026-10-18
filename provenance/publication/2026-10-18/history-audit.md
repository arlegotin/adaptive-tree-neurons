# Reachable-history publication privacy audit — 2026-10-18

**Verdict:** no credential or privacy publication blocker found within the audited scope. This is a bounded local audit, not a guarantee that no secret exists. Audited HEAD `7af00e80ab33b6f26a9edc8aaefcf350c9d1101d`; all 76 reachable commits, 1,355 unique blobs (57,504,202 bytes).

## Scanner findings

Gitleaks 8.30.1 redacted report `history-secrets.json` contains 71 generic-api-key findings. Every finding was inspected with `git show <recorded-commit>:<recorded-path>` and mapped to its actual property/assertion; repeated findings are preserved as separate indexed entries with columns in `history-classifications.json`. There are 59 distinct scanner fingerprints because several candidates share one JSON line. No true credentials were identified.

| Type | Findings | Evidence |
|---|---:|---|
| Configuration SHA-256 identity | 16 | `full_config_key`, `config_key`, and `assert fullkey` are experiment identities from canonical JSON; all 16 independently recomputed against reachable configuration objects. |
| Source-file SHA-256 inventory | 52 | Properties are `src/treeneuron/baselines/api.py` or `tests/test_access.py`; every candidate independently equals SHA-256 of reachable Git blob content. |
| Archive access-ledger SHA-256 inventory | 3 | Three retained-ledger digest values in external/local/unknown-retention manifests under `files[protocol/access/<configuration-hash>/access.jsonl]`; neighboring entries are artifact SHA-256 inventories. Original ledger prefix is absent as a standalone reachable blob, so payload equality was **not** recomputed. No credential/authentication use is present. |

All 71 values are exactly 64 lowercase hexadecimal characters; independent cryptographic verification succeeded for 68. Hash format alone was not used to classify any finding. Machine JSON records fingerprint, property, category, format check, recomputation status and rationale without candidate values.

## Supplementary history review

- Credential-like filenames (`.env*`, private SSH keys, credentials/secrets stores, `.netrc`, cloud/SSH directories, PEM/P12/PFX/key/keystore files): none tracked across reachable history. Private-key PEM headers: none in text blobs.
- Hardware records deliberately omit serial/UUID fields. The historical machine record retains Mac model/chip/core/memory and firmware/OS versions for scientific provenance. Only fake test serial/UUID literals appear in doctor/backend redaction tests; they exercise the scrubber. The IPv4-shaped hits were version strings, not operational network addresses.
- HTTP(S) hostnames and query parameter names were reviewed. They are upstream source/package/docs/research/license links and upstream public dataset downloads, including Dropbox data links in vendored NODE code/notebooks. Queries are bibliographic `id`, public-download `dl`, language `lang`, GitHub `recursive`, and license-reference `p`; no credential query parameters or URL userinfo found. No nonpublic operational URLs or client data identified. Links were not contacted.
- Author/committer identity, owner Gmail, local owner paths and scientific timing/environment provenance remain intact as authorized. Dependency license names/emails/copyright notices and upstream attribution are required attribution, not new private client information.

## Versioned binary artifacts

Current tracked tree and historical path-version inventory contain 26 `.pt`/`.npz` files totaling 6,424,413 bytes: 24 `.pt` files / 5,199,222 bytes and 2 `.npz` files / 1,225,191 bytes. Copies account for three repeated blobs; there are 23 unique binary blobs totaling 5,276,601 bytes (21 PT + 2 NPZ). Detailed per-path sizes and archive metadata are in machine JSON.

ZIP members, pickle opcode strings/global references and NPY headers were inspected **without unpickling or executing model payloads**. PT global references are only standard collections and PyTorch tensor/storage reconstruction. NPZ arrays are numeric float64/int64 feature/target/prediction/row/group data, with no object or string dtype. These are synthetic experiment and regression artifacts: retained `diagnose.py` explicitly seeds PyTorch and uses `torch.randn` for the widened rounding case; its payload is identical to the test fixture. The causal fixture's companion Markdown identifies the toy recipe. Native-final resolved configs use `data.dataset=synthetic`; old-source reference provenance identifies teacher/split seeds and the independent exploratory E stream. The scientific term “protected” denotes held-out synthetic experiment labels, not private personal/client data.

## Narrow scanner handling proposed

Keep normal scanning enabled. Safest historical disposition is an exact immutable-commit fingerprint entry for each of the 59 reviewed fingerprints, derived from all 71 classification records; preserve the classification report for audit. This covers only these reviewed historical occurrences and does not suppress new commits or entire directories.

If a rule-local allowlist is preferred, scope it to `generic-api-key`, the exact reviewed artifact paths, and exact reviewed digest values together; use separate entries for configuration identities, the two source-file inventory properties, and the retained-ledger inventory. Restrict the two `assert fullkey` occurrences to their exact wrapper paths. Future digest values should continue to trigger review. Do not allowlist arbitrary 64hex values, generic `key` fields, all JSON, audit directories, or a whole rule. Gitleaks match fragments truncate some property names (e.g. `pi.py`), so substring-only match regexes are especially unsafe. No scanner configuration was changed by this audit.

## Limits

Scope is locally reachable Git history at the recorded HEAD and the supplied scanner report; unreachable/deleted objects, reflogs, remote-only refs, untracked/ignored working data, remote account configuration, GitHub server metadata, and future edits are excluded. Pattern scanning and manual contextual classification cannot prove absence of novel/encoded credentials. Binary structure and producer provenance support synthetic origin but numeric payloads were not regenerated or exhaustively checked for covert data. Three retained-ledger hashes could not be recomputed because their exact source payloads are not versioned. Public-link status and third-party data-sharing permissions were not revalidated online. Existing public identity and scientific provenance are intentionally retained.
