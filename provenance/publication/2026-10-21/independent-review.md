# Publication review and disposition

A fresh read-only `gpt-6-astra` review covered the publication delta from
`a3c0414`, the publication spec/plan, README, citation, notices, reproduction and
pause wording, selected reports, canonical M3 arithmetic, scan dispositions and
the deliberate tracked-artifact boundary. It did not repeat the previous
scientific implementation review or run a broad software suite.

The reviewer found the reusable value and negative hypothesis both understandable.
It independently recomputed the sixteen M3 problem averages, the 15.5995% and
1.25056% headline differences, verified all 104 classified file-hash occurrences
against their Git blobs, and checked the current-report binding and 155 tracked
WP10 software-evidence files / 3,983,949 bytes. No mismatch was found.

## Findings and the single correction pass

- **Critical: none.**
- **Important: nested AUC Mu attribution.** The retained RealMLP source declares
  Ross Kleiman's 2019 MIT component, while the initial notice only named
  PyTabKit's Apache license. Added the exact original MIT notice from upstream
  commit `8586753cb225424c63310983b10858a30cde769a` under
  `licenses/AUC-Mu-MIT.txt` and linked the exception in THIRD_PARTY_NOTICES.md.
  The upstream source lock and snapshots were not changed. The executing agent
  checked the fetched full grant, copyright, local bytes and notice link.
- **Minor: exact scoring versus greedy selection.** README/CFF initially called
  this an exact selector. The fixed-set ridge score is exact; subset selection
  is conditional greedy. Both now say so explicitly.
- **Preliminary tooling:** a scratch classification helper counted prose among
  verified files and omitted the later manual context step. It was removed from
  the public file set before commitment. The final disposition retains the
  actual 104 file-hash, 14 context-hash and four prose occurrences, plus the
  initial redacted scan. No supported public classifier tool is asserted.

The verdict was ready after the MIT notice correction, final scan and public
verification. This is the original review plus executing-agent corrections;
no second independent review is claimed.

## Declined scope and executing-agent rulings

- Prior scientific implementation, broad-suite failures, full raw authentication
  and cross-platform installation: retain the existing bounded evidence and
  limitations. This publication changes no scientific inputs.
- Push/public-byte verification: execute after the review and retain its receipt.
- Live stopped-state verification: the reviewer's process read was sandbox-denied;
  the executing agent checks the previously stopped PIDs independently. No
  research job is restarted by publication.
- Legal conclusions beyond retained component notices: no new legal assurance is
  claimed. Existing AGPL/CC BY grants and third-party terms remain explicit.
