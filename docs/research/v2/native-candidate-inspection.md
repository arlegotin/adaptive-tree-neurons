# Primary native candidate inspection

`research.native_candidate_v2.inspect_primary_native_candidate` composes three
previously separate checks: the current-source managed generation chain, a fresh
closed-model prediction replay and primary coordinator V arithmetic. It starts
no training or HPO. Native workload/support qualification and paid-search/freeze
authority remain separate; every returned quality/HPO/native-qualification flag
stays false.

Supply the exact generation arguments accepted by `audit_native_generation`, an
immutable generation container holding its request and native output, a fresh
external inspection directory, the qualified interpreter and pinned overlays.
The output must be outside both the immutable container and prepared data store.
The caller must also keep it outside any owning frozen queue.

The inspector authenticates generation before creating replay inputs. It copies
the complete closed snapshot, feature-only query and retained predictions into
new, byte-verified inputs. The fresh replay process imports only that method's
adapter, uses the declared numerical threads and owned RSS guard, has no wall
deadline, and must reproduce prediction dtype/values exactly. It receives no V
or T labels. All original generation files remain unchanged.

Only after successful generation and independent inference does the coordinator
open primary V through the existing logged scorer. The inspector then rechecks
the original generation container/source/roles, copied inputs, replay/process
inventories and **every validation source, including V labels**. Changed bytes
fail the inspection. A resource-stopped replay returns its paid failure and
never reads V. A failure after starting validation records that boundary;
`V_labels_read_by_coordinator=null` means validation was invoked but did not
complete, rather than asserting nonaccess.

The result includes all component evidence, the containing function time and a
separately named nested replay-supervisor span. The caller must additionally pay
fitting, publication, return and surrounding search work. Neither nested times
nor shared parents can be added twice. This component does not independently
reexecute training, authenticate acquisition, qualify native implementations or
turn earlier engineering models into eligible HPO candidates.

Eight inspection cases plus generation, replay and V-scoring regressions passed
in the71-case affected suite. Actual tiny native P13 fits exercise fresh-process
replay, V arithmetic and old-container immutability; synthetic sealed roles do
not authenticate official dataset acquisition. The original six missing-module
failures, sandboxed `/bin/ps` refusal, controlled late-evidence failures and
recoveries remain retained in the [receipt](WP08-native-candidate-inspection-receipt.json).
Two late-mutation regressions exposed a real assembly gap before release:
generation checks deliberately omit V labels, and fresh replay evidence needed
its own final inventory check. Both now reject. No real study artifact was
mutated and the held native baseline sources are unchanged.

Production still needs the native/workload/support registry, proposal-journal
binding, complete candidate/selected-model assembly and the final protocol
barrier. There is no production search, final freeze or protected T access here.
