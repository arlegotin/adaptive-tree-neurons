# Protecting registry evidence during output creation

The native registry's own directory is immutable evidence, alongside its input
proofs and six working datasets. New searches, preparation artifacts and model
seals must stay outside those directories and the qualified interpreter.

`native_registry_storage_roots` lists those boundaries from complete metadata;
the helper explicitly grants no qualification. Actual public entry points still
require the genuine registry first. The qualified callback now checks its search
location before constructing the inner worker bridge. Preparation, worker-body
and model-seal guards use the same complete boundaries. New registry publication
also protects working data and the current interpreter before opening component
proofs. Disjoint legitimate output paths retain their behavior.

The affected six-module suite passed **105 checks in 51.91 seconds**. Pure
metadata fixtures cover the registry container, working data, input proofs,
source and interpreter paths without creating them. An initial helper integration
mistake misread the existing validator's tuple return; the seven failures and
correction are retained. Two later publication tests failed before the working
data/interpreter checks were added. The [receipt](WP08-native-registry-output-paths-receipt.json)
pins source, commands, logs and the unqualified metadata observation.

These changes concern future output locations. They change no completed model
or predictive result, and do not qualify the current source for production.
The held-source NODE queue and its environment remain unchanged. The production
launcher must still preflight its control directory before the generic process
observer creates handshake artifacts; this component supplies the boundaries.
