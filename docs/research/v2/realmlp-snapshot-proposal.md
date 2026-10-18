# P06 native RealMLP inference snapshot proposal

Historical status at representation approval: proposed closed inference
representation before implementation. The reviewed implementation and final
results are now described in realmlp-qualification.md. Numeric and mixed-category 3-epoch engineering fits now match
direct official predictions exactly; they remain quality-ineligible. No portable
model loader is implemented or claimed. Original MSE-key failures remain retained.
The current diagnostic NPZ is expressly not a portable snapshot.

## What the executed graphs contain

The numeric fixture has22 tensors/540,380 raw tensor bytes; the mixed binary and
11-level categorical fixture has25 tensors/550,024 bytes. Both native fitted
models and `creator.static_model` must be represented: static categorical one-hot
encoding lives outside the fitted model's state_dict. The fitted model contains
robust scaling/smooth clipping, numerical embeddings, learned categorical
embedding, NTP factors, parametric Mish, inactive evaluation dropout, native
output inverse scaling and F-min/max clamp. Generic state_dict/pickle is
insufficient; retained F labels are unnecessary for this inference representation.

## Proposed closed graph and tensors

Use versioned JSON plus strict allow_pickle=False NPZ, with the two graphs stored
separately and every tensor's key/shape/dtype/bytes/hash/trainable/hyper_factors
recorded. Permit only explicitly coded constructors for these pinned official
classes: SequentialLayer, IdentityLayer, ConcatParallelLayer, FilterTensorsLayer,
RenameTensorLayer, FunctionLayer, BiasLayer, ScaleLayer, WeightLayer,
ParametricActivationLayer, DropoutLayer, NormalizeOutputLayer, ClampLayer,
PLREmbeddingsLayer, PLREmbeddingsLayerCosBias, EncodingLayer, SingleOneHotLayer,
SingleEmbeddingLayer. Reject unknown classes/fields/callables rather than import
a class name from the JSON. Admitted functions are exact pinned smooth_clip_func
with its finite max_abs_value and torch.relu/torch.selu/native mish. Preserve
all original factors, one-hot flags/onoff values and embedding flags.

Construct official native Layer objects using the closed dispatch table and
Variable constructor; call their exact native forward_tensors in evaluation
mode. Training-only fitter metadata and scheduling/controller references are
not needed by this dictionary-tensor prediction path. Record those omissions
explicitly and reject training/resume operations. Dropout's inactive evaluation
getter receives its recorded finite probability through a native constant
HyperparamManager; this is an inference-only representation, never a recovered
training state. Store context scope and variable hyper_factors for audit, with
training still prohibited. No eval, generic pickle, model retraining or retained
F/G observations on load.

The F-fitted official converter contributes schema, category order and exact
OrdinalEncoder category arrays. Reconstruct only its explicit F vocabulary via
a synthetic vocabulary carrier with zero numeric fields, then assert native
codes/cardinalities/order; this fits no numeric statistics. The carrier contains
at most max(F category cardinalities) rows and no original observations or
targets. Query encoding remains the official converter, not a handwritten
replacement. Typed canonicalization/unknown0 semantics remain the declared
F-only boundary adaptation. A numeric-only carrier needs one zero row.

Run static forward_tensors once before the native1024-row prediction batches,
matching pinned get_predict_dataloader ordering. Batch via native
ParallelDictDataLoader and concatenate native outputs in the exact same order.
Require bit-identical predictions against the official trained interface and
a fresh-process loaded snapshot, including unknown categories and constant
numeric/target controls. Each admitted none/PL/PBLD/PLR and activation graph
requires its own tiny engineering check; unqualified graphs fail explicitly.
No full256-epoch quality anchor follows from these tests.

## Binding, validation and costs

Require a trusted external expected manifest SHA256. Check source, exact private
overlay/core distributions and selected actual module/native-image RECORD bytes
on every fresh load before native use. Bind resolved factory config, mode, seed,
F-only converter vocabulary, model/static graph, every tensor and selected
native RMSE epoch to one immutable manifest. Validate nonnegative finite MSE/RMSE
histories, full declared epoch/update counts and native latest-best<= selection;
engineering fixtures cannot become quality checkpoints through edited mode flags.
Model dimensions/functions/embedding/layout must agree with resolved native
parameters, not merely a self-rehashed JSON declaration. Preserve failed loads.

Storage is native tensors plus NPZ headers and bounded JSON graph/vocabulary/
provenance; measure exact file bytes and fresh load time/RSS in fixtures. The
observed tensors are approximately0.52MiB; no data archive or training replay is
required. Formal deployment/resource and native training recovery qualification
remain pending GateD. This proposal introduces no extra package dependency.

## Approved finite engineering graph matrix

Root approved this representation before persistence implementation. Enumerate
12 graph fixtures in canonical embedding order none/PL/PBLD/PLR and activation
order ReLU/SELU/Mish. At case index i=0..11, use hidden layout
`[[256,256,256],[64,64,64,64,64],[512]][i % 3]`, front-scale
`bool(i % 2)`, mixed-category input for odd i, numeric-only input for even i.
Each has exactly three engineering epochs, seed17 and the declared synthetic
fixture stream2026101819. This is a finite graph coverage matrix, not HPO or
independent predictive evidence. Existing default-PBLD/Mish numeric and mixed
fixtures plus numeric constant-target/latest-tie control remain separate.
Every case compares direct official and fresh-process loaded predictions exactly;
all failures/cumulative unvisited F rows remain visible. No full256-epoch anchor.
