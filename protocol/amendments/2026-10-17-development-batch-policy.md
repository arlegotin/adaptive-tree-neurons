# Pre-freeze development batch-policy amendment

Date: 17 October 2026. No genuine pilot, frozen H1 protocol or protected test
access exists. No independent cohort is being adapted after its results.

The original complete-epoch throughput heuristic selected a requested batch of
4096. Actual complete-fit measurements show that this point benchmark does not
capture full-study retention. The reviewed checkpoint correction preserves raw
logs; the 4096/0.005 parent still retains 155.2 MB of them, versus 10.3 MB for
128/0.005 on the same development problem. Their descriptive G minima and
single-run fit clocks do not show an advantage sufficient to make the larger
retention a required engineering choice.

Use complete-pipeline development learning, time and storage to choose the
common batch; proceed with 128 for finite calibration. Preserve the historical
95% epoch result, all runs and all raw row streams. Keep logical and physical
batch equal, CPU float32 training/float64 scoring, shared optimizer treatment,
paired data order, common allowances and the original H1 contrast.

[Root Ruling 17](../../audit/implementation/evidence/2026-10-17/ruling-17-complete-study-batch-policy.md)
records the measurements, exact next calibration and cost if wrong. The final
protocol must state the chosen recipe and its boundary. This amendment makes no
claim about other batches or a general advantage for selective allocation.
