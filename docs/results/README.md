# Recorded diagnostics

See `../validation.md` and each run's `manifest.json` for the protocol.
`metrics.csv` contains per-step simulator reward and body-frame command-tracking errors.
`summary.json` contains completed episodes and forecast error. `predictions.npz`
contains normalized predicted/actual states and the sample timestep.

7030 media and data use the final 640×480 presentation recording. 3710 uses the
320×240 diagnostic recording. Dynamics settings are otherwise matched. Small
floating-point differences between reruns are expected. These are single-seed
results, not a benchmark or paper reproduction.

Regenerate plots: `python scripts/plot_results.py docs/results/7030`.
