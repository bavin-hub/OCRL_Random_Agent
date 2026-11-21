# Validation and limitations

Validation date: November 21, 2026. Historical base commit:
`7f500c3891dbfbbe37dde79c28d749df2700784b` (`wm_on_the_fly_training`).
The original working tree contained changes to training, reward scale, evaluation,
and configuration; those were preserved before cleanup outside the repository.
Historical checkpoints cannot be attributed to a fully captured configuration.

## What was checked

- Policy 3710 and world-model 3710 deserialize on CPU and load into their architectures.
- GPU evaluation and MP4 recording run on an RTX 5090 with clean MJLab 1.2.0 and
  RSL-RL 5.0.1 wheels, using the existing environment's other pinned dependencies.
- A four-iteration, eight-environment GPU training test reaches imagination PPO
  updates and writes policy/world-model checkpoints and CSV metrics.
- A separate three-iteration GPU initialization run restored both historical
  models and optimizer states and completed new updates/saves.
- CPU tests cover observation layout, model forward/backward, circular replay
  chronology, reset boundaries, and insufficient history.
- Checksum manifests identify selected historical checkpoints.

This is not yet a full installation test in an environment with every dependency
installed from scratch. Interactive viewer behavior has a shared preprocessing
path but was not manually exercised; offscreen recording was exercised.

## Evaluation protocol

One environment, seed 42, 1000 steps at 0.02 seconds/step, fixed command
`(0.5, 0, 0)`, scaled previous-action input, no observation corruption or pushes,
no command curriculum. Startup and reset randomization remain enabled. Contact
capacity is 128 to avoid the overflow observed during an initial exploratory run.

The 7030 checkpoint stayed upright until the 1000-step timeout, but its forward
velocity error was approximately 0.499 m/s against a 0.5 m/s command: it mostly
stood still. Checkpoint 3710 had 36 non-timeout episode endings, averaging 27.44
steps per completed episode. These are single-seed diagnostics, not a benchmark
or evidence of successful command-following locomotion.

Forecast plots use one uninterrupted trajectory: 32 warmup states followed by
8 prediction steps (0.16 seconds). Predictions are deterministic conditional means
with recorded simulator actions. The aggregate normalized RMSE is sensitive to
state scaling and trajectory selection; it is not a validation-set generalization
estimate. Ground truth and predicted arrays are retained with the results.

## Changes made during cleanup

- Removed tracked bytecode, dead model variants, empty/scratch code, and a broken
  transformer experiment; added ignore rules for generated artifacts.
- Deleted only byte-identical copies from `logs (copy)` after SHA-256 comparison.
  Distinct checkpoints and runs remain local.
- Added CLI controls, per-run output directories, saved config/arguments, metrics,
  explicit checkpoint paths, and deterministic evaluation seeding.
- Unified headless and viewer policy preprocessing; introduced video and data export.
- Fixed replay-buffer chronology after wraparound and rejected insufficient history.
- Detached world-model loss before accumulating it for reporting.
- Preserved real-simulator action history across training iterations instead of
  clearing it every iteration; refresh real observations after imagination.
- Clarified checkpoint initialization versus exact resume.

These corrections change new training behavior. Historical checkpoint files have
not been modified. No long retraining run has been performed after cleanup.

## Algorithm issues still requiring a research pass

1. Historical real rollouts feed scaled previous joint targets to the policy, while
   imagination feeds normalized targets. Both evaluation conventions are exposed
   explicitly; the default reproduces the historical real-rollout path.
2. Custom reward code receives normalized world-model states but uses constants
   expressed in physical units. Its inferred joint count is also inconsistent with
   the 96-dimensional state layout. This needs a deliberate reward/normalization
   correction followed by retraining, not a cosmetic adjustment to reported results.
3. Real simulator termination flags are replaced by custom termination flags in the
   historical training loop. Simulator resets and replay sequence boundaries need
   to be reconciled before relying on a new long training run.
4. The world model stores world-frame velocities while velocity commands are
   body-frame. Reward-frame conventions need an explicit design and tests.
5. Exact resume requires saving/restoring RNG, replay, simulator, and rollout state.
6. Historical full learning curves are unavailable. Only newly recorded metrics
   can be plotted; existing checkpoint loss values do not reconstruct policy learning.

These limitations are intentionally visible. A training smoke test demonstrates
execution, not algorithm correctness or convergence. The next research milestone
is consistent observations/rewards/resets, followed by repeatable locomotion
improvement across seeds and a matched simulator-only PPO baseline.

## Before publication

- The owner selected publication of the current experimental snapshot; a successful
  locomotion result or paper reproduction is not claimed.
- Upload the selected checkpoint bundle and replace the local-only checkpoint note
  with the actual download URL. No public download URL is fabricated.
- The obsolete copied dependency scripts have been retired. The project
  license remains undecided at the owner's request; dependencies retain their licenses.

## Repository restructure validation

The supported implementation now lives in `src/ocrl`, installed through
`pyproject.toml`. Command wrappers are under `scripts`, configuration is under
`configs`, and report/media files are grouped by purpose under `docs`. Obsolete
offline/baseline code and the uninitialized baseline submodule have been removed.
Historical checkpoint files were moved intact to ignored `artifacts/historical`.

Final checks passed: editable package installation; six CPU tests; the same tests
from a relocated source copy; CLI help; all six historical checkpoint checksums;
four-iteration GPU fresh training; three-iteration GPU checkpoint initialization;
80-step GPU evaluation with video and forecast export; plotting; syntax, links,
and machine-specific path scanning. New run manifests store relative paths.
The final smoke checks do not change the previously reported experimental results.

## Locomotion baseline and pipeline animation

Two separate simulator-trained G1 policies (1199 and 1000) were evaluated with
standard MJLab actor observations, seed 42, a fixed 0.5 m/s forward command, and
1000 steps. Both completed the 20-second episode without a fall. Checkpoint 1199
had forward-velocity RMSE 0.232 m/s versus 0.253 m/s for 1000, and was selected
for the animation. Its side-view rerun averaged 0.271 m/s forward velocity.
These diagnostics establish a usable baseline demonstration, not reliable tracking
across seeds or evidence that the online world-model training produced this policy.

`docs/baselines.json` records checkpoint hashes. Baseline 1199 data and manifest
are in `docs/results/baseline-1199`. The 67-second pipeline video uses the report's
architecture figures, animated data/optimization flow, and real-time simulator
footage. It is captioned and silent. Seven CPU tests pass, including a check that
the baseline adapter preserves the original actor observations. The rendered MP4
was checked for dimensions, duration, decode errors, and representative chapter frames.

The pipeline animation now presents the requested intended sequence on a white
background with moving transitions, active nodes, GRU recurrence, optimizer
feedback, and an explicit simulator-to-world-model PPO switch. Initial simulator
PPO updates are an intended step, not a claim about the current warm-start code,
which only collects transitions and fits the world model.
