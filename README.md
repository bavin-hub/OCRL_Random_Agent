# OCRL_Random_Agent

OCRL project — GRU & MLP world models with privileged contact prediction.

---

## Training

### MLP

```bash
python train_mlp.py \
  --db_dir_name "1000000_transitions_good"
```

Checkpoints saved to `checkpoints/wm_mlp/wm_mlp_<TIMESTAMP>/`.

### GRU

```bash
python main.py \
  --run_mode train \
  --model_type wm_gru \
  --db_dir_name "1000000_transitions_good"
```

Checkpoints saved to `logs/ckpts/wm_gru_<TIMESTAMP>/`.

Optional: override rollout horizon with `--N <int>`.

---

## Evaluation

> Both scripts default to `--eval_seed 42` and automatically use `data/<db_dir_name>_eval/` for eval data.

### MLP

```bash
python eval_mlp.py \
  --wm-checkpoint checkpoints/wm_mlp/<TAB-COMPLETE-PATH>.pth \
  --db_dir_name "1000000_transitions_good" \
  --wm-model-name <EVAL_NAME> \
  --all-joints
```

- `--wm-model-name` sets **both** the plot filename prefix and the output folder under `logs/plots/`.
- Plots saved to `logs/plots/<EVAL_NAME>/`.

**Example:**
```bash
python eval_mlp.py \
  --wm-checkpoint checkpoints/wm_mlp/wm_mlp_2026-05-06_20:29:23/wm_mlp-ckpt-epoch_1.pth \
  --db_dir_name "1000000_transitions_good" \
  --wm-model-name mlp_w_contacts_e1 \
  --all-joints
```

### GRU

```bash
python main.py \
  --run_mode eval \
  --model_type wm_gru \
  --wm-checkpoint logs/ckpts/<TAB-COMPLETE-PATH>.pth \
  --db_dir_name "1000000_transitions_good"
```

- `--wm-checkpoint` = full path to checkpoint file (tab-completable!).
- Plot folder name is auto-derived from the checkpoint's parent directory.
- Plots saved to `logs/plots/<parent_dir_name>/`.

**Example:**
```bash
python main.py \
  --run_mode eval \
  --model_type wm_gru \
  --wm-checkpoint logs/ckpts/wm_gru_2026-05-06_21:38:19/wm_gru-ckpt-epoch_50.pth \
  --db_dir_name "1000000_transitions_good"
```

---

## Comparing MLP vs GRU

Both eval scripts use `--eval_seed 42` by default so they select the **exact same trajectory window**. To compare on a different trajectory, pass the same seed to both:

```bash
python eval_mlp.py  --eval-seed 123 ...
python main.py      --eval_seed 123 --run_mode eval ...
```

Plots are saved to `logs/plots/`. Metrics (RMSE) are saved as `*_metrics.json` alongside the plots.
