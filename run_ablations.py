"""Run a sweep of N-value ablations for MLP and/or GRU world models.
For each N, train + eval, then collect rollout-MSE metrics into a summary JSON + printed table.

Usage:
    # MLP only (default)
    python run_ablations.py --db_dir_name 1000000_transitions_good

    # GRU only
    python run_ablations.py --db_dir_name 1000000_transitions_good --model_type wm_gru

    # Both MLP and GRU
    python run_ablations.py --db_dir_name 1000000_transitions_good --model_type wm_mlp wm_gru

    # Custom N values
    python run_ablations.py --db_dir_name 1000000_transitions_good --N_values 1 8

    # Skip training, eval only (uses most recent checkpoint)
    python run_ablations.py --db_dir_name 1000000_transitions_good --skip-train
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# ── Checkpoint discovery ─────────────────────────────────────────────────────

def _find_latest_run_dir(model_type: str, since_ts: float) -> Path | None:
    """Find the most recently modified run directory created after `since_ts`."""
    search_dirs = []
    if model_type == "wm_mlp":
        search_dirs.append(Path("checkpoints") / model_type)
    else:
        search_dirs.append(Path("logs") / "ckpts")

    for base in search_dirs:
        if not base.is_dir():
            continue
        candidates = [d for d in base.iterdir() if d.is_dir() and d.stat().st_mtime >= since_ts - 5]
        if candidates:
            return max(candidates, key=lambda d: d.stat().st_mtime)
    return None


def _last_epoch_ckpt(run_dir: Path) -> Path | None:
    """Find the checkpoint with the highest epoch number in a run directory."""
    pat = re.compile(r"-ckpt-epoch_(\d+)\.pth$")
    candidates = []
    for p in run_dir.iterdir():
        m = pat.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    return max(candidates, key=lambda kv: kv[0])[1]


def find_existing_run(model_type: str) -> Path | None:
    """Best-effort lookup of an already-trained run (used with --skip-train)."""
    search_dirs = []
    if model_type == "wm_mlp":
        search_dirs.append(Path("checkpoints") / model_type)
    else:
        search_dirs.append(Path("logs") / "ckpts")

    for base in search_dirs:
        if not base.is_dir():
            continue
        runs = [d for d in base.iterdir() if d.is_dir()]
        if runs:
            return max(runs, key=lambda d: d.stat().st_mtime)
    return None


# ── Training ─────────────────────────────────────────────────────────────────

def run_training(N: int, db_dir: str, model_type: str) -> Path:
    cmd = [
        sys.executable, "main.py",
        "--run_mode", "train",
        "--model_type", model_type,
        "--db_dir_name", db_dir,
        "--N", str(N),
    ]
    print(f"[ablation {model_type} N={N}] training: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    run_dir = _find_latest_run_dir(model_type, since_ts=t0)
    if run_dir is None:
        raise RuntimeError(f"[ablation {model_type} N={N}] no run dir found after training")
    print(f"[ablation {model_type} N={N}] training done in {time.time() - t0:.0f}s, run_dir={run_dir}")
    return run_dir


# ── Evaluation ───────────────────────────────────────────────────────────────

def run_eval_mlp(ckpt_path: Path, db_dir: str, tag: str, eval_seed: int) -> dict:
    cmd = [
        sys.executable, "eval_mlp.py",
        "--wm-checkpoint", str(ckpt_path),
        "--db_dir_name", db_dir,
        "--wm-model-name", tag,
        "--all-joints",
        "--eval-seed", str(eval_seed),
    ]
    print(f"[ablation tag={tag}] eval MLP: {' '.join(cmd)}")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(cmd, check=True, env=env)
    metrics_path = Path("logs") / "plots" / tag / f"{tag}_metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"[ablation tag={tag}] metrics file missing at {metrics_path}")
    with open(metrics_path) as f:
        return json.load(f)


def run_eval_gru(ckpt_path: Path, db_dir: str, tag: str, eval_seed: int) -> dict:
    cmd = [
        sys.executable, "main.py",
        "--run_mode", "eval",
        "--model_type", "wm_gru",
        "--wm-checkpoint", str(ckpt_path),
        "--model_dir_name", tag,
        "--model_name", tag,
        "--db_dir_name", db_dir,
        "--eval_seed", str(eval_seed),
    ]
    print(f"[ablation tag={tag}] eval GRU: {' '.join(cmd)}")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(cmd, check=True, env=env)
    metrics_path = Path("logs") / "plots" / tag / f"{tag}_metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"[ablation tag={tag}] metrics file missing at {metrics_path}")
    with open(metrics_path) as f:
        return json.load(f)


def run_eval(model_type: str, ckpt_path: Path, db_dir: str, tag: str, eval_seed: int) -> dict:
    if model_type == "wm_mlp":
        return run_eval_mlp(ckpt_path, db_dir, tag, eval_seed)
    else:
        return run_eval_gru(ckpt_path, db_dir, tag, eval_seed)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="N-value ablation sweep for MLP and/or GRU world models.")
    p.add_argument("--db_dir_name", required=True,
                   help="Subfolder under data/ containing rollout .db files.")
    p.add_argument("--N_values", nargs="+", type=int, default=[1, 8],
                   help="N values to sweep (default: 1 8)")
    p.add_argument("--model_type", nargs="+", default=["wm_mlp"],
                   help="Model type(s) to ablate: wm_mlp, wm_gru, or both (default: wm_mlp)")
    p.add_argument("--out", default="ablations_summary.json",
                   help="Path to write aggregate summary JSON.")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; use most recent checkpoint per model type.")
    p.add_argument("--eval-seed", type=int, default=42,
                   help="RNG seed for eval trajectory selection (default 42).")
    args = p.parse_args()

    summary: list[dict] = []

    for model_type in args.model_type:
        tag_prefix = f"{model_type}_ablation"
        for N in args.N_values:
            tag = f"{tag_prefix}_N{N}"
            entry: dict = {"model_type": model_type, "N": N, "tag": tag}
            try:
                t0 = time.time()
                if args.skip_train:
                    run_dir = find_existing_run(model_type)
                    if run_dir is None:
                        raise RuntimeError(f"--skip-train: no run dir found for {model_type}")
                    print(f"[ablation {model_type} N={N}] --skip-train, using run_dir={run_dir}")
                else:
                    run_dir = run_training(N, args.db_dir_name, model_type)

                ckpt = _last_epoch_ckpt(run_dir)
                if ckpt is None:
                    raise RuntimeError(f"no -ckpt-epoch_*.pth found in {run_dir}")
                entry["ckpt"] = str(ckpt)
                entry["run_dir"] = str(run_dir)

                metrics = run_eval(model_type, ckpt, args.db_dir_name, tag, eval_seed=args.eval_seed)
                entry["metrics"] = metrics
                entry["wall_time_s"] = time.time() - t0
            except Exception as e:
                print(f"[ablation {model_type} N={N}] FAILED: {e}")
                entry["error"] = str(e)
            finally:
                summary.append(entry)
                with open(args.out, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"[ablation] partial summary -> {args.out}")

    # ── Print table ──
    print("\n" + "=" * 109)
    print(f"{'model':>8} {'N':>3} | {'lin_vel':>10} {'ang_vel':>10} {'proj_grav':>10} "
          f"{'jt_pos':>10} {'jt_vel':>10} {'jt_tau':>10} {'contact':>10} {'foot_c':>10} | {'time(s)':>8}")
    print("=" * 109)
    for r in summary:
        if "error" in r:
            print(f"{r['model_type']:>8} {r['N']:>3} | FAILED: {r['error']}")
            continue
        m = r.get("metrics", {})
        print(f"{r['model_type']:>8} {r['N']:>3} | "
              f"{m.get('base_lin_vel_rmse', float('nan')):>10.5f} "
              f"{m.get('base_ang_vel_rmse', float('nan')):>10.5f} "
              f"{m.get('proj_gravity_rmse', float('nan')):>10.5f} "
              f"{m.get('joint_pos_rmse', float('nan')):>10.5f} "
              f"{m.get('joint_vel_rmse', float('nan')):>10.5f} "
              f"{m.get('joint_tau_rmse', float('nan')):>10.5f} "
              f"{m.get('contact_rmse', float('nan')):>10.5f} "
              f"{m.get('foot_contact_rmse', float('nan')):>10.5f} | "
              f"{r.get('wall_time_s', 0):>8.0f}")
    print("=" * 109)
    print(f"\nfull summary written to {args.out}")


if __name__ == "__main__":
    main()
