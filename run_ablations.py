"""Run a sweep of N-value ablations: for each N, train with train_mlp + eval with eval_mlp,
then collect rollout-MSE metrics into a single summary JSON + printed table.

Usage:
    python run_ablations.py --db_dir_name 1000000_transitions_good
    python run_ablations.py --db_dir_name <dir> --N_values 1 8 16
    python run_ablations.py --db_dir_name <dir> --N_values 4 8 --skip-train  # eval only

Each ablation gets its own checkpoint subdir under checkpoints/wm_mlp/<timestamped_run>/
and its own plot dir under logs/plots/wm_mlp_ablation_N<N>/.
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


def _find_latest_run_dir(model_type: str, since_ts: float) -> Path | None:
    base = Path("checkpoints") / model_type
    if not base.is_dir():
        return None
    candidates = [d for d in base.iterdir() if d.is_dir() and d.stat().st_mtime >= since_ts - 5]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _last_epoch_ckpt(run_dir: Path) -> Path | None:
    pat = re.compile(r"-ckpt-epoch_(\d+)\.pth$")
    candidates = []
    for p in run_dir.iterdir():
        m = pat.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    return max(candidates, key=lambda kv: kv[0])[1]


def run_training(N: int, db_dir: str, model_type: str = "wm_mlp") -> Path:
    cmd = [
        sys.executable, "main.py",
        "--run_mode", "train",
        "--model_type", model_type,
        "--db_dir_name", db_dir,
        "--N", str(N),
    ]
    print(f"[ablation N={N}] training: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    run_dir = _find_latest_run_dir(model_type, since_ts=t0)
    if run_dir is None:
        raise RuntimeError(f"[ablation N={N}] no run dir found under checkpoints/{model_type}/ after training")
    print(f"[ablation N={N}] training done in {time.time() - t0:.0f}s, run_dir={run_dir}")
    return run_dir


def run_eval(ckpt_path: Path, db_dir: str, tag: str) -> dict:
    cmd = [
        sys.executable, "eval_mlp.py",
        "--wm-checkpoint", str(ckpt_path),
        "--db_dir_name", db_dir,
        "--wm-model-name", tag,
        "--wm-model-dir-name", tag,
    ]
    print(f"[ablation tag={tag}] eval: {' '.join(cmd)}")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(cmd, check=True, env=env)
    metrics_path = Path("logs") / "plots" / tag / f"{tag}_metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"[ablation tag={tag}] metrics file missing at {metrics_path}")
    with open(metrics_path) as f:
        return json.load(f)


def find_existing_run(model_type: str, N: int) -> Path | None:
    """Best-effort lookup of an already-trained run for this N (used with --skip-train).
    No metadata is currently saved alongside the checkpoint, so we can't filter by N reliably;
    returns the most recent run dir under checkpoints/<model_type>/. Caller should verify.
    """
    base = Path("checkpoints") / model_type
    if not base.is_dir():
        return None
    runs = [d for d in base.iterdir() if d.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda d: d.stat().st_mtime)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db_dir_name", required=True,
                   help="Subfolder under data/ containing rollout .db files.")
    p.add_argument("--N_values", nargs="+", type=int, default=[1, 8, 16],
                   help="N values to sweep (default: 1 8 16)")
    p.add_argument("--model_type", default="wm_mlp")
    p.add_argument("--tag-prefix", default="wm_mlp_ablation",
                   help="Prefix for plot directory / model name tags.")
    p.add_argument("--out", default="ablations_summary.json",
                   help="Path to write aggregate summary JSON.")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; locate latest run dir per N (untagged, so you must "
                        "have just trained one model with the matching N).")
    args = p.parse_args()

    summary: list[dict] = []
    for N in args.N_values:
        tag = f"{args.tag_prefix}_N{N}"
        entry: dict = {"N": N, "tag": tag}
        try:
            t0 = time.time()
            if args.skip_train:
                run_dir = find_existing_run(args.model_type, N)
                if run_dir is None:
                    raise RuntimeError(f"--skip-train: no run dir found under checkpoints/{args.model_type}/")
                print(f"[ablation N={N}] --skip-train, using run_dir={run_dir}")
            else:
                run_dir = run_training(N, args.db_dir_name, args.model_type)

            ckpt = _last_epoch_ckpt(run_dir)
            if ckpt is None:
                raise RuntimeError(f"no -ckpt-epoch_*.pth found in {run_dir}")
            entry["ckpt"] = str(ckpt)
            entry["run_dir"] = str(run_dir)

            metrics = run_eval(ckpt, args.db_dir_name, tag)
            entry["metrics"] = metrics
            entry["wall_time_s"] = time.time() - t0
        except Exception as e:
            print(f"[ablation N={N}] FAILED: {e}")
            entry["error"] = str(e)
        finally:
            summary.append(entry)
            # Write partial summary after every iteration so a crash doesn't lose earlier results.
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[ablation] partial summary -> {args.out}")

    print("\n" + "=" * 99)
    print(f"{'N':>4} | {'lin_vel':>10} {'ang_vel':>10} {'proj_grav':>10} "
          f"{'jt_pos':>10} {'jt_vel':>10} {'jt_tau':>10} {'contact':>10} {'foot_c':>10} | {'time(s)':>8}")
    print("=" * 99)
    for r in summary:
        if "error" in r:
            print(f"{r['N']:>4} | FAILED: {r['error']}")
            continue
        m = r.get("metrics", {})
        print(f"{r['N']:>4} | "
              f"{m.get('base_lin_vel_rmse', float('nan')):>10.5f} "
              f"{m.get('base_ang_vel_rmse', float('nan')):>10.5f} "
              f"{m.get('proj_gravity_rmse', float('nan')):>10.5f} "
              f"{m.get('joint_pos_rmse', float('nan')):>10.5f} "
              f"{m.get('joint_vel_rmse', float('nan')):>10.5f} "
              f"{m.get('joint_tau_rmse', float('nan')):>10.5f} "
              f"{m.get('contact_rmse', float('nan')):>10.5f} "
              f"{m.get('foot_contact_rmse', float('nan')):>10.5f} | "
              f"{r.get('wall_time_s', 0):>8.0f}")
    print("=" * 99)
    print(f"\nfull summary written to {args.out}")


if __name__ == "__main__":
    main()
