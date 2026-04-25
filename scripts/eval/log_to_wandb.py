#!/usr/bin/env python3
"""Stream per-episode SIMPLE eval results into Weights & Biases.

Polls a per-task log directory for ``episode_N/head_stereo_*_{success,failed}.mp4``
files newer than ``--leg-start-ts`` and emits a wandb point per episode. Exits
when ``--done-flag`` exists or all ``--num-episodes`` are seen, then logs
final aggregate stats parsed from ``--eval-stats``.

This script is designed to run inside an ephemeral ``psi0:latest`` container so
that wandb is on the venv path and only the host ``logs/eval/`` directory needs
to be bind-mounted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import wandb

EPISODE_RE = re.compile(r"^episode_(\d+)$")
RESULT_RE = re.compile(r"_(success|failed)\.mp4$")
STATS_LINE_RE = re.compile(r"^episode_(\d+):\s*(True|False)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--mode", required=True, choices=["baseline", "linexa"])
    p.add_argument("--task", required=True)
    p.add_argument("--dr", required=True)
    p.add_argument("--num-episodes", type=int, required=True)
    p.add_argument("--watch-dir", required=True,
                   help="Path containing episode_N/ subdirs (e.g. .../psi0_decoupled_wbc/<task>/<dr>)")
    p.add_argument("--eval-stats", required=True,
                   help="Path to logs/eval/eval_stats.txt; parsed from end-of-run as ground truth.")
    p.add_argument("--done-flag", required=True,
                   help="File whose existence signals the eval orchestrator finished.")
    p.add_argument("--leg-start-ts", type=float, required=True,
                   help="Unix timestamp; mp4s with mtime <= this are ignored (avoid prior-run dirs).")
    p.add_argument("--repo-commit", required=True)
    p.add_argument("--linexa-layers", default="")
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--max-wait-s", type=float, default=6 * 3600)
    return p.parse_args()


def detect_episode(watch_dir: Path, ep_idx: int, leg_start_ts: float) -> tuple[bool | None, float | None]:
    """Return (success?, mtime) if episode ep_idx has produced a result mp4 newer than leg_start_ts."""
    ep_dir = watch_dir / f"episode_{ep_idx}"
    if not ep_dir.is_dir():
        return None, None
    best_mtime = -1.0
    success = None
    for mp4 in ep_dir.glob("head_stereo_*_*.mp4"):
        m = RESULT_RE.search(mp4.name)
        if not m:
            continue
        try:
            mtime = mp4.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime <= leg_start_ts:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            success = (m.group(1) == "success")
    if success is None:
        return None, None
    return success, best_mtime


def parse_final_stats(eval_stats_path: Path, leg_start_ts: float) -> dict[int, bool]:
    """Parse the LAST `===` block of eval_stats.txt (newer than leg_start_ts) into {ep: success}."""
    if not eval_stats_path.is_file():
        return {}
    if eval_stats_path.stat().st_mtime <= leg_start_ts:
        return {}
    text = eval_stats_path.read_text()
    blocks = [b for b in text.split("================") if b.strip()]
    if not blocks:
        return {}
    last = blocks[-1]
    out: dict[int, bool] = {}
    for line in last.splitlines():
        m = STATS_LINE_RE.match(line.strip())
        if m:
            out[int(m.group(1))] = (m.group(2) == "True")
    return out


def main() -> int:
    args = parse_args()
    watch_dir = Path(args.watch_dir)
    eval_stats = Path(args.eval_stats)
    done_flag = Path(args.done_flag)

    wandb.init(
        project=args.project,
        name=args.name,
        config={
            "mode": args.mode,
            "task": args.task,
            "dr": args.dr,
            "num_episodes": args.num_episodes,
            "linexa_ttt_enabled": int(args.mode == "linexa"),
            "linexa_layers": args.linexa_layers or "all",
            "linexa_write_scale": 0.0,
            "linexa_decay": 0.0,
            "linexa_clip": 0.0,
            "repo_commit": args.repo_commit,
            "phase": "phase0",
            "leg_start_ts": args.leg_start_ts,
        },
        tags=["phase0", args.mode, args.task, args.dr],
    )
    print(f"[wandb-logger] init done. project={args.project} name={args.name} mode={args.mode}", flush=True)

    seen: dict[int, dict] = {}
    last_mtime = args.leg_start_ts
    deadline = time.time() + args.max_wait_s

    def running_stats():
        n = len(seen)
        succ = sum(1 for v in seen.values() if v["success"])
        return n, succ

    try:
        while time.time() < deadline:
            for ep in range(args.num_episodes):
                if ep in seen:
                    continue
                success, mtime = detect_episode(watch_dir, ep, args.leg_start_ts)
                if success is None:
                    continue
                wall_s = max(0.0, mtime - last_mtime)
                last_mtime = mtime
                seen[ep] = {"success": success, "mtime": mtime, "wall_s": wall_s}
                n, succ = running_stats()
                point = {
                    "episode": ep,
                    "success": int(success),
                    "episode_wall_s": wall_s,
                    "running_success": succ,
                    "running_episodes": n,
                    "running_success_rate": succ / max(1, n),
                }
                wandb.log(point, step=ep)
                print(
                    f"[wandb-logger] episode={ep} success={success} "
                    f"wall={wall_s:.1f}s running={succ}/{n}",
                    flush=True,
                )
            if len(seen) >= args.num_episodes:
                print("[wandb-logger] all episodes accounted for", flush=True)
                break
            if done_flag.exists():
                # Eval finished but some episodes may still be missing (e.g. crash). Wait briefly,
                # then proceed to final-stats parse so wandb still gets the truth from eval_stats.txt.
                print("[wandb-logger] done-flag present; final wait before stats parse", flush=True)
                time.sleep(min(15.0, args.poll_interval * 5))
                break
            time.sleep(args.poll_interval)

        final = parse_final_stats(eval_stats, args.leg_start_ts)
        for ep, ok in final.items():
            if ep in seen:
                continue
            seen[ep] = {"success": ok, "mtime": time.time(), "wall_s": float("nan")}
            wandb.log({"episode": ep, "success": int(ok),
                       "running_episodes": len(seen),
                       "running_success": sum(1 for v in seen.values() if v["success"]),
                       "running_success_rate": sum(1 for v in seen.values() if v["success"]) / max(1, len(seen)),
                       "from_eval_stats_only": 1}, step=ep)
            print(f"[wandb-logger] final-fill episode={ep} success={ok} (from eval_stats.txt)", flush=True)

        n, succ = running_stats()
        summary = {
            "final/num_episodes_observed": n,
            "final/num_episodes_target": args.num_episodes,
            "final/success_count": succ,
            "final/success_rate": succ / max(1, n),
            "final/incomplete": int(n < args.num_episodes),
        }
        wall_s = [v["wall_s"] for v in seen.values()
                  if isinstance(v["wall_s"], float) and v["wall_s"] == v["wall_s"] and v["wall_s"] > 0]
        if wall_s:
            summary["final/episode_wall_s_mean"] = sum(wall_s) / len(wall_s)
            summary["final/episode_wall_s_max"] = max(wall_s)
            summary["final/episode_wall_s_min"] = min(wall_s)
        for k, v in summary.items():
            wandb.run.summary[k] = v
        print("[wandb-logger] summary:", json.dumps(summary, indent=2), flush=True)
    finally:
        wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
