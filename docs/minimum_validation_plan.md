# Minimum Validation Plan — LiNexa

Goal: decide as cheaply as possible whether the episode-level fast-memory idea is worth pursuing.

## Phase 0 — baselines and infrastructure (no LiNexa code yet)

### 0.1 Reproduce the base Psi0 number on SIMPLE

Before adding anything:

1. Use the **smallest** SIMPLE task first. Candidate: `simple/G1WholebodyBendPick-v0` (it is the default in `extern/psi0/examples/simple/simple_eval.py:65`) or an even simpler tabletop pick if the `configs/tasks/simple_pick.yaml` target is smaller. Pick one. Confirm env_id and max_episode_steps (defaults to 360 in simple_eval.py:79).
2. Run the existing Psi0 checkpoint through `python extern/psi0/examples/simple/simple_eval.py --run-dir … --num-episodes 20 …`. Record closed-loop success rate on 20 episodes. This is the **reference success rate**.
3. Record per-episode `frame_idx` and `duration_seconds` from `EvalEpisodeResult` — these tell us the per-step wall clock budget for any added TTT compute.

**Stop condition at phase 0:** if the base Psi0 checkpoint does not reproduce on SIMPLE within expected bounds, fix infrastructure before building anything. No amount of fast-memory magic rescues a broken baseline.

### 0.2 Open-loop teacher-forced evaluation

Use `extern/psi0/examples/simple/openloop_eval.ipynb` (listed alongside simple_eval.py). In open-loop we replay a logged episode and measure per-step action MSE / gripper accuracy between predicted and ground-truth actions. No env feedback involved.

This gives us a cheap reference curve: "what does the frozen base policy look like when everything is teacher-forced?" It's the upper bound on what better *prediction* alone can buy.

## Phase 1 — first cheap signal (no retraining)

The absolute cheapest smoke test we can run to see *any* signal from an online write:

### 1.1 Identity-write, identity-read sanity

- No new learned parameters. Set `ttt_proj = I`, `ttt_conv = identity` (kernel-size-1 with weight 1), `ttt_lr` small (e.g. 0.01).
- Target `t = h` (write the key back onto itself, pure associative memory). This is the degenerate case: each forward pass's read retrieves something like `Σ_s ⟨h_t, h_s⟩ · h_s` added to the normal MLP output.
- Run 20 episodes closed-loop. Compare success rate to base Psi0.

**What to expect:** degradation or neutral. A positive delta here would be almost magical — the fast weights are untrained and untargeted.

**Why bother:** this verifies (i) the hook points work, (ii) resetting on episode boundary works, (iii) the eval harness runs end-to-end with a modified policy. It is an *infrastructure* smoke test, not a scientific test. Time budget: 1 day.

### 1.2 First real target — proprio delta, no training

- `Φ(outcome_t) = linear(Δproprio_t)` with the linear layer initialized to something reasonable (e.g. mean-centered PCA over logged rollouts) and frozen.
- `ttt_proj = I` still. `ttt_lr ∈ {1e-3, 1e-2, 1e-1}`.
- 20 episodes × 3 LR × {with / without TTT} = 120 episodes. Still one day.

**What to expect:** neutral at best; possibly small positive on tasks where proprio shift is strongly diagnostic (e.g. locomotion pick where stance drift is large). This is the first point where we could see weak evidence of the hypothesis.

**Stop condition:** if 1.1 crashes the policy (large regression on most episodes even at `ttt_lr = 1e-3`) or 1.2 is indistinguishable from noise across *all* tasks, the hypothesis that retrieval + random outcome embeddings help is falsified. This is the first serious go/no-go.

## Phase 2 — closed-loop with a lightly trained write head

Only proceed here if Phase 1 is neutral-or-better and does not regress.

### 2.1 Train a write head offline from Psi0 rollouts

- Freeze Psi0 entirely. Add `ttt_proj` (hidden→hidden linear) and a small `Φ` that maps (proprio delta, obs-delta summary, optionally progress) to a vector of dimension = hidden.
- **Training objective (hypothesis, to be validated):** minimize per-step action error at step `t+k` conditioned on (keys from [t, t+k-1], outcomes from [t, t+k-1]). Concretely: replay logged episodes, simulate the fast-memory writes the current `Φ` and `ttt_proj` would have produced, forward the action head with those fast weights, compute flow-matching loss against ground-truth action. Backprop through `Φ` and `ttt_proj` only.
- Compute budget: one GPU-day to start; limit to the same small task as Phase 1.

### 2.2 Closed-loop eval

- 50 episodes × {base Psi0, LiNexa-trained}.
- Report: success rate, per-step action MSE, step-to-failure distribution.

**Early positive signal:** +5 percentage points success rate or better **while not regressing tail behavior**. A LiNexa run that improves mean success but has more catastrophic failures (e.g. self-collisions, fall-off-table) is not a win.

**Honest falsification:** if after Phase 2 the gap vs. base Psi0 is within noise (<2 pp across 50 episodes, bootstrap CI overlapping zero), the idea as currently formulated does not work on this task. Two options:
1. Try one different task where the hypothesis is stronger (e.g. a bimanual handover task where episode-specific pose shift is large). One task only.
2. Stop.

## Phase 3 — ablations and scope (only if Phase 2 shows promise)

If and only if Phase 2 produces a positive signal:

- **Which block(s)?** Ablate TTT in {block 0}, {block 3}, {block 5}, {all 6}. Expect a clear "middle or late" pattern.
- **Reset sanity.** Disable the episode reset. If performance changes by less than noise, the fast memory is not doing anything episode-specific — a red flag.
- **Target ablation.** Replace outcome-aligned `Φ` with a zero target. If performance is similar, the signal is from the hook, not the targets. Red flag.
- **Decay.** `W_t = γ · W_{t-1} + ΔW` with γ ∈ {1.0, 0.95, 0.9}.
- **Learning rate curve.** A clear sweet spot for `ttt_lr` should exist.
- **Longer eval.** 200 episodes instead of 50. Confirm the early 50-episode signal is not a seed artifact.

## Metrics that matter at each phase

| Phase | Primary | Secondary | Debugging |
|---|---|---|---|
| 0 | Reference success rate (20 eps) | Wall time/step | Per-step action MSE (open loop) |
| 1.1 | Success rate Δ vs base | Any catastrophic regression flag | Hook exercised (verify by logging) |
| 1.2 | Success rate Δ vs base | Regression tail count | Per-step fast-weight norm; saturation |
| 2.2 | Success rate Δ with 95% CI | Step-to-failure CDF | Write/read stats, key similarity distribution |
| 3 | All of the above, averaged over seeds | Ablation ordering | Layer-wise attribution |

## What counts as an early positive signal

- Phase 1.2: the *sign* of the success-rate delta is consistently positive across 3 `ttt_lr` values and the variance is not larger than the baseline variance.
- Phase 2.2: +5 pp mean success rate on 50 eps with the lower bound of a 95% bootstrap CI being >0, and no increase in catastrophic failures.

## What counts as a stop signal

- Any phase: a significant *regression* that cannot be attributed to a specific, fixable bug (e.g. wrong reset wiring).
- Phase 2.2: gap within noise on the chosen task AND on one alternative task.
- Any phase: the write head silently collapses to zero norm, or produces saturating fast-weight norms (divergence) regardless of `ttt_lr`. This indicates the hypothesis that outcome-aligned writes are well-defined is broken.

## Compute and calendar budget

| Phase | GPU-days | Calendar |
|---|---|---|
| 0 | 0.5 | ~1 day |
| 1.1 | 0.3 | ~0.5 day |
| 1.2 | 0.5 | ~1 day |
| 2.1 + 2.2 | 1–2 | ~3 days |
| 3 | 3–5 | ~1 week |
| **Total to decision point (end of Phase 2)** | **~3 GPU-days** | **~5–6 days** |

Go/no-go after Phase 2 is the decision point. Do not commit to Phase 3 until then.
