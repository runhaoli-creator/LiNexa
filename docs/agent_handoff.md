# LiNexa Agent Handoff

**Audience:** the next coding agent picking up this repo.
**Goal of this document:** let you understand the project, the current state of
Phase 0, and how to run the minimal validation experiment without regressing
the repo.

If something in this document conflicts with `plan.md`, `plan.md` wins. This
file is a synthesis; the canonical design discussion lives in:

- `idea.md` — research idea + scope
- `plan.md` — implementation plan, decisions, open questions
- `code_reading_map.md` — exact file:line pointers for psi0 / In-Place TTT
- `minimum_validation_plan.md` — go/no-go criteria across phases

---

## 1. Project Goal

### Core idea (one paragraph)

A frozen VLA (Vision-Language-Action) policy is deployed with fixed weights.
Each episode introduces episode-local shifts the policy never saw in this
exact combination — calibration drift, contact dynamics, friction, occlusion,
failed-grasp recovery. We propose maintaining a **resettable per-episode
fast-weight delta** `ΔW_episode` inside selected MLP layers of the action
expert, supervised by **observed environment outcomes** rather than the
policy's own predictions:

```
W_effective = W_frozen + ΔW_episode

reset ΔW_episode = 0 at episode start
read W_frozen + ΔW_episode every action-head forward
write ΔW_episode rarely, between env steps, from outcome signal Φ(o_{t+1} - o_t, ...)
```

This is structurally inspired by **In-Place TTT** for LLMs
(`extern/in_place_ttt/inference_model/hf_qwen3/modeling_qwen3.py:117-152`),
but the supervision target is environment outcome, not next-token / hidden
state. The base parameters are never mutated; clearing the cache restores
baseline behavior. Full rationale: `idea.md`.

### What the minimal experiment verifies

**Phase 0 is an infrastructure validation, not a research validation.** The
outcome-aligned write rule (Mode B) is **not implemented yet**. The minimal
experiment proves only:

1. The fast-weight wrapper can be installed on Ψ₀'s action expert without
   changing computation when `ΔW_episode = 0` (bit-exact in the empty-cache
   fast path; ≤ `1e-5` fp32 in the slow path with zero delta).
2. The wrapper is reset on the existing `history={"reset": True}` signal.
3. The default code path (with `LINEXA_TTT_ENABLED=0`) is byte-identical to
   the pre-LiNexa baseline.
4. Latency overhead in closed-loop eval is acceptable.

Phase 0 does **not** test whether episode-level TTT helps. That requires Mode
B (`Φ` + write rule + outcome plumbing), which is deferred. See `plan.md` §7
for the design TODOs.

---

## 2. Current Repo Status

### What is implemented (commit `fef78c5` on `main`)

| Layer | Module | Behavior |
|---|---|---|
| Config | `src/linexa/ttt/config.py` | `LinexaConfig.from_env()` reads `LINEXA_TTT_*` env vars |
| Cache | `src/linexa/ttt/fast_weight.py` | `FastWeightCache` (per-layer ΔW dict + reset). `commit()` raises `NotImplementedError` (Phase 0) |
| Wrapper | `src/linexa/ttt/fast_ff_wrapper.py` | `FastFFWrapper` around Diffusers `FeedForward`. Empty cache → delegates (bit-exact). Non-empty → manual `net[0] → net[1] → F.linear(h, base_w + ΔW, bias)` |
| Φ stub | `src/linexa/ttt/outcome_projector.py` | `IdentityPhi`, `LinearPhi` placeholders (Phase 1+) |
| Adapter | `src/linexa/adapters/psi0.py` | `install / uninstall / reset / commit_outcome` on a Psi0 model. Discovers `VLATransformerBlock` instances via `isinstance`. Fails loudly on nonzero write knobs |
| Server | `src/linexa/eval/serve_psi0_linexa.py` | Monkey-patches `psi.deploy.psi0_serve_simple.Server` then calls `upstream.main()`. **Zero submodule edits.** |
| Entry | `docker/psi0-server-entrypoint.sh` | Branches on `LINEXA_TTT_ENABLED`. Default execs upstream `serve_psi0` console script unchanged |
| Compose | `docker/docker-compose.yml` | Mounts `${LINEXA_ROOT}/src` at `/linexa-src` (read-only); passes `LINEXA_TTT_*` env into the server container |
| Tests | `tests/test_*.py` | 13 tests. Cover config parsing, install-time safety, wrapper equivalence (bit-exact + 1e-5 slow path), layout rejection |

### Files added or modified by `fef78c5`

```
A  src/linexa/adapters/psi0.py
A  src/linexa/eval/serve_psi0_linexa.py
A  src/linexa/ttt/config.py
A  src/linexa/ttt/fast_ff_wrapper.py
A  src/linexa/ttt/fast_weight.py
A  src/linexa/ttt/outcome_projector.py
A  tests/test_adapter_safety.py
A  tests/test_config.py
A  tests/test_fast_ff_wrapper.py
M  .gitignore                            # +.nfs*, +issue.md, +*.swp/swo
M  README.md                             # status + Phase 0 section
M  docker/.env.sample                    # +LINEXA_TTT_* block
M  docker/docker-compose.yml             # +LiNexa env passthrough + /linexa-src mount
M  docker/psi0-server-entrypoint.sh      # +LiNexa branch (default unchanged)
M  src/linexa/ttt/__init__.py            # re-exports LinexaConfig only (torch-free)
```

### The new code path

Triggered only when `LINEXA_TTT_ENABLED=1` in `docker/.env`. Flow:

```
docker/psi0-server-entrypoint.sh
  → exec /opt/venv-psi/bin/python -m linexa.eval.serve_psi0_linexa  (...)
      → linexa.eval.serve_psi0_linexa.main()
          → patches psi.deploy.psi0_serve_simple.Server
          → calls psi.deploy.psi0_serve_simple.main()
              → tyro CLI → instantiates patched LinexaServer
                  → super().__init__(...)  # upstream model load
                  → linexa.adapters.psi0.install(self.model, cfg)
                      → wraps every (or selected) VLATransformerBlock.ff_act
              → server.run() → FastAPI on port PSI0_SERVER_PORT
              → on each /act: peek payload["history"]["reset"] → cache.reset()
              → upstream predict_action() runs as before
                  (FastFFWrapper.forward delegates because cache is empty)
```

### Original code paths that MUST remain unchanged

These are the load-bearing invariants. **Do not break them:**

1. **Default behavior:** with `LINEXA_TTT_ENABLED=0` (the default), the
   entrypoint script execs the upstream `serve_psi0` console script
   unchanged. No LiNexa code is imported. The model is never wrapped. The
   inference path is byte-identical to pre-LiNexa.
2. **No submodule source edits.** `extern/psi0`, `extern/simple`, and
   `extern/in_place_ttt` are untouched. `patches/psi0/` and `patches/simple/`
   are empty in Phase 0. If you find yourself opening a file under `extern/`
   to edit, stop and reconsider — re-read `plan.md` §3 "Submodule Patch
   Policy".
3. **Base parameters.** `ff_act.net[2].weight` is **never mutated.** Writes
   in Mode B must go to a delta tensor in `FastWeightCache`. If you find
   yourself writing into the base parameter, stop.
4. **Existing reset signal.** The `history={"reset": True}` field on `/act`
   already exists in the upstream client/server. Use it; do not invent a new
   reset endpoint or signal in Phase 0/1.
5. **Test invariants.** `tests/test_fast_ff_wrapper.py` asserts bit-exact
   passthrough on empty cache and `< 1e-5` slow-path equivalence. Any change
   that breaks these invalidates the safety contract.

---

## 3. Clean Implementation Rules

These are the rules under which Phase 0 was built. Future work continues under
them.

1. **Minimal changes only.** A bug fix does not need a refactor. A feature does
   not need a redesign of unrelated modules.
2. **Do not modify original code unless absolutely necessary.** Submodule
   sources under `extern/` are off-limits. If a hook is genuinely required,
   write a unified diff under `patches/<sub>/` and apply via
   `scripts/apply_patches.sh`. In-process monkey-patching from
   `src/linexa/eval/serve_psi0_linexa.py` is preferred over patches when
   feasible (Phase 0 needed zero patches).
3. **Wrappers / adapters / config flags only.** New behavior lives under
   `src/linexa/`. Cross-module wiring goes through env vars + the
   `LinexaConfig` dataclass.
4. **Optional and disabled by default.** Every new feature ships with an
   off-by-default flag. Default behavior of the repo must remain the upstream
   Ψ₀ baseline.
5. **No hardcoded absolute paths.** Use `${LINEXA_ROOT}`, `LINEXA_SRC`, or
   relative paths anchored at the repo root via
   `git rev-parse --show-toplevel`. Container-side paths use the mount points
   defined in `docker/docker-compose.yml`.
6. **No scattered logs / outputs / caches / data / checkpoints.** Per-run
   outputs go under `logs/`, `results/`, `data/`, or `checkpoints/`, all of
   which are `.gitignore`'d at the top level (see `.gitignore`).
7. **No generated files committed.** Always `git status` before staging. The
   following must never be committed: `__pycache__/`, `.pyc`, `.nfs*`,
   `docker/.env`, `wandb/`, `outputs/`, `logs/`, `tmp/`, `data/*`,
   `checkpoints/*`, `issue.md`, `CLAUDE.md`. The `.gitignore` already covers
   these — if you see one in `git status`, fix the ignore rule.
8. **No `python` / `pip` directly on the DGX host.** All execution happens
   inside a Docker container. Check `docker ps` first; reuse a running
   container with `docker exec` rather than launching a new one without
   permission. See `~/.claude/rules/dgx-server.md`.

---

## 4. Minimal Experiment Instructions

### Hard rule: run everything in Docker

**Do not run `python`, `pip`, `pytest`, or any project script directly on
the DGX host.** The host is shared and has no project venv. All execution
happens inside the existing Docker containers:

- **`linexa-psi0-server`** (image `psi0:latest`, venv at `/opt/venv-psi`) —
  use `docker exec linexa-psi0-server …` for the model server, pytest runs,
  and any one-off Python introspection of psi0 / Diffusers.
- **`simple-eval`** (image `simple:latest`) — driven through
  `scripts/eval/run_simple_eval.sh`, which manages its own `docker compose`
  invocation. Do not edit the SIMPLE container's source from the host.

Before launching anything new, run `docker ps` and reuse a running container
when one matches. If no container is running, start the server with
`bash scripts/eval/serve_psi0.sh`. Never call `docker compose up` directly
unless you explicitly need a configuration change. See
`~/.claude/rules/dgx-server.md` for the full DGX shared-server policy.

### What you are about to run

The Phase 0 minimal validation has two layers:

- **Layer A — unit-level wrapper equivalence** (already automated, fast,
  required to pass before anything else). Runs in seconds.
- **Layer B — closed-loop wrapper safety** (slow, optional confirmation). Runs
  20 episodes baseline + 20 episodes LiNexa-enabled on a SIMPLE task, compares
  success rate and latency.

**Always pass Layer A before doing Layer B.** If Layer A fails, Layer B is
meaningless.

### Prerequisites (one-time)

1. Repo cloned with submodules at `${LINEXA_ROOT}` (this repo).
2. Docker images `psi0:latest` and `simple:latest` already on the host.
3. `docker/.env` populated:
   ```bash
   cp docker/.env.sample docker/.env
   # edit: HF_TOKEN, PSI0_GPUS, EVAL_GPUS at minimum
   ```
4. Ψ₀ checkpoint downloaded:
   ```bash
   REMOTE_SUBDIR=psi0/simple-checkpoints/<your-run-id> bash scripts/eval/download_psi0_ckpt.sh
   # auto-populates PSI0_RUN_DIR + PSI0_CKPT_STEP in docker/.env
   ```
5. Eval data downloaded for the chosen task:
   ```bash
   TASK=<task-name> bash scripts/eval/download_eval_data.sh
   ```

### Layer A — unit-level wrapper equivalence (run first, every time)

Inside the running `linexa-psi0-server` container:

```bash
# If pytest is not installed in the container venv:
docker exec linexa-psi0-server bash -lc \
  "VIRTUAL_ENV=/opt/venv-psi /root/.local/bin/uv pip install -q pytest"

# Copy tests + src into the container (they're not normally mounted as a unit):
docker cp src/linexa  linexa-psi0-server:/tmp/linexa
docker cp tests       linexa-psi0-server:/tmp/tests

# Run:
docker exec linexa-psi0-server bash -lc \
  "PYTHONPATH=/tmp /opt/venv-psi/bin/python -m pytest /tmp/tests -v"

# Cleanup:
docker exec linexa-psi0-server rm -rf /tmp/linexa /tmp/tests /tmp/.pytest_cache
```

**Pass condition:** all 13 tests pass. Critical assertions:

- `test_empty_cache_passthrough_is_bit_exact` → `max|Δ| = 0.0`
- `test_zero_delta_slow_path_within_tolerance` → `max|Δ| < 1e-5`
- `test_reset_restores_fast_path` → cache empties, fast path bit-exact
- `test_nonzero_knob_raises[*]` → `RuntimeError` on any nonzero write knob

### Layer B — closed-loop wrapper safety (slow, in tmux)

This compares baseline vs. LiNexa-enabled-with-ΔW=0 across 20 episodes.

**Choose one task.** TODO: pin the canonical Phase 0 task. Candidate based on
`plan.md` §9: a `BendPick`-class non-`Teleop` task at `level-0` DR. Confirm
availability via `ls data/evals/simple-eval/` after `download_eval_data.sh`.

Use tmux because each run takes 8–18 min and you want it to survive shell
disconnects:

```bash
tmux new -s linexa-phase0
```

Inside the tmux session:

```bash
# === BASELINE ===
# Ensure docker/.env has LINEXA_TTT_ENABLED=0 (default).
bash scripts/eval/stop.sh -v             # wipe caches between configurations
bash scripts/eval/serve_psi0.sh
# wait for "Uvicorn running on http://0.0.0.0:22085"
docker logs linexa-psi0-server | grep -E "Uvicorn|FATAL"

TASK=<TASK> DR=level-0 NUM_EPISODES=20 \
  bash scripts/eval/run_simple_eval.sh \
  2>&1 | tee logs/eval/phase0_baseline_$(date +%Y%m%d_%H%M%S).log

bash scripts/eval/stop.sh

# === LINEXA-ENABLED, ΔW=0 ===
# Edit docker/.env: set LINEXA_TTT_ENABLED=1. Leave write knobs at 0.0.
bash scripts/eval/serve_psi0.sh
docker logs linexa-psi0-server | grep -E "linexa: installed|Uvicorn|FATAL"
# expect: "linexa: installed FastFFWrapper on N / N action-expert blocks"

TASK=<TASK> DR=level-0 NUM_EPISODES=20 \
  bash scripts/eval/run_simple_eval.sh \
  2>&1 | tee logs/eval/phase0_linexa_$(date +%Y%m%d_%H%M%S).log

bash scripts/eval/stop.sh
```

Detach with `Ctrl-b d`. Reattach with `tmux attach -t linexa-phase0`.

### Where logs / results land

| Artifact | Path |
|---|---|
| Server logs | `logs/eval/<agent>/<task>/` (set by `run_simple_eval.sh`) |
| Per-episode rollout videos | same path |
| Aggregated `eval_stats.txt` | same path |
| tmux tee'd console log | `logs/eval/phase0_*_<timestamp>.log` |
| Pytest output (Layer A) | container stdout, not persisted |

All of `logs/`, `data/`, `checkpoints/`, `outputs/`, `wandb/` are gitignored.
**Do not** create result directories elsewhere.

### What indicates the experiment is positive

For **Layer A** (must pass):

- All 13 tests green.

For **Layer B** (Phase 0 safety, not research signal):

- Wrapper-enabled run shows `"linexa: installed FastFFWrapper on N / N action-expert blocks"` in server logs at startup.
- Wrapper-enabled run shows at least one `"linexa: reset cache (count=K)"` log line per episode (K monotonically increasing).
- 20-episode success rate of LiNexa run is within `±1` success of baseline
  (Ψ₀ inference is largely deterministic at fixed seed; small variation can
  come from sim nondeterminism).
- Per-step latency overhead `< 5%` (compare server-side `Return Action` log
  cadence between baseline and LiNexa runs).
- No new error / traceback in the LiNexa server logs that does not appear
  in the baseline logs.

A negative Layer B result (regression in success rate, latency blowup,
crashes) means the wrapper or hooking has a bug — go to §5.

**Phase 0 deliberately cannot give a positive research signal**, because the
write rule is not implemented. A Phase 0 "positive" only means the
modification path is safe to use for Phase 1 experiments.

---

## 5. Debugging Rules

If a run fails:

1. **Identify the root cause before patching.**
   - Read the actual traceback. Check `docker logs linexa-psi0-server`
     and the tee'd log file.
   - Distinguish: (a) install-time failure (wrapper not installed),
     (b) runtime failure inside `predict_action`, (c) eval-side failure
     in the SIMPLE worker, (d) infra failure (Docker, GPU, port conflict).
   - Reproduce on the smallest unit possible. If Layer A passes but Layer B
     fails, the bug is in install / reset / monkey-patch wiring or
     SIMPLE-side, not in the wrapper math.
2. **Apply the smallest safe fix.**
   - Prefer fixing in `src/linexa/` over editing `extern/` or `docker/`.
   - Do not silence errors with broad `try/except`. Catch the specific
     exception that you understand.
   - Do not bypass safety checks (e.g. don't comment out
     `_check_phase0_safe`; instead set the offending env var to `0`).
3. **Keep a record of every fix.**
   - Add an entry to a local `debug_log.md` at the repo root (gitignore it
     if you want, or commit it as part of the bugfix). Format:
     ```
     ## YYYY-MM-DD HH:MM — <short title>
     symptom: <what failed and where>
     root cause: <one sentence>
     fix: <one sentence + file:line>
     reproducer: <command>
     verified: <how>
     ```
4. **Re-run the minimal test until it completes.**
   - After every fix, re-run Layer A (seconds). If Layer A still fails,
     the fix is wrong; do not move on.
   - Once Layer A passes, re-run only the failing portion of Layer B,
     not the whole thing.
5. **Common failure modes (reference):**
   - `ModuleNotFoundError: No module named 'linexa'` in server logs →
     `LINEXA_SRC` not exported or `/linexa-src` mount missing. Check
     `docker-compose.yml` and the entrypoint script's `case` block.
   - `RuntimeError: linexa: nonzero write knobs set` → expected if any of
     `LINEXA_TTT_WRITE_SCALE / DECAY / CLIP` is nonzero in `docker/.env`.
     Set them back to `0.0`. Mode B is not implemented.
   - `TypeError: FastFFWrapper expected ... .net of length 3` → upstream
     Diffusers changed the FeedForward layout. Re-introspect with the
     snippet in `plan.md` §5 and update `FastFFWrapper.__init__` if the
     layout is now different but still has a clear "final projection"
     index.
   - `linexa: install() called twice` → benign warning; ignore.

---

## 6. Live Supervision & Progress Reporting

You are not just running commands — you are **supervising** the experiment.
Use the skills your environment exposes to do this efficiently.

### Use your available skills

Check your environment's skill list before improvising. The skills relevant to
this work typically include (names may vary by harness):

- **`experiment-monitor`** — track GPU utilization, container health,
  training/eval logs in flight.
- **`experiment-sweep`** — manage multi-config / multi-seed runs.
- **`analyze-results`** — generate statistical analysis tables once
  `eval_stats.txt` is produced.
- **`code-debugger` / `pua`** — when something fails repeatedly, switch into
  systematic debugging instead of retrying the same approach.

Prefer skill-driven workflows over ad-hoc shell loops. Do not invoke a skill
just for show — invoke it when its description matches what you are doing.

### Maintain a live experiment report

Create `reports/phase0_<YYYYMMDD_HHMM>.md` **before** you start the first
command. Update it continuously while the experiment runs, so the human
supervisor can read the file at any moment and know the current state without
asking. Treat it as a single source of truth.

Required structure (extend with timestamps as the run progresses):

```markdown
# Phase 0 Minimal Experiment — <YYYY-MM-DD>

## Status
[one of: NOT_STARTED | LAYER_A_RUNNING | LAYER_A_PASS | LAYER_A_FAIL |
         LAYER_B_BASELINE_RUNNING | LAYER_B_LINEXA_RUNNING |
         LAYER_B_DONE | BLOCKED | COMPLETE]
Last update: <YYYY-MM-DD HH:MM:SS>

## Configuration
- Repo commit: <git rev-parse HEAD>
- TASK: <e.g. G1WholebodyXMovePick-v0>
- DR: <e.g. level-0>
- NUM_EPISODES: 20
- LINEXA_TTT_LAYERS: <empty = all>

## Commands run (verbatim, with timestamps)
- [HH:MM] <command 1>
- [HH:MM] <command 2>
- ...

## Bugs encountered & fixes applied
- [HH:MM] symptom → root cause → fix (file:line) → verification
- ...

## Logs / results paths
- Layer A (pytest): <stdout captured to ...>
- Layer B baseline: logs/eval/<agent>/<task>/...
- Layer B linexa:   logs/eval/<agent>/<task>/...
- Tee'd console:    logs/eval/phase0_*_<ts>.log

## Intermediate observations
- [HH:MM] <e.g. "Server reports 6/6 blocks wrapped at startup">
- [HH:MM] <e.g. "Episode 5/20 baseline: success">
- ...

## Current concern / blocker (if any)
- ...

## Next planned action
- ...
```

Update cadence:

- After **every command you run**, append the command + timestamp.
- After **every fix**, append a bug entry with verification.
- After **every episode boundary** in Layer B (visible in server logs as a
  "linexa: reset cache" or upstream equivalent), append a one-line
  observation.
- Whenever the **status field** changes, update it immediately (don't batch).

If the experiment runs unattended for a long stretch, set a periodic check
(e.g. every 5 minutes) that re-reads the latest log lines and writes a
heartbeat observation into the report. This is exactly what
`experiment-monitor` is for.

### Discipline rules for the live report

- **Append-only by default.** Don't rewrite history. If you discover an
  earlier observation was wrong, add a new entry that corrects it.
- **No claims of success in the report until the data is in hand.** "Layer A
  passed" is only true after the pytest summary line shows
  `13 passed`. "Layer B safety positive" is only true after both runs
  finished and you computed the deltas.
- **Be precise about uncertainty.** If the baseline log shows
  `success_baseline = 14/20` but you only have `8/20` finished for the LiNexa
  run, the report says exactly that — not a projection.

---

## 7. Final Reporting Requirements

After the experiment finishes, write a final summary at the **bottom** of the
same `reports/phase0_<YYYYMMDD_HHMM>.md` file (under a `## Final summary`
heading). The human supervisor will read this section first.

Required content:

1. **Commands used.** Verbatim, in order. Include any deviation from §4
   (e.g. you ran 10 episodes instead of 20, or used a different `TASK`).
2. **Files changed.** Output of `git diff --stat <baseline-commit>..HEAD`,
   plus a one-line description of each file's purpose.
3. **Bugs fixed.** Each entry from your live `## Bugs encountered & fixes
   applied` log (symptom, root cause, fix, file:line, verification command).
4. **Logs / results path.** Absolute paths under `logs/eval/` for both the
   baseline and LiNexa-enabled runs, plus the tee'd console log paths.
   Include the contents of any `eval_stats.txt` produced verbatim — do not
   paraphrase numbers.
5. **Whether the idea appears to work.** This is the most important bullet.
   The rules:
   - Phase 0 is an **infrastructure validation, not a research validation**
     (see §1). Do not claim "the idea works" or "the idea fails" based on
     Phase 0 alone — Phase 0 only proves the wrapper path is safe.
   - State explicitly:
     - Layer A: pass / fail (with which specific test failed if any).
     - Layer B: pass / fail against the criteria in §4. Be numeric:
       `success_baseline = X/N, success_linexa = Y/N, latency_overhead = Z%`.
   - **Anchor every claim to data you actually observed.** If you write
     "wrapper introduces no regression," cite the success-rate numbers from
     `eval_stats.txt` and the latency numbers from server logs. If you have
     no data for a claim, do not make it.
   - **If results are inconclusive, say so.** Inconclusive is a valid
     finding. Forced positives are not.
6. **Remaining risks or TODOs.** Examples to consider:
   - Mode B / outcome write rule still not implemented (`plan.md` §7) —
     Phase 0 alone cannot test the research hypothesis.
   - The Phase 0 task in §4 is currently a TODO — pin it after the first run.
   - Latency measurement is approximate from log cadence; a proper
     per-`predict_action` timer is a Phase 1 prerequisite.
   - Phase 1 outcome plumbing channel (option C in `plan.md` §8) needs a
     small `patches/simple/<name>.patch` to extend the `/act` payload.

Format: same `reports/phase0_<YYYYMMDD_HHMM>.md` file (live report + final
summary appended). Do not commit it unless instructed; the `reports/` path
is **not** currently in `.gitignore`, so add it there if you create the
directory and want it ignored.

---

## Quick reference — files you will touch most

| Task | File |
|---|---|
| Toggle LiNexa on/off | `docker/.env` (`LINEXA_TTT_ENABLED`) |
| Adjust which blocks to wrap | `docker/.env` (`LINEXA_TTT_LAYERS=0,2,4` etc.) |
| Read wrapper math | `src/linexa/ttt/fast_ff_wrapper.py` |
| Change install behavior | `src/linexa/adapters/psi0.py` |
| Change server hook | `src/linexa/eval/serve_psi0_linexa.py` |
| Add a unit test | `tests/test_*.py` |
| Document a Phase 1 design decision | `plan.md` (do not edit `idea.md`) |

## Final note

If you finish Phase 0 successfully and are tempted to start Phase 1: read
`plan.md` §6 (read-many / write-rarely), §7 (Mode B scope), §8 (outcome
plumbing option C), and the open TODOs in `idea.md` §6. Mode B has
five unresolved design decisions — pin them in `plan.md` first, get sign-off
from the human, then implement.
