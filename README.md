# LiNexa

Research scaffolding for **episode-level fast-weight adaptation** in humanoid VLA models, inspired by In-Place TTT.

> **Status:** Phase 0 landed — optional, no-op-equivalent fast-weight wrapper around the Ψ₀ action expert's `ff_act`. Default behavior is unchanged. The outcome-aligned write rule (Mode B) is not implemented; see [`docs/plan.md`](docs/plan.md) §7.

## Goal

Investigate whether per-episode test-time training (TTT) — i.e. applying fast-weight updates *within* a single episode — can improve closed-loop behavior of VLA policies on humanoid manipulation tasks without damaging the underlying pretrained model.

Concretely, we want to study:

- **Fast weights vs. frozen weights:** what to adapt and where (FF vs. attention, which layers).
- **Episode-scoped adaptation:** reset-to-base at episode boundaries; optional decay within an episode.
- **Open-loop vs. closed-loop eval:** whether TTT helps beyond teacher-forced rollouts.

## Why these submodules

This repo stays small. The heavy components live as git submodules in `extern/`:

| Submodule | Path | Role |
|---|---|---|
| [Psi0](https://github.com/physical-superintelligence-lab/Psi0) | `extern/psi0` | Base VLA model and training stack. Source of the policy we adapt. |
| [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE) | `extern/simple` | Humanoid manipulation benchmark / environments we evaluate on. |
| [In-Place TTT](https://github.com/ByteDance-Seed/In-Place-TTT) | `extern/in_place_ttt` | Reference implementation of the in-place test-time training mechanism we build on. |

Submodule internals are **not modified in-tree**. Any required adjustments live as patches in `patches/` and are applied via `scripts/apply_patches.sh`.

## Repo layout

```
linexa/
├── configs/         # experiment configs (base, tasks, ablations)
├── docs/            # idea, plan, agent handoff, code-reading map, validation plan
├── src/linexa/      # our code: ttt, adapters, eval, utils, cli
├── patches/         # patches applied to extern/ submodules
├── scripts/         # shell entry points (setup, baseline, ttt runs)
│   └── eval/        # psi0-on-SIMPLE orchestration (Docker)
├── docker/          # psi0-server Dockerfile + compose + .env
├── notebooks/       # exploratory notebooks
├── results/         # open_loop / closed_loop / figures
├── data/            # eval task bundles (gitignored; fetched by scripts)
├── checkpoints/     # model weights (gitignored; fetched by scripts)
├── logs/            # eval + server logs (gitignored)
└── extern/          # submodules: psi0, simple, in_place_ttt
```

## Quickstart

```bash
git clone --recursive <this-repo>
cd linexa
bash scripts/setup_submodules.sh
bash scripts/apply_patches.sh
# then see configs/ and scripts/ for runs
```

See [`docs/`](docs/) for design notes, the implementation plan, and the agent handoff.

## Evaluate psi0 on SIMPLE

Runs the Ψ₀ policy against a SIMPLE humanoid task using two containers:
one for the psi0 inference server, one for the SIMPLE / IsaacSim
rollout. Both containers use pre-built images that already exist on the
DGX host (`psi0:latest`, `simple:latest`) — **no image build, no host
Python/pip, no manual venv setup**. All paths resolve under the repo
root; nothing is downloaded outside the repo.

**Requirements:** Docker + NVIDIA Container Toolkit, an RTX-class GPU,
a HuggingFace token with access to `USC-PSI-Lab/{psi-data, psi-model}`,
and the two images above. On a fresh host, build them once via
`extern/simple/Dockerfile` (for SIMPLE — see the SIMPLE README) and from
[Psi0 repo docker setup](extern/psi0/Dockerfile). On this DGX they are
already present and tagged.

### 0. One-time setup

```bash
cp docker/.env.sample docker/.env
# edit docker/.env: at minimum set HF_TOKEN and PSI0_GPUS / EVAL_GPUS.
# PSI0_RUN_DIR + PSI0_CKPT_STEP are auto-populated in step 1.
```

### 1. Prepare data and checkpoints

```bash
# Eval task bundle → data/evals/simple-eval/<task>/{level-0,level-1,level-2}/
TASK=G1WholebodyXMovePickTeleop-v0 bash scripts/eval/download_eval_data.sh

# psi0 SIMPLE checkpoint → checkpoints/psi0/simple-checkpoints/<run>/
# (defaults to the full 6-task tree; narrow with REMOTE_SUBDIR for one task)
REMOTE_SUBDIR=psi0/simple-checkpoints/g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205 \
  bash scripts/eval/download_psi0_ckpt.sh
# ↑ auto-runs find_run_dir.sh which writes PSI0_RUN_DIR + PSI0_CKPT_STEP
#   into docker/.env. Rerun scripts/eval/find_run_dir.sh to switch runs.
```

### 2. Start the psi0 server

```bash
bash scripts/eval/serve_psi0.sh
# wait for the log line "Uvicorn running on http://0.0.0.0:22085" — typically 1–2 min
docker logs -f linexa-psi0-server
# sanity check (separate terminal)
curl -s http://localhost:22085/health    # → {"status":"ok"}
```

### 3. Run SIMPLE eval end-to-end

```bash
TASK=G1WholebodyXMovePickTeleop-v0 DR=level-0 NUM_EPISODES=10 \
  bash scripts/eval/run_simple_eval.sh
```

Auto-dispatches by task suffix: `*Teleop-v0` → `eval-decoupled-wbc` +
`psi0_decoupled_wbc`, `*MP-v0` → `eval` + `psi0`. Rollout videos,
`eval_stats.txt`, and per-episode logs land under
`logs/eval/<agent>/<task>/`.

> **First run downloads ~4.5 GB of SIMPLE assets** (scenes, robots,
> materials, Objaverse/HSSD USD files) into `data/{scenes,robots,assets,vMaterials_2}/`
> — one-shot fetch from SIMPLE's asset registry on first reset(). Subsequent
> runs skip this. Total wall time for a first 1-episode run: ~18 min
> (3 min IsaacSim startup + 8 min asset download + 5 min physics rollout).
> Later single-episode runs take ~8–12 min.

### Cleanup / iteration

```bash
bash scripts/eval/stop.sh          # shut down psi0-server (and any workers)
bash scripts/eval/stop.sh -v       # also wipe the cuRobo / JIT cache volumes
```

See `docker/.env.sample` for every knob (ports, action horizon, GPU
selection, RTC on/off).

## LiNexa Phase 0: optional fast-weight wrapper

The Phase 0 implementation adds a resettable fast-weight wrapper around the
final projection of `VLATransformerBlock.ff_act` in Ψ₀'s action expert (see
[`docs/plan.md`](docs/plan.md) §5). It does **not** implement an outcome-aligned
write rule yet — that is Mode B, gated behind explicit knobs and currently
`NotImplementedError`.

### Default behavior is unchanged

With `LINEXA_TTT_ENABLED=0` (the default in `docker/.env.sample`), the
entrypoint script execs the upstream `serve_psi0` console script directly.
No LiNexa code is imported and the original Ψ₀ inference path is byte-identical
to the pre-LiNexa baseline.

### Enable the wrapper

Set in `docker/.env`:

```dotenv
LINEXA_TTT_ENABLED=1
# Optional: comma-separated VLATransformerBlock indices to wrap (empty = all).
LINEXA_TTT_LAYERS=
# Mode B knobs — must stay 0.0 in Phase 0; nonzero raises at install time.
LINEXA_TTT_WRITE_SCALE=0.0
LINEXA_TTT_DECAY=0.0
LINEXA_TTT_CLIP=0.0
LINEXA_TTT_LOG_STATS=0
```

Then restart the server (`scripts/eval/stop.sh && scripts/eval/serve_psi0.sh`).
The entrypoint will launch `python -m linexa.eval.serve_psi0_linexa`, which
monkey-patches `psi.deploy.psi0_serve_simple.Server` in process to install
wrappers after model load and clear them on each `history={"reset": True}`
request — no submodule patches required.

### Phase 0 safety contract

- Empty fast-weight cache → wrapper delegates to the wrapped module
  (bit-exact baseline).
- Zero ΔW slow path → matches the wrapped module within `max|Δ| < 1e-5` (fp32).
- `cache.reset()` clears all per-layer deltas and is invoked on the existing
  `history.reset` signal.
- Setting any nonzero write knob raises a clear `RuntimeError` at server
  startup — Mode B is not silently enabled.

### Tests

The suite under `tests/` covers config parsing, Phase 0 install-time safety,
and the wrapper-equivalence contract. From any environment that has `torch`
and `diffusers` installed (e.g. inside the `psi0:latest` container):

```bash
PYTHONPATH=src pytest tests/
```

`tests/test_config.py` and `tests/test_adapter_safety.py` need only
`torch + pytest`; `tests/test_fast_ff_wrapper.py` additionally needs
`diffusers`. Tests `importorskip` so missing optional deps skip rather than
fail.

### Notes on custom Docker work

- **No Dockerfile is added in this repo.** We reuse the pre-built
  `psi0:latest` and `simple:latest` images. The psi0 image already has
  a uv-managed venv at `/opt/venv-psi` with `serve_psi0` registered;
  the SIMPLE image has `/workspace/SIMPLE/.venv`. Our compose file
  mounts `extern/psi0` at `/workspace` and `extern/simple/src` on top
  of the SIMPLE image's source tree, so the code under eval comes from
  these submodules (not from whatever was baked into the image).
- **To rebuild the images from scratch** (on a new host or if image
  versions drift), follow the instructions in
  `extern/simple/README.md` (for `simple:latest`) and in the upstream
  Psi0 repo (for `psi0:latest`). Tag them exactly as `psi0:latest`
  and `simple:latest`.
- **Shared DGX hosts:** both services use `network_mode: host`. Set
  `PSI0_SERVER_PORT` to something non-default in `docker/.env` if the
  default 22085 is already taken.

## License

TBD.
