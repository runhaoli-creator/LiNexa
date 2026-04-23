# LiNexa

Research scaffolding for **episode-level fast-weight adaptation** in humanoid VLA models, inspired by In-Place TTT.

> **Status:** scaffolding only. No method implementation yet.

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
├── docs/            # idea, theory, setup, experiment plan
├── src/linexa/      # our code: ttt, adapters, eval, utils, cli
├── patches/         # patches applied to extern/ submodules
├── scripts/         # shell entry points (setup, baseline, ttt runs)
├── notebooks/       # exploratory notebooks
├── results/         # open_loop / closed_loop / figures
├── checkpoints/     # local model checkpoints (gitignored)
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

See `docs/setup.md` for environment details.

## License

TBD.
