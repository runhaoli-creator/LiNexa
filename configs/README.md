# Configs

- `base.yaml` — shared defaults (model, optimizer, logging).
- `psi0_simple.yaml` — Psi0 policy + SIMPLE environments.
- `tasks/` — per-task configs (pick, handover, xmove_pick).
- `ablations/` — TTT ablations (no_ttt, ttt_fast_ff, ttt_fast_ff_decay).

Configs are plain YAML; resolution strategy (Hydra / OmegaConf / custom) TBD.
