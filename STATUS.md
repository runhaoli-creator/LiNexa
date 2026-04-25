# Status

**Phase:** Phase 0 (infrastructure validation) — implemented, awaiting closed-loop wrapper-safety eval
**Last updated:** 2026-04-24
**Updated by:** dgx session

## Current Focus
Validate the Phase 0 fast-weight wrapper end-to-end on a SIMPLE task and pin
the Phase 1 (Mode B) write-rule design before any further code lands.

## Recent Progress
- 2026-04-22: Initialized repo and Claude config.
- 2026-04-23: Eval infrastructure (Docker compose, psi0 server, SIMPLE worker, eval scripts) end-to-end functional.
- 2026-04-24: Phase 0 fast-weight wrapper landed (`fef78c5`). 13 unit tests pass; bit-exact wrapper equivalence on the empty-cache fast path; install-time guard against nonzero write knobs.

## Next Actions
- [ ] Pin a canonical SIMPLE task for Phase 0 closed-loop wrapper-safety run (`BendPick`-class candidate per `docs/plan.md` §9).
- [ ] Run baseline (LINEXA_TTT_ENABLED=0) vs LiNexa-enabled (ΔW=0) for 20 episodes; compare success rate, latency, and traceback rate.
- [ ] Commit a Phase 0 experiment report under `reports/` (gitignored) and copy the verdict into this file.
- [ ] Pin Mode B design TODOs in `docs/plan.md` §7 (outcome target `Φ`, write rule, attribution timing) before any Phase 1 code.

## Blockers / Open Questions
- Canonical Phase 0 SIMPLE task not yet pinned — `docs/agent_handoff.md` §4 marks this as TODO.
- Latency budget for "real-time" on the chosen task — `docs/plan.md` §10 open question.

## Notes for next agent
- Read `docs/agent_handoff.md` first.
- Default Ψ₀ inference path must remain byte-identical when `LINEXA_TTT_ENABLED=0` — this is the load-bearing invariant.
- Mode B (outcome write rule) is **not** implemented; `commit_outcome` and `FastWeightCache.commit` raise `NotImplementedError`. Do not silently turn this on.
