# Status

**Phase:** Phase 0 (infrastructure validation) — **COMPLETE**. Wrapper path verified safe; Mode B (outcome write rule) not started.
**Last updated:** 2026-04-25
**Updated by:** dgx session

## Current Focus
Phase 0 closed-loop wrapper-safety eval on `G1WholebodyXMovePickTeleop-v0` `level-0` is complete. Next: pin the five Mode B design decisions in `docs/plan.md` §7 before any Phase 1 code lands.

## Recent Progress
- 2026-04-22: Initialized repo and Claude config.
- 2026-04-23: Eval infrastructure (Docker compose, psi0 server, SIMPLE worker, eval scripts) end-to-end functional.
- 2026-04-24: Phase 0 fast-weight wrapper landed (`fef78c5`). 13 unit tests pass; bit-exact wrapper equivalence on the empty-cache fast path; install-time guard against nonzero write knobs.
- 2026-04-25: Phase 0 Layer B end-to-end verified. **Layer A 13/13 PASS; Layer B baseline 10/10 = linexa-with-ΔW=0 10/10.** Per-/act count identical (85=85); 10 reset-cache calls (one per episode). Two bugs in the linexa serving path found and fixed (logger-output invisibility; missing FastAPI body annotation), plus an OpenAPI schema-check guard added to the orchestrator. Full report: `reports/phase0_20260424_2343.md`. W&B: https://wandb.ai/uscgvl/linexa-phase0 (runs `3psp1yxx` baseline, `n0xc978m` linexa).

## Key Results
- **Phase 0 verdict:** infrastructure-validation PASS. Wrapper path is safe to use for Phase 1.
- **Closed-loop bit-exact pattern:** 10/10 success in both legs on the same 10-trajectory level-0 lerobot bundle. Reproduces published Ψ₀ baseline (paper: 10/10) within stochastic variance of 0.
- **Wrapper overhead:** below per-episode-wall resolution. Warm-mean wall — baseline 558 s, linexa 547 s (Δ = -2.0 %, sim noise dominates).
- **Reset hook:** 10 monotonic `linexa: reset cache (count=N)` lines (N=1..10), one per episode boundary.

## Decisions Made
- 2026-04-25: **Pinned Phase 0 task = `G1WholebodyXMovePickTeleop-v0` `level-0`, N=10 episodes per leg.** The lerobot eval bundle for this task at level-0 contains exactly 10 trajectories — `--num-episodes 20` is silently capped. Matches published Ψ₀ paper N. (Was a TODO in `docs/agent_handoff.md` §4.)
- 2026-04-25: All 6 action-expert blocks wrapped (idx=[0,1,2,3,4,5], `LINEXA_TTT_LAYERS=` empty). Phase 0 stress-tested the install path on the full set; Phase 1 / Mode B may narrow this once the write rule is pinned.

## Open Questions
- [ ] Mode B `Φ` choice: analytic (`Δ proprio`?), frozen-random, or learned offline? — `state_of_idea.md` §6, `docs/plan.md` §7
- [ ] Mode B write-rule form: algebraic (analogue of In-Place TTT `modeling_qwen3.py:117-152`) or something different?
- [ ] Action-chunk outcome attribution: which env-step's outcome supervises a chunk's write?
- [ ] Layer subset for Mode B (Phase 0 wrapped all 6; Mode B may want just 2-3).
- [ ] Outcome-payload plumbing channel (`docs/plan.md` §8 option C) — small `patches/simple/` patch?

## Next Steps
- [ ] Pin the five Mode B design decisions in `docs/plan.md` §7 with explicit rationale; get human sign-off before any code.
- [ ] Add per-`predict_action` timing instrumentation (Phase 1 prerequisite) — wrap the upstream method, log `time.perf_counter()` deltas. Phase 0 latency claims are per-episode-wall proxies, not per-step.
- [ ] Replace mp4-watcher Monitor's `-newer console.log` with `find -newermt "@$ts"` against a frozen orchestrator-start timestamp (cosmetic; W&B sidecar already uses a frozen ts and was unaffected).
- [ ] Optional: extend Phase 0 verification to `level-1` (10 eps) and `level-2` (6 eps) on the same task, for a 26-episode bit-exact contract check across all DR levels.
- [ ] Optional: commit the Phase 0 patches (`src/linexa/adapters/psi0.py`, `src/linexa/eval/serve_psi0_linexa.py`) and the new `scripts/eval/log_to_wandb.py` + `scripts/eval/run_phase0_layer_b.sh` — currently uncommitted, awaiting human review.

## Notes for next agent
- Read `docs/agent_handoff.md` first; `reports/phase0_20260424_2343.md` is the canonical Phase 0 evidence file.
- Default Ψ₀ inference path must remain byte-identical when `LINEXA_TTT_ENABLED=0` — this is the load-bearing invariant.
- The `LinexaServer.predict_action` override **must** keep the upstream signature `(self, payload: Dict[str, Any])`. Dropping the annotation = every /act → 422. Inline comment in the source flags this; orchestrator now has an OpenAPI guard.
- Mode B (outcome write rule) is **not** implemented; `commit_outcome` and `FastWeightCache.commit` raise `NotImplementedError`. Do not silently turn this on.
