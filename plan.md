# Minimal Modification Plan

This is the working plan for adding an optional LiNexa online-adaptation path
while keeping the original Psi0/SIMPLE inference path clean and unchanged by
default.

The plan is intentionally conservative: first prove that a fast-weight wrapper
can be inserted safely, then add outcome-update plumbing, then test a nonzero
write rule.

---

## 1. Requirements

1. Keep the original repo clean. Do not edit upstream submodule code directly
   unless a tiny hook becomes unavoidable.
2. Put new LiNexa code under `src/linexa/`.
3. Prefer wrappers, adapters, config flags, and separate modules.
4. Keep the original Psi0/VLA inference path unchanged by default.
5. Make online adaptation optional and easy to disable.
6. Use the official In-Place TTT implementation as a structural reference:
   cached fast weights, selected MLP layers, resettable per-session state, and
   base weights left intact.
7. Do not invent an outcome-aligned loss or write rule before it is defined and
   validated.

---

## 2. Current Code Facts

Confirmed from this repo:

- LiNexa-owned code currently lives under `src/linexa/`, but it is scaffolding.
- Submodules live under `extern/psi0`, `extern/simple`, and
  `extern/in_place_ttt`.
- Repo convention is to keep submodule changes as patches under `patches/`,
  applied by `scripts/apply_patches.sh`.
- Psi0 serving is orchestrated through `docker/docker-compose.yml` and
  `docker/psi0-server-entrypoint.sh`.
- Psi0 server code loads the model in
  `extern/psi0/src/psi/deploy/psi0_serve_simple.py`.
- Psi0 inference calls `Psi0Model.predict_action(...)`.
- Psi0 has an action expert with `VLATransformerBlock` blocks and an
  action-stream feed-forward path named `ff_act`.
- `ff_act` is imported from Diffusers as `diffusers.models.attention.FeedForward`.
- SIMPLE evaluation has an explicit rollout loop:
  `policy.get_action(...)` followed by `env.step(action)`.
- SIMPLE/Psi0 already carries an episode reset signal through policy history,
  but LiNexa still needs code that actually clears any new fast-weight cache.

Confirmed from official In-Place TTT code:

- Inference-time adaptation uses cached fast weights, not permanent mutation of
  base model parameters.
- The official inference path does not compute a scalar test-time loss or call
  `backward()`.
- TTT is enabled by config-like flags: selected layers, mode, write scale, and
  chunking.
- The LM target does not transfer directly to VLA outcome adaptation.

---

## 3. Clean Code Strategy

### New LiNexa Modules

Add new implementation code under `src/linexa/`:

- `src/linexa/ttt/fast_weight.py`
  - resettable fast-weight/cache state
  - base-weight plus delta/cached-weight read behavior
  - norm/stat logging helpers

- `src/linexa/ttt/fast_ff_wrapper.py`
  - wrapper for selected action-expert feed-forward modules
  - default behavior must be identical to the wrapped module
  - optional nonzero write path behind explicit flags

- `src/linexa/ttt/config.py`
  - typed/dataclass config for LiNexa TTT flags
  - examples: enabled, layer indices, write scale, decay, clipping, logging

- `src/linexa/ttt/outcome_projector.py`
  - placeholder home for `Φ`, the outcome-to-target projection
  - may start as analytic or frozen for smoke tests
  - learned `Φ` is a later phase, not assumed

- `src/linexa/adapters/psi0.py`
  - install wrappers into a loaded Psi0 model
  - reset fast-weight state
  - expose an explicit `commit_outcome(...)` function for later phases

- `src/linexa/eval/serve_psi0_linexa.py`
  - optional LiNexa-owned server entrypoint/wrapper
  - should delegate to existing Psi0 serving logic as much as possible

Keep these modules independent of SIMPLE where possible. SIMPLE should only
need a small payload extension or callback if outcome updates require it.

### Docker / Import Wiring

Current Docker services mount `extern/psi0` and `extern/simple`, but not
`src/linexa`. The clean repo-owned change is:

- mount `${LINEXA_ROOT}/src` into the Psi0 server container
- add that mount to `PYTHONPATH`
- add LiNexa flags to `docker/.env.sample`

This avoids copying LiNexa code into the Psi0 submodule.

### Submodule Patch Policy

Avoid direct edits under `extern/psi0` and `extern/simple`.

Preferred Phase 0 approach:

- use a LiNexa-owned server entrypoint/wrapper to install fast-weight wrappers
  after Psi0 loads
- keep `extern/psi0` untouched
- keep default serving path unchanged when LiNexa is disabled

If a hook becomes unavoidable:

- create a small patch under `patches/psi0/` or `patches/simple/`
- keep the default behavior unchanged
- guard all LiNexa behavior behind explicit flags
- apply through `scripts/apply_patches.sh`

Expected patch need:

- Phase 0 should need no submodule patches.
- Phase 1 may need one small SIMPLE/Psi0-client patch to pass previous outcome
  information back to the server.

---

## 4. Optional Inference Path

The original Psi0 inference path must remain unchanged unless LiNexa is enabled.

Default behavior:

```text
LINEXA_TTT_ENABLED=0
```

With default settings:

- no wrapper is installed, or wrapper is installed in strict no-op mode
- no fast-weight state is written
- `Psi0Model.predict_action(...)` should behave like the existing baseline

Optional behavior:

```text
LINEXA_TTT_ENABLED=1
```

Then:

- selected action-expert feed-forward modules are wrapped
- fast weights are stored separately from base parameters
- episode reset clears fast-weight cache
- nonzero writes are enabled only when a write rule is explicitly configured

Recommended Phase 0 config source:

- use `docker/.env` / environment variables first
- keep YAML/Hydra integration for later, once behavior stabilizes

---

## 5. Fast-Weight Design Boundary

For the first implementation, follow the official In-Place TTT boundary:

- base weights remain frozen
- adapted weights are cache-local
- clearing the cache recovers base behavior
- selected MLP layers only
- no VLM backbone changes
- no final action projection changes
- no inference-time `backward()` unless later justified

For Psi0, the conceptual target is the action-stream `ff_act` path inside
`VLATransformerBlock`.

Important implementation note:

- Psi0 uses Diffusers `FeedForward`, not Qwen's explicit
  `gate_proj/up_proj/down_proj` module.
- Runtime structure verified inside the live `linexa-psi0-server` container
  (Diffusers `FeedForward(dim=1536, dim_out=1536, activation_fn="gelu-approximate")`):

```text
ff_act.net = nn.ModuleList([
    net[0]: diffusers...GELU      # holds Linear(1536, 6144) + GELU activation
    net[1]: nn.Dropout
    net[2]: nn.Linear(6144, 1536) # the final projection
])
```

- Pinned for v1 implementation:
  - **fast-weight target** = `ff_act.net[2].weight` (shape `1536 × 6144`)
  - **key signal `h`** = output of `ff_act.net[0]` (post-GELU, shape `(B, T, 6144)`)
- These are the analogues of `down_proj.weight` and
  `act_fn(gate_proj(x)) * up_proj(x)` in
  `extern/in_place_ttt/inference_model/hf_qwen3/modeling_qwen3.py:117-152`.
- Re-confirm against the active Psi0 checkpoint at install time
  (parameter shape, dtype, device), but treat the structure above as the
  design assumption.

Note on shared blocks:

- The `VLATransformerBlock` instances are shared between `predict_action`
  (`psi0.py:1642`) and the RTC variants (`psi0.py:1736`,
  `psi0_serve_simple.py:144`). Wrapping `ff_act` once therefore covers both
  serving paths; no separate RTC handling is needed in Phase 0.

Open decisions:

- represent fast state as a full cached weight or as `ΔW_episode`
  (recommended: `ΔW_episode` so reset is `ΔW = 0` and the base parameter is
  textually never touched)
- decide selected layer indices for Mode B (Phase 0 wraps all 6 blocks to
  stress the install path)

---

## 6. Read / Write Timing

Psi0 action generation uses an internal denoising/flow loop. The wrapper may be
called many times within one environment step.

Phase 0 timing rule:

```text
read many, write never
```

That means:

- wrappers read `W_base + ΔW_episode`
- `ΔW_episode = 0`
- no outcome commit is called

First nonzero-write timing rule:

```text
read many, write rarely
```

That means:

- wrappers read on every action-head forward call
- `ΔW_episode` is mutated only by an explicit `commit_outcome(...)` call
- `commit_outcome(...)` should happen outside the denoising loop, between
  environment steps or before the next `/act` response

Avoid:

- writing inside every denoising step
- changing weights mid-denoising-loop within a single action prediction

This keeps the online path closer to official In-Place TTT's cache-local update
pattern while respecting Psi0's different inference structure.

---

## 7. Write Rule Scope

Supported now by official-code analogy:

- no-gradient algebraic write
- selected-layer cached fast weights
- resettable cache

Not yet supported by current LiNexa/Psi0 code:

- an outcome-aligned target
- a learned `Φ`
- an online loss for post-step adaptation
- an env-step callback or payload field carrying
  `(obs_t, action_t, obs_{t+1}, info_{t+1})`

Therefore the first implementation separates two modes.

### Mode A: No-Op / Instrumentation

Purpose:

- prove wrappers and reset do not break baseline inference
- collect hidden-key/stat logs needed for later design

Behavior:

- install wrappers
- collect optional key/norm/timing stats
- keep `ΔW_episode = 0`

### Mode B: Explicit Experimental Write

Purpose:

- test the first nonzero write rule after Mode A passes

Behavior:

- enabled only by explicit config
- starts with the simplest candidate from `state_of_idea.md`
- uses stability knobs from day one:
  - write scale
  - EMA decay
  - norm clipping
  - optional write gate

TODO before Mode B:

- define exact first write rule
- define exact outcome target
- define `Φ`
- define how outcomes are attributed to action chunks
- define whether the first write is analytic, frozen, or learned offline

Do not treat a random frozen `Φ` as a validated method. It is acceptable only
as a plumbing or stress test.

---

## 8. Minimum Code Change Set

### Phase 0: Wiring and No-Op Wrapper

Repo-owned additions:

- `src/linexa/ttt/fast_weight.py`
- `src/linexa/ttt/fast_ff_wrapper.py`
- `src/linexa/ttt/config.py`
- `src/linexa/adapters/psi0.py`
- `src/linexa/eval/serve_psi0_linexa.py`

Repo-owned Docker/config additions:

- mount LiNexa `src/` in the Psi0 server container
- add LiNexa `src/` to `PYTHONPATH`
- add LiNexa flags to `docker/.env.sample`
- branch in `docker/psi0-server-entrypoint.sh`:
  - default: existing `serve_psi0`
  - optional: `python -m linexa.eval.serve_psi0_linexa`

Patch expectation:

- no submodule patches for Phase 0

### Phase 1: Outcome Update Plumbing

Only after Phase 0 passes:

- expose previous transition data needed for `commit_outcome(...)`
- prefer reusing the existing `/act` request path before adding a new endpoint
- prefer putting previous outcome data in `history` or another existing request
  field, guarded by step id for idempotence

Likely small patch:

- `patches/simple/<outcome-payload>.patch`
  - extend Psi0 client adapter to include previous outcome in the next `/act`
    request

Avoid for Phase 1 unless necessary:

- broad SIMPLE rollout-loop refactors
- a separate `/update` endpoint
- sending full fast-weight tensors over HTTP

---

## 9. Fastest Validation

### Smallest Experiment First

Run a **no-op wrapper safety test** on Psi0/SIMPLE.

Question:

> Can we insert LiNexa fast-weight wrappers and reset logic without changing
> practical baseline Psi0 behavior?

Procedure:

1. Run existing frozen Psi0 baseline on one SIMPLE task.
2. Enable LiNexa wrapper mode with writes disabled.
3. Run the same eval configuration.
4. Compare success rate, action statistics, logs, and latency.

This does not prove the idea works. It proves the modification path is safe.

### Better Unit-Level Check

Before long rollouts, run a captured-request or single-call check:

1. Capture one valid server request or construct one from existing eval data.
2. Run `predict_action` with LiNexa disabled.
3. Run `predict_action` with wrappers enabled and `ΔW_episode = 0`.
4. Compare returned action tensors within an agreed tolerance.

This is a cleaner equivalence check than relying only on full rollout success,
because full rollouts may include stochastic sampling, scheduler randomness, and
simulation nondeterminism.

### Baseline

Primary baseline:

- existing frozen Psi0 inference path, no LiNexa wrappers, no TTT

Secondary control:

- LiNexa wrapper installed with `ΔW_episode = 0`

### Metrics

For Phase 0:

- no meaningful regression in closed-loop success on the selected SIMPLE task
- captured-request action difference within tolerance when writes are disabled
- latency overhead small enough to keep eval practical
- reset path confirmed by logs
- wrapper logs confirm selected layers were wrapped and reset

For the first nonzero write experiment:

- closed-loop success delta versus frozen Psi0
- catastrophic failure count
- fast-weight norm boundedness
- latency overhead

### Code Needed For Fastest Verification

Phase 0 only:

- LiNexa fast-weight wrapper module
- Psi0 wrapper installer
- optional LiNexa server entrypoint
- reset hook from existing history signal
- Docker/PYTHONPATH wiring

No outcome-target code is required for Phase 0.

### Positive Signal

Phase 0 positive signal:

- wrapper no-op mode is safe enough to use for experiments
- reset logs show episode-local state is cleared
- latency overhead is acceptable for iterative eval

First nonzero-write positive signal:

- success rate is neutral or better than frozen Psi0 on the same task
- no increase in catastrophic failures
- fast-weight norms stay bounded
- at least one write-scale setting shows a consistent positive trend

TODO:

- choose exact episode count and confidence criterion before claiming a
  scientific result
- choose the first SIMPLE task based on available checkpoint/data and baseline
  reproducibility

---

## 10. Open Design Questions

- What is the exact first outcome target?
- Is `Δ proprio` enough, or is visual/contact/progress information required?
- Should `Φ` be analytic, frozen for smoke testing, random for stress testing,
  or trained offline?
- Which layer indices should Mode B use?
- How should action-chunk outcomes be attributed?
- Should outcome updates piggyback through `/act`, or eventually use a separate
  endpoint?
- What latency budget qualifies as real-time for the chosen SIMPLE task?

Resolved (pinned for Phase 0):

- **No-op equivalence tolerance.** In the captured-request equivalence check
  (§9), the wrapper-with-`ΔW=0` output and the baseline output must satisfy
  `max |Δ| < 1e-5` in fp32 across the full 24-step action chunk. This
  allows for accumulator-order differences between `wrapped.net[2](h)` and
  `F.linear(h, base_w + 0)` while still being tight enough to catch any
  real wrapper bug. Tighten to `< 1e-6` if both runs use the same kernel
  selection.

---

## 11. Non-Goals For The First Implementation

- no VLM backbone adaptation
- no permanent finetuning during rollout
- no online optimizer or gradient step unless later justified
- no broad SIMPLE refactor
- no modification to unrelated baselines
- no claim of outcome-aligned success until a nonzero write rule is tested
