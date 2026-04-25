# Code Reading Map — LiNexa

Exact file paths and why each matters. Priority order: in_place_ttt → psi0 → simple.

## 1. `extern/in_place_ttt/` — the design reference

### Tier 1 — must read to understand the mechanism

| File | Lines | Why it matters |
|---|---|---|
| `inference_model/hf_qwen3/modeling_qwen3.py` | 83–152, 302–378, 453–557, 576–587 | **The inference-time TTT.** `Qwen3MLP.forward` (83–152) is the fast-weight update: key `h = act(gate(x)) * up(x)`, target `t`, convolution over t, outer-product write `ΔW = ttt_lr · (ttt_proj · conv(t))ᵀ · h`, read `W · h`. `Qwen3DecoderLayer.forward` (316–378) caches the carry state and handles chunk boundaries. `TTTDynamicCache` (52–59) is the per-session memory container; resetting it = resetting fast memory. `Qwen3ForCausalLM.generate` (576–587) enforces `bs=1` and wires the TTT cache into HF generation. This is the reference design for the read/write mechanics of a LiNexa fast memory. |
| `hf_models/hf_qwen3/modeling_qwen3.py` | 83–138, 295–338, 620–633 | **The training-time TTT.** Same mechanism but vectorized across chunks via `cumsum` (line 136). Shows that the *only* training loss is standard causal LM loss (line 631) — "LM-aligned" is entirely a consequence of end-to-end training with TTT on, not an auxiliary objective. This clarifies what "outcome-aligned" would have to mean for us: we'd need to train with the TTT path active and with a loss that propagates through it. |
| `configs/pretrain/qwen3_longct.yaml` | all 59 lines | Hyperparameters: `ttt_layers: [0, 6, 12, 18, 24, 30, 36]` (every 6th layer of Qwen3-8B), `ttt_lr: 3` (not a typo — the update is large), `ttt_chunk: 4096`, `ttt_proj: true`. Useful prior for choosing comparable sparsity and magnitudes in the action head. |

### Tier 2 — useful but not critical for a first prototype

| File | Why |
|---|---|
| `tasks/train_torch.py` | Continual-pretraining loop via VeOmni. Not directly portable — we'd use Psi0's trainer — but shows the training cadence and the fact that there is *no* TTT-specific auxiliary loss. |
| `hf_models/hf_qwen3/configuration_qwen3.py`, `hf_models/hf_llama/configuration_llama.py` | Config fields (`ttt_mode`, `ttt_layers`, `ttt_lr`, `ttt_chunk`, `ttt_proj`, `ttt_target`) — the exact surface that was added to stock HF configs. |
| `inference_model/hf_llama3/modeling_llama.py` | Parallel implementation for Llama — diff against Qwen version to see which parts are architecture-agnostic. |

### Tier 3 — can skip for now

`eval_config/*.py`, `eval.sh`, `scripts/merge_dcp_to_hf.py`, `tasks/infer.py` — RULER long-context eval and HF format conversion; LM-specific, not needed for VLA.

## 2. `extern/psi0/` — the model to adapt

### Tier 1 — the insertion point

| File | Lines | Why it matters |
|---|---|---|
| `src/psi/models/psi0.py` | 1480–1601 (`Psi0Model.__init__` / `from_pretrained`), 1602–1639 (`forward`), 1641–1733 (`predict_action`) | **The host model.** Shows VLM backbone (frozen) + action expert (trainable). `predict_action` runs VLM once then `num_inference_steps=10` denoising sweeps of the action head (line 1714). This 10× per-step amplification is why the action head is the right insertion point. |
| `src/psi/models/psi0.py` | 814–1006 (`VLATransformerBlock`) | **The block where fast memory would live.** Each block has `self.ff_act = FeedForward(dim=1536, ..., activation_fn="gelu-approximate")` (line 900) — a Linear → GELU → Linear stack. Its second Linear is the direct analog of Qwen3MLP's `down_proj`. The forward (921–1006) cleanly isolates the action-stream FF (962–967) from the obs-stream FF (998–1004); a prototype touches only `ff_act`. |
| `src/psi/models/psi0.py` | 1008–1349 (`ActionTransformerModel`, `DiTActionTransformerModel`) | Stacks of `VLATransformerBlock`. `forward` accepts `joint_attention_kwargs={action_hidden_embeds, views, obs, traj2ds}`. Shows where to intercept per-block hidden states if we need custom routing for the write path. |
| `src/psi/config/model_psi0.py` | all 99 lines | Model config. `num_blocks: int = 6` (line 51), `hidden_dim: int = 1536` (line 50), `nhead: int = 24` (line 53), `eval_diffusion_steps: int = 10` (line 28). Tells us the scale of a fast-memory clone (e.g., for one `ff_act` Linear of shape `[1536, 6144]`: ~9.4M params per copy — cheap). |

### Tier 2 — deployment and rollout integration

| File | Lines | Why it matters |
|---|---|---|
| `src/psi/deploy/psi0_serve_simple.py` | 26–169, 126 | **HTTP server that hosts the model for SIMPLE eval.** `Server.predict_action` is the per-step entry point. **Line 126 is the current episode-reset signal**: `if self.previous_action is None or "reset" in history_dict`. LiNexa can piggyback on this exact keyword to reset fast memory server-side. |
| `examples/simple/simple_eval.py` | all 159 lines | End-to-end wrapper that starts the server and runs `EvalRunner` against it. Perfect template for a LiNexa evaluation script — only changes are (a) the model serving path and (b) adding the post-step outcome-update round trip. |

### Tier 3 — training, only when we get there

| File | Why |
|---|---|
| `src/psi/trainers/finetune.py` | 598 lines; the finetune loop. If/when we train the `ttt_proj`/`ttt_conv` analog, this is the trainer to extend. |
| `src/psi/config/train/finetune_simple_psi0_config.py` | Launch config for SIMPLE finetune — env, data, optimizer, checkpoint. |
| `src/psi/trainers/trainer.py` | Base trainer — read before modifying `finetune.py`. |
| `src/psi/tokenizer/fast_action_tokenizer.py` | Action tokenization; may be irrelevant for DiT/flow path but good to confirm. |

### Deliberately out of scope for prototype

- `src/psi/deploy/psi_serve_rtc-*.py` — RTC variants; extra complexity we do not need for a clean first test.
- `baselines/` (act, dp, egovla, gr00t-n1.6, h-rdt, internvla-m1, pi05) — other VLA baselines; not needed to validate the idea inside Psi0.
- `real/` — real-robot teleop/deploy; we are simulation-only.
- `src/openpi/`, `src/gr00t/`, `src/h_rdt/`, `src/InternVLA-M1/`, `src/egovla/`, `src/act/`, `src/dp/`, `src/fast/` — vendored baseline source trees.

## 3. `extern/simple/` — the rollout harness

### Tier 1 — the episode loop

| File | Lines | Why it matters |
|---|---|---|
| `src/simple/evals/env_runner.py` | 282–352 (`run_episode`), 146–231 (`run`), 98–113 (`_PolicyObjectAdapter`) | **The closed-loop rollout.** `run_episode` resets the env, calls `policy.reset(**reset_kwargs)` (line 313 — **the fast-memory-reset hook**), then loops `action = policy.get_action(...)` + `env.step(action)` until terminated/truncated. LiNexa's memory write fires between the step return and the next `get_action`. |
| `src/simple/baselines/psi0.py` | 39–185 | **The policy client adapter for Psi0.** `Psi0Agent.reset()` (174–184) already zeroes per-episode state and sets `_reset_history = True`. `get_action()` (70–172) sends observation + `history={"reset": True, ...}` on the first call of each episode. This is where LiNexa adds the post-step outcome packet to the request — or, for an inline prototype, where we replace the HTTP client with a direct `Psi0Model` call. |
| `src/simple/cli/eval.py` | 43–646 | The CLI entry (`run_eval`, line 340; `main`, line 601). Launches worker processes that call `EvalRunner`. Useful as the reproduction target for the baseline number. |

### Tier 2 — policy interface and env plumbing

| File | Why |
|---|---|
| `src/simple/agents/policy_agent.py`, `src/simple/agents/base_agent.py`, `src/simple/agents/primitive_agent.py` | The agent base classes Psi0Agent inherits from. Confirm the `reset()` / `get_action()` contract and the action-queue mechanics. |
| `src/simple/envs/__init__.py`, `src/simple/envs/wrappers/` | Env IDs and wrappers. `max_episode_steps` defaults to 360 in the Psi0 eval script — important for planning rollout budget. |
| `src/simple/datasets/lerobot.py`, `src/simple/datasets/rlds.py` | Episode loading — determines what the init env_conf looks like at reset. |
| `src/simple/engines/mujoco.py`, `src/simple/engines/isaacsim.py` | The physics backends. Only relevant if MuJoCo-only runs are needed to avoid Isaac Sim overhead during rapid iteration. |

### Tier 3 — reference only

- `src/simple/tasks/*.py` — individual task definitions (pick / handover / bend-pick / locomotion variants). Useful for choosing the smallest task for first eval.
- `src/simple/robots/*.py` — robot abstractions.
- `src/simple/baselines/` (other files) — competing policies; not needed to validate the core idea.

## 4. The smallest safe insertion point

If we eventually build a prototype, the minimal diff set would touch (read-only today — listed only so the boundary is explicit):

- **New files in `src/linexa/`:**
  - `ttt/fast_memory.py` — `FastMemoryState` + `write(h, t)` + `read(h)` + `reset()`.
  - `ttt/hooks.py` — a monkey-patch or wrapper around `ff_act` of selected `VLATransformerBlock`s that reads the fast memory on forward and buffers keys for the next write.
  - `adapters/psi0_linexa.py` — subclass `simple.baselines.psi0.Psi0Agent`; after each `env.step` (via a callback from `env_runner`), compute target `t = Φ(o_{t+1} − o_t, Δproprio)` and push the write into the fast memory.
  - `eval/run_simple.py` — a LiNexa copy of `extern/psi0/examples/simple/simple_eval.py` that runs the adapted agent either in-process or against an extended server.
- **Patches in `patches/psi0/` (via `patches/`, not in-tree):**
  - Minimal hook in `VLATransformerBlock.forward` to expose `h = act(gate(x)) * up(x)` and the post-`ff_act` projection as tap points. Ideally the hook is off by default and controlled by a config flag.

### What should not be touched in a first prototype

- **VLM backbone (`self.vlm_model`).** Frozen in Psi0; stays frozen.
- **`ActionTransformerModel`/`DiTActionTransformerModel` architecture.** Do not add/remove blocks or heads; only tap into existing FF modules.
- **Flow-matching scheduler and denoising loop.** Writes/reads happen once per *control step*, not once per diffusion step, even though the action head runs 10 sweeps.
- **Attention sublayers.** Keep the fast-weight story local to the FF, per In-Place TTT's design. Touching attention invites issues with the AdaLN conditioning and the joint action/obs cross-attention pattern.
- **HTTP client/server protocol** (if possible). Use the existing `history={"reset": True}` signal. Only extend the payload when unavoidable.
- **Other `baselines/` in both repos.** They are not part of the experiment.
