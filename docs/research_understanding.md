# Research Understanding — LiNexa

Read-only analysis phase. No implementation. All claims tagged `[code]`, `[paper]`, or `[hypothesis]`.

## 1. The idea in plain language

LiNexa wants a humanoid VLA policy that **adapts within a single episode** by maintaining a tiny "fast memory" inside the action pathway. The fast memory is:

- **reset at each new episode** — no cross-episode leakage, no catastrophic forgetting
- **updated after every control step** using real feedback from the next observation
- **read during future forward passes** to bias the policy toward corrections it has already "written down" for similar states

The full policy (VLM backbone + action expert) stays frozen. Only the fast memory moves at runtime.

### Why this might help on humanoid VLA

Humanoid manipulation suffers from *episode-specific* shifts that are not new skills:
object pose / camera shift / control delay / friction / bimanual coordination bias / progress-dependent hidden state. These are **local corrections**, not new capabilities. A small temporary memory is a natural fit — it can absorb the local correction without touching what the base policy already knows.

### Why this is not "copy In-Place TTT into robotics"

In-Place TTT `[paper: arxiv/2604.06169; code: extern/in_place_ttt/inference_model/hf_qwen3/modeling_qwen3.py]` updates the MLP down-projection weight *within a single forward pass* using **post-attention hidden states as targets**, chunked over a long context. The update is **LM-aligned** only because of training with TTT active — the gradient of next-token loss flows through `ttt_proj` and `ttt_conv` (code lines 86–91) and teaches them to issue updates that reduce future token loss. The targets themselves are internal features, not external outcomes.

Two mismatches with our setting:

| Aspect | In-Place TTT | LiNexa (as proposed) |
|---|---|---|
| Update trigger | Chunk boundary inside one forward pass | After each `env.step`, between forward passes |
| Target signal | Post-attention hidden state of the same layer `[code]` | Real environment feedback: next-obs/proprio delta `[hypothesis]` |
| Supervision | End-to-end next-token loss fine-shapes `ttt_proj`, `ttt_conv` | Episode-level outcome alignment, training objective TBD |
| Reset scope | End of generation / "clear" command | End of each episode |

So "outcome-aligned" is the critical divergence. In-Place TTT's mechanism is reusable; its semantics need replacement.

### Why episode-level and resettable

- Humanoid episodes are short (seconds to minutes) and independent. Inter-episode conditions differ dramatically (object poses, lighting, task instance).
- Persistent memory across episodes would be a slow learner, not a fast adapter — and would need *forgetting* logic to avoid drift.
- Reset = clean semantics: memory stores "what I've discovered about this specific rollout", nothing more.
- The codebase already has an episode-reset hook for the policy: `simple.baselines.psi0.Psi0Agent.reset()` sets `_reset_history = True` (extern/simple/src/simple/baselines/psi0.py:174–184) `[code]`.

### Why outcome-aligned targets

Reconstruction targets (In-Place TTT style: "predict your own post-attention hidden state") require matched training. If LiNexa tries to bolt TTT onto a *frozen* Psi0 without retraining `ttt_proj`/`ttt_conv`, updates would be random noise in weight space and probably hurt. `[hypothesis — this is the central risk of Path A below.]`

Outcome-aligned targets (next-obs-change, proprio delta, a progress-like signal) carry direct information about *what the world did in response to the action* and are well-defined without retraining the backbone — though you still need to learn a write function that maps (state, outcome) into a useful weight update. This is the design space LiNexa proposes to explore.

## 2. Technical formulation of the idea

### 2.1 In-Place TTT mechanics we can reuse (grounded in code)

From `extern/in_place_ttt/inference_model/hf_qwen3/modeling_qwen3.py`:

- **Fast weight:** a per-session clone of `down_proj.weight` (line 123), shape `[hidden_size, intermediate_size]`. Stored in `TTTDynamicCache.ttt_states[layer_idx] = (past_h, past_t, past_w)` (lines 52–59).
- **Key (`h`):** output of gate/up activation `h = act_fn(gate_proj(x)) * up_proj(x)` (line 118). One "key" per token position, dim = `intermediate_size`.
- **Target (`t`):** post-attention hidden state passed in as `target_states` (Qwen3DecoderLayer.forward, lines 347–348).
- **Convolution over targets:** depthwise `Conv1d(kernel=5, padding=2)` zero-initialized (lines 102–104, 406–408) — smooths t across a small temporal window before using it.
- **Learned write projection:** optional `ttt_proj` linear layer on t (lines 98, 146).
- **Read:** `y = W · h` (line 141) — vanilla linear with the *current* fast weight.
- **Write (per chunk):** `ΔW = ttt_lr · (ttt_proj · t_conv)ᵀ · h` (line 146) — an outer-product update (classic fast-weight / linear-attention memory).
- **Chunk-wise scheduling:** chunk `i` reads from `W_i`, writes `ΔW_i`, next chunk reads from `W_{i+1} = W_i + ΔW_i` (lines 140–150). Causal at chunk granularity.

**The whole thing is a linear associative memory:** `W_t = W_0 + Σ_s outer(h_s, proj(t_s))`. A retrieval `W_t · h_query` returns `W_0 · h_query + Σ_s ⟨h_s, h_query⟩ · proj(t_s)`. So similar prior `h`'s retrieve their stored `proj(t)` values, scaled by similarity. `[paper: Schlag et al. "Linear Transformers are Secretly Fast Weight Programmers"]`.

### 2.2 Mapping to the Psi0 action head (grounded in code)

Psi0 `[code: extern/psi0/src/psi/models/psi0.py]`:
- VLM backbone `self.vlm_model` = Qwen3-VL-2B-Instruct (lines 1512, 1529).
- Action expert `self.action_header` = 6× `VLATransformerBlock` (lines 814–1006) of hidden dim 1536, each with a `FeedForward(dim, activation_fn="gelu-approximate")` on the action stream (`self.ff_act`, line 900) — a Linear → GELU → Linear stack whose last Linear is the direct analog of Qwen3MLP's `down_proj`.
- Per inference step, `Psi0Model.predict_action` (line 1641) runs the VLM once, then `num_inference_steps=10` denoising sweeps of the action head over a random `(B, Tp, Da)` action chunk. Each sweep is a forward pass of all 6 blocks.

Proposed role assignment for a LiNexa prototype:

| In-Place TTT role | LiNexa analog |
|---|---|
| Fast weight `W` | `ff_act.down_proj.weight` clone in selected action-head blocks `[hypothesis]` |
| Key `h` | output of `ff_act.up_proj * act(gate_proj)` at each action-token position |
| Target `t` | **Path A:** post-attention hidden state in the same block (In-Place TTT faithful). **Path B:** embedding of `(o_{t+1} − o_t, Δproprio, progress)` broadcast to action-token positions, fused via a small write head |
| Update moment | **Path A:** within each denoising-sweep forward pass. **Path B:** after `env.step`, once per control step |
| Reset | `Psi0Agent.reset()` → `_reset_history=True` triggers server-side clear |

### 2.3 Why insert into the action pathway, not the VLM backbone

Five grounded reasons:

1. **Compute.** The VLM backbone is Qwen3-VL-2B (~2B params). A TTT insertion there means cloning a `down_proj` of shape `[hidden × intermediate]` per selected layer per episode — manageable but burns VRAM we don't need. The action head at 1536 hidden × 4× FF is much smaller per layer.
2. **Rate.** The VLM runs **once per control step**; the action head runs **`num_inference_steps = 10`** times per step (line 1714). More forward passes per unit of env feedback = more opportunities for the read path to exploit the write.
3. **Training scope.** Psi0's finetune pipeline `[extern/psi0/src/psi/trainers/finetune.py]` freezes the VLM (`tune_vlm: bool = False`, config/model_psi0.py:70). Adding trainable TTT slow params (`ttt_proj`, `ttt_conv`) to the action head stays within the currently-trainable surface; doing the same to the VLM forces unfreezing.
4. **Locality of corrections.** The class of shifts the idea targets (pose, delay, friction, bimanual bias) are largely **motor-level** — they live closer to action generation than to vision-language grounding. Writing corrections into the action head is a cleaner hypothesis than writing them into the VLM. `[hypothesis]`
5. **Safety.** The VLM is the part of the model most expensive to get right; leaving it untouched reduces the risk of breaking language-conditioning or visual grounding.

### 2.4 Where outcome-aligned targets come from (per step)

Inside `simple.evals.env_runner.EnvRunner.run_episode` (extern/simple/src/simple/evals/env_runner.py:282–352) `[code]`:

```python
observation, info = env.reset(...)                              # episode start
policy.reset(**reset_kwargs)                                    # ← fast memory reset
while not episode_over:
    action = policy.get_action(observation, info=info, instruction=instruction)
    observation, reward, terminated, truncated, info = env.step(action)  # ← outcome observed here
```

Between iterations `i` and `i+1`, three things are known: `obs_i`, `action_i`, `obs_{i+1}` (with its proprio). LiNexa would compute a **target embedding** `t_i = Φ(obs_{i+1} − obs_i, Δproprio, progress)` and write it into the fast memory using the keys `h_i` that the action head produced when generating `action_i`. `[hypothesis — Φ is a new learned module.]`

### 2.5 Assumptions under which this can help

1. **Correction-shaped shifts.** Episode-specific error is structured: there exists a small correction ΔW such that base_policy(x) + correction closes most of the gap for a meaningful fraction of episodes.
2. **Key locality.** Future states within an episode that *need* the correction have hidden representations `h` similar to earlier states where the correction was learned. If retrieval misses, the memory is dead weight.
3. **Targets are informative.** Next-obs deltas and proprio changes carry enough signal about the correction. For some shifts this is true (contact friction → joint torque trace differs; object slip → relative hand-object pose differs). For others, maybe not.
4. **Write function exists.** There is a `Φ` (and a learned `ttt_proj` style matrix) such that writing `Φ(outcome)` keyed by `h` reduces future action error *on average*. This can be trained offline from logged rollouts before claiming anything at evaluation time.

### 2.6 Assumptions under which this fails

- **Shifts are perceptual, not motor.** E.g. wrong task semantics from instruction — fast memory in the action head cannot fix an incorrectly-grounded goal. Would need VLM-side memory.
- **Keys are not stable.** If the action head's intermediate features at similar physical states look very different across timesteps (e.g. due to diffusion noise at different denoising sweeps), retrieval quality collapses.
- **Update drowns in denoising.** Each action-chunk prediction runs 10 denoising sweeps; writing from diffusion-step-k states and reading at diffusion-step-k' states may be inconsistent.
- **Outcome targets are too noisy.** Observation delta at high control rates is dominated by sensor noise rather than the signal we care about.
- **Scale gap.** `ttt_lr · outer(h, proj(t))` at small episode lengths (say 360 steps × 10 denoising = 3600 writes but writes are tied to control steps only — so ~360 writes) may be too few to accumulate meaningful structure.
- **Contamination across chunks.** Psi0 predicts `Tp` actions per call and executes `Ta` of them; writes tied to the wrong action in the chunk become systematically misattributed.

## 3. Theoretical claim to test

> If, within an episode, a state `x_t` shares its action-head key `h_t ≈ h_s` with a prior state `x_s` where an outcome-aligned write `Φ(outcome_s)` was installed in the fast memory, then the action predicted at `x_t` will incorporate a bias term proportional to `⟨h_t, h_s⟩ · Φ(outcome_s)`, reducing future action error on average relative to the frozen base policy.

This claim is **not yet proven**. It depends on all four assumptions in §2.5 holding simultaneously — and we have no direct code evidence that they do in humanoid VLA. The validation plan (§4) is designed to decide fast whether pursuing this is worth it.
