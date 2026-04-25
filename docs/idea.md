# Outcome-Aligned In-Place TTT for VLA Online Adaptation

## 1. One-Sentence Idea

Convert **In-Place Test-Time Training (TTT)** from a language-model
next-token/hidden-state memory mechanism into an **episode-local
next-outcome memory** mechanism for robot policies.

The goal is real-time online adaptation: a frozen VLA should use feedback from
the current rollout to adjust its behavior later in the same episode, without
permanently finetuning the base model.

---

## 2. Problem

VLA and WAM-style robot policies are usually trained offline and deployed with
fixed weights. During a real episode, the policy may encounter a specific
combination of conditions not seen during training:

- camera or calibration drift
- object pose, mass, friction, or contact variation
- embodiment mismatch or gripper offset
- partial occlusion
- long-horizon progress state
- repeated failed attempts that require local recovery

The research question is:

> Can a frozen pretrained robot policy acquire useful **episode-local
> information** during execution while preserving its general skill prior?

---

## 3. Core Hypothesis

Split the deployed policy into two timescales:

```text
slow weights = pretrained robot skill prior          frozen
fast weights = temporary memory for this episode     resettable
```

Inside selected MLP layers of the action expert, use:

```text
W_effective = W_frozen + ΔW_episode
```

`ΔW_episode` is reset at the start of each episode and updated only from
information observed during that episode.

The intended behavior is:

```text
general policy knowledge       -> W_frozen
scene/task/dynamics memory     -> ΔW_episode
```

---

## 4. Relation to Official In-Place TTT

The official In-Place TTT code provides the structural reference:

- selected Transformer MLP layers are TTT-enabled
- the fast weight is associated with the MLP down-projection
- the base parameter is not permanently overwritten during inference
- adapted fast weights live in a per-session cache
- resetting the cache restores base-model behavior
- inference-time adaptation is an algebraic write rule, not optimizer-based
  gradient descent

Important difference:

```text
Official In-Place TTT:
    target comes from LM hidden/input states
    training alignment comes from language-model loss

This idea:
    target should come from observed robot outcomes
    alignment objective is still a TODO
```

So the reusable part is the **cached fast-weight mechanism**, not the LM target.

The interesting consequence of this swap is not just "a different target". It
is that the write target now lives in the environment rather than inside the
model. If the write rule is aligned, the fast weights can store information
about *this scene's* physics, contact, and progress, instead of only recycling
the frozen policy's internal predictions. This is the scientific bet that makes
outcome supervision more than a cosmetic relabel.

---

## 5. Proposed Episode Loop

The method must be causal:

```text
reset ΔW_episode = 0

for each control step t:
    action_t = policy(obs_t; W_frozen + ΔW_episode)
    obs_{t+1}, info_{t+1} = env.step(action_t)
    ΔW_episode = update(ΔW_episode, obs_t, action_t, obs_{t+1}, info_{t+1})
```

The action at step `t` can only use fast-weight information from steps `< t`.
The outcome of step `t` may affect behavior from step `t+1` onward.

---

## 6. Outcome-Aligned Target

The central design question is how to convert observed robot feedback into a
useful fast-weight update.

Candidate outcome signals:

- `Δ proprio`
- `Δ visual_latent`
- contact state or slip signal
- progress signal
- task success/failure signal

Current placeholder:

```text
target_t = Φ(obs_t, action_t, obs_{t+1}, info_{t+1})
```

**v1 starting bet (a concrete default to commit to before exploring):**

```text
target_t = Δ proprio_{t -> t+1}
Φ        = simplest projection from R^|proprio| into the action-expert hidden space
write    = In-Place TTT-style algebraic write rule applied to the final
           projection inside selected action-expert feed-forward layers
```

Pick this not because it is right but because it is the cheapest falsifiable
specialization of `Φ`. Phase 1 of `minimum_validation_plan.md` already
tests this exact configuration, so the doc and the plan should stay aligned.

TODO:

- confirm `Δ proprio` is enough signal, or whether `Δ visual_latent` must be
  added to make the write informative
- define the representation space of `target_t` (proprio dim vs. hidden dim
  of the action-expert MLP — `Φ` bridges these)
- define how `target_t` maps into a fast-weight update (the algebraic write
  rule from In-Place TTT is the v1 default; alternatives are open)
- decide whether `Φ` is analytic, frozen/random for a smoke test, or learned
  offline in Phase 2
- define the training/alignment objective for `Φ` once it is learned
- define when an update should be skipped or gated (e.g. tiny proprio delta,
  contact instability, missing progress signal)
- **chunked-action / denoising-loop timing.** Psi0's action expert
  flow-matches over many internal denoising steps per control tick and emits
  an action *chunk*. This raises two open questions that are part of the
  method, not implementation detail:
  - where in the denoising loop is `W_frozen + ΔW_episode` actually
    applied — every denoising step, or only the final read?
  - which env-step's outcome supervises the write — the first action in
    the chunk, the last, or an aggregate? Misattribution here breaks the
    causal contract from §5.

Do not claim that real environment feedback is automatically corrective. It is
grounded, but it can still encode failed or misleading behavior.

---

## 7. Intended Placement

For a first version, keep the scope narrow:

- use a VLA with a clear VLM + action-expert split
- keep the VLM backbone frozen and untouched
- do not modify the final action projection
- place fast weights only in a small number of action-expert MLP layers
- reset fast weights at episode boundaries

In this repo, Psi0 is the natural first candidate because it has:

- a frozen VLM backbone plus action expert
- action-expert transformer blocks with MLP feed-forward layers
- a SIMPLE rollout loop with an explicit `get_action -> env.step` structure
- an existing episode reset signal in the policy/server path

Resolved (per `code_reading_map.md`):

- The conceptual fast-weight insertion point is the action-stream feed-forward
  path `ff_act` inside `VLATransformerBlock`, defined in
  `extern/psi0/src/psi/models/psi0.py:814-1006`.
- The exact parameter to wrap should be the final projection inside Diffusers'
  `FeedForward` module, confirmed at runtime against the active installed
  Diffusers version and checkpoint structure.
- The action expert has 6 such blocks (1536-dim each). v1 wraps a small
  middle subset, not all six.
- Episode reset is already wired through `Psi0Agent.reset()`
  (`extern/simple/src/simple/baselines/psi0.py:174-184`) and the existing
  `history={"reset": True}` server signal. The reset signal exists; model/server
  code still needs to clear the fast-weight cache.

Still open:

- confirm the wrapping integrates cleanly with the active Psi0 checkpoint
  at runtime (parameter shapes, dtype, sharding)
- confirm whether the write happens once per control step, once per action
  chunk, or once per internal denoising step (see §6, chunked-action TODO)
- measure added latency before making any real-time claim

---

## 8. Smallest Safe Development Plan

The first code change should test infrastructure, not scientific performance.

Minimum safe path:

1. Add an optional fast-weight wrapper around selected action-expert MLP layers.
2. Store fast weights separately from base parameters.
3. Reset the fast-weight cache at each episode boundary.
4. Start with fast-weight updates disabled or zero-scaled.
5. Verify that wrapped Psi0 matches the frozen baseline when `ΔW_episode = 0`.
6. Log hidden keys, fast-weight norms, action outputs, and latency.
7. Only then try a nonzero write rule.

This mirrors the official In-Place TTT safety pattern: base behavior must remain
recoverable by clearing the cache or disabling TTT.

**Stability levers to have ready before turning on a nonzero write rule:**

- a small write-step / scaling factor on every update to `ΔW_episode`
- EMA decay so old per-step writes fade rather than accumulate forever
- norm clipping on the cumulative `ΔW_episode`
- per-update gating that can skip writes when the proprio delta is below a
  noise floor, contact is unstable, or a progress signal is missing

These are not separate research questions. They are required for a closed-loop
rollout to not diverge once a nonzero write rule is enabled, and they should
exist as configurable knobs from the first nonzero experiment, not be added
after the first divergence is observed.

---

## 9. What Must Be Tested First

Weak assumptions:

- similar within-episode states must produce similar action-expert hidden keys
- outcome signals must contain information useful for future action correction
- updates must not destabilize diffusion/flow-style action generation
- action-chunk predictions must not misattribute an outcome to the wrong action
- the update must be cheap enough for online use

First empirical checks:

- baseline Psi0 closed-loop success on a SIMPLE task
- no-op fast-weight wrapper preserves baseline outputs
- hidden-state/key similarity has useful within-episode structure
- fast-weight norms remain bounded under any proposed write rule
- latency stays within the control budget

---

## 10. Contribution Statement

> We propose **Outcome-Aligned In-Place TTT** for VLA policies: a resettable
> episode-local fast-weight memory inside selected action-expert MLP layers,
> updated from observed robot outcomes rather than language-model hidden-state
> targets. The frozen backbone preserves general skill, while the fast weights
> aim to store temporary information about the current scene, dynamics, and task
> progress.

Novelty in one line:

```text
In-Place TTT for LLMs:  hidden-state / next-token-aligned memory
This work:             observed-outcome memory for robot control
```

Open TODO:

> Define and validate the outcome-aligned write rule. This is the core research
> uncertainty, not an implementation detail.
