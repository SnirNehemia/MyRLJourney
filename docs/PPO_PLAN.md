# PPO Implementation & A2C Comparison Plan

The working plan for the next version: implement **PPO incrementally** and demonstrate why it beats
**A2C** on `Hopper-v4`. Companion to [../CLAUDE.md](../CLAUDE.md).

## Goal & deliverable

Learn how PPO works and *why* it differs from A2C, by building it one ingredient at a time and
measuring what each ingredient adds. The output is a **LinkedIn post** comparing A2C vs PPO with:

1. **Sample efficiency** — score vs **environment steps** (the fair x-axis for on-policy methods).
2. **Stability across seeds** — mean ± std shaded region over multiple seeds.
3. **Final performance + a side-by-side GIF** — best A2C policy vs best PPO policy hopping.

## The conceptual bridge: A2C → PPO

The A2C agent here is already *most of the way* to PPO. `A2CAgent.learn_from_batch` collects an
n-step rollout across `num_envs`, computes GAE advantages, and even runs **K critic epochs** per
rollout. The v1.5.2 ablation ("can K critic epochs rescue λ=0.95?") is literally asking *how close
A2C can get to PPO*.

What A2C **cannot** safely do: reuse the same rollout for **multiple actor updates**. After one
gradient step the policy has moved, so the collected actions are now off-policy and the gradient
is wrong. PPO's core idea fixes exactly this:

- Track the **probability ratio** `r(θ) = π_new(a|s) / π_old(a|s)` (requires storing `log π_old` at
  collection time — A2C never does this).
- **Clip** the ratio to `[1−ε, 1+ε]` so a single batch can be reused for several epochs without the
  policy lurching too far. This is the one mechanism that turns "many epochs = instability" (A2C)
  into "many epochs = sample efficiency" (PPO).

Everything else PPO adds (minibatching, value clipping, advantage norm, KL stop) is refinement on
top of that ratio-clipping core.

## Environment: Hopper-v4 (MuJoCo)

```powershell
pip install mujoco        # add to requirements.txt
```

Add this entry under `environments:` in `config.yaml` (Hopper: 11-dim obs, 3-dim continuous action):

```yaml
  "Hopper-v4":
    state_size: 11
    action_size: 3
    is_continuous: true
    win_condition: 2500.0   # not an official "solved" threshold; a strong-policy target
    network:
      hidden_size: [64, 64] # 64x64 is the standard PPO MuJoCo size
    v_plot_range: [0, 3000] # for the critic value plot in GIFs
```

Notes:
- `Hopper-v5` is the newer Gymnasium MuJoCo API; `-v4` is fine and matches the chosen target. Pick one
  and keep it consistent across A2C and PPO runs so the comparison is apples-to-apples.
- Validate the PPO implementation on cheap `Pendulum-v1` (minutes) before committing to long Hopper
  runs — it catches sign/shape bugs fast.

## Incremental build (each stage = a config flag, so it's ablatable)

Add a `PPOAgent` in `agent.py` and a `ppo(...)` loop in `train.py` (same signature & 5-tuple return
as `a2c`). Gate each ingredient behind a flag in `config.agent` so the ablation runner can sweep
them and show the contribution of each. Suggested flags: `ppo_clip_eps`, `ppo_epochs`,
`num_minibatches`, `clip_value_loss`, `normalize_advantage`, `target_kl`.

**Stage 0 — Importance ratio, no clip (the "naive multi-epoch A2C").**
Store `log π_old` during rollout; do K actor epochs using the ratio `r(θ)` (unclipped surrogate).
*Expected result:* unstable / collapses with `ppo_epochs > 1`. This is the motivating failure that
justifies clipping — capture its curve for the post.

**Stage 1 — Clipped surrogate objective (the core).**
`L = E[ min(r·A, clip(r, 1−ε, 1+ε)·A) ]`, ε ≈ 0.2. Now multi-epoch updates are stable.
*Expected result:* the first clearly-better-than-A2C curve.

**Stage 2 — Minibatch multi-epoch updates.**
Flatten the rollout, shuffle, split into `num_minibatches`, do `ppo_epochs` passes. This is the
engine of PPO's sample efficiency. Builds on the existing K-epoch loop structure.

**Stage 3 — Value clipping + single shared loss/optimizer.**
Combine actor + value + entropy into one loss with one optimizer (A2C currently uses *separate*
actor/critic optimizers — note this difference). Optionally clip the value update symmetrically.

**Stage 4 — Advantage normalization + KL early-stop.**
Per-minibatch advantage normalization, and stop the epoch loop early when approximate KL exceeds
`target_kl`. The robustness layer — supports the "PPO is reliable" narrative.

Each stage is its own `ablation_study.experiments` entry → `run_ablation.py` already produces the
mean±std overlay plot comparing them.

## Comparison methodology

- **Reuse the ablation runner.** Define experiments for `A2C` and the PPO stages, set a `seeds:` list
  (3+ for a credible band), point `active_env` at `Hopper-v4`, and run `python ablation_study.py`.
  `plot_ablation_statistics` already draws the mean±std shaded comparison.
- **Sample-efficiency x-axis (needs a small change):** the loops currently log
  `grad_updates_at_score` (cumulative gradient updates per completed episode). For "score vs env
  steps", add a global **environment-step counter** (per on-policy iteration, `+= n_steps * num_envs`)
  and return it alongside scores, then plot score vs env-steps. This is the headline plot — make it
  the fair axis for the A2C-vs-PPO claim.
- **GIFs:** `make_gif.py` / `test_ablation.py` already build comparison GIFs from best checkpoints;
  the side-by-side A2C-vs-PPO hopping clip is the visual hook.

## Suggested starting hyperparameters (Hopper PPO)

Add under `config.agent` (tune from here):

```yaml
  # PPO
  ppo_clip_eps: 0.2
  ppo_epochs: 10
  num_minibatches: 32
  clip_value_loss: true
  normalize_advantage: true
  target_kl: 0.03        # null/0 to disable early-stop
  # reuse existing: gamma 0.99, gae_lambda 0.95, num_envs, n_steps (rollout length),
  # entropy_weight_*, lr (try ~3e-4), critic_loss_weight 0.5
```

## Checklist

- [ ] `pip install mujoco`; add to `requirements.txt`; add `Hopper-v4` to `config.yaml`.
- [ ] Smoke-test the env: short A2C run on `Hopper-v4` to confirm the continuous pipeline works.
- [ ] `PPOAgent` in `agent.py`; `ppo(...)` in `train.py` (match signature + 5-tuple return).
- [ ] Validate PPO on `Pendulum-v1` before long Hopper runs.
- [ ] Build stages 0→4, each behind a flag, each an ablation experiment.
- [ ] Add the global env-step counter for the sample-efficiency plot.
- [ ] Run multi-seed `ablation_study.py`: A2C vs PPO stages on Hopper.
- [ ] Generate side-by-side GIF; write `discussions/Version_X-Y-Z.md`; bump version; update README.
