# CLAUDE.md

Guidance for working in this repo. Keep this file lean — deep dives live in `docs/` and `discussions/`.

## What this project is

A **from-scratch RL learning repo** (PyTorch + Gymnasium). The owner is learning RL by
implementing algorithms one at a time and studying them with ablations. Progression so far:
DQN → DDQN → Dueling → PER → REINFORCE → **A2C (with GAE, n-step, K-critic-epochs)**.

**Current goal: implement PPO incrementally and demonstrate its advantage over A2C on
`Hopper-v4` (MuJoCo, continuous control), for a LinkedIn write-up.** Full plan:
[docs/PPO_PLAN.md](docs/PPO_PLAN.md).

The point is *understanding*, not just a working agent. Favor readable, well-commented code
that makes the mechanism visible over clever/terse code. Match the existing comment density in
`agent.py` (the A2C `learn_from_batch` is the style reference).

## Architecture (the mental model)

Everything is driven by **`config.yaml`** — the single source of truth.

- **`config.yaml`** — project/env/agent/training/ablation settings. `active_env` picks the env;
  `agent.algorithm` (a string) picks the training loop.
- **Dispatch**: the algorithm string is resolved to a function in `train.py` by name via
  `getattr(train_module, algo_to_run)`. So `algorithm: "a2c"` runs `train.a2c(...)`. **To add
  PPO you add a `ppo(...)` function in `train.py` and a `PPOAgent` in `agent.py` — no dispatcher
  edits needed.**
- **`agent.py`** — agent classes: `Agent` (DQN/DDQN/Dueling/PER), `A2CAgent`, `ReinforceAgent`,
  plus `ReplayBuffer`. Each agent owns its network (defined inline as a nested class for the
  policy-gradient agents) and its learning rule.
- **`brain.py`** — `QNetwork` only (the DQN value net, supports dueling). Policy/actor-critic
  nets live inside their agent classes in `agent.py`.
- **`train.py`** — one training-loop function per algorithm (`dqn`, `a2c`, `reinforce`). Each has
  the **same signature** and returns the **same 5-tuple**:
  `(scores, q_values, avg_value_history, grad_updates_at_score, loss_histories_dict)`.
  Keep this contract when adding `ppo` so the runners and plotters work unchanged.
- **Pipeline scripts**: `run_ablation.py` (train all `ablation_study.experiments` × seeds, then
  plot mean±std), `test_ablation.py` (evaluate best checkpoints), `make_gif.py` (comparison GIFs).
  `ablation_study.py` runs all three as one pipeline.
- **`run_experiment.py`** — multi-seed single-config runs.

## Conventions

- **Continuous vs discrete** is per-env via `is_continuous` in `config.yaml`. Continuous agents use
  a `Normal` policy (tanh-mean + softplus-std); discrete use `Categorical`. `DiscretizeBoxWrapper`
  in `train.py` lets DQN-family agents handle Box action spaces by binning.
- **Outputs** go to `raw_results/{env}/{version}/{run_type}/[{study}/]{run_name}/`. Every run saves
  its exact `run_config.yaml` for reproducibility. `*_best.pth` / `*_local_best.pth` / `*_last.pth`
  checkpoints.
- **Seeds**: passed explicitly; env reset uses `seed + i_episode`. Ablation/experiment runners loop
  over a `seeds:` list and aggregate.
- **Per-version discussions**: each major version gets a `discussions/Version_X-Y-Z.md` with the
  theory + ablation findings (see `discussions/Version_1-5-2.md` for the format). Write one for PPO.
- **README version history** is updated per release; bump `project.version` in `config.yaml`.
- Adding a new algorithm = **new `Agent` class + new `train.py` function + set `agent.algorithm`**.
  Optionally add an `ablation_study.experiments` block to compare it against others.

## Running

```powershell
# Single training run (uses config.agent.algorithm + config.active_env)
python train.py

# Full ablation pipeline (train all experiments × seeds → test → GIFs)
python ablation_study.py
```

Edit `config.yaml` first: set `active_env`, `agent.algorithm`, and the `ablation_study` block.

## Gotchas

- **Windows + PowerShell**; the venv is in `venv/`. `requirements.txt` is UTF-16 — read it with an
  editor, not naive byte tools.
- **`Hopper-v4` needs MuJoCo**: `pip install mujoco` (not yet in `requirements.txt`; add it). No
  config entry for Hopper exists yet — see [docs/PPO_PLAN.md](docs/PPO_PLAN.md) for the one to add.
- **A2C/PPO use `gym.vector.AsyncVectorEnv`** (`num_envs` parallel envs). On Windows this spawns
  processes — keep env construction picklable and guard entry points under `if __name__`.
- torchrl emits FutureWarnings; they're filtered at the top of `agent.py`.
- This is a **single-developer learning repo**: there is no test suite. "Verify" means run the
  training loop for a short run and watch the score/diagnostics curves.
