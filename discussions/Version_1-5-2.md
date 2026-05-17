# Version 1.5.2 Discussion — A2C and the Search for Why λ=0.95 Fails

## Version Features

**Changes:**
- Implemented REINFORCE (Monte Carlo policy gradient) — a clean baseline with no critic.
- Implemented Advantage Actor-Critic (A2C) with:
  - Generalized Advantage Estimation (GAE) with configurable λ
  - n-step returns across `num_envs` parallel environments
  - Separate actor and critic networks with orthogonal initialization
  - Optional shared backbone (`share_network`)
  - Separate optimizers with independent learning rates
  - Entropy regularization with linear annealing schedule
  - TD(λ) critic targets (`returns = advantages + V(s)`)
  - Gradient clipping per head
- Structured ablation studies (4 sequential) to diagnose why A2C underperforms REINFORCE with the standard λ=0.95
- **K critic epochs per rollout** — restructured `learn_from_batch` into three phases to allow the critic to converge before the actor update

---

## Algorithm Theory

### REINFORCE

REINFORCE is the simplest policy gradient algorithm. At the end of each episode, it computes the Monte Carlo return for every timestep:

$$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

and updates the policy by gradient ascent on the log-probability of each action, weighted by its return:

$$\nabla_\theta J \approx \sum_t G_t \nabla_\theta \log \pi_\theta(a_t | s_t)$$

**Pros:** No critic to train, no bias, mathematically clean.  
**Cons:** Very high variance — $G_t$ integrates all future randomness for the entire episode.

In practice, a baseline (e.g., the mean return) is subtracted to reduce variance without changing the expected gradient. Our implementation normalizes returns to unit variance per episode.

---

### Advantage Actor-Critic (A2C)

A2C replaces the raw Monte Carlo return with an *advantage estimate* $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$: how much better was this action than average? The actor gradient becomes:

$$\nabla_\theta J \approx \sum_t A(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t | s_t)$$

The critic learns $V(s)$, providing a lower-variance baseline. The architecture uses:
- **Actor network**: outputs a Categorical (discrete) or Normal (continuous) distribution over actions.
- **Critic network**: outputs a scalar $V(s)$.
- **Two independent optimizers** (Adam), each with its own learning rate.

#### Generalized Advantage Estimation (GAE)

Rather than a single-step TD error or a full MC return, GAE uses an exponentially-weighted sum of multi-step TD errors:

$$A_t^{GAE} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

The λ parameter continuously trades off bias vs. variance:
- **λ=0**: pure one-step TD (high bias, low variance — only as good as $V$)
- **λ=1**: full n-step Monte Carlo (no bias from bootstrapping, high variance from reward stochasticity)
- **λ∈(0,1)**: smooth interpolation controlled by the *effective horizon* — the number of future steps that contribute meaningfully:

$$\text{Effective horizon} \approx \frac{1}{1 - \lambda\gamma}$$

#### TD(λ) Critic Targets

The critic is trained on returns computed as:

$$G_t^{TD(\lambda)} = A_t^{GAE} + V(s_t)$$

These are the same targets used to compute advantages. The critic minimizes Smooth-L1 loss against them.

#### Orthogonal Initialization

All hidden layers use orthogonal initialization with gain $\sqrt{2}$ (appropriate for ReLU). Output layers use smaller gains: 0.01 for the actor output (near-uniform initial policy) and 1.0 for the critic output (reasonable value scale). This prevents vanishing/exploding gradients and ensures stable early training.

#### Vectorized Environments

A2C collects rollouts from `num_envs=8` parallel environments simultaneously. Each environment produces an independent trajectory, so the `n_steps × num_envs` transitions in a single batch are largely uncorrelated. This reduces variance in gradient estimates and dramatically speeds up wall-clock convergence.

---

## The Ablation Journey: Why Does λ=0.95 Fail?

The textbook recommendation for GAE is λ=0.95. All 4 ablation studies below investigate why this standard value fails in this environment while λ=1.0 succeeds.

---

### Study 1 — Critic Weight and Entropy Sweep

**Question:** Do the default critic loss weight and entropy schedule cause the plateau at ~75 reward?

**Configuration:** Varied `critic_loss_weight` (0.001 vs 0.005) and `entropy_weight_start` (0.05 vs 0.1). All runs used λ=0.95, 8000 episodes.

**Findings:**
- **Root cause 1 — Critic starved of gradient.** `critic_loss_weight=0.001–0.005` gives the critic ~1000–3000× less gradient than the actor. The actor changes rapidly while the critic cannot track it, producing biased advantages.
- **Root cause 2 — Entropy floor too low.** `entropy_weight_end=0.0003` (not overridden in the ablation) collapsed policy entropy by episode 6000. All configs converged to nearly the same entropy level regardless of their starting values.
- **Fix applied:** Restored canonical A2C default `critic_loss_weight: 0.5`; added per-experiment `entropy_weight_end` overrides; raised `critic_lr` from 0.0003 → 0.001.

---

### Study 2 — REINFORCE vs A2C

**Question:** Can A2C beat REINFORCE at all? Which λ values succeed?

**Configuration:** REINFORCE (no critic, full MC), A2C with λ∈{0.95, 0.97, 0.99, 1.0}. Canonical critic settings restored.

**Findings:**
- λ=1.0 and λ=0.99 both reach 200+ (problem solved). λ=0.97 plateaus ~115. λ=0.95 plateaus ~75 — barely better than chance.
- The monotonic degradation as λ → 0.95 confirmed the effective-horizon hypothesis.

---

### Study 3 — Lambda Sweep

**Question:** Where exactly does A2C break down? What is the mechanism?

**Configuration:** λ∈{0.95, 0.97, 0.99, 1.0}, all with canonical settings, 10000 episodes.

**Key findings:**

#### The Effective Horizon Problem

Lunar Lander's terminal landing bonus (+100 to +200) arrives ~200–400 steps into a trajectory. With λ=0.95, an action that eventually leads to landing receives advantage weight $(0.9405)^{150} \approx 1.5\times10^{-4}$ of the landing bonus — effectively zero credit. The actor is learning a purely myopic signal ("survive the next 17 steps") and converges to hovering.

| λ | Effective horizon (γ=0.99) | Performance |
|---|---|---|
| 0.95 | ~17 steps | Plateaus ~75 |
| 0.97 | ~25 steps | Plateaus ~115 |
| 0.99 | ~50 steps | Reaches 200 |
| 1.0 | 256 steps (full rollout) | Reaches 200+ |

#### The High-EV Trap

Counterintuitively, λ=0.97 showed the *highest* Explained Variance (EV ≈ 0.85) despite third-best performance.

EV measures how well the critic predicts its own training targets, **not** true $V(s)$. With λ=0.97, the training target is a 25-step smoothed return — low variance, easy to predict, high EV. But that well-predicted target simply does not encode events >25 steps away. The critic is "succeeding" at a task that does not matter for the actor. **High EV with low performance is a red flag, not a good sign.**

#### EV Collapse as a Convergence Artifact

For λ=1.0, EV sharply drops from ~0.6 to deeply negative around episode 9000. This is NOT a training failure. The EV formula is:

$$EV = 1 - \frac{\text{Var}(G - V)}{\text{Var}(G) + \varepsilon}$$

Once the policy reliably lands (scores ~200–260), return variance collapses: $\text{Var}(G) \to \varepsilon$. Even tiny prediction errors make $\text{Var}(G - V)/\varepsilon$ very large, driving EV deeply negative. The performance chart shows scores staying above 200 with *narrowing* confidence bands at the exact same time — the policy is not getting worse; the metric is ill-conditioned.

---

### Study 4 — Gamma Sweep

**Question:** Does increasing γ toward 1 rescue λ=0.95 by enriching each TD error with more far-future signal?

**Mechanism:** Each TD error carries the landing bonus in proportion to $\gamma^{\text{steps to landing}}$:

| γ | Landing bonus at step 200 | V half-life |
|---|---|---|
| 0.99 | $0.99^{200} \approx$ **13%** | ~69 steps |
| 0.995 | $0.995^{200} \approx$ **37%** | ~138 steps |
| 0.999 | $0.999^{200} \approx$ **82%** | ~693 steps |

**Configuration:** λ=0.95 with γ∈{0.99, 0.995, 0.999}, plus λ=1.0/γ=0.99 as reference. 6000 episodes.

**Findings:** λ=1.0 still won. Higher γ gave modest improvement to λ=0.95 but did not close the gap.

**Why γ alone cannot fix it — the noise accumulation ratio:**

With λ=0.95, GAE is a weighted sum of ~17 TD errors. Each TD error contains noise $\varepsilon_t = V(s_t) - V^*(s_t)$ (critic estimation error). The cumulative noise variance is:

$$\sigma^2_{\text{noise}} \propto \sum_{k=0}^{\infty} (\gamma\lambda)^{2k} = \frac{1}{1 - (\gamma\lambda)^2} \approx 8.7$$

For λ=1.0, telescoping cancellation eliminates all intermediate $V(s_t)$ terms — only $V(s_0)$ and $V(s_N)$ survive. The structural noise multiplier is **1** regardless of critic quality. The 8.7× noise penalty for λ=0.95 persists no matter how informative each individual TD error becomes from higher γ.

---

### Study 5 — K Critic Epochs Per Rollout (Current)

**Question:** If the critic is trained K times per rollout before the actor updates, does λ=0.95 finally work?

**Hypothesis:** With K=1, the critic never converges against the frozen TD-lambda targets before the actor consumes its noisy estimates. K>1 gives the critic time to fit before the actor updates.

**On-policy validity:** The on-policy constraint applies to $\log \pi(a|s)$ in the *actor* loss — actions must be sampled from the current policy. The critic loss is pure supervised regression against frozen $(s, G)$ pairs with no $\pi$ dependence. Multiple critic passes on fixed targets do not violate the on-policy assumption. The actor still updates exactly once per rollout.

#### Implementation: Three-Phase `learn_from_batch`

```
Phase 1 (no_grad): Compute frozen TD-lambda returns from initial critic.
                   These targets are fixed for all K critic epochs.

Phase 2 (K times): Critic forward → Smooth-L1 loss → backward → critic_optimizer.step().
                   Actor parameters receive no gradient; actor_optimizer is never stepped.

Phase 3 (once):   Fresh forward pass for actor distributions.
                   Recompute advantages using the now-improved critic (detached values).
                   Actor loss = -(log_probs × advantages) − entropy_weight × H(π).
                   actor_optimizer.step() only.
```

**Configuration:** λ=0.95, K∈{1, 4, 8} and λ=1.0/K=1 as reference. 6000 episodes.

**Expected result:**
- If K=8 lets λ=0.95 match λ=1.0 → the textbook λ=0.95 recommendation was correct; the failure was an implementation artifact (insufficient critic training per batch).
- If K=8 still fails → the effective-horizon gap (17 vs 256 steps) is the irreducible bottleneck; no amount of critic training can put the landing bonus back into a 17-step advantage window.

The EV chart is the key diagnostic: EV improvement from K=1 → K=8 measures how much critic quality improves; performance improvement (or lack of it) then shows whether critic quality was actually the binding constraint.

---

## Future Ideas

- **PPO:** Proximal Policy Optimization solves the same K-epoch problem for the *actor* by clipping the importance ratio $\text{clip}(\pi_\theta / \pi_{\theta_\text{old}}, 1\pm\varepsilon)$, allowing multiple actor update epochs per rollout without divergence. The natural next step after this version.
- **Adaptive K:** Monitor per-epoch critic loss to stop early when EV stops improving, rather than running a fixed K.
- **Separate critic LR schedule:** The critic's task difficulty changes with the policy (non-stationary target). A curriculum-style LR warmup for the critic at the start of training could reduce early noise.
- **Compare K-epoch A2C vs PPO:** With K critic epochs + one actor epoch, our A2C is structurally close to PPO minus the clip. A direct comparison would quantify how much the clip matters.
