# MMoE Gates in W&B

This note explains the gate metrics logged in [scripts/train.py](scripts/train.py#L478).

## What is logged?

During validation, the following values are written:

- `val/gate_entropy`
- `val/gate_entropy_norm`
- `val/gate_max_prob`
- `val/gate_effective_experts`
- `val/gate_usage/e0 ... e{n_experts-1}`

The source is `last_gate_weights` from the MMoE model.

## Metric meanings

### 1) `val/gate_usage/e*`

Mean gate probability per expert, averaged across batch and traits.

- The sum over all experts is approximately 1.
- With 6 experts, a perfectly uniform level is about $1/6 \approx 0.1667$.

Interpretation:

- All `e*` close to 0.1667: little specialization, routing is almost uniform.
- Some `e*` clearly higher/lower: experts are used unevenly.

### 2) `val/gate_entropy`

Shannon entropy of the gate distribution per sample/trait, then averaged.

$$
H(g) = -\sum_i g_i \log(g_i)
$$

Interpretation:

- High: broad distribution, closer to uniform.
- Low: sharper distribution, more decisive expert selection.

### 3) `val/gate_entropy_norm`

Normalized entropy:

$$
H_{norm} = \frac{H(g)}{\log(n_{experts})}
$$

Interpretation:

- Close to 1.0: nearly uniform.
- Clearly below 1.0: more specialization.

### 4) `val/gate_max_prob`

Largest gate probability per sample/trait, then averaged.

$$
\max_i g_i
$$

Interpretation:

- Higher: the gate often prefers one dominant expert.
- Lower: the distribution is broader.

### 5) `val/gate_effective_experts`

Effective number of experts being used:

$$
N_{eff} = e^{H(g)}
$$

Interpretation:

- With 6 experts, the maximum is close to 6.
- Values near 6 mean most experts contribute similarly per sample.
- Smaller values mean fewer experts dominate.

## Quick reading rules for W&B

If you want to see specialization, typically:

- `gate_entropy_norm` should go down,
- `gate_max_prob` should go up,
- `gate_effective_experts` should go down,
- and `gate_usage/e*` should spread out more.

If everything stays close to uniform, MMoE often behaves more like a shared MTL backbone without strong expert routing.

## How to read your plots

Your curves show, qualitatively:

- `gate_entropy_norm` stays very high, close to 1,
- `gate_effective_experts` stays high, close to the number of experts,
- `gate_usage/e*` stay close to each other,
- `gate_max_prob` is only moderate.

That suggests weak gate specialization and fairly even expert usage.

## Important

This is not automatically a problem.

- If `val/loss` is good, this soft mixing can still be useful.
- If MMoE is barely better than MTL, this gate pattern is a plausible explanation.
