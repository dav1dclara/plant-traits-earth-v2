# Model Comparison Guidance

This note summarizes what the current results suggest and how we should move forward.

## What the current results say

We now have a useful first comparison across five runs: three model families, two loss variants, and Daniel's benchmark numbers.

The main pattern is:

- ResPatch is the most reliable baseline.
- MTL and MMoE are not clearly better yet.
- The uncertainty-weighted loss is not giving a consistent advantage in this untuned setup.
- Some traits are already modeled well by all three models.
- A smaller group of traits remains consistently difficult, especially some wood-, conduit-, and structure-related traits.

So the issue is probably not a broken training pipeline. The models are learning meaningful signal. The issue is more likely that the current multitask setup is too generic for the task mix we are asking it to solve.

## What we should not conclude yet

We should not immediately conclude that MTL or MMoE are useless.

The current result only says they are not yet extracting enough task-specific benefit to beat ResPatch consistently. That can happen for several reasons:

- the tasks are too heterogeneous,
- the losses are not balanced well enough,
- the gate structure is too weak,
- the training schedule is not tuned,
- or the shared encoder is already strong enough that the heads do not add much.

This is a tuning and design problem, not necessarily a conceptual failure.

## Recommended next steps

### 1. Use ResPatch as the reference baseline

Keep ResPatch as the baseline model for now. It is the simplest and currently the most stable point of comparison.

Any new MTL or MMoE change should be judged against:

- macro Pearson's r,
- macro R2,
- macro RMSE,
- and the per-trait table.

Use the global metrics as supporting information, but not as the only decision criterion.

### 2. Compare traits in groups, not only overall

We should not treat all traits as one homogeneous prediction problem.

A better approach is to split traits into rough groups:

- leaf traits,
- root traits,
- wood / conduit traits,
- seed / dispersal traits,
- structural traits such as height or diameter.

Then inspect whether a model is consistently better or worse within a group. This is where MTL or MMoE might start to show value.

### 3. Separate "easy" and "hard" traits

Some traits are already above Daniel's benchmark or very close to it. Others are clearly harder.

We should label traits into:

- well modeled,
- borderline,
- persistently hard.

The hard group is especially interesting, because it may be the group where a specialized architecture or a different loss design can still help.

### 4. Tune the loss before changing the architecture again

The current evidence suggests that the loss setup matters more than simply adding more model complexity.

We should try:

- fixed weights versus uncertainty weighting,
- different trait-level weighting schemes,
- possibly per-group weights,
- and maybe stronger down-weighting of noisy or unstable targets.

If the loss is not aligned with the actual task difficulty, MTL and MMoE will not get a fair chance.

### 5. Give MMoE a better chance to specialize

The current gate behavior looks too close to uniform. That usually means the experts are not specializing enough.

For MMoE, we should explore:

- fewer or more structured experts,
- stronger gate regularization,
- trait-group-specific experts,
- or training tasks in a way that encourages separation.

If the gates stay nearly uniform, MMoE behaves too much like a slightly more complex shared-head model.

### 6. Run multiple seeds

The differences between the models are not huge.

That means a single run is not enough to make a strong decision. We should run at least a few seeds per configuration before claiming a winner.

## How to approach the next experiments

I would recommend the following sequence:

1. Keep the current best ResPatch run as the anchor.
2. Re-run the best MTL and MMoE configurations with the fixed loss.
3. Compare per-trait results against Daniel, not just the averages.
4. Group traits and inspect whether one model family helps specific groups.
5. If MMoE still does not specialize, simplify or rework the gate design.
6. Only then decide whether to keep MTL/MMoE or stick with ResPatch.

## My current conclusion

The most defensible conclusion right now is:

- the pipeline is basically working,
- the data contains learnable signal,
- ResPatch is the strongest stable baseline,
- MTL and MMoE need more targeted design choices to become useful,
- and the next improvement will probably come from better task grouping, better loss balancing, and more controlled specialization rather than from simply making the model larger.

In short: do not abandon MTL/MMoE yet, but do not assume they will improve performance automatically. They need a more task-aware setup.
