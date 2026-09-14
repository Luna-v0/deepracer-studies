# TRPO vs PPO on the feature-vector policy

Completion rate of the final evaluation, 5 seeds per cell, 64 parallel envs,
300k environment steps unless stated. `p` is an exact permutation test on the
difference of means (one-sided, 252 partitions).

| track | steps | TRPO | PPO | p |
|---|---|---|---|---|
| reinvent_base | 300k | **0.781** ± 0.043 | 0.486 ± 0.195 | 0.004 |
| Oval_track | 300k | **0.888** ± 0.072 | 0.711 ± 0.168 | 0.056 |
| 2022_reinvent_champ | 300k | **0.412** ± 0.308 | 0.223 ± 0.095 | 0.175 |
| 2022_reinvent_champ | 1M | 0.436 ± 0.223 | not run | — |

Per-seed values:

```
reinvent_base  300k  TRPO  0.759 0.842 0.816 0.766 0.720
                     PPO   0.646 0.646 0.154 0.611 0.372
Oval_track     300k  TRPO  0.825 0.937 0.970 0.783 0.926
                     PPO   0.689 0.972 0.494 0.813 0.588
champ          300k  TRPO  0.179 0.226 0.093 0.703 0.859
                     PPO   0.222 0.266 0.107 0.143 0.376
champ          1M    TRPO  0.133 0.654 0.395 0.722 0.277
```

## What holds

TRPO leads on the mean of all three tracks, and on mean return as well. On
`reinvent_base` the separation is clean: its worst seed (0.720) beats PPO's best
(0.646), with no overlap.

## What does not

**The stability advantage is track-dependent.** On `reinvent_base` TRPO is four
times tighter than PPO (sd 0.043 vs 0.195), which is the behavior the trust
region is meant to buy. On `2022_reinvent_champ` it inverts: TRPO spreads from
0.09 to 0.86 (sd 0.308) while PPO stays uniformly low (sd 0.095). TRPO wins
there on the strength of two good seeds, not on consistency.

**Only one track is individually significant.** Oval is borderline, champ is
not. The consistent direction across three tracks is the stronger evidence.

**Both algorithms largely fail on `2022_reinvent_champ`** (0.41 and 0.22). That
track is 33.3 m long with a 0.46 m minimum turn radius, against 17.7 m and
0.57 m for `reinvent_base`. Raising TRPO's budget to 1M moved the mean from
0.412 to 0.436 and did not resolve the spread — seed 0 stays at 0.13 with three
times the training, so the failure is not only a budget shortfall.

**PPO is untuned.** It runs rsl-rl defaults (`lr=3e-4`, adaptive KL schedule)
while TRPO sizes its own step by construction. A learning-rate sweep for PPO is
the first objection this comparison has to answer.

## Open

- PPO at 1M on `2022_reinvent_champ`, to complete that row.
- `New_York_Track` and `Tokyo_Training_track` at 1M, for two track profiles not
  yet covered (narrow; wide with a tight corner).
- Five more seeds on `Oval_track` to settle its p-value.
