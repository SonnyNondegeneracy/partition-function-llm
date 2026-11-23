# Source data of the revised figures

Written by the scripts in `plotting/` (folder `figureN` = paper Fig. N, script `figureN.py`). Counts come from
`measurements/counts/main` (main experiment) and `measurements/counts/revision` (revision runs) of the data archive.

| File | Content |
|---|---|
| `pooled_c.json` | Calibration factor per task, fitted jointly on the three training seeds (T1 3.036, T2 3.451) |
| `recalibrated/` | Fig. 2 and Fig. 3 source data of the paper recomputed at the three-seed c (`recalibrate.py`); same schemas as the frozen paper source data, columns tied to the single-run calibration interval dropped |
| `figure2/panel_a_transitions.csv` | Fig. 2a: predicted and measured T_ij - Q_ij, main experiment, all rounds |
| `figure2/panel_b_seeds.csv` | Fig. 2b: r of the two further training seeds (enter the error bars) |
| `figure3/seeds_reward_entropy.csv` | Fig. 3: reward and entropy of the two further seeds (enter the error bars) |
| `figure4/seeds_T1.csv` | Fig. 4: r, R, H of the two further T1 seeds (enter the main-protocol error bars) |
| `figure4/additional_series.json` | Fig. 4: Gemma-4-E4B (merged weights) and reward interpolation alpha = 0.25-4, values and 500-draw bootstrap intervals |
| `figure5/branches.csv` | Calibration factor and per-round change of 1/beta_eff for every training branch, including the training-amount branches reported in Methods |
| `figure5/panel_a_pairs.csv` | Fig. 5a: calibration relation in effective units, main experiment and KL scan |
| `figure5/panel_b_r.csv` | Fig. 5b: r of the KL branches at their own beta_eff and deviation from the main-experiment curve |
| `figure5/panel_c_calibration.csv` | Fig. 5c: c versus KL coefficient; main-experiment seed half-width |

Error bars of the main experiment: half-width 1.96 * sqrt(s_boot^2 + s_seed^2), s_boot = 95% bootstrap width / 3.92,
s_seed = standard deviation over the three training seeds.
