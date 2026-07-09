# Visual Gallery

This gallery collects the visual evidence used to explain the humanoid walking control project.

## Overview Dashboard

![Overview dashboard](portfolio_retrain_bc_improved/report_overview_dashboard.svg)

## Long-horizon walking keyframes

![Long-horizon walking keyframes](portfolio_retrain_bc_improved/keyframes/bc_improved_keyframes.png)

Eight frames sampled from the final demo video, covering 14.0s to 265.7s.

## Closed-loop performance comparison

![Closed-loop performance comparison](portfolio_retrain_bc_improved/report_steps_comparison.svg)

Average episode length improves from the original BC retrain to the rollout-selected enhanced BC.

## Enhanced BC training loss

![Enhanced BC training loss](portfolio_retrain_bc_improved/report_training_loss.svg)

Training and validation MSE curves for the normalized behavior cloning policy.

## Rollout-based checkpoint selection

![Rollout-based checkpoint selection](portfolio_retrain_bc_improved/report_rollout_selection.svg)

MuJoCo closed-loop rollout is used to select the checkpoint, instead of relying only on validation MSE.

## Episode stability diagnostics

![Episode stability diagnostics](portfolio_retrain_bc_improved/report_stability_diagnostics.svg)

Episode steps and action-delta diagnostics help identify whether the controller is stable or only lucky.

## Robustness and action smoothing sweep

![Robustness and action smoothing sweep](portfolio_retrain_bc_improved/report_robustness_sweep.svg)

Naive deployment-time action smoothing reduces performance for this learned gait.
