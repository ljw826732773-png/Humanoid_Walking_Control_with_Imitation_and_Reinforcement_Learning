# Model Card: Enhanced BC Humanoid Walking Controller

## Model Summary

The main portfolio model is an enhanced behavior cloning policy for the Loco-MuJoCo `HumanoidTorque.run` task. It maps a 36-dimensional humanoid state vector to a 13-dimensional continuous torque/action vector.

The model is trained from the Loco-MuJoCo perfect expert dataset and deployed in a closed-loop MuJoCo simulation.

## Intended Use

- Demonstrate an imitation-learning based humanoid walking control pipeline.
- Support portfolio, resume, interview, and course-project discussion.
- Provide a reproducible baseline for later PPO/SAC fine-tuning or online imitation-learning research.

## Not Intended For

- Real robot deployment.
- Safety-critical control.
- Claims of fully solved long-horizon humanoid locomotion.
- Generalization to unseen terrain, morphology, sensors, or dynamics without additional validation.

## Architecture

| Item | Value |
| --- | --- |
| Input dimension | 36 |
| Output dimension | 13 |
| Network | MLP: 36 -> 256 -> 128 -> 13 |
| Activations | ReLU |
| Policy type | Deterministic behavior cloning |
| Normalization | State normalization and action normalization |
| Checkpoint selection | Closed-loop MuJoCo rollout average steps |

## Main Results

| Metric | Value |
| --- | ---: |
| Original BC retrain average steps | 97.5 |
| Enhanced BC average steps | 839.2 |
| Enhanced BC best steps | 1000 |
| Stability diagnostic average steps | 820.3 |
| Raw-action robustness sweep average steps | 930.2 |
| Final demo video duration | 279.7 s |

## Evaluation Protocol

The main result is evaluated with:

- 10 MuJoCo closed-loop episodes.
- Maximum 1000 steps per episode.
- Deterministic policy output.
- Action clipping to `[-1, 1]`.
- Additional stability diagnostics for torso height, action norm, and action delta.
- Robustness sweep over several seeds and action smoothing values.

## Limitations

- The policy is still imitation-learning based and can fail when it enters states far from the expert dataset distribution.
- The result is sensitive to checkpoint selection; validation MSE alone is not reliable for closed-loop locomotion quality.
- Simple action smoothing at deployment time hurts performance in the current controller.
- PPO/SAC fine-tuning and true online DAgger are not claimed as completed main results.

## Reproducibility

Recommended checks:

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_cli.py validate
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_cli.py showcase
```

Additional commands and result boundaries are documented in `EXPERIMENTS.md`.
