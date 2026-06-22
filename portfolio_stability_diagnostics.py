import argparse
import csv
import os

import loco_mujoco
import numpy as np
import torch
import torch.nn as nn


STATE_DIM = 36
ACTION_DIM = 13


class BCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, x):
        return self.network(x)


def load_normalizer(path):
    data = np.load(path)
    return data["state_mean"], data["state_std"], data["action_mean"], data["action_std"]


def act(model, obs, state_mean_t, state_std_t, action_mean, action_std, device):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    norm_obs = (obs_t - state_mean_t) / state_std_t
    norm_action = model(norm_obs).squeeze(0).detach().cpu().numpy()
    return np.clip(norm_action * action_std + action_mean, -1.0, 1.0)


def run_diagnostics(model_path, normalizer_path, output_dir, episodes, max_steps, device):
    os.makedirs(output_dir, exist_ok=True)
    env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
    state_mean, state_std, action_mean, action_std = load_normalizer(normalizer_path)
    state_mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)

    model = BCModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    rows = []
    with torch.no_grad():
        for episode in range(1, episodes + 1):
            obs = env.reset()
            total_reward = 0.0
            steps = 0
            heights = []
            action_norms = []
            action_deltas = []
            prev_action = None

            for _ in range(max_steps):
                action = act(model, obs, state_mean_t, state_std_t, action_mean, action_std, device)
                obs, reward, done, _ = env.step(action)
                total_reward += float(reward)
                steps += 1

                try:
                    heights.append(float(env._data.qpos[2]))
                except Exception:
                    heights.append(float(obs[0]))

                action_norms.append(float(np.linalg.norm(action)))
                if prev_action is not None:
                    action_deltas.append(float(np.linalg.norm(action - prev_action)))
                prev_action = action.copy()

                if done:
                    break

            rows.append(
                {
                    "episode": episode,
                    "steps": steps,
                    "reward": total_reward,
                    "avg_height": float(np.mean(heights)),
                    "min_height": float(np.min(heights)),
                    "avg_action_norm": float(np.mean(action_norms)),
                    "avg_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
                }
            )
            print(f"episode {episode}/{episodes}, steps={steps}, reward={total_reward:.2f}")

    out_path = os.path.join(output_dir, "stability_diagnostics.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(output_dir, "stability_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episodes",
                "avg_steps",
                "best_steps",
                "avg_reward",
                "avg_height",
                "min_height",
                "avg_action_norm",
                "avg_action_delta",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "episodes": episodes,
                "avg_steps": float(np.mean([r["steps"] for r in rows])),
                "best_steps": int(np.max([r["steps"] for r in rows])),
                "avg_reward": float(np.mean([r["reward"] for r in rows])),
                "avg_height": float(np.mean([r["avg_height"] for r in rows])),
                "min_height": float(np.min([r["min_height"] for r in rows])),
                "avg_action_norm": float(np.mean([r["avg_action_norm"] for r in rows])),
                "avg_action_delta": float(np.mean([r["avg_action_delta"] for r in rows])),
            }
        )

    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="portfolio_retrain_bc_improved/bc_improved_best.pth")
    parser.add_argument("--normalizer", default="portfolio_retrain_bc_improved/bc_improved_normalizer.npz")
    parser.add_argument("--output-dir", default="portfolio_retrain_bc_improved")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_diagnostics(args.model, args.normalizer, args.output_dir, args.episodes, args.max_steps, device)


if __name__ == "__main__":
    main()
