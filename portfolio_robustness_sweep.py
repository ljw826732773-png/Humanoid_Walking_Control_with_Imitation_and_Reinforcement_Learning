import argparse
import csv
import os
from collections import defaultdict

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


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def reset_env(env, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def predict_action(model, obs, state_mean_t, state_std_t, action_mean, action_std, device):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    norm_obs = (obs_t - state_mean_t) / state_std_t
    norm_action = model(norm_obs).squeeze(0).detach().cpu().numpy()
    action = norm_action * action_std + action_mean
    return np.clip(action, -1.0, 1.0)


def evaluate_episode(
    env,
    model,
    state_mean_t,
    state_std_t,
    action_mean,
    action_std,
    device,
    max_steps,
    smoothing,
    seed,
):
    obs = reset_env(env, seed)
    total_reward = 0.0
    steps = 0
    prev_action = None
    action_norms = []
    action_deltas = []
    heights = []

    with torch.no_grad():
        for _ in range(max_steps):
            raw_action = predict_action(model, obs, state_mean_t, state_std_t, action_mean, action_std, device)
            if prev_action is None or smoothing <= 0:
                action = raw_action
            else:
                action = (1.0 - smoothing) * raw_action + smoothing * prev_action
                action = np.clip(action, -1.0, 1.0)

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

    return {
        "steps": steps,
        "reward": total_reward,
        "success": int(steps >= max_steps),
        "avg_height": float(np.mean(heights)),
        "min_height": float(np.min(heights)),
        "avg_action_norm": float(np.mean(action_norms)),
        "avg_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    state_mean, state_std, action_mean, action_std = load_normalizer(args.normalizer)
    state_mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)

    model = BCModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
    smoothing_values = parse_float_list(args.smoothing_values)

    rows = []
    for smoothing in smoothing_values:
        for seed_index in range(args.num_seeds):
            seed_base = args.base_seed + seed_index * 1000
            for episode in range(1, args.episodes_per_seed + 1):
                episode_seed = seed_base + episode
                result = evaluate_episode(
                    env,
                    model,
                    state_mean_t,
                    state_std_t,
                    action_mean,
                    action_std,
                    device,
                    args.max_steps,
                    smoothing,
                    episode_seed,
                )
                row = {
                    "smoothing": smoothing,
                    "seed": seed_base,
                    "episode": episode,
                    "episode_seed": episode_seed,
                    **result,
                }
                rows.append(row)
                print(
                    f"smoothing={smoothing:.2f}, seed={seed_base}, episode={episode}, "
                    f"steps={result['steps']}, reward={result['reward']:.2f}"
                )

    detail_path = os.path.join(args.output_dir, "robustness_sweep.csv")
    write_csv(detail_path, rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["smoothing"]].append(row)

    summary_rows = []
    for smoothing, group in sorted(grouped.items()):
        summary_rows.append(
            {
                "smoothing": smoothing,
                "episodes": len(group),
                "avg_steps": float(np.mean([r["steps"] for r in group])),
                "std_steps": float(np.std([r["steps"] for r in group])),
                "best_steps": int(np.max([r["steps"] for r in group])),
                "success_rate": float(np.mean([r["success"] for r in group])),
                "avg_reward": float(np.mean([r["reward"] for r in group])),
                "avg_height": float(np.mean([r["avg_height"] for r in group])),
                "min_height": float(np.min([r["min_height"] for r in group])),
                "avg_action_norm": float(np.mean([r["avg_action_norm"] for r in group])),
                "avg_action_delta": float(np.mean([r["avg_action_delta"] for r in group])),
            }
        )

    summary_path = os.path.join(args.output_dir, "robustness_summary.csv")
    write_csv(summary_path, summary_rows)
    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="portfolio_retrain_bc_improved/bc_improved_best.pth")
    parser.add_argument("--normalizer", default="portfolio_retrain_bc_improved/bc_improved_normalizer.npz")
    parser.add_argument("--output-dir", default="portfolio_retrain_bc_improved")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--episodes-per-seed", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--smoothing-values", default="0.0,0.2,0.4")
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
