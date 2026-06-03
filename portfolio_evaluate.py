import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import cv2
import loco_mujoco
import mujoco
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


STATE_DIM = 36
ACTION_DIM = 13


class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, state):
        return self.network(state)


class PPOPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )
        self.log_std = nn.Parameter(torch.ones(ACTION_DIM) * 0.5)

    def forward(self, state, episode=0):
        mean = torch.tanh(self.network(state))
        log_std = torch.clamp(self.log_std, -2.3, 5)
        std = torch.exp(torch.clamp(log_std - 0.0001 * episode, -20, 5))
        return mean, std

    def deterministic_action(self, state):
        mean, _ = self.forward(state)
        return mean


class SACPolicy(nn.Module):
    def __init__(self, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(128, ACTION_DIM)
        self.log_std_layer = nn.Linear(128, ACTION_DIM)

    def forward(self, state, deterministic=False):
        x = self.network(state)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        std = torch.exp(log_std)
        if deterministic:
            return mean * self.max_action
        dist = Normal(mean, std)
        normal_action = dist.rsample()
        action = torch.tanh(normal_action)
        return action * self.max_action


@dataclass
class EvalResult:
    model_name: str
    weight_file: str
    episodes: int
    max_steps: int
    avg_steps: float
    best_steps: int
    avg_reward: float
    best_reward: float
    success_rate: float
    video_path: str


def load_policy(model_type: str, weight_file: str, device: torch.device, max_action: float):
    if model_type == "bc":
        policy = BCPolicy()
        action_fn = lambda p, s: p(s)
    elif model_type == "ppo":
        policy = PPOPolicy()
        action_fn = lambda p, s: p.deterministic_action(s)
    elif model_type == "sac":
        policy = SACPolicy(max_action=max_action)
        action_fn = lambda p, s: p(s, deterministic=True)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state = torch.load(weight_file, map_location=device, weights_only=True)
    policy.load_state_dict(state, strict=True)
    policy.to(device)
    policy.eval()
    return policy, action_fn


def get_max_action(env) -> float:
    try:
        ctrlrange = env._model.actuator_ctrlrange
        return float(np.max(np.abs(ctrlrange)))
    except Exception:
        return 1.0


def evaluate_model(
    model_name: str,
    model_type: str,
    weight_file: str,
    episodes: int,
    max_steps: int,
    output_dir: str,
    record_video: bool,
    device: torch.device,
) -> EvalResult:
    env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
    max_action = get_max_action(env)
    policy, action_fn = load_policy(model_type, weight_file, device, max_action)

    video_path = ""
    renderer = None
    writer = None
    if record_video:
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f"{model_name}_demo.mp4")
        renderer = mujoco.Renderer(env._model, width=640, height=480)
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
        if not writer.isOpened():
            writer = None
            video_path = ""

    rewards: List[float] = []
    steps_list: List[int] = []

    with torch.no_grad():
        for episode in range(episodes):
            obs = env.reset()
            episode_reward = 0.0
            steps = 0

            for _ in range(max_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = action_fn(policy, obs_t).squeeze(0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0)
                obs, reward, done, _ = env.step(action)
                episode_reward += float(reward)
                steps += 1

                if writer is not None:
                    mujoco.mj_forward(env._model, env._data)
                    renderer.update_scene(env._data, camera="track")
                    frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                    writer.write(frame)

                if done:
                    break

            rewards.append(episode_reward)
            steps_list.append(steps)
            print(
                f"{model_name}: episode {episode + 1}/{episodes}, "
                f"steps={steps}, reward={episode_reward:.2f}"
            )

    if writer is not None:
        writer.release()

    return EvalResult(
        model_name=model_name,
        weight_file=weight_file,
        episodes=episodes,
        max_steps=max_steps,
        avg_steps=float(np.mean(steps_list)),
        best_steps=int(np.max(steps_list)),
        avg_reward=float(np.mean(rewards)),
        best_reward=float(np.max(rewards)),
        success_rate=float(np.mean([s >= max_steps for s in steps_list])),
        video_path=video_path,
    )


def write_results(results: List[EvalResult], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    csv_path = os.path.join(output_dir, "evaluation_metrics.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-dir", default="portfolio_results")
    parser.add_argument("--demo-model", default="bc_best")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    candidates = [
        ("bc_last", "bc", "bc_model.pth"),
        ("bc_best", "bc", "bc_best_model.pth"),
        ("ppo_best_steps", "ppo", "best_ppo_model.pth"),
        ("ppo_best_reward", "ppo", "best_reward_model.pth"),
        ("ppo_longest", "ppo", "longest_steps_model.pth_11"),
        ("sac", "sac", "sac_model.pth"),
    ]

    results = []
    for model_name, model_type, weight_file in candidates:
        if not os.path.exists(weight_file):
            print(f"Skip missing weight file: {weight_file}")
            continue
        results.append(
            evaluate_model(
                model_name=model_name,
                model_type=model_type,
                weight_file=weight_file,
                episodes=args.episodes,
                max_steps=args.max_steps,
                output_dir=args.output_dir,
                record_video=(model_name == args.demo_model),
                device=device,
            )
        )

    write_results(results, args.output_dir)


if __name__ == "__main__":
    main()
