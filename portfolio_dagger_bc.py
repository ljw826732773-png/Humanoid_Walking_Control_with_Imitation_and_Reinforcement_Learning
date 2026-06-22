import argparse
import csv
import os
import time

import cv2
import loco_mujoco
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


STATE_DIM = 36
ACTION_DIM = 13
EPS = 1e-6


class BCDataset(Dataset):
    def __init__(self, states, actions, state_mean, state_std, action_mean, action_std):
        self.states = torch.tensor((states - state_mean) / state_std, dtype=torch.float32)
        self.actions = torch.tensor((actions - action_mean) / action_std, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index]


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


def save_normalizer(path, state_mean, state_std, action_mean, action_std):
    np.savez(path, state_mean=state_mean, state_std=state_std, action_mean=action_mean, action_std=action_std)


def policy_action(model, obs, state_mean_t, state_std_t, action_mean, action_std, device):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    norm_obs = (obs_t - state_mean_t) / state_std_t
    norm_action = model(norm_obs).squeeze(0).detach().cpu().numpy()
    return np.clip(norm_action * action_std + action_mean, -1.0, 1.0)


def collect_rollout_states(model, normalizer_path, episodes, max_steps, device):
    env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
    state_mean, state_std, action_mean, action_std = load_normalizer(normalizer_path)
    state_mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)

    visited_states = []
    episode_steps = []
    model.eval()
    with torch.no_grad():
        for episode in range(episodes):
            obs = env.reset()
            steps = 0
            for _ in range(max_steps):
                visited_states.append(obs.copy())
                action = policy_action(model, obs, state_mean_t, state_std_t, action_mean, action_std, device)
                obs, _, done, _ = env.step(action)
                steps += 1
                if done:
                    break
            episode_steps.append(steps)
            print(f"collect rollout {episode + 1}/{episodes}, steps={steps}")

    return np.asarray(visited_states), episode_steps


def label_with_nearest_expert(visited_states, expert_states, expert_actions, state_mean, state_std, chunk_size=64):
    norm_expert_states = ((expert_states - state_mean) / state_std).astype(np.float32)
    norm_visited_states = ((visited_states - state_mean) / state_std).astype(np.float32)
    expert_actions = expert_actions.astype(np.float32)
    expert_sq_norm = np.sum(norm_expert_states ** 2, axis=1)

    labels = []
    nearest_distances = []
    for start in range(0, len(norm_visited_states), chunk_size):
        chunk = norm_visited_states[start : start + chunk_size]
        chunk_sq_norm = np.sum(chunk ** 2, axis=1, keepdims=True)
        dist_sq = chunk_sq_norm + expert_sq_norm[None, :] - 2.0 * chunk @ norm_expert_states.T
        dist_sq = np.maximum(dist_sq, 0.0)
        nearest_idx = np.argmin(dist_sq, axis=1)
        labels.append(expert_actions[nearest_idx])
        nearest_distances.append(np.sqrt(dist_sq[np.arange(len(chunk)), nearest_idx]))

    labeled_actions = np.concatenate(labels, axis=0)
    mean_distance = float(np.concatenate(nearest_distances).mean())
    return labeled_actions, mean_distance


def evaluate_model(model, normalizer_path, episodes, max_steps, device):
    env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
    state_mean, state_std, action_mean, action_std = load_normalizer(normalizer_path)
    state_mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)

    steps_list = []
    rewards = []
    model.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs = env.reset()
            total_reward = 0.0
            steps = 0
            for _ in range(max_steps):
                action = policy_action(model, obs, state_mean_t, state_std_t, action_mean, action_std, device)
                obs, reward, done, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
                if done:
                    break
            steps_list.append(steps)
            rewards.append(total_reward)

    return {
        "avg_steps": float(np.mean(steps_list)),
        "best_steps": int(np.max(steps_list)),
        "avg_reward": float(np.mean(rewards)),
        "best_reward": float(np.max(rewards)),
    }


def train_on_aggregated_data(
    model,
    states,
    actions,
    normalizer_path,
    output_dir,
    epochs,
    batch_size,
    device,
    eval_every,
    selection_episodes,
    max_steps,
):
    state_mean, state_std, action_mean, action_std = load_normalizer(normalizer_path)
    train_states, val_states, train_actions, val_actions = train_test_split(
        states, actions, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(
        BCDataset(train_states, train_actions, state_mean, state_std, action_mean, action_std),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        BCDataset(val_states, val_actions, state_mean, state_std, action_mean, action_std),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    best_path = os.path.join(output_dir, "bc_dagger_best.pth")
    last_path = os.path.join(output_dir, "bc_dagger_last.pth")
    training_log = os.path.join(output_dir, "bc_dagger_training_log.csv")

    best_rollout = -1.0
    start = time.time()
    with open(training_log, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "rollout_avg_steps", "rollout_best_steps", "selected", "elapsed_sec"],
        )
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for state_batch, action_batch in train_loader:
                state_batch = state_batch.to(device)
                action_batch = action_batch.to(device)
                pred = model(state_batch)
                loss = criterion(pred, action_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for state_batch, action_batch in val_loader:
                    state_batch = state_batch.to(device)
                    action_batch = action_batch.to(device)
                    val_loss += criterion(model(state_batch), action_batch).item()

            rollout = {"avg_steps": "", "best_steps": ""}
            selected = False
            if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
                rollout = evaluate_model(model, normalizer_path, selection_episodes, max_steps, device)
                selected = rollout["avg_steps"] > best_rollout
                if selected:
                    best_rollout = rollout["avg_steps"]
                    torch.save(model.state_dict(), best_path)
                print(
                    f"epoch {epoch}/{epochs}, train_loss={train_loss / len(train_loader):.6f}, "
                    f"val_loss={val_loss / len(val_loader):.6f}, rollout_avg={rollout['avg_steps']:.1f}, "
                    f"rollout_best={rollout['best_steps']}, selected={selected}"
                )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss / len(train_loader),
                    "val_loss": val_loss / len(val_loader),
                    "rollout_avg_steps": rollout["avg_steps"],
                    "rollout_best_steps": rollout["best_steps"],
                    "selected": selected,
                    "elapsed_sec": time.time() - start,
                }
            )
            f.flush()

    torch.save(model.state_dict(), last_path)
    return best_path, last_path


def record_demo(model_path, normalizer_path, output_dir, episodes, max_steps, device):
    env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
    state_mean, state_std, action_mean, action_std = load_normalizer(normalizer_path)
    state_mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)

    model = BCModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    video_path = os.path.join(output_dir, "bc_dagger_demo.mp4")
    eval_path = os.path.join(output_dir, "bc_dagger_eval.csv")
    renderer = mujoco.Renderer(env._model, width=640, height=480)
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))

    rows = []
    with torch.no_grad():
        for episode in range(1, episodes + 1):
            obs = env.reset()
            total_reward = 0.0
            steps = 0
            for _ in range(max_steps):
                action = policy_action(model, obs, state_mean_t, state_std_t, action_mean, action_std, device)
                obs, reward, done, _ = env.step(action)
                total_reward += float(reward)
                steps += 1

                mujoco.mj_forward(env._model, env._data)
                renderer.update_scene(env._data, camera="track")
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
                if done:
                    break

            rows.append({"episode": episode, "steps": steps, "reward": total_reward})
            print(f"eval episode {episode}/{episodes}, steps={steps}, reward={total_reward:.2f}")

    video_writer.release()
    with open(eval_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "steps", "reward"])
        writer.writeheader()
        writer.writerows(rows)

    steps = np.asarray([r["steps"] for r in rows], dtype=np.float32)
    rewards = np.asarray([r["reward"] for r in rows], dtype=np.float32)
    with open(os.path.join(output_dir, "bc_dagger_summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episodes", "avg_steps", "best_steps", "avg_reward", "best_reward", "video_path"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "episodes": episodes,
                "avg_steps": float(steps.mean()),
                "best_steps": int(steps.max()),
                "avg_reward": float(rewards.mean()),
                "best_reward": float(rewards.max()),
                "video_path": video_path,
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="portfolio_retrain_bc_improved/bc_improved_best.pth")
    parser.add_argument("--normalizer", default="portfolio_retrain_bc_improved/bc_improved_normalizer.npz")
    parser.add_argument("--output-dir", default="portfolio_dagger_bc")
    parser.add_argument("--collect-episodes", type=int, default=8)
    parser.add_argument("--collect-max-steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--selection-episodes", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
    expert = env.create_dataset()
    expert_states = expert["states"]
    expert_actions = expert["actions"]
    state_mean, state_std, action_mean, action_std = load_normalizer(args.normalizer)

    model = BCModel().to(device)
    model.load_state_dict(torch.load(args.base_model, map_location=device, weights_only=True))

    visited_states, rollout_steps = collect_rollout_states(
        model, args.normalizer, args.collect_episodes, args.collect_max_steps, device
    )
    labeled_actions, mean_nn_distance = label_with_nearest_expert(
        visited_states, expert_states, expert_actions, state_mean, state_std
    )

    aggregate_states = np.concatenate([expert_states, visited_states], axis=0)
    aggregate_actions = np.concatenate([expert_actions, labeled_actions], axis=0)
    np.savez(
        os.path.join(args.output_dir, "bc_dagger_aggregate_dataset.npz"),
        visited_states=visited_states,
        labeled_actions=labeled_actions,
        rollout_steps=np.asarray(rollout_steps),
        mean_nn_distance=mean_nn_distance,
    )

    print(f"collected_states={len(visited_states)}, mean_nearest_expert_distance={mean_nn_distance:.4f}")
    best_path, _ = train_on_aggregated_data(
        model,
        aggregate_states,
        aggregate_actions,
        args.normalizer,
        args.output_dir,
        args.epochs,
        args.batch_size,
        device,
        args.eval_every,
        args.selection_episodes,
        args.max_steps,
    )
    record_demo(best_path, args.normalizer, args.output_dir, args.eval_episodes, args.max_steps, device)


if __name__ == "__main__":
    main()
