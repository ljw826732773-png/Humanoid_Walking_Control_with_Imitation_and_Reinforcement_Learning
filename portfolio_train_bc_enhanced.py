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


class NormalizedBCDataset(Dataset):
    def __init__(self, states, actions, state_mean, state_std, action_mean, action_std, noise_std=0.0):
        self.states = torch.tensor((states - state_mean) / state_std, dtype=torch.float32)
        self.actions = torch.tensor((actions - action_mean) / action_std, dtype=torch.float32)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state = self.states[index]
        if self.noise_std > 0:
            state = state + torch.randn_like(state) * self.noise_std
        return state, self.actions[index]


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


def compute_normalizer(train_states, train_actions):
    state_mean = train_states.mean(axis=0)
    state_std = train_states.std(axis=0) + EPS
    action_mean = train_actions.mean(axis=0)
    action_std = train_actions.std(axis=0) + EPS
    return state_mean, state_std, action_mean, action_std


def save_normalizer(path, state_mean, state_std, action_mean, action_std):
    np.savez(
        path,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )


def load_normalizer(path):
    data = np.load(path)
    return data["state_mean"], data["state_std"], data["action_mean"], data["action_std"]


def train_bc(epochs, batch_size, output_dir, device, noise_std):
    os.makedirs(output_dir, exist_ok=True)
    env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
    dataset = env.create_dataset()
    states = dataset["states"]
    actions = dataset["actions"]

    train_states, val_states, train_actions, val_actions = train_test_split(
        states, actions, test_size=0.2, random_state=42
    )
    state_mean, state_std, action_mean, action_std = compute_normalizer(train_states, train_actions)
    normalizer_path = os.path.join(output_dir, "bc_improved_normalizer.npz")
    save_normalizer(normalizer_path, state_mean, state_std, action_mean, action_std)

    train_loader = DataLoader(
        NormalizedBCDataset(
            train_states,
            train_actions,
            state_mean,
            state_std,
            action_mean,
            action_std,
            noise_std=noise_std,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        NormalizedBCDataset(val_states, val_actions, state_mean, state_std, action_mean, action_std),
        batch_size=batch_size,
        shuffle=False,
    )

    model = BCModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_path = os.path.join(output_dir, "bc_improved_best.pth")
    last_path = os.path.join(output_dir, "bc_improved_last.pth")
    log_path = os.path.join(output_dir, "bc_improved_training_log.csv")

    start = time.time()
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "elapsed_sec"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for state_batch, action_batch in train_loader:
                state_batch = state_batch.to(device)
                action_batch = action_batch.to(device)
                pred_actions = model(state_batch)
                loss = criterion(pred_actions, action_batch)
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
                    pred_actions = model(state_batch)
                    val_loss += criterion(pred_actions, action_batch).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            elapsed = time.time() - start
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "elapsed_sec": elapsed,
                }
            )
            f.flush()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_path)

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(
                    f"epoch {epoch}/{epochs}, train_loss={avg_train_loss:.6f}, "
                    f"val_loss={avg_val_loss:.6f}, best_val_loss={best_val_loss:.6f}, "
                    f"elapsed={elapsed:.1f}s"
                )

    torch.save(model.state_dict(), last_path)
    return best_path, last_path, normalizer_path


def record_bc(model_path, normalizer_path, output_dir, device, episodes=10, max_steps=1000):
    env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
    state_mean, state_std, action_mean, action_std = load_normalizer(normalizer_path)

    state_mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)

    model = BCModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    video_path = os.path.join(output_dir, "bc_improved_demo.mp4")
    metrics_path = os.path.join(output_dir, "bc_improved_eval.csv")
    summary_path = os.path.join(output_dir, "bc_improved_summary.csv")
    renderer = mujoco.Renderer(env._model, width=640, height=480)
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))

    rows = []
    with torch.no_grad():
        for episode in range(1, episodes + 1):
            obs = env.reset()
            total_reward = 0.0
            steps = 0

            for _ in range(max_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                norm_obs = (obs_t - state_mean_t) / state_std_t
                norm_action = model(norm_obs).squeeze(0).cpu().numpy()
                action = norm_action * action_std + action_mean
                action = np.clip(action, -1.0, 1.0)

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

    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "steps", "reward"])
        writer.writeheader()
        writer.writerows(rows)

    steps = np.array([r["steps"] for r in rows], dtype=np.float32)
    rewards = np.array([r["reward"] for r in rows], dtype=np.float32)
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episodes", "avg_steps", "best_steps", "avg_reward", "best_reward", "video_path"],
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

    print(f"wrote {video_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="portfolio_retrain_bc_improved")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--noise-std", type=float, default=0.0)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    print(f"state noise std: {args.noise_std}")

    best_path, _, normalizer_path = train_bc(
        args.epochs,
        args.batch_size,
        args.output_dir,
        device,
        args.noise_std,
    )
    record_bc(
        best_path,
        normalizer_path,
        args.output_dir,
        device,
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
