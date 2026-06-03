import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import loco_mujoco
from sklearn.model_selection import train_test_split
import mujoco
import cv2
import os

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 加载环境和完美数据集
env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
dataset = env.create_dataset()

# 提取状态和动作
states = dataset["states"]
actions = dataset["actions"]
print(f"Dataset loaded: {states.shape} states, {actions.shape} actions")

# 2. 数据预处理
train_states, val_states, train_actions, val_actions = train_test_split(
    states, actions, test_size=0.2, random_state=42
)

class BCDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

train_dataset = BCDataset(train_states, train_actions)
val_dataset = BCDataset(val_states, val_actions)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 3. 定义神经网络模型
class BCModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

state_dim = states.shape[-1]
action_dim = actions.shape[-1]
model = BCModel(state_dim, action_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for state_batch, action_batch in train_loader:
        state_batch, action_batch = state_batch.to(device), action_batch.to(device)
        pred_actions = model(state_batch)
        loss = criterion(pred_actions, action_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for state_batch, action_batch in val_loader:
            state_batch, action_batch = state_batch.to(device), action_batch.to(device)
            pred_actions = model(state_batch)
            loss = criterion(pred_actions, action_batch)
            val_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# 6. 保存模型
torch.save(model.state_dict(), "bc_model.pth")
print("Model saved as bc_model.pth")

# 7. 测试并录制视频
def record_test(model, env, video_dir, prefix="bc"):
    model.eval()
    total_reward = 0
    max_steps_per_episode = 1000
    num_episodes = 10
    total_steps = 0

    video_path = os.path.join(video_dir, f"{prefix}.mp4")
    os.makedirs(video_dir, exist_ok=True)
    print(f"Video will be saved to: {video_path}")

    model_mujoco = env._model
    data = env._data
    viewer = mujoco.Renderer(model_mujoco, width=640, height=480)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    if not video_writer.isOpened():
        print("Error: VideoWriter failed to initialize")
        return total_reward, total_steps

    with torch.no_grad():
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            step = 0
            print(f"\nStarting {prefix} Episode {episode + 1}/{num_episodes}")

            while step < max_steps_per_episode:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = model(state_tensor).cpu().numpy()[0]
                action = np.clip(action, -1.0, 1.0)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state

                data = env._data
                mujoco.mj_forward(model_mujoco, data)
                viewer.update_scene(data, camera="track")
                frame = viewer.render()
                if frame is not None and frame.size > 0:
                    if frame.shape != (480, 640, 3):
                        frame = cv2.resize(frame, (640, 480))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame)
                else:
                    print(f"Warning: Empty frame at Episode {episode + 1}, Step {step + 1}")

                step += 1
                total_steps += 1
                print(f"{prefix} Episode {episode + 1}, Step {step}, Reward: {reward:.4f}, Done: {done}, Info: {info}")

                if done:
                    print(f"{prefix} Episode {episode + 1} terminated at step {step}, Reward: {episode_reward:.2f}")
                    break

            total_reward += episode_reward
            if step >= max_steps_per_episode:
                print(f"{prefix} Episode {episode + 1} reached max steps, Reward: {episode_reward:.2f}")

    video_writer.release()
    print(f"{prefix} Total Reward across {num_episodes} episodes: {total_reward:.2f}, Total Steps: {total_steps}")
    return total_reward, total_steps

# 执行测试并录制视频
video_dir = r"E:\mujoco\new_test_videos"
print("\nTesting BC model with video recording:")
bc_reward, bc_steps = record_test(model, env, video_dir, "bc")