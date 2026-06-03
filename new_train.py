import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import loco_mujoco
import os
import cv2
import mujoco
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 定义PPO策略网络
class PPOPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(output_dim))  # 可训练标准差，初始值为 0

    def forward(self, state):
        mean = self.network(state)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, state):
        mean, std = self(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

# 2. 定义值函数网络
class PPOValue(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)

# 3. 计算折扣回报
def compute_returns(rewards, gamma=0.95):  # 降低 gamma
    returns = []
    discounted_reward = 0
    for reward in reversed(rewards):
        discounted_reward = reward + gamma * discounted_reward
        returns.insert(0, discounted_reward)
    return torch.tensor(returns, dtype=torch.float32)

# 4. PPO更新函数
def ppo_update(policy, value_net, optimizer, value_optimizer, states, actions, log_probs_old, returns, bc_actions, bc_weight=0.5, clip_eps=0.2, epochs=20, entropy_coef=0.01):  # 增加 epochs
    policy.train()
    value_net.train()
    for _ in range(epochs):
        mean, std = policy(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = value_net(states).squeeze()

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_probs - log_probs_old.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        entropy = dist.entropy().mean()

        # 计算 BC 损失
        bc_loss = ((mean - bc_actions) ** 2).mean()
        actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy + bc_weight * bc_loss
        value_loss = (returns - values).pow(2).mean()

        optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        value_optimizer.step()

        # 调试输出
        print(f"Actor Loss: {actor_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, BC Loss: {bc_loss.item():.4f}")

# 5. 加载环境和预训练模型
env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
state_dim = 36
action_dim = 13

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PPOPolicy(state_dim, action_dim).to(device)
value_net = PPOValue(state_dim).to(device)

bc_model_path = "bc_model.pth"
state_dict = torch.load(bc_model_path, weights_only=True)
policy.load_state_dict(state_dict, strict=False)
print("Loaded pre-trained BC model from bc_model.pth")

# 只冻结第一层
for name, param in policy.named_parameters():
    if "network.0" in name:
        param.requires_grad = False

optimizer = optim.Adam(policy.parameters(), lr=1e-5)
value_optimizer = optim.Adam(value_net.parameters(), lr=3e-5)

# 加载预训练 BC 模型用于生成专家动作
bc_policy = PPOPolicy(state_dim, action_dim).to(device)
bc_policy.load_state_dict(state_dict, strict=False)
bc_policy.eval()

episode_rewards_record = []
actor_loss_record = []
value_loss_record = []
bc_loss_record = []

# 6. PPO训练循环
num_episodes = 1000
max_steps = 1000
gamma = 0.95  # 与 compute_returns 一致
batch_states, batch_actions, batch_log_probs, batch_returns, batch_bc_actions = [], [], [], [], []

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    states, actions, rewards, log_probs_old, bc_actions = [], [], [], [], []

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action, log_prob = policy.get_action(state_tensor)
        action = action.cpu().numpy()[0]

        # 使用 BC 模型生成专家动作
        with torch.no_grad():
            bc_mean, _ = bc_policy(state_tensor)
            bc_action = bc_mean

        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        states.append(state_tensor)
        actions.append(torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device))
        rewards.append(reward)
        log_probs_old.append(log_prob)
        bc_actions.append(bc_action)

        state = next_state
        if done:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {len(rewards)}, Info: {info}")
            break

    # 使用折扣回报
    returns = compute_returns(rewards, gamma=0.97).to(device)

    batch_states.extend(states)
    batch_actions.extend(actions)
    batch_log_probs.extend(log_probs_old)
    batch_returns.extend(returns)
    batch_bc_actions.extend(bc_actions)

    # 每 2 个 episode 更新一次
    if (episode + 1) % 2 == 0:  # 增加更新频率
        batch_states = torch.cat(batch_states)
        batch_actions = torch.cat(batch_actions)
        batch_log_probs = torch.cat(batch_log_probs)
        batch_returns = torch.tensor(batch_returns, dtype=torch.float32).to(device)
        batch_bc_actions = torch.cat(batch_bc_actions)

        # BC 损失权重减慢衰减
        bc_weight = 0.5 * (1 - episode / (2 * num_episodes))
        ppo_update(policy, value_net, optimizer, value_optimizer, batch_states, batch_actions, batch_log_probs, batch_returns, batch_bc_actions, bc_weight)
        batch_states, batch_actions, batch_log_probs, batch_returns, batch_bc_actions = [], [], [], [], []

    print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {len(rewards)}")

# 7. 保存训练后的模型
torch.save(policy.state_dict(), "ppo_model.pth")
print("PPO model saved as ppo_model.pth")



# 8. 测试并录制视频（使用 mujoco 渲染）
def record_test(policy, env, video_dir, prefix="test"):
    policy.eval()
    total_reward = 0
    max_steps_per_episode = 1000  # 每轮最大步数
    num_episodes = 10  # 测试 10 轮
    total_steps = 0

    video_path = os.path.join(video_dir, f"{prefix}.mp4")
    os.makedirs(video_dir, exist_ok=True)
    print(f"Video will be saved to: {video_path}")

    # 获取 MuJoCo 模型和数据
    model = env._model
    data = env._data
    viewer = mujoco.Renderer(model, width=640, height=480)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    if not video_writer.isOpened():
        print("Error: VideoWriter failed to initialize")
        return total_reward, total_steps

    with torch.no_grad():
        for episode in range(num_episodes):
            state = env.reset()  # 每轮开始时重置环境
            episode_reward = 0
            step = 0
            print(f"\nStarting {prefix} Episode {episode + 1}/{num_episodes}")

            while step < max_steps_per_episode:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                mean, _ = policy(state_tensor)
                action = mean.cpu().numpy()[0]
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state

                # 更新仿真并渲染
                data = env._data
                mujoco.mj_forward(model, data)
                viewer.update_scene(data, camera="track")  # 使用 "track" 相机
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


# 测试PPO模型
print("\nTesting PPO model:")
video_dir = r"E:\mujoco\new_test_videos"
ppo_reward, ppo_steps = record_test(policy, env, video_dir, "PPO")

# 测试BC模型
print("\nTesting BC model for comparison:")
bc_policy = PPOPolicy(state_dim, action_dim).to(device)
bc_policy.load_state_dict(torch.load(bc_model_path, weights_only=True), strict=False)
bc_reward, bc_steps = record_test(bc_policy, env, video_dir, "BC")