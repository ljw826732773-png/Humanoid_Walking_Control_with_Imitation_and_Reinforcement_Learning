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
from matplotlib import font_manager
#最终版本
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['font.size'] = 12  # 五号字体，10pt
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
        self.log_std = nn.Parameter(torch.ones(output_dim) * 0.5)  # 可训练标准差，初始值为 0.5

    def forward(self, state, episode=0):
        mean = torch.tanh(self.network(state))  # 对均值应用 tanh 激活
        # 裁剪 log_std 确保标准差在合理范围内，最小标准差为 0.1
        log_std_clipped = torch.clamp(self.log_std, -2.3, 5)  # exp(-2.3) ≈ 0.1
        std = torch.exp(torch.clamp(log_std_clipped - 0.0001 * episode, -20, 5))
        return mean, std

    def get_action(self, state, episode=0):
        mean, std = self(state, episode)
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
def compute_returns(rewards, gamma=0.95):
    returns = []
    discounted_reward = 0
    for reward in reversed(rewards):
        discounted_reward = reward + gamma * discounted_reward
        returns.insert(0, discounted_reward)
    return torch.as_tensor(returns, dtype=torch.float32)

# 4. 软更新目标网络
def soft_update(target, source, tau=0.05):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# 5. PPO更新函数（加入temporal smooth loss）
def ppo_update(policy, value_net, target_value_net, optimizer, value_optimizer, states, actions, log_probs_old, returns, episode,
               clip_eps=0.1, epochs=10, entropy_coef=0.1, smooth_coef=0.2, temporal_smooth_coef=0.1):
    policy.train()
    value_net.train()
    actor_losses = []
    value_losses = []

    for _ in range(epochs):
        mean, std = policy(states, episode)  # 传递 episode 参数
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = value_net(states).squeeze()

        with torch.no_grad():
            target_values = target_value_net(states).squeeze()

        advantages = returns - target_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -10, 10)

        ratio = torch.clamp(torch.exp(log_probs - log_probs_old.detach()), 0.01, 100.0)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        entropy = dist.entropy().mean()

        # 调试信息
        print(f"Ratio mean: {ratio.mean().item():.10f}, std: {ratio.std().item():.10f}")
        print(f"Advantages mean: {advantages.mean().item():.10f}, std: {advantages.std().item():.10f}")
        print(f"Entropy: {entropy.item():.10f}")
        print(f"Surr1 mean: {surr1.mean().item():.10f}, Surr2 mean: {surr2.mean().item():.10f}")
        print(f"Value net output mean: {values.mean().item():.10f}, std: {values.std().item():.10f}")
        print(f"Target value net output mean: {target_values.mean().item():.10f}, std: {target_values.std().item():.10f}")
        print(f"Value diff mean: {(values - target_values.detach()).mean().item():.10f}, std: {(values - target_values.detach()).std().item():.10f}")

        smooth_loss = 0
        if len(actions) > 1:
            action_diff = (actions[1:] - actions[:-1]).pow(2).mean()
            smooth_loss = smooth_coef * action_diff

        temporal_smooth_loss = 0
        if len(actions) > 1:
            temporal_diff = torch.diff(actions, dim=0).pow(2).mean()
            temporal_smooth_loss = temporal_smooth_coef * temporal_diff

        actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy + smooth_loss + temporal_smooth_loss
        value_loss = (values - target_values.detach()).pow(2).mean()

        print(f"Actor Loss components: surr={(-torch.min(surr1, surr2).mean()).item():.10f}, "
              f"entropy={(-entropy_coef * entropy).item():.10f}, "
              f"smooth={smooth_loss.item():.10f}, temporal={temporal_smooth_loss.item():.10f}")
        print(f"Actor Loss: {actor_loss.item():.10f}, Value Loss: {value_loss.item():.10f}")

        optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_grad_norm = torch.nn.utils.clip_grad_norm_(value_net.parameters(), 10.0)
        value_optimizer.step()

        soft_update(target_value_net, value_net, tau=0.05)

        actor_losses.append(actor_loss.item())
        value_losses.append(value_loss.item())
        print(f"Actor Grad Norm: {actor_grad_norm:.10f}, Value Grad Norm: {value_grad_norm:.10f}")

    return np.mean(actor_losses), np.mean(value_losses)

# 6. 加载环境
env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
state_dim = 36
action_dim = 13

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PPOPolicy(state_dim, action_dim).to(device)
value_net = PPOValue(state_dim).to(device)
target_value_net = PPOValue(state_dim).to(device)
target_value_net.load_state_dict(value_net.state_dict())

optimizer = optim.Adam(policy.parameters(), lr=1e-4, weight_decay=1e-5)
value_optimizer = optim.Adam(value_net.parameters(), lr=5e-4, weight_decay=1e-5)

episode_rewards_record = []
actor_loss_record = []
value_loss_record = []
success_rates = []
stability_metrics = []

# 初始化最长步数模型跟踪
max_steps = 0
longest_model_path = "longest_steps_model.pth_11"

# 7. PPO训练循环（保留短回合并施加惩罚，添加生存奖励）
num_episodes = 7000
max_steps_per_episode = 200
gamma = 0.95
batch_states, batch_actions, batch_log_probs, batch_returns = [], [], [], []

running_reward_mean = 0.0
running_reward_std = 1.0
running_reward_count = 0
alpha = 0.1
action_norm_penalty = 0.01
short_episode_penalty = 0.8
long_episode_bonus = 4.0
survival_reward = 2.0

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    states, actions, rewards, log_probs_old = [], [], [], []
    episode_success = 0
    episode_stability = []

    raw_rewards = []
    for step in range(max_steps_per_episode):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action, log_prob = policy.get_action(state_tensor, episode)  # 传递 episode 参数
        action = action.cpu().numpy()[0]

        next_state, reward, done, info = env.step(action)
        action_norm = np.linalg.norm(action)
        if action_norm > 2.0:
            reward -= action_norm_penalty * action_norm

        if not done:
            reward += survival_reward
        raw_rewards.append(reward)
        episode_reward += reward

        if reward > 0:
            episode_success += 1
        if step > 0:
            action_diff = np.linalg.norm(action - prev_action)
            episode_stability.append(action_diff)
        prev_action = action

        states.append(state_tensor)
        actions.append(torch.as_tensor(action, dtype=torch.float32).unsqueeze(0).to(device))
        rewards.append(reward)
        log_probs_old.append(log_prob)

        state = next_state
        if done:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {len(rewards)}, Info: {info}")
            break

    # 保存最长步数的模型
    episode_steps = len(raw_rewards)
    if episode_steps > max_steps:
        max_steps = episode_steps
        torch.save(policy.state_dict(), longest_model_path)
        print(f"Saved model with longest steps {max_steps} at episode {episode + 1}")

    # 短回合惩罚
    if len(raw_rewards) < 3:
        penalty = short_episode_penalty / len(raw_rewards)
        raw_rewards = [r - penalty for r in raw_rewards]
        episode_reward -= penalty

    # 长回合奖励
    if len(raw_rewards) > 3:
        bonus = long_episode_bonus * np.log(len(raw_rewards))
        raw_rewards = [r + bonus for r in raw_rewards]
        episode_reward += bonus

    # 奖励标准化
    if raw_rewards:
        episode_rewards = np.array(raw_rewards)
        episode_mean = np.mean(episode_rewards)
        episode_std = np.std(episode_rewards) + 1e-8

        if running_reward_count == 0:
            running_reward_mean = episode_mean
            running_reward_std = episode_std
        else:
            running_reward_mean = (1 - alpha) * running_reward_mean + alpha * episode_mean
            running_reward_std = (1 - alpha) * running_reward_std + alpha * episode_std
        running_reward_count += 1

        rewards = [(r - running_reward_mean) / running_reward_std for r in raw_rewards]
    else:
        rewards = raw_rewards

    success_rate = episode_success / len(rewards) if rewards else 0
    stability = np.mean(episode_stability) if episode_stability else 0
    success_rates.append(success_rate)
    stability_metrics.append(stability)
    episode_rewards_record.append(episode_reward)

    returns = compute_returns(rewards, gamma=0.97).to(device)

    batch_states.extend(states)
    batch_actions.extend(actions)
    batch_log_probs.extend(log_probs_old)
    batch_returns.extend(returns)

    if (episode + 1) % 60 == 0:
        batch_states = torch.cat(batch_states)
        batch_actions = torch.cat(batch_actions)
        batch_log_probs = torch.cat(batch_log_probs)
        batch_returns = torch.as_tensor(batch_returns, dtype=torch.float32).to(device)

        actor_loss, value_loss = ppo_update(policy, value_net, target_value_net, optimizer, value_optimizer, batch_states,
                                            batch_actions, batch_log_probs, batch_returns, episode, temporal_smooth_coef=0.1)
        actor_loss_record.append(actor_loss)
        value_loss_record.append(value_loss)
        batch_states, batch_actions, batch_log_probs, batch_returns = [], [], [], []

    print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {len(rewards)}")

# 8. 可视化
output_dir = r"E:\mujoco\table\ppo_train"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards_record, label="原始总奖励", alpha=0.5)
avg_rewards = [np.mean(episode_rewards_record[max(0, i - 10):i + 1]) for i in range(len(episode_rewards_record))]
plt.plot(avg_rewards, label="平滑平均奖励（窗口=10）", linestyle="--")
plt.xlabel("训练轮数")
plt.ylabel("奖励")
plt.title("PPO奖励变化曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "PPO奖励变化曲线.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(actor_loss_record, label="策略损失")
plt.plot(value_loss_record, label="价值损失")
plt.xlabel("更新次数")
plt.ylabel("损失")
plt.title("PPO损失函数曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "PPO损失函数曲线.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(stability_metrics, label="动作稳定性（动作差的均值）")
plt.xlabel("训练轮数")
plt.ylabel("稳定性")
plt.title("PPO动作稳定性变化曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "PPO动作稳定性曲线.png"))
plt.close()

# 9. 测试并录制视频
def record_test(policy, env, video_dir, prefix="ppo_1"):
    policy.eval()
    total_reward = 0
    max_steps_per_episode = 200
    num_episodes = 10
    total_steps = 0
    max_steps = 0
    max_episode = 0
    max_episode_joint_angles = []
    max_episode_actions = []
    max_episode_states = []
    episode_steps = []  # 记录每回合的步数

    # 与 ppo_v7.py 原有可视化路径一致
    table_dir = r"E:\mujoco\table\ppo_train"
    video_path = os.path.join(video_dir, f"{prefix}.mp4")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    print(f"视频将保存至：{video_path}")

    model = env._model
    data = env._data
    viewer = mujoco.Renderer(model, width=640, height=480)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    if not video_writer.isOpened():
        print("错误：VideoWriter 初始化失败")
        return total_reward, total_steps

    # 加载专家数据集以获取专家轨迹
    dataset = env.create_dataset()
    dataset_states = dataset["states"]
    dataset_actions = dataset["actions"]

    with torch.no_grad():
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            step = 0
            episode_frames = []
            episode_joint_angles = []
            episode_actions = []
            episode_states = []
            print(f"\n开始 {prefix} 第 {episode + 1}/{num_episodes} 回合")

            while step < max_steps_per_episode:
                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                mean, _ = policy(state_tensor, episode)
                action = mean.cpu().numpy()[0]
                action = np.clip(action, -1.0, 1.0)  # 确保动作在合理范围内
                next_state, reward, done, info = env.step(action)
                if not done:
                    reward += survival_reward
                episode_reward += reward
                state = next_state

                data = env._data
                mujoco.mj_forward(model, data)
                viewer.update_scene(data, camera="track")
                frame = viewer.render()
                joint_angles = data.qpos[:19].copy()  # 提取19个关节角度
                episode_joint_angles.append(joint_angles)
                episode_actions.append(action)
                episode_states.append(state)

                if frame is not None and frame.size > 0:
                    if frame.shape != (480, 640, 3):
                        frame = cv2.resize(frame, (640, 480))
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                    episode_frames.append(frame_bgr)
                else:
                    print(f"警告：第 {episode + 1} 回合，第 {step + 1} 步为空帧")

                step += 1
                total_steps += 1
                print(f"{prefix} 第 {episode + 1} 回合，第 {step} 步，奖励: {reward:.4f}, 完成: {done}, 信息: {info}")

                if done:
                    print(f"{prefix} 第 {episode + 1} 回合在第 {step} 步终止，奖励: {episode_reward:.2f}")
                    break

            total_reward += episode_reward
            episode_steps.append(step)

            # 保存当前回合的关键帧组图（4x6）
            if episode_frames:
                import math
                frames_per_image = 24  # 4x6
                sampled_frames = episode_frames[::1]  # 每个时间步取一帧
                num_images = math.ceil(len(sampled_frames) / frames_per_image)

                for img_idx in range(num_images):
                    fig, axes = plt.subplots(6, 4, figsize=(16, 24))  # 横4竖6
                    for i in range(6):
                        for j in range(4):
                            frame_idx = img_idx * frames_per_image + i * 4 + j
                            if frame_idx < len(sampled_frames):
                                axes[i, j].imshow(cv2.cvtColor(sampled_frames[frame_idx], cv2.COLOR_BGR2RGB))
                                axes[i, j].set_title(f'帧 {frame_idx * 1 + 1}')
                                axes[i, j].axis('off')
                            else:
                                axes[i, j].axis('off')
                    plt.suptitle(f'PPO第{episode + 1}回合关键帧（图 {img_idx + 1}/{num_images}）', fontsize=16)
                    plt.savefig(os.path.join(table_dir, f'PPO第{episode + 1}回合_关键帧_{img_idx + 1}.png'))
                    plt.close()

            if step > max_steps:
                max_steps = step
                max_episode = episode + 1
                max_episode_joint_angles = episode_joint_angles
                max_episode_actions = episode_actions
                max_episode_states = episode_states

            if step >= max_steps_per_episode:
                print(f"PPO{prefix} 第 {episode + 1} 回合达到最大步数，奖励: {episode_reward:.2f}")

    video_writer.release()
    print(f"{prefix} 在 {num_episodes} 回合中的总奖励: {total_reward:.2f}, 总步数: {total_steps}")

    # 绘制十轮测试的时间步曲线
    plt.figure(figsize=(10, 6))
    episodes = np.arange(1, num_episodes + 1)
    plt.plot(episodes, episode_steps, label='每回合时间步数', marker='o')
    avg_steps = np.mean(episode_steps)
    plt.axhline(y=avg_steps, color='r', linestyle='--', label=f'平均时间步数: {avg_steps:.2f}')
    plt.title('PPO十轮测试时间步曲线')
    plt.xlabel('回合')
    plt.ylabel('时间步数')
    plt.legend(frameon=False)
    plt.grid(True)
    plt.savefig(os.path.join(table_dir, 'PPO十轮测试时间步曲线.png'))
    plt.close()

    # 寻找最接近的专家轨迹
    def find_closest_expert_trajectory(states, dataset_states, num_steps):
        min_dist = float('inf')
        closest_traj_idx = 0
        states = np.array(states)
        for i in range(0, len(dataset_states) - num_steps + 1, num_steps):
            if i + num_steps <= len(dataset_states):
                dist = np.linalg.norm(states - dataset_states[i:i + num_steps, :], axis=1).mean()
                if dist < min_dist:
                    min_dist = dist
                    closest_traj_idx = i
        return closest_traj_idx

    # 绘制最大步数回合的关节信息图片，包含专家数据
    if max_episode_joint_angles:
        joint_angles = np.array(max_episode_joint_angles)  # 形状 (num_steps, 19)
        time_steps = np.arange(len(joint_angles))

        # 找到最接近的专家轨迹（假设前19维是qpos）
        max_episode_states = np.array(max_episode_states)
        closest_traj_idx = find_closest_expert_trajectory(max_episode_states, dataset_states, len(max_episode_states))
        expert_joint_angles = dataset_states[closest_traj_idx:closest_traj_idx + len(max_episode_states), :19]

        # 调试：打印维度
        print(f"预测关节角度的形状：{joint_angles.shape}")
        print(f"专家关节角度的形状：{expert_joint_angles.shape}")

        # 第一张图：关节6和11
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, joint_angles[:, 6], label=f'预测关节 {model.joint(6).name}')
        plt.plot(time_steps, joint_angles[:, 11], label=f'预测关节 {model.joint(11).name}')
        plt.plot(time_steps, expert_joint_angles[:, 6], '--', label=f'专家关节 {model.joint(6).name}')
        plt.plot(time_steps, expert_joint_angles[:, 11], '--', label=f'专家关节 {model.joint(11).name}')
        plt.title(f'PPO第 {max_episode} 回合关节角度（关节6和11）')
        plt.xlabel('时间步')
        plt.ylabel('关节角度（弧度）')
        plt.legend(frameon=False)
        plt.grid(True)
        plt.savefig(os.path.join(table_dir, f'PPO第{max_episode}回合_关节角度_6_11.png'))
        plt.close()

        # 第二张图：关节9和14
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, joint_angles[:, 9], label=f'预测关节 {model.joint(9).name}')
        plt.plot(time_steps, joint_angles[:, 14], label=f'预测关节 {model.joint(14).name}')
        plt.plot(time_steps, expert_joint_angles[:, 9], '--', label=f'专家关节 {model.joint(9).name}')
        plt.plot(time_steps, expert_joint_angles[:, 14], '--', label=f'专家关节 {model.joint(14).name}')
        plt.title(f'PPO第 {max_episode} ，回合关节角度（关节9和14）')
        plt.xlabel('时间步')
        plt.ylabel('关节角度（弧度）')
        plt.legend(frameon=False)
        plt.grid(True)
        plt.savefig(os.path.join(table_dir, f'PPO第{max_episode}回合_关节角度_9_14.png'))
        plt.close()

    return total_reward, total_steps

# 测试最长步数PPO模型
print("\nTesting longest steps PPO model:")
video_dir = r"E:\mujoco\new_test_videos"
policy.load_state_dict(torch.load(longest_model_path, weights_only=True))
ppo_reward, ppo_steps = record_test(policy, env, video_dir, "longest_steps_ppo_11")
print(f"Model with longest steps loaded from {longest_model_path}")