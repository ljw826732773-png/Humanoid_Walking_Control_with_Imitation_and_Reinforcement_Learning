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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 定义SAC策略网络
class SACPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, max_action=1.0):
        super(SACPolicy, self).__init__()
        self.max_action = max_action
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(128, output_dim)
        self.log_std_layer = nn.Linear(128, output_dim)

    def forward(self, state, deterministic=False):
        x = self.network(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
        else:
            dist = Normal(mean, std)
            normal_action = dist.rsample()
            action = torch.tanh(normal_action)
            log_prob = dist.log_prob(normal_action).sum(dim=-1)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
            log_prob = log_prob.unsqueeze(-1)
            return action * self.max_action, log_prob
        return action * self.max_action

# 2. 定义Q网络
class SACQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(SACQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

# 3. 定义值函数网络
class SACValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SACValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.network(state)

# 4. 计算折扣回报
def compute_returns(rewards, gamma=0.98):
    returns = []
    discounted_reward = 0
    for reward in reversed(rewards):
        discounted_reward = reward + gamma * discounted_reward
        returns.insert(0, discounted_reward)
    return torch.tensor(returns, dtype=torch.float32)

# 5. 软更新目标网络
def soft_update(target, source, tau=0.005):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# 6. SAC更新函数
def sac_update(policy, q1_net, q2_net, target_q1_net, target_q2_net, value_net, target_value_net,
               optimizer, q_optimizer, value_optimizer, states, actions, rewards, next_states, dones,
               gamma=0.95, alpha=0.1):
    policy.train()
    q1_net.train()
    q2_net.train()
    value_net.train()

    with torch.no_grad():
        next_action, next_log_prob = policy(next_states)
        target_q1 = target_q1_net(next_states, next_action)
        target_q2 = target_q2_net(next_states, next_action)
        target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
        target_value = rewards + (1 - dones) * gamma * target_q

    q1_pred = q1_net(states, actions)
    q2_pred = q2_net(states, actions)
    q1_loss = (q1_pred - target_value).pow(2).mean()
    q2_loss = (q2_pred - target_value).pow(2).mean()
    q_loss = q1_loss + q2_loss

    q_optimizer.zero_grad()
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(q1_net.parameters()) + list(q2_net.parameters()), 5.0)
    q_optimizer.step()

    action, log_prob = policy(states)
    q1_new = q1_net(states, action)
    q2_new = q2_net(states, action)
    q_new = torch.min(q1_new, q2_new)
    policy_loss = (alpha * log_prob - q_new).mean()

    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    optimizer.step()

    value_pred = value_net(states)
    with torch.no_grad():
        action, log_prob = policy(states)
        q1_value = q1_net(states, action)
        q2_value = q2_net(states, action)
        q_value = torch.min(q1_value, q2_value)
        target_value_net_value = q_value - alpha * log_prob
    value_loss = (value_pred - target_value_net_value).pow(2).mean()

    value_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 5.0)
    value_optimizer.step()

    soft_update(target_q1_net, q1_net, tau=0.005)
    soft_update(target_q2_net, q2_net, tau=0.005)
    soft_update(target_value_net, value_net, tau=0.005)

    return policy_loss.item(), q_loss.item(), value_loss.item()

# 7. 加载环境
env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
state_dim = 36
action_dim = 13

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = SACPolicy(state_dim, action_dim).to(device)
value_net = SACValueNetwork(state_dim).to(device)
target_value_net = SACValueNetwork(state_dim).to(device)
q1_net = SACQNetwork(state_dim, action_dim).to(device)
q2_net = SACQNetwork(state_dim, action_dim).to(device)
target_q1_net = SACQNetwork(state_dim, action_dim).to(device)
target_q2_net = SACQNetwork(state_dim, action_dim).to(device)

target_value_net.load_state_dict(value_net.state_dict())
target_q1_net.load_state_dict(q1_net.state_dict())
target_q2_net.load_state_dict(q2_net.state_dict())

optimizer = optim.Adam(policy.parameters(), lr=1e-5, weight_decay=1e-4)
q_optimizer = optim.Adam(list(q1_net.parameters()) + list(q2_net.parameters()), lr=2e-5, weight_decay=2e-4)
value_optimizer = optim.Adam(value_net.parameters(), lr=2e-5, weight_decay=2e-4)

episode_rewards_record = []
policy_loss_record = []
q_loss_record = []
value_loss_record = []
stability_metrics = []

# 8. SAC训练循环
num_episodes = 8000
max_steps = 1000
gamma = 0.95
batch_size = 32
batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []

running_reward_mean = 0.0
running_reward_std = 1.0
running_reward_count = 0
alpha = 0.1
action_norm_penalty = 0.01
survival_reward = 4.0
forward_reward_scale = 8.0
gait_reward_scale = 8.0
foot_landing_scale = 0.1
stability_scale = 1
uptarget_pelvis_tz = 5

optimizer = optim.Adam(policy.parameters(), lr=1e-5, weight_decay=1e-4)
q_optimizer = optim.Adam(list(q1_net.parameters()) + list(q2_net.parameters()), lr=2e-5, weight_decay=2e-4)
value_optimizer = optim.Adam(value_net.parameters(), lr=2e-5, weight_decay=2e-4)

max_steps_achieved = 0

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    states, actions, rewards, next_states, dones = [], [], [], [], []
    episode_stability = []
    prev_pelvis_tx = env._data.qpos[0]
    prev_hip_flexion_diff = 0
    prev_action = np.zeros(action_dim)

    raw_rewards = []
    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action, _ = policy(state_tensor)
        if episode < 1000:
            noise = torch.randn_like(action) * 0.05
            action = (action + noise).clamp(-1.0, 1.0)
        action = action.detach().cpu().numpy()[0]

        next_state, _, done, info = env.step(action)

        current_pelvis_tx = env._data.qpos[0]
        forward_distance = current_pelvis_tx - prev_pelvis_tx
        forward_reward = forward_reward_scale * forward_distance

        hip_flexion_r = env._data.qpos[6]
        hip_flexion_l = env._data.qpos[11]
        current_hip_flexion_diff = abs(hip_flexion_r - hip_flexion_l)
        gait_reward = gait_reward_scale * np.exp(-abs(current_hip_flexion_diff - prev_hip_flexion_diff))
        gait_continuity = 0.1 * np.exp(-np.linalg.norm(action - prev_action)) if step > 0 else 0

        ankle_angle_r = env._data.qpos[10]
        ankle_angle_l = env._data.qpos[15]
        foot_landing_reward = foot_landing_scale * (1 - np.abs(ankle_angle_r) - np.abs(ankle_angle_l))

        pelvis_tz = env._data.qpos[1]
        pelvis_ty = env._data.qpos[2]
        stability_reward = -stability_scale * ((pelvis_tz - uptarget_pelvis_tz) ** 2 + pelvis_ty ** 2)

        pelvis_tilt = env._data.qpos[3]
        lumbar_extension = env._data.qpos[16]
        upright_reward = 4.0 * (np.exp(-abs(pelvis_tilt)) + np.exp(-abs(lumbar_extension)))

        action_norm = np.linalg.norm(action)
        action_penalty = -action_norm_penalty * action_norm

        survival = survival_reward if not done else 0.0

        reward = forward_reward + gait_reward + gait_continuity + foot_landing_reward + stability_reward + upright_reward + action_penalty + survival
        raw_rewards.append(reward)
        episode_reward += reward

        prev_pelvis_tx = current_pelvis_tx
        prev_hip_flexion_diff = current_hip_flexion_diff
        prev_action = action.copy()

        states.append(state_tensor)
        actions.append(torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device))
        rewards.append(reward)
        next_states.append(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device))
        dones.append(done)

        state = next_state
        if done or step >= 200:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {len(rewards)}, Info: {info}")
            break

    if len(rewards) > max_steps_achieved:
        max_steps_achieved = len(rewards)
        torch.save(policy.state_dict(), os.path.join(r"E:\mujoco", f"sac_model_steps_{max_steps_achieved}.pth"))
        print(f"New max steps achieved: {max_steps_achieved}, model saved as sac_model_steps_{max_steps_achieved}.pth")

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

    stability = np.mean([np.linalg.norm(a.cpu().numpy() - prev_action.cpu().numpy()) for a, prev_action in zip(actions[1:], actions[:-1])]) if len(actions) > 1 else 0
    stability_metrics.append(stability)
    episode_rewards_record.append(episode_reward)

    if states:
        batch_states = torch.cat(states)
        batch_actions = torch.cat(actions)
        batch_rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
        batch_next_states = torch.cat(next_states)
        batch_dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)

        policy_loss, q_loss, value_loss = sac_update(
            policy, q1_net, q2_net, target_q1_net, target_q2_net, value_net, target_value_net,
            optimizer, q_optimizer, value_optimizer, batch_states, batch_actions, batch_rewards,
            batch_next_states, batch_dones
        )
        policy_loss_record.append(policy_loss)
        q_loss_record.append(q_loss)
        value_loss_record.append(value_loss)

    print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {len(rewards)}")

# 9. 可视化
output_dir = r"E:\mujoco\table\sac_train"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards_record, label="原始总奖励", alpha=0.5)
avg_rewards = [np.mean(episode_rewards_record[max(0, i - 10):i + 1]) for i in range(len(episode_rewards_record))]
plt.plot(avg_rewards, label="平滑平均奖励（窗口=10）", linestyle="--")
plt.xlabel("训练轮数")
plt.ylabel("奖励")
plt.title("奖励变化曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "奖励变化曲线.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(policy_loss_record, label="策略损失")
plt.plot(q_loss_record, label="Q函数损失")
plt.plot(value_loss_record, label="值函数损失")
plt.xlabel("更新次数")
plt.ylabel("损失")
plt.title("损失函数曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "损失函数曲线.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(stability_metrics, label="动作稳定性（动作差的均值）")
plt.xlabel("训练轮数")
plt.ylabel("稳定性")
plt.title("动作稳定性变化曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "动作稳定性曲线.png"))
plt.close()

# 10. 测试并录制视频
def record_test(policy, env, video_dir, prefix="sac_1000"):
    policy.eval()
    total_reward = 0
    max_steps_per_episode = 1000
    num_episodes = 1000  # 修改为1000轮测试
    total_steps = 0
    max_steps = 0
    max_episode_frames = []
    max_episode = 0
    max_episode_joint_angles = []
    max_episode_actions = []
    max_episode_states = []
    episode_steps = []

    table_dir = r"E:\mujoco\table\sac_train"
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

    dataset = env.create_dataset()
    dataset_states = dataset["states"]
    dataset_actions = dataset["actions"]

    target_steps = 34  # 目标步数
    target_frames_saved = False  # 标记是否已保存目标步数的关键帧

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

            prev_pelvis_tx = env._data.qpos[0]

            while step < max_steps_per_episode:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy(state_tensor, deterministic=True)
                action = action.detach().cpu().numpy()[0]
                next_state, reward, done, info = env.step(action)

                current_pelvis_tx = env._data.qpos[0]
                forward_distance = current_pelvis_tx - prev_pelvis_tx
                forward_reward = forward_reward_scale * forward_distance

                hip_flexion_r = env._data.qpos[6]
                hip_flexion_l = env._data.qpos[11]
                gait_diff = abs(hip_flexion_r + hip_flexion_l)
                gait_reward = gait_reward_scale * np.exp(-gait_diff)

                ankle_angle_r = env._data.qpos[10]
                ankle_angle_l = env._data.qpos[15]
                foot_landing_reward = foot_landing_scale * (1 - np.abs(ankle_angle_r) - np.abs(ankle_angle_l))

                pelvis_tz = env._data.qpos[1]
                pelvis_ty = env._data.qpos[2]
                stability_reward = -stability_scale * ((pelvis_tz - uptarget_pelvis_tz) ** 2 + pelvis_ty ** 2)

                pelvis_tilt = env._data.qpos[3]
                lumbar_extension = env._data.qpos[16]
                upright_reward = 4.0 * (np.exp(-abs(pelvis_tilt)) + np.exp(-abs(lumbar_extension)))

                action_norm = np.linalg.norm(action)
                action_penalty = -action_norm_penalty * action_norm

                survival = survival_reward if not done else 0.0

                reward = forward_reward + gait_reward + foot_landing_reward + stability_reward + upright_reward + action_penalty + survival
                episode_reward += reward

                prev_pelvis_tx = current_pelvis_tx
                state = next_state

                data = env._data
                mujoco.mj_forward(model, data)
                viewer.update_scene(data, camera="track")
                frame = viewer.render()
                joint_angles = data.qpos[:19].copy()
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
            if step > max_steps:
                max_steps = step
                max_episode_frames = episode_frames
                max_episode = episode + 1
                max_episode_joint_angles = episode_joint_angles
                max_episode_actions = episode_actions
                max_episode_states = episode_states

            # 检查是否达到目标步数34并生成关键帧大图
            if step == target_steps and not target_frames_saved:
                import math
                frames_per_image = 24
                sampled_frames = episode_frames[::2]
                num_images = math.ceil(len(sampled_frames) / frames_per_image)

                for img_idx in range(num_images):
                    fig, axes = plt.subplots(6, 4, figsize=(16, 24))
                    for i in range(6):
                        for j in range(4):
                            frame_idx = img_idx * frames_per_image + i * 4 + j
                            if frame_idx < len(sampled_frames):
                                axes[i, j].imshow(cv2.cvtColor(sampled_frames[frame_idx], cv2.COLOR_BGR2RGB))
                                axes[i, j].set_title(f'帧 {frame_idx * 2+ 1}')
                                axes[i, j].axis('off')
                            else:
                                axes[i, j].axis('off')
                    plt.suptitle(f'第5回合关键帧（步数 {step}，图 {img_idx + 1}/{num_images}）', fontsize=16)
                    plt.savefig(os.path.join(table_dir, f'第{episode + 1}回合_关键帧_{img_idx + 1}_步数{step}.png'))
                    plt.close()
                print(f"已生成步数为 {step} 的关键帧大图，测试停止")
                target_frames_saved = True
                video_writer.release()
                return total_reward, total_steps  # 停止测试

            if step >= max_steps_per_episode:
                print(f"{prefix} 第 {episode + 1} 回合达到最大步数，奖励: {episode_reward:.2f}")

    video_writer.release()
    print(f"{prefix} 在 {num_episodes} 回合中的总奖励: {total_reward:.2f}, 总步数: {total_steps}")

    # 绘制测试的时间步曲线
    plt.figure(figsize=(10, 6))
    episodes = np.arange(1, len(episode_steps) + 1)
    plt.plot(episodes, episode_steps, label='每回合时间步数', marker='o')
    avg_steps = np.mean(episode_steps)
    plt.axhline(y=avg_steps, color='r', linestyle='--', label=f'平均时间步数: {avg_steps:.2f}')
    plt.title('SAC测试时间步曲线')
    plt.xlabel('回合')
    plt.ylabel('时间步数')
    plt.legend(frameon=False)
    plt.grid(True)
    plt.savefig(os.path.join(table_dir, 'SAC测试时间步曲线.png'))
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
        joint_angles = np.array(max_episode_joint_angles)
        time_steps = np.arange(len(joint_angles))

        max_episode_states = np.array(max_episode_states)
        closest_traj_idx = find_closest_expert_trajectory(max_episode_states, dataset_states, len(max_episode_states))
        expert_joint_angles = dataset_states[closest_traj_idx:closest_traj_idx + len(max_episode_states), :19]

        print(f"预测关节角度的形状：{joint_angles.shape}")
        print(f"专家关节角度的形状：{expert_joint_angles.shape}")

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, joint_angles[:, 6], label=f'预测关节 {model.joint(6).name}')
        plt.plot(time_steps, joint_angles[:, 11], label=f'预测关节 {model.joint(11).name}')
        plt.plot(time_steps, expert_joint_angles[:, 6], '--', label=f'专家关节 {model.joint(6).name}')
        plt.plot(time_steps, expert_joint_angles[:, 11], '--', label=f'专家关节 {model.joint(11).name}')
        plt.title(f'SAC第 回合关节角度（关节6和11）')
        plt.xlabel('时间步')
        plt.ylabel('关节角度（弧度）')
        plt.legend(frameon=False)
        plt.grid(True)
        plt.savefig(os.path.join(table_dir, f'SAC第{max_episode}回合_关节角度_6_11.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, joint_angles[:, 9], label=f'预测关节 {model.joint(9).name}')
        plt.plot(time_steps, joint_angles[:, 14], label=f'预测关节 {model.joint(14).name}')
        plt.plot(time_steps, expert_joint_angles[:, 9], '--', label=f'专家关节 {model.joint(9).name}')
        plt.plot(time_steps, expert_joint_angles[:, 14], '--', label=f'专家关节 {model.joint(14).name}')
        plt.title(f'SAC第 {max_episode} 回合关节角度（关节9和14）')
        plt.xlabel('时间步')
        plt.ylabel('关节角度（弧度）')
        plt.legend(frameon=False)
        plt.grid(True)
        plt.savefig(os.path.join(table_dir, f'SAC第{max_episode}回合_关节角度_9_14.png'))
        plt.close()

    return total_reward, total_steps

# 测试SAC模型
print("\nTesting SAC model:")
video_dir = r"E:\mujoco\new_test_videos"
sac_reward, sac_steps = record_test(policy, env, video_dir, "sac_1000")