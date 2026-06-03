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
from collections import deque
import random
from gym.spaces import Box
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state_t = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_t = action if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        reward_t = reward if isinstance(reward, torch.Tensor) else torch.tensor(reward, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        next_state_t = next_state if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        done_t = done if isinstance(done, torch.Tensor) else torch.tensor(done, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        self.buffer.append((state_t, action_t, reward_t, next_state_t, done_t))

    def sample(self, batch_size, device):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        state = torch.cat(state).to(device)
        action = torch.cat(action).to(device)
        reward = torch.cat(reward).to(device)
        next_state = torch.cat(next_state).to(device)
        done = torch.cat(done).to(device)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# 2. 定义SAC策略网络
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
            action = mean
        else:
            dist = Normal(mean, std)
            normal_action = dist.rsample()
            action = torch.tanh(normal_action)
            log_prob = dist.log_prob(normal_action).sum(dim=-1, keepdim=True)
            tanh_correction = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            log_prob -= tanh_correction

        action = action * self.max_action
        if deterministic:
            return action
        else:
            return action, log_prob

# 3. 定义Q网络
class SACQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(SACQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

# 4. 定义值函数网络
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

# 5. 软更新目标网络
def soft_update(target, source, tau=0.005):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# 6. SAC更新函数
def sac_update(policy, q1_net, q2_net, target_q1_net, target_q2_net, value_net, target_value_net,
               optimizer, q_optimizer, value_optimizer, alpha_optimizer, log_alpha, target_entropy,
               states, actions, rewards, next_states, dones, gamma=0.99, tau=0.005):
    policy.train()
    q1_net.train()
    q2_net.train()
    value_net.train()
    alpha = torch.exp(log_alpha)

    with torch.no_grad():
        next_action, next_log_prob = policy(next_states)
        target_q1 = target_q1_net(next_states, next_action)
        target_q2 = target_q2_net(next_states, next_action)
        target_q = torch.min(target_q1, target_q2) - alpha.detach() * next_log_prob
        next_v_target = target_q
        q_target = rewards + (1 - dones) * gamma * next_v_target

    q1_pred = q1_net(states, actions)
    q2_pred = q2_net(states, actions)
    q1_loss = (q1_pred - q_target).pow(2).mean()
    q2_loss = (q2_pred - q_target).pow(2).mean()
    q_loss = q1_loss + q2_loss

    q_optimizer.zero_grad()
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(q1_net.parameters()) + list(q2_net.parameters()), 5.0)
    q_optimizer.step()

    action, log_prob = policy(states)
    q1_new = q1_net(states, action)
    q2_new = q2_net(states, action)
    q_new = torch.min(q1_new, q2_new)
    policy_loss = (alpha.detach() * log_prob - q_new).mean()

    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    optimizer.step()

    alpha_loss = (-alpha * (log_prob.detach() + target_entropy)).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    value_pred = value_net(states)
    with torch.no_grad():
        q1_value = q1_net(states, action)
        q2_value = q2_net(states, action)
        q_value = torch.min(q1_value, q2_value)
        value_target = q_value - alpha.detach() * log_prob

    value_loss = (value_pred - value_target).pow(2).mean()
    value_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 5.0)
    value_optimizer.step()

    soft_update(target_q1_net, q1_net, tau)
    soft_update(target_q2_net, q2_net, tau)
    soft_update(target_value_net, value_net, tau)

    return policy_loss.item(), q_loss.item(), value_loss.item(), log_prob.mean().item(), alpha.item(), alpha_loss.item()

# 7. 加载环境和初始化
env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
state_dim = 36
action_dim = 13

# 修复动作空间
model = env._model
try:
    ctrlrange = model.actuator_ctrlrange
    max_action = float(np.max(np.abs(ctrlrange)))
    action_space = Box(low=ctrlrange[:, 0], high=ctrlrange[:, 1], dtype=np.float32)
    print(f"Action space: {action_space}, Max action: {max_action}")
except Exception as e:
    print(f"Failed to get action_space/max_action: {e}")
    max_action = 1.0
    action_space = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    print(f"Using fallback: max_action={max_action}, action_space={action_space}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

policy = SACPolicy(state_dim, action_dim, max_action=max_action).to(device)
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
q_optimizer = optim.Adam(list(q1_net.parameters()) + list(q2_net.parameters()), lr=5e-5, weight_decay=1e-4)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-5, weight_decay=2e-4)
log_alpha = torch.tensor(math.log(0.1), requires_grad=True, device=device)
alpha_optimizer = optim.Adam([log_alpha], lr=5e-6)
target_entropy = -1.0
alpha = torch.exp(log_alpha).item()

replay_buffer = ReplayBuffer(capacity=1000000)
batch_size = 256

episode_rewards_record = []
policy_loss_record = []
q_loss_record = []
value_loss_record = []
stability_metrics = []
entropy_record = []
action_mean_record = []
action_std_record = []
reward_mean_record = []
reward_std_record = []
alpha_record = []
alpha_loss_record = []
episode_steps_counted_record = []

# 手动计步初始化
data = env._data
LEFT_FOOT_GEOM_NAME = 'foot_box_l'
RIGHT_FOOT_GEOM_NAME = 'foot_box_r'

GROUND_GEOM_NAME = 'floor'
# 在初始化后打印左右脚位置
# print(f"左足初始位置: {data.geom_xpos[left_foot_geom_id]}")
# print(f"右足初始位置: {data.geom_xpos[right_foot_geom_id]}")
left_foot_geom_id = -1
right_foot_geom_id = -1
ground_geom_id = -1

try:
    left_foot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FOOT_GEOM_NAME)
    right_foot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FOOT_GEOM_NAME)
    ground_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM_NAME)

    geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
    print(f"Available geom names: {geom_names}")

    if left_foot_geom_id == -1: print(f"Warning: Geom '{LEFT_FOOT_GEOM_NAME}' not found.")
    if right_foot_geom_id == -1: print(f"Warning: Geom '{RIGHT_FOOT_GEOM_NAME}' not found.")
    if ground_geom_id == -1: print(f"Warning: Geom '{GROUND_GEOM_NAME}' not found.")

    if left_foot_geom_id != -1 and right_foot_geom_id != -1 and ground_geom_id != -1:
        print(f"Manual step counting enabled. Foot/Ground IDs: left={left_foot_geom_id}, right={right_foot_geom_id}, ground={ground_geom_id}")
        manual_step_counting_enabled = True
    else:
        print("Manual step counting disabled: Failed to find all required geom IDs.")
        manual_step_counting_enabled = False
except Exception as e:
    print(f"Error during geom ID lookup: {e}")
    print("Manual step counting disabled.")
    manual_step_counting_enabled = False

# 填充回放缓冲区
print("Filling replay buffer with random actions...")
state = env.reset()
for _ in range(batch_size * 50):
    action = action_space.sample()
    action = np.clip(action, -0.5, 0.5)
    next_state, reward, done, info = env.step(action)
    replay_buffer.push(
        torch.tensor(state, dtype=torch.float32).unsqueeze(0),
        torch.tensor(action, dtype=torch.float32).unsqueeze(0),
        reward,
        torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
        done
    )
    state = next_state if not done else env.reset()
print(f"Replay buffer filled with {len(replay_buffer)} samples.")

# 8. SAC训练循环
num_episodes = 10000
max_steps = 1000
gamma = 0.99
action_norm_penalty_weight = 0.0005
survival_reward_amount = 10.0
step_reward_amount = 50.0
alternating_reward_amount = 0.0
cycle_reward_amount = 0.0

for episode in range(num_episodes):
    state = env.reset()
    # 修复初始姿态，调整脚部位置
    data.qpos[:] = 0
    data.qpos[2] = 0.9  # 骨盆初始高度
    data.qpos[3:7] = [0, 0, 0, 1]  # 骨盆旋转（四元数）
    # 原代码中左右腿初始角度相同，可能导致不平衡
    data.qpos[7:10] = [0.0, -0.2, 0.0]  # 左腿：hip, knee, ankle
    data.qpos[10:13] = [0.0, -0.2, 0.0]  # 右腿
    mujoco.mj_forward(model, data)

    # 动态调整骨盆高度直到脚部接近地面
    l_foot_z = data.geom_xpos[left_foot_geom_id][2]
    r_foot_z = data.geom_xpos[right_foot_geom_id][2]
    target_ground_z = 0.01
    max_adjustments = 100
    adjustment_step = 0.01
    adjustment_count = 0

    while (l_foot_z > target_ground_z or r_foot_z > target_ground_z) and data.qpos[
        2] > 0.5 and adjustment_count < max_adjustments:
        # 同时降低骨盆高度直到双脚都触地
        data.qpos[2] -= adjustment_step
        mujoco.mj_forward(model, data)
        l_foot_z = data.geom_xpos[left_foot_geom_id][2]
        r_foot_z = data.geom_xpos[right_foot_geom_id][2]
        adjustment_count += 1

    print(f"Episode {episode + 1}: Adjusted pelvis height to {data.qpos[2]:.3f}, Left foot z={l_foot_z:.3f}, Right foot z={r_foot_z:.3f}, Adjustments={adjustment_count}")

    # 检查初始接触
    initial_contacts = [c for c in data.contact if (
        (c.geom1 == left_foot_geom_id and c.geom2 == ground_geom_id) or
        (c.geom2 == left_foot_geom_id and c.geom1 == ground_geom_id) or
        (c.geom1 == right_foot_geom_id and c.geom2 == ground_geom_id) or
        (c.geom2 == right_foot_geom_id and c.geom1 == ground_geom_id)
    ) and c.dist < -0.005]
    print(f"Episode {episode + 1}: Initial foot contacts: {len(initial_contacts)}")

    state[0] = data.qpos[2]
    episode_reward = 0
    episode_stability = []
    raw_rewards_this_episode = []
    episode_actions = []
    total_steps_counted_episode = 0
    prev_left_foot_contact = False
    prev_right_foot_contact = False
    last_step_foot = None
    steps_since_last_contact = 0
    gait_cycle_count = 0
    prev_pelvis_tx = 0.0

    if episode % 5 == 0:
        print(f"Episode {episode + 1} initial state: height={state[0]:.3f}, tilt={state[1]:.3f}, list={state[2]:.3f}")
        print(f"Left foot pos: {data.geom_xpos[left_foot_geom_id]}, Right foot pos: {data.geom_xpos[right_foot_geom_id]}")
        print(f"Initial qpos: {data.qpos[:13]}")

    prev_action = np.zeros(action_dim)

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        if len(replay_buffer) < batch_size * 50 or episode < 5000:
            action_np = action_space.sample()
            action_np = np.clip(action_np, -0.5, 0.5)
        else:
            with torch.no_grad():
                action_tensor, _ = policy(state_tensor, deterministic=False)
            action_np = action_tensor.squeeze(0).cpu().numpy()

        next_state, reward, done, info = env.step(action_np)
        steps_since_last_contact += 1

        # 手动接触检测与奖励整形
        steps_gained_this_step = 0
        height_reward = 0
        posture_penalty = 0
        non_foot_contacts = []

        if manual_step_counting_enabled:
            data = env._data
            current_left_foot_contact = False
            current_right_foot_contact = False
            contact_count = 0
            contact_details = []

            for i in range(data.ncon):
                contact = data.contact[i]
                if (contact.geom1 == ground_geom_id or contact.geom2 == ground_geom_id) and contact.dist < -0.005:
                    other_geom_id = contact.geom1 if contact.geom2 == ground_geom_id else contact.geom2
                    other_geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id)
                    print(f"接触 {i}: {other_geom_name} (距离={contact.dist:.3f})")
                    contact_details.append(f"Contact {i}: geom1={contact.geom1} ({mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)}), geom2={contact.geom2} ({other_geom_name}), dist={contact.dist:.3f}")
                    if other_geom_id == left_foot_geom_id:
                        current_left_foot_contact = True
                        contact_count += 1
                    elif other_geom_id == right_foot_geom_id:
                        current_right_foot_contact = True
                        contact_count += 1
                    else:
                        non_foot_contacts.append(other_geom_name)

            # 计步逻辑
            if (current_left_foot_contact and not prev_left_foot_contact) or (current_right_foot_contact and not prev_right_foot_contact):
                steps_gained_this_step += 1.0
                steps_since_last_contact = 0

            total_steps_counted_episode += steps_gained_this_step
            reward += steps_gained_this_step * step_reward_amount

            if episode % 5 == 0:
                print(f"Episode {episode + 1}, Step {step + 1}: Contacts={contact_count}, Left={current_left_foot_contact}, Right={current_right_foot_contact}")
                for detail in contact_details:
                    print(detail)

            prev_left_foot_contact = current_left_foot_contact
            prev_right_foot_contact = current_right_foot_contact

        # 高度奖励
        pelvis_height = data.qpos[2]
        target_height = 0.9
        height_reward = -abs(pelvis_height - target_height) * 10.0
        if pelvis_height < 0.7:
            reward -= 100.0
            done = True
        reward += height_reward

        # 姿态惩罚
        posture_penalty = -np.sum(np.abs(data.qpos[3:7])) * 2.0
        reward += posture_penalty

        # 动作范数惩罚
        action_norm = np.linalg.norm(action_np)
        reward -= action_norm_penalty_weight * action_norm

        # 生存奖励
        if not done:
            reward += survival_reward_amount

        # 非脚部接触终止
        if non_foot_contacts:
            reward -= 50.0
            done = True
            if episode % 5 == 0:
                print(f"Non-foot contact detected: {non_foot_contacts}")

        raw_rewards_this_episode.append(reward)
        episode_reward += reward
        episode_actions.append(action_np)

        if step > 0:
            action_diff = np.linalg.norm(action_np - prev_action)
            episode_stability.append(action_diff)
        prev_action = action_np

        replay_buffer.push(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0),
            torch.tensor(action_np, dtype=torch.float32).unsqueeze(0),
            reward,
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
            done
        )

        state = next_state
        state[0] = data.qpos[2]

        if done and episode % 5 == 0:
            print(f"Episode {episode + 1} terminated: Height={pelvis_height:.3f}, Non-foot contacts={non_foot_contacts}, Info={info}")
            print(f"Termination qpos: {data.qpos[:13]}")
            print(f"Termination qvel: {data.qvel[:13]}")

        if len(replay_buffer) >= batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size, device)
            policy_loss, q_loss, value_loss, entropy, alpha, alpha_loss = sac_update(
                policy, q1_net, q2_net, target_q1_net, target_q2_net, value_net, target_value_net,
                optimizer, q_optimizer, value_optimizer, alpha_optimizer, log_alpha, target_entropy,
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones,
                gamma=gamma
            )
            policy_loss_record.append(policy_loss)
            q_loss_record.append(q_loss)
            value_loss_record.append(value_loss)
            entropy_record.append(entropy)
            alpha_record.append(alpha)
            alpha_loss_record.append(alpha_loss)

        if done:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Sim Steps: {step+1}, Actual Steps: {total_steps_counted_episode}, Height: {pelvis_height:.3f}")
            break

    stability = np.mean(episode_stability) if episode_stability else 0
    stability_metrics.append(stability)
    episode_rewards_record.append(episode_reward)
    episode_steps_counted_record.append(total_steps_counted_episode)

    if episode_actions:
        episode_actions = np.array(episode_actions)
        action_mean = np.mean(episode_actions)
        action_std = np.std(episode_actions)
        action_mean_record.append(action_mean)
        action_std_record.append(action_std)
    else:
        action_mean_record.append(0)
        action_std_record.append(0)

    if raw_rewards_this_episode:
        reward_mean = np.mean(raw_rewards_this_episode)
        reward_std = np.std(raw_rewards_this_episode)
        reward_mean_record.append(reward_mean)
        reward_std_record.append(reward_std)
    else:
        reward_mean_record.append(0)
        reward_std_record.append(0)

    entropy_val = entropy_record[-1] if entropy_record else 0.0
    alpha_val = alpha_record[-1] if alpha_record else torch.exp(log_alpha).item()
    if not done:
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Sim Steps: {step+1}, Actual Steps: {total_steps_counted_episode}, Entropy: {entropy_val:.2f}, Alpha: {alpha_val:.4f}, Height: {pelvis_height:.3f}")

# 9. 可视化
output_dir = r"E:\mujoco\table\sac_train_optimized_v6_515"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards_record, label="累积整形奖励", alpha=0.5)
avg_rewards = [np.mean(episode_rewards_record[max(0, i - 10):i + 1]) for i in range(len(episode_rewards_record))]
plt.plot(avg_rewards, label="平滑平均累积整形奖励（窗口=10）", linestyle="--")
plt.xlabel("训练轮数")
plt.ylabel("奖励")
plt.title("累积整形奖励变化曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "累积整形奖励变化曲线_515.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(policy_loss_record, label="策略损失", alpha=0.7)
plt.plot(q_loss_record, label="Q函数损失", alpha=0.7)
plt.plot(value_loss_record, label="价值损失", alpha=0.7)
plt.xlabel("更新次数")
plt.ylabel("损失")
plt.title("损失函数曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "损失函数曲线_515.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(stability_metrics, label="动作稳定性（动作差的均值）")
plt.xlabel("训练轮数")
plt.ylabel("稳定性")
plt.title("动作稳定性变化曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "动作稳定性曲线_515.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(entropy_record, label="策略熵", alpha=0.8)
plt.axhline(y=target_entropy, color='r', linestyle='--', label="目标熵")
plt.xlabel("更新次数")
plt.ylabel("熵")
plt.title("策略熵变化曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "策略熵变化曲线_515.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(action_mean_record, label="动作均值", alpha=0.8)
plt.plot(action_std_record, label="动作标准差", alpha=0.8)
plt.xlabel("训练轮数")
plt.ylabel("值")
plt.title("动作均值和标准差曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "动作均值方差曲线_515.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(reward_mean_record, label="奖励均值", alpha=0.8)
plt.plot(reward_std_record, label="奖励标准差", alpha=0.8)
plt.xlabel("训练轮数")
plt.ylabel("值")
plt.title("样本奖励分布曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "样本奖励分布曲线_515.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(alpha_record, label="Alpha值", alpha=0.8)
plt.xlabel("更新次数")
plt.ylabel("Alpha")
plt.title("Alpha值变化曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Alpha值变化曲线_515.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(alpha_loss_record, label="Alpha损失", alpha=0.8)
plt.xlabel("更新次数")
plt.ylabel("损失")
plt.title("Alpha损失变化曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Alpha损失变化曲线_515.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(episode_steps_counted_record, label="实际行走步数")
plt.xlabel("训练轮数")
plt.ylabel("步数")
plt.title("实际行走步数变化曲线")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "实际行走步数变化曲线_515.png"))
plt.close()

# 10. 保存模型
model_save_path = "sac_model_optimized_v6_515.pth"
torch.save(policy.state_dict(), model_save_path)
print(f"SAC policy model saved as {model_save_path}")

# 11. 测试并录制视频
def record_test(policy, env, video_dir, prefix, left_foot_geom_id, right_foot_geom_id, ground_geom_id):
    policy.eval()
    total_reward = 0
    max_steps_per_episode = 1000
    num_episodes = 10
    total_sim_steps = 0
    total_actual_steps = 0

    video_path = os.path.join(video_dir, f"{prefix}.mp4")
    os.makedirs(video_dir, exist_ok=True)
    print(f"Test video will be saved to: {video_path}")

    viewer = None
    video_writer = None
    try:
        model = env._model
        data = env._data
        viewer = mujoco.Renderer(model, width=640, height=480)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        if not video_writer.isOpened():
            print("Error: VideoWriter failed to initialize")
            video_writer = None
    except Exception as e:
        print(f"Warning: Could not initialize MuJoCo viewer/recorder: {e}")

    with torch.no_grad():
        for episode in range(num_episodes):
            state = env.reset()
            data.qpos[:] = 0
            data.qpos[2] = 0.9
            data.qpos[3:7] = [0, 0, 0, 1]
            data.qpos[7:10] = [0, -0.1, 0]
            data.qpos[10:13] = [0, -0.1, 0]
            mujoco.mj_forward(model, data)
            l_foot_z = data.geom_xpos[left_foot_geom_id][2]
            r_foot_z = data.geom_xpos[right_foot_geom_id][2]
            target_ground_z = 0.01
            max_adjustments = 100
            adjustment_step = 0.01
            adjustment_count = 0

            while (l_foot_z > target_ground_z or r_foot_z > target_ground_z) and data.qpos[2] > 0.5 and adjustment_count < max_adjustments:
                data.qpos[2] -= adjustment_step
                mujoco.mj_forward(model, data)
                l_foot_z = data.geom_xpos[left_foot_geom_id][2]
                r_foot_z = data.geom_xpos[right_foot_geom_id][2]
                adjustment_count += 1

            print(f"Test Episode {episode + 1}: Adjusted pelvis height to {data.qpos[2]:.3f}, Left foot z={l_foot_z:.3f}, Right foot z={r_foot_z:.3f}, Adjustments={adjustment_count}")
            state[0] = data.qpos[2]
            episode_reward = 0
            episode_sim_steps = 0
            test_total_steps_counted_episode = 0
            test_prev_left_foot_contact = False
            test_prev_right_foot_contact = False

            if left_foot_geom_id != -1 and right_foot_geom_id != -1 and ground_geom_id != -1:
                data = env._data
                initial_contacts = [c for c in data.contact if (
                    (c.geom1 == left_foot_geom_id and c.geom2 == ground_geom_id) or
                    (c.geom2 == left_foot_geom_id and c.geom1 == ground_geom_id) or
                    (c.geom1 == right_foot_geom_id and c.geom2 == ground_geom_id) or
                    (c.geom2 == right_foot_geom_id and c.geom1 == ground_geom_id)
                ) and c.dist < -0.005]
                test_prev_left_foot_contact = any(c.geom1 == left_foot_geom_id or c.geom2 == left_foot_geom_id for c in initial_contacts)
                test_prev_right_foot_contact = any(c.geom1 == right_foot_geom_id or c.geom2 == right_foot_geom_id for c in initial_contacts)

            print(f"\nStarting {prefix} Episode {episode + 1}/{num_episodes}")

            while episode_sim_steps < max_steps_per_episode:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy(state_tensor, deterministic=True).squeeze(0).cpu().numpy()
                next_state, reward, done, info = env.step(action)

                steps_gained_this_step = 0
                if left_foot_geom_id != -1 and right_foot_geom_id != -1 and ground_geom_id != -1:
                    data = env._data
                    current_left_foot_contact = False
                    current_right_foot_contact = False

                    for i in range(data.ncon):
                        contact = data.contact[i]
                        if (contact.geom1 == ground_geom_id or contact.geom2 == ground_geom_id) and contact.dist < -0.005:
                            other_geom_id = contact.geom1 if contact.geom2 == ground_geom_id else contact.geom2
                            if other_geom_id == left_foot_geom_id:
                                current_left_foot_contact = True
                            elif other_geom_id == right_foot_geom_id:
                                current_right_foot_contact = True

                    if (current_left_foot_contact and not test_prev_left_foot_contact) or (current_right_foot_contact and not test_prev_right_foot_contact):
                        steps_gained_this_step += 1.0

                    test_total_steps_counted_episode += steps_gained_this_step
                    test_prev_left_foot_contact = current_left_foot_contact
                    test_prev_right_foot_contact = current_right_foot_contact

                if not done:
                    reward += 10.0

                episode_reward += reward
                state = next_state
                state[0] = data.qpos[2]

                if viewer and video_writer:
                    try:
                        viewer.update_scene(env._data, camera="track")
                        frame = viewer.render()
                        if frame is not None and frame.size > 0:
                            if frame.shape != (480, 640, 3):
                                frame = cv2.resize(frame, (640, 480))
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            video_writer.write(frame)
                        else:
                            print(f"Warning: Empty frame at Episode {episode + 1}, Sim Step {episode_sim_steps + 1}")
                    except Exception as e:
                        print(f"Rendering/Recording error: {e}")
                        if video_writer:
                            video_writer.release()
                        video_writer = None

                episode_sim_steps += 1
                total_sim_steps += 1

                if done:
                    print(f"{prefix} Episode {episode + 1} terminated at sim step {episode_sim_steps}, Actual Steps: {test_total_steps_counted_episode}, Reward: {episode_reward:.2f}")
                    total_actual_steps += test_total_steps_counted_episode
                    break

            if episode_sim_steps >= max_steps_per_episode:
                print(f"{prefix} Episode {episode + 1} reached max sim steps, Actual Steps: {test_total_steps_counted_episode}, Reward: {episode_reward:.2f}")
                total_actual_steps += test_total_steps_counted_episode

            total_reward += episode_reward

    if video_writer:
        video_writer.release()
    elif viewer:
        print("Video recording was skipped or failed.")

    avg_reward_per_episode = total_reward / num_episodes if num_episodes > 0 else 0
    avg_actual_steps_per_episode = total_actual_steps / num_episodes if num_episodes > 0 else 0
    print(f"\n--- Test Summary ({prefix}) ---")
    print(f"Total Episodes Tested: {num_episodes}")
    print(f"Average Reward per Episode: {avg_reward_per_episode:.2f}")
    print(f"Average Actual Steps per Episode: {avg_actual_steps_per_episode:.2f}")
    print(f"Total Simulation Steps: {total_sim_steps}")
    print("----------------------------")

    return total_reward, total_sim_steps

print("\nStarting Test Phase:")
video_dir = r"E:\mujoco\new_test_videos_optimized_v6_515"
test_video_prefix = "sac_test_optimized_v6_515"
sac_reward, sac_sim_steps = record_test(policy, env, video_dir, test_video_prefix,
                                        left_foot_geom_id, right_foot_geom_id, ground_geom_id)

print("\nTraining and testing complete.")