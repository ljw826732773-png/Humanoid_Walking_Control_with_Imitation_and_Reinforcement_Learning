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
import matplotlib.pyplot as plt
import math

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 默认中文字体
plt.rcParams['font.size'] = 13  # 五号字体，10pt
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 加载环境和完美数据集
env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
dataset = env.create_dataset()

# 提取状态和动作
states = dataset["states"]
actions = dataset["actions"]
print(f"数据集加载完成：{states.shape} 个状态，{actions.shape} 个动作")

# 打印关节名称和索引
print("\n关节名称和索引：")
model_mujoco = env._model
for i in range(model_mujoco.njnt):
    joint_name = model_mujoco.joint(i).name
    print(f"关节 {i}: {joint_name}")

# 调试：打印状态和关节角度的维度
print(f"模型中的关节数量：{model_mujoco.nq}")
data = env._data
print(f"data.qpos 的形状：{data.qpos.shape}")

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

# 5. 训练模型并记录损失
num_epochs = 800
best_val_loss = float('inf')
train_losses = []
val_losses = []

# 定义 table_dir
table_dir = r"E:\mujoco\table\bc_better_step"
os.makedirs(table_dir, exist_ok=True)

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

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "bc_best_model.pth")

    print(
        f"轮次 {epoch + 1}/{num_epochs}, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.title('行为克隆训练和验证损失曲线')
plt.xlabel('轮次')
plt.ylabel('均方误差损失')
plt.legend(frameon=False)
plt.grid(True)
plt.savefig(os.path.join(table_dir, '行为克隆训练和验证损失曲线.png'))  # 保存到 table_dir
plt.close()

# 6. 保存最后一个模型
torch.save(model.state_dict(), "bc_model.pth")
print("最后一个模型保存为 bc_model.pth")
print("最佳模型保存为 bc_best_model.pth")


# 7. 测试并录制视频，同时保存最长回合的帧和关节信息
def record_test(model, env, video_dir, table_dir, prefix="bc"):
    model.eval()
    total_reward = 0
    max_steps_per_episode = 1000
    num_episodes = 10
    total_steps = 0
    max_steps = 0
    max_episode_frames = []
    max_episode = 0
    max_episode_joint_angles = []
    max_episode_actions = []
    max_episode_states = []
    episode_steps = []  # 记录每回合的步数

    video_path = os.path.join(video_dir, f"{prefix}.mp4")
    table_dir = os.path.join(table_dir, f"{prefix}_step")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    print(f"视频将保存至：{video_path}")

    model_mujoco = env._model
    data = env._data
    viewer = mujoco.Renderer(model_mujoco, width=640, height=480)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    if not video_writer.isOpened():
        print("错误：VideoWriter 初始化失败")
        return total_reward, total_steps

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
            episode_steps.append(step)  # 记录回合步数
            if step > max_steps:
                max_steps = step
                max_episode_frames = episode_frames
                max_episode = episode + 1
                max_episode_joint_angles = episode_joint_angles
                max_episode_actions = episode_actions
                max_episode_states = episode_states

            if step >= max_steps_per_episode:
                print(f"{prefix} 第 {episode + 1} 回合达到最大步数，奖励: {episode_reward:.2f}")

    video_writer.release()
    print(f"{prefix} 在 {num_episodes} 回合中的总奖励: {total_reward:.2f}, 总步数: {total_steps}")

    # 保存最长回合的帧为4x6大图（横4竖6，每3步抽帧）
    if max_episode_frames:
        frames_per_image = 24  # 4x6
        sampled_frames = max_episode_frames[::3]  # 每三个时间步取一帧
        num_images = math.ceil(len(sampled_frames) / frames_per_image)

        for img_idx in range(num_images):
            fig, axes = plt.subplots(6, 4, figsize=(16, 24))  # 横4竖6
            for i in range(6):
                for j in range(4):
                    frame_idx = img_idx * frames_per_image + i * 4 + j
                    if frame_idx < len(sampled_frames):
                        axes[i, j].imshow(cv2.cvtColor(sampled_frames[frame_idx], cv2.COLOR_BGR2RGB))
                        axes[i, j].set_title(f'帧 {frame_idx * 3 + 1}')
                        axes[i, j].axis('off')
                    else:
                        axes[i, j].axis('off')
            plt.suptitle(f'行为克隆第 {max_episode} 回合关键帧（图 {img_idx + 1}/{num_images}）', fontsize=16)
            plt.savefig(os.path.join(table_dir, f'第{max_episode}回合_关键帧_{img_idx + 1}.png'))
            plt.close()

            # 绘制十轮测试的时间步曲线
            plt.figure(figsize=(10, 6))
            episodes = np.arange(1, num_episodes + 1)
            plt.plot(episodes, episode_steps, label='每回合时间步数', marker='o')
            avg_steps = np.mean(episode_steps)
            plt.axhline(y=avg_steps, color='r', linestyle='--', label=f'平均时间步数: {avg_steps:.2f}')
            plt.title('行为克隆十轮测试时间步曲线')
            plt.xlabel('回合')
            plt.ylabel('时间步数')
            plt.legend(frameon=False)
            plt.grid(True)
            plt.savefig(os.path.join(table_dir, '行为克隆十轮测试时间步曲线.png'))
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
        closest_traj_idx = find_closest_expert_trajectory(max_episode_states, states, len(max_episode_states))
        expert_joint_angles = states[closest_traj_idx:closest_traj_idx + len(max_episode_states), :19]

        # 调试：打印维度
        print(f"预测关节角度的形状：{joint_angles.shape}")
        print(f"专家关节角度的形状：{expert_joint_angles.shape}")

        # 第一张图：关节6和11
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, joint_angles[:, 6], label=f'预测关节 {model_mujoco.joint(6).name}')
        plt.plot(time_steps, joint_angles[:, 11], label=f'预测关节 {model_mujoco.joint(11).name}')
        plt.plot(time_steps, expert_joint_angles[:, 6], '--', label=f'专家关节 {model_mujoco.joint(6).name}')
        plt.plot(time_steps, expert_joint_angles[:, 11], '--', label=f'专家关节 {model_mujoco.joint(11).name}')
        plt.title(f'行为克隆第 {max_episode} 回合关节角度（关节6和11）')
        plt.xlabel('时间步')
        plt.ylabel('关节角度（弧度）')
        plt.legend(frameon=False)
        plt.grid(True)
        plt.savefig(os.path.join(table_dir, f'第{max_episode}回合_关节角度_6_11.png'))
        plt.close()

        # 第二张图：关节9和14
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, joint_angles[:, 9], label=f'预测关节 {model_mujoco.joint(9).name}')
        plt.plot(time_steps, joint_angles[:, 14], label=f'预测关节 {model_mujoco.joint(14).name}')
        plt.plot(time_steps, expert_joint_angles[:, 9], '--', label=f'专家关节 {model_mujoco.joint(9).name}')
        plt.plot(time_steps, expert_joint_angles[:, 14], '--', label=f'专家关节 {model_mujoco.joint(14).name}')
        plt.title(f'行为克隆第 {max_episode} 回合关节角度（关节9和14）')
        plt.xlabel('时间步')
        plt.ylabel('关节角度（弧度）')
        plt.legend(frameon=False)
        plt.grid(True)
        plt.savefig(os.path.join(table_dir, f'第{max_episode}回合_关节角度_9_14.png'))
        plt.close()

        # 绘制专家轨迹复现度（L2距离）
        expert_actions = actions[closest_traj_idx:closest_traj_idx + len(max_episode_actions)]
        pred_actions = np.array(max_episode_actions)
        l2_distances = np.linalg.norm(pred_actions - expert_actions, axis=1)
        avg_l2_distance = np.mean(l2_distances)

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, l2_distances, label='预测动作与专家动作的L2距离')
        plt.axhline(y=avg_l2_distance, color='r', linestyle='--', label=f'平均L2距离: {avg_l2_distance:.4f}')
        plt.title(f'行为克隆第 {max_episode} 回合专家轨迹复现准确度')
        plt.xlabel('时间步')
        plt.ylabel('L2距离')
        plt.legend(frameon=False)
        plt.grid(True)
        plt.savefig(os.path.join(table_dir, f'行为克隆第{max_episode}回合_动作L2距离.png'))
        plt.close()


    return total_reward, total_steps


# 执行测试并录制视频
video_dir = r"E:\mujoco\new_test_videos"
print("\n测试行为克隆模型并录制视频：")
bc_reward, bc_steps = record_test(model, env, video_dir, table_dir, "bc_better")