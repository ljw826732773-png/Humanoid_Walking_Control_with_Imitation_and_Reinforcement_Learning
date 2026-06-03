import gym
import numpy as np
# import loco_mujoco
#
# # 创建环境
# env = loco_mujoco.LocoEnv.make("HumanoidTorque.run", dataset_type="perfect")
# obs = env.reset()
#
# # 获取专家数据集（通过私有变量）
# expert_dataset = env._dataset
#
# # 尝试访问底层 mujoco 环境，获取关节名
# if hasattr(env, "_env") and hasattr(env._env, "sim"):
#     sim = env._env.sim
#     joint_names = sim.model.actuator_names
#     print("=== 关节名称（动作对应） ===")
#     for i, name in enumerate(joint_names):
#         print(f"{i}: {name.decode('utf-8')}")
# else:
#     print("无法访问 sim 模拟器，可能不是标准 MuJoCo 环境。")
#
# # 打印专家动作数据（前5帧）
# print("\n=== 专家数据集中的动作样本（前5帧） ===")
# if "actions" in expert_dataset:
#     actions = expert_dataset["actions"]
#     for i in range(min(5, len(actions))):
#         print(f"第{i}帧动作: {actions[i]}")
# else:
#     print("未在专家数据集中找到 'actions' 字段")
#
# print("专家数据集字段:", expert_dataset.keys())
#
# # 打印专家观察数据（前5帧）
# print("\n=== 专家数据集中的状态值（前5帧） ===")
# for i in range(5):
#     print(f"第{i}帧状态: {expert_dataset['states'][i]}")
#
# state = expert_dataset["states"][0]
# print(f"状态向量总长度: {len(state)}")
# state = expert_dataset['states'][0]
#
# print(f"=== 状态向量总长度: {len(state)} ===\n")
# print("=== 状态各维度含义（推测） ===")
#
# # 如果能访问 sim.model（可能无法）
# model = env._env.model
# for i, name in enumerate(model.joint_names):
#     print(f"joint[{i}] = {name}")
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体为宋体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun']  # Windows系统宋体
plt.rcParams['font.size'] = 14  # 全局字体大小

# 创建数据和标签
algorithms = ['PPO', 'SAC', '行为克隆']
time_steps = [8.8, 15.9, 68.3]

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
ax.set_facecolor('white')

# 绘制柱状图 - 使用浅蓝色 (#87CEEB)
bars = ax.bar(algorithms, time_steps, color='#87CEEB', width=0.6)

# 设置标题和标签，调整标题的 pad 参数以更靠近上界
ax.set_title('三种算法测试平均时间步数比较', fontsize=13, pad=5)
ax.set_xlabel('使用算法', fontsize=13, labelpad=5)
ax.set_ylabel('时间步数', fontsize=13, labelpad=5)

# 设置纵轴范围
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))

# 在柱子顶部添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=13)

# 调整布局
plt.tight_layout()

# 创建保存目录（如果不存在）
save_dir = r"E:\mujoco\table"
os.makedirs(save_dir, exist_ok=True)

# 保存图片到指定路径
save_path = os.path.join(save_dir, "algorithm_comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {save_path}")
# 显示图形（可选）
# plt.show()
