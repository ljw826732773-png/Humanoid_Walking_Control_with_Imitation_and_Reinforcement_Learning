# 基于强化学习与模仿学习的人形机器人行走控制

本项目基于 MuJoCo / Loco-MuJoCo 搭建人形机器人连续控制实验环境，围绕 `HumanoidTorque.run` 任务实现并评估行为克隆（BC）、PPO、SAC 等策略在人形机器人短时步态控制中的表现。

项目重点不是宣称“已经实现长期稳定行走”，而是完整展示一个强化学习/模仿学习控制实验的工程流程：环境搭建、专家数据读取、策略网络训练、模型评估、视频录制、指标统计和实验对比。

## 项目亮点

- 基于 Loco-MuJoCo 构建 36 维状态到 13 维连续动作的人形机器人控制任务。
- 使用 PyTorch 实现行为克隆策略网络，并完成 800 epoch 完整重训。
- 整理 PPO、SAC、BC 等多组模型权重和测试结果，对比不同算法的短时步态表现。
- 新增简历展示用评估脚本，可自动加载权重、统计平均步数/奖励并录制 MuJoCo 视频。
- 输出训练日志、评估指标、展示视频和 README，便于复现实验与面试讲解。

## 技术栈

- Python
- PyTorch
- MuJoCo / Loco-MuJoCo
- Gymnasium / Stable-Baselines3
- OpenCV
- Matplotlib
- 行为克隆、PPO、SAC、连续动作控制、仿真视频录制

## 任务设置

| 项目 | 内容 |
| --- | --- |
| 环境 | `HumanoidTorque.run` |
| 状态维度 | 36 |
| 动作维度 | 13 |
| 数据来源 | Loco-MuJoCo perfect expert dataset |
| 控制目标 | 根据机器人状态输出连续关节控制动作，使人形机器人尽可能稳定地向前移动 |

## 目录结构

```text
.
├── bc_4.py                         # 原始 BC 训练与测试脚本
├── ppo_v8.py                       # PPO 训练与测试脚本
├── sac_11.py                       # SAC 训练与测试脚本
├── portfolio_train_bc.py           # 新增：独立 BC 重训脚本
├── portfolio_train_bc_enhanced.py  # 新增：归一化增强版 BC 训练脚本
├── portfolio_evaluate.py           # 新增：统一模型评估脚本
├── requirements_portfolio.txt      # 当前验证可运行的核心依赖
├── *.pth                           # 已训练模型权重
├── new_test_videos/                # 原项目测试视频
├── table/                          # 训练曲线、测试曲线、关键帧图
├── portfolio_results/              # 简历展示/对比评估结果
├── portfolio_retrain_bc/           # 800 epoch BC 重训结果
└── portfolio_retrain_bc_improved/  # 归一化增强版 BC 最佳结果
```

## 环境准备

本机验证可用环境为：

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe
```

安装核心依赖：

```powershell
python -m pip install -r requirements_portfolio.txt
```

注意：`loco-mujoco==0.3.0` 依赖 `mujoco==2.3.7`。如果直接安装 `mujoco==3.3.3`，会和 `loco-mujoco==0.3.0` 产生版本冲突。

## 一键评估已有权重

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_evaluate.py --episodes 10 --max-steps 1000 --demo-model bc_best
```

运行后会生成：

- `portfolio_results/evaluation_metrics.csv`
- `portfolio_results/evaluation_metrics.json`
- 多回合调试视频

说明：多回合调试视频会把多个 episode 连续录在一起，因此会出现摔倒和 reset；它适合调试，不适合作为简历主展示视频。

## 完整重训 BC

为了验证项目不只是依赖已有视频，新增了独立 BC 重训脚本：

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_train_bc.py --epochs 800 --eval-episodes 10 --output-dir portfolio_retrain_bc
```

该脚本不会覆盖原始模型，会把重训产物单独保存到 `portfolio_retrain_bc/`：

- `bc_retrained_best.pth`
- `bc_retrained_last.pth`
- `bc_training_log.csv`
- `bc_retrained_eval.csv`
- `bc_retrained_demo.mp4`

## 重训结果

在 CPU 上完成一次 800 epoch BC 重训，实测结果如下：

| 指标 | 数值 |
| --- | ---: |
| 训练耗时 | 约 9 分 55 秒 |
| 评估回合 | 10 |
| 平均步数 | 97.5 |
| 最好步数 | 133 |
| 平均奖励 | 93.76 |
| 最好奖励 | 128.35 |
| 评估视频 | `portfolio_retrain_bc/bc_retrained_demo.mp4` |
| 视频信息 | 975 帧，30 FPS，约 32.5 秒，640x480 |

各回合评估结果：

| Episode | Steps | Reward |
| ---: | ---: | ---: |
| 1 | 80 | 77.09 |
| 2 | 70 | 68.08 |
| 3 | 133 | 128.35 |
| 4 | 68 | 66.30 |
| 5 | 102 | 98.70 |
| 6 | 119 | 113.68 |
| 7 | 105 | 100.05 |
| 8 | 95 | 90.80 |
| 9 | 99 | 95.36 |
| 10 | 104 | 99.18 |

## 已完成改进：归一化 + 闭环选模

针对原始 BC 闭环控制容易失稳的问题，新增 `portfolio_train_bc_enhanced.py`，实现了两项直接改进：

- **状态归一化**：使用训练集状态均值和标准差标准化输入，降低不同状态维度量纲差异。
- **动作归一化**：训练网络预测标准化动作，再在部署时反归一化回 MuJoCo 控制动作。
- **闭环 rollout 选模**：训练过程中定期在 MuJoCo 中真实 rollout，用平均存活步数选择最佳 checkpoint，避免只按验证集 MSE 选模型。

运行方式：

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_train_bc_enhanced.py
```

默认配置为当前实测稳定设置：

- `epochs=50`
- `noise_std=0.0`
- `eval_episodes=10`
- 输出目录：`portfolio_retrain_bc_improved/`

若要复现实测最优结果，可运行闭环选模版本：

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_train_bc_enhanced.py --epochs 80 --eval-episodes 10 --output-dir portfolio_retrain_bc_improved --noise-std 0.0 --select-by-rollout --selection-eval-every 5 --selection-episodes 5 --selection-max-steps 1000
```

最终增强版结果如下：

| 指标 | 原始 BC 重训 | 增强版 BC |
| --- | ---: | ---: |
| 训练轮数 | 800 | 80 |
| 状态/动作归一化 | 否 | 是 |
| 闭环 rollout 选模 | 否 | 是 |
| 平均步数 | 97.5 | 839.2 |
| 最好步数 | 133 | 1000 |
| 跑满 1000 步回合数 | 0/10 | 3/10 |
| 平均奖励 | 93.76 | 776.75 |
| 最好奖励 | 128.35 | 951.50 |
| 评估视频 | `portfolio_retrain_bc/bc_retrained_demo.mp4` | `portfolio_retrain_bc_improved/bc_improved_demo.mp4` |

增强版各回合结果：

| Episode | Steps | Reward |
| ---: | ---: | ---: |
| 1 | 731 | 680.52 |
| 2 | 720 | 650.11 |
| 3 | 740 | 682.95 |
| 4 | 1000 | 950.69 |
| 5 | 1000 | 944.27 |
| 6 | 1000 | 951.50 |
| 7 | 817 | 732.12 |
| 8 | 869 | 777.40 |
| 9 | 712 | 658.30 |
| 10 | 803 | 739.68 |

本次改进把平均步数从 97.5 提升到 839.2，最好步数从 133 提升到 1000，并有 3/10 个评估回合跑满 1000 步，已经从“短时步态”提升到“接近稳定行走”的阶段。

对照实验也显示：并不是训练越久越好。增强版在 100 epoch 和 800 epoch 设置下闭环控制反而退化，说明该任务中验证集 MSE 更低不一定意味着仿真闭环更稳定，需要用实际 episode 步数作为最终评估指标。

## 已有权重评估结果

使用 `portfolio_evaluate.py --episodes 10 --max-steps 1000 --demo-model bc_best` 对现有权重进行评估：

| 模型 | 权重文件 | 平均步数 | 最好步数 | 平均奖励 | 最好奖励 |
| --- | --- | ---: | ---: | ---: | ---: |
| BC last | `bc_model.pth` | 68.3 | 85 | 65.70 | 81.79 |
| BC best | `bc_best_model.pth` | 80.3 | 123 | 76.53 | 118.54 |
| PPO best steps | `best_ppo_model.pth` | 4.4 | 7 | 3.56 | 5.89 |
| PPO best reward | `best_reward_model.pth` | 3.7 | 4 | 2.99 | 3.90 |
| PPO longest | `longest_steps_model.pth_11` | 6.4 | 9 | 4.68 | 6.55 |
| SAC | `sac_model.pth` | 15.7 | 24 | 10.47 | 18.26 |

## 结果分析

从当前实验结果看，BC 在该任务上明显优于当前 PPO/SAC 权重。原因是 BC 直接从专家轨迹中学习动作映射，在短时步态复现上更稳定；而 PPO/SAC 需要从奖励信号中探索高维连续控制策略，训练难度更高，当前权重仍较容易失稳。

BC 的局限也很明显：它能学到短时步态，但长期稳定性不足。模型一旦进入专家数据覆盖较少的状态，误差会逐步累积，最终导致摔倒。这是模仿学习中常见的 distribution shift 问题。

因此，本项目更适合表述为“人形机器人行走控制算法实验与对比”，而不是“成熟稳定的人形机器人行走控制系统”。

## 后续可改进方向

- 使用增强版 BC 作为预训练策略，再接 PPO/SAC 进行强化学习微调。
- 使用 DAgger 或在线数据聚合缓解 BC 的分布偏移问题。
- 优化奖励函数，加入躯干姿态、足底接触、动作平滑和前向速度约束。
- 将 3/10 个满步回合提升到 10/10 个回合都稳定跑满 1000 步。
- 对不同随机种子重复实验，进一步验证增强版结果的稳定性。

## 简历写法示例

**基于强化学习与模仿学习的人形机器人行走控制实验系统**

- 基于 MuJoCo / Loco-MuJoCo 搭建人形机器人连续控制环境，完成 36 维状态到 13 维连续动作的策略学习与仿真评估。
- 使用 PyTorch 实现 BC、PPO、SAC 等策略网络，基于专家轨迹和奖励反馈训练人形机器人短时行走控制策略。
- 构建统一评估流程，支持模型权重加载、平均步数/奖励统计、MuJoCo 渲染录制和实验结果可视化。
- 完成状态/动作归一化与闭环 rollout 选模，将 BC 评估平均步数从 97.5 提升到 839.2，最好步数从 133 提升到 1000。
- 对比不同算法在短时步态稳定性和存活步数上的表现，分析归一化、训练轮数、验证集 MSE 和闭环控制稳定性之间的关系。
