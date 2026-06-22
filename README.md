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
├── portfolio_stability_diagnostics.py # 新增：稳定性诊断脚本
├── portfolio_robustness_sweep.py   # 新增：多 seed 与动作平滑鲁棒性评估
├── portfolio_generate_report.py    # 新增：自动生成实验报告与 SVG 图表
├── portfolio_extract_keyframes.py  # 新增：从展示视频提取关键帧总览图
├── portfolio_dagger_bc.py          # 新增：近似 DAgger 实验入口
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

## 已完成改进：归一化 + 闭环选模 + 稳定性诊断

针对原始 BC 闭环控制容易失稳的问题，新增 `portfolio_train_bc_enhanced.py`，实现了两项直接改进：

- **状态归一化**：使用训练集状态均值和标准差标准化输入，降低不同状态维度量纲差异。
- **动作归一化**：训练网络预测标准化动作，再在部署时反归一化回 MuJoCo 控制动作。
- **闭环 rollout 选模**：训练过程中定期在 MuJoCo 中真实 rollout，用平均存活步数选择最佳 checkpoint，避免只按验证集 MSE 选模型。
- **稳定性诊断**：额外统计躯干高度、动作幅度和动作变化量，用于判断策略是否只是偶然跑得远。

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

## 稳定性诊断

运行方式：

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_stability_diagnostics.py
```

诊断输出：

- `portfolio_retrain_bc_improved/stability_diagnostics.csv`
- `portfolio_retrain_bc_improved/stability_summary.csv`

当前增强版策略的 10 回合稳定性诊断结果：

| 指标 | 数值 |
| --- | ---: |
| 平均步数 | 820.3 |
| 最好步数 | 1000 |
| 跑满 1000 步回合数 | 4/10 |
| 平均奖励 | 756.08 |
| 平均动作 L2 范数 | 0.540 |
| 平均动作变化量 | 0.100 |

诊断结果与主评估结果同量级，说明增强版策略的提升不是单个视频偶然现象。

## 鲁棒性与动作平滑 Sweep

新增 `portfolio_robustness_sweep.py`，用于在多个随机种子和不同动作平滑系数下重复评估增强版 BC 策略：

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_robustness_sweep.py --num-seeds 3 --episodes-per-seed 2 --smoothing-values 0.0,0.2,0.4 --max-steps 1000
```

输出文件：

- `portfolio_retrain_bc_improved/robustness_sweep.csv`
- `portfolio_retrain_bc_improved/robustness_summary.csv`

实测结果：

| 动作平滑系数 | 回合数 | 平均步数 | 步数标准差 | 最好步数 | 跑满 1000 步比例 | 平均奖励 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 6 | 930.2 | 79.9 | 1000 | 50.0% | 853.36 |
| 0.2 | 6 | 333.3 | 151.6 | 528 | 0.0% | 298.32 |
| 0.4 | 6 | 164.2 | 54.0 | 269 | 0.0% | 147.74 |

该实验说明：对当前策略直接做动作平滑后处理并不会提升稳定性，反而破坏了已学到的闭环步态节律。因此最终展示策略保留原始归一化 BC 输出，动作平滑应作为训练目标或奖励约束设计，而不是简单部署后处理。

## 自动实验报告

新增 `portfolio_generate_report.py`，可以从已有 CSV 自动生成项目总结和可视化图表：

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_generate_report.py
```

输出文件：

- `portfolio_retrain_bc_improved/portfolio_summary.md`
- `portfolio_retrain_bc_improved/report_steps_comparison.svg`
- `portfolio_retrain_bc_improved/report_training_loss.svg`
- `portfolio_retrain_bc_improved/report_rollout_selection.svg`
- `portfolio_retrain_bc_improved/report_stability_diagnostics.svg`
- `portfolio_retrain_bc_improved/report_robustness_sweep.svg`

图表脚本使用 Python 标准库直接写 SVG，不依赖 Matplotlib，避免部分 Windows/conda 环境中绘图库原生崩溃的问题。

## 展示视频关键帧

为了避免只凭一个视频文件难以判断效果，新增 `portfolio_extract_keyframes.py`，可以从最终展示视频中抽取多个时间点并生成关键帧总览图：

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_extract_keyframes.py --video portfolio_retrain_bc_improved/bc_improved_demo.mp4 --output-dir portfolio_retrain_bc_improved/keyframes --num-frames 8 --columns 4
```

输出文件：

- `portfolio_retrain_bc_improved/keyframes/bc_improved_keyframes.png`
- `portfolio_retrain_bc_improved/keyframes/bc_improved_keyframes.csv`

当前最终展示视频共 8392 帧，30 FPS，约 279.7 秒。关键帧覆盖 14.0s 到 265.7s，能直观看到策略在较长时间范围内保持连续步态，而不是只展示摔倒前的短片段。

## 实验入口：近似 DAgger 数据聚合

新增 `portfolio_dagger_bc.py` 作为 DAgger 方向的实验入口。它会：

1. 使用当前增强版 BC 策略在 MuJoCo 中 rollout。
2. 收集策略偏离专家轨迹后的访问状态。
3. 用专家数据集中最近邻状态的动作作为近似专家标签。
4. 合并专家数据和访问状态数据继续训练。

说明：这不是严格意义上的 DAgger，因为当前项目没有在线专家控制器，只能用专家数据最近邻近似标注。该脚本已作为后续研究入口保留，但未纳入主结果；主结果仍以已验证的归一化 + 闭环选模 BC 为准。

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
- 将近似 DAgger 扩展为真正带专家控制器的在线数据聚合。
- 优化奖励函数，加入躯干姿态、足底接触、动作平滑和前向速度约束。
- 将动作平滑从部署后处理改为训练期正则项或奖励项，避免破坏已有闭环步态节律。
- 将 3/10 个满步回合提升到 10/10 个回合都稳定跑满 1000 步。
- 对不同随机种子重复实验，进一步验证增强版结果的稳定性。

## 简历写法示例

**基于强化学习与模仿学习的人形机器人行走控制实验系统**

- 基于 MuJoCo / Loco-MuJoCo 搭建人形机器人连续控制环境，完成 36 维状态到 13 维连续动作的策略学习与仿真评估。
- 使用 PyTorch 实现 BC、PPO、SAC 等策略网络，基于专家轨迹和奖励反馈训练人形机器人短时行走控制策略。
- 构建统一评估流程，支持模型权重加载、平均步数/奖励统计、MuJoCo 渲染录制和实验结果可视化。
- 完成状态/动作归一化与闭环 rollout 选模，将 BC 评估平均步数从 97.5 提升到 839.2，最好步数从 133 提升到 1000。
- 增加稳定性诊断指标，统计躯干高度、动作幅度和动作变化量，辅助分析策略长期失稳原因。
- 设计多 seed 鲁棒性与动作平滑 sweep 实验，验证原始增强版 BC 在 6 回合补充评估中平均达到 930.2 步，并发现简单动作平滑会显著削弱步态稳定性。
- 编写自动报告生成脚本，将训练曲线、闭环选模、稳定性诊断和鲁棒性 sweep 输出为 Markdown 与 SVG 图表，提升实验复现和展示效率。
- 生成最终展示视频关键帧总览图，用 14.0s 至 265.7s 的多时间点画面辅助说明模型具备较长时间连续行走能力。
- 对比不同算法在短时步态稳定性和存活步数上的表现，分析归一化、训练轮数、验证集 MSE 和闭环控制稳定性之间的关系。
