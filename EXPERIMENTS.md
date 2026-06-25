# 实验复现与结果说明

本文档整理项目中推荐使用的复现实验命令、产物位置和结论边界。主结果以 `portfolio_retrain_bc_improved/` 中保存的增强版 BC 为准。

## 快速检查

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_cli.py validate
```

该命令会检查 README、核心脚本、模型权重、视频、关键帧、CSV 指标和视频元数据，并输出：

- `portfolio_retrain_bc_improved/artifact_validation.json`
- `portfolio_retrain_bc_improved/artifact_validation.csv`

无需安装 MuJoCo / PyTorch / OpenCV 的轻量静态检查：

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_cli.py static-check
```

该命令也是 GitHub Actions 的检查入口。

## 展示材料生成

```powershell
C:\Users\ASUS\.conda\envs\rl_env\python.exe portfolio_cli.py showcase
```

该命令会依次生成最终视频关键帧、Markdown/SVG 报告，并校验核心产物。

## 主实验命令

| 目的 | 命令 | 主要输出 |
| --- | --- | --- |
| 原始 BC 重训 | `python portfolio_train_bc.py --epochs 800 --eval-episodes 10 --output-dir portfolio_retrain_bc` | `portfolio_retrain_bc/` |
| 增强 BC 闭环选模 | `python portfolio_train_bc_enhanced.py --epochs 80 --eval-episodes 10 --output-dir portfolio_retrain_bc_improved --noise-std 0.0 --select-by-rollout --selection-eval-every 5 --selection-episodes 5 --selection-max-steps 1000` | `portfolio_retrain_bc_improved/` |
| 稳定性诊断 | `python portfolio_stability_diagnostics.py` | `stability_*.csv` |
| 鲁棒性 sweep | `python portfolio_robustness_sweep.py --num-seeds 3 --episodes-per-seed 2 --smoothing-values 0.0,0.2,0.4 --max-steps 1000` | `robustness_*.csv` |
| 自动报告 | `python portfolio_generate_report.py` | `portfolio_summary.md` 与 SVG 图 |
| 视频关键帧 | `python portfolio_extract_keyframes.py` | `keyframes/bc_improved_keyframes.png` |

## 当前主结果

| 指标 | 数值 |
| --- | ---: |
| 原始 BC 平均步数 | 97.5 |
| 增强版 BC 平均步数 | 839.2 |
| 增强版 BC 最好步数 | 1000 |
| 稳定性诊断平均步数 | 820.3 |
| 鲁棒性 sweep 原始动作平均步数 | 930.2 |
| 最终展示视频长度 | 8392 帧，约 279.7 秒 |

## 结论边界

- 已完成并验证：状态归一化、动作归一化、闭环 rollout 选模、稳定性诊断、鲁棒性 sweep、自动报告、关键帧展示。
- 已提供实验入口但不作为主结果：近似 DAgger 数据聚合。
- 尚未作为主结果声称完成：真正在线专家 DAgger、PPO/SAC 长时间微调、奖励函数系统重设计。

这样的表述更适合简历和答辩：它既展示了工程闭环和可复现实验，也避免把未充分验证的方向包装成最终成果。
