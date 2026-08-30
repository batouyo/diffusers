# FLUX-Kontext 早期轨迹搜索与 LoRA 蒸馏最小验证报告

## 结论摘要

本实验验证了一个最小闭环：在 Flux-Kontext 生成的最早两个扩散步骤分别生成 4 个候选，用官方图像编辑评分模型筛选更好的候选，再用筛选出的轨迹训练一个小型 LoRA。

结论是：早期候选搜索在当前 8 个测试样本上有初步正向趋势，但 LoRA 强度存在明显波动，不能宣称强度越大越好。当前结果适合作为机制验证，不足以作为大规模统计结论。

## 已完成

- PIE-Bench：16 个训练样本、8 个测试样本。
- source 与生成图像统一为 512x512；另完成 1024x1024 基准对照。
- 前两个有效扩散步骤为调度器索引 1、2。
- 修复了直接套用同步随机微分方程漂移公式导致图像退化的问题。
- 完成 P1 两个生成种子，共 16 条样本记录。
- 完成 16/16 个官方评分筛选的教师缓存。
- 完成 rank=4、250 步 selective LoRA 训练。
- 完成 LoRA 加载、禁用和重复性验证。
- 完成 P3 四种 LoRA 强度，共 64 条结果记录。

## 分辨率对照

| 设置 | 官方评分 |
|---|---:|
| 512x512 基准 | 8.1388 |
| 1024x1024 基准 | 7.2664 |

后续主实验统一使用 512x512。生成图像和 source conditioning 均记录为 1024 个图像 token。

## P1：早期搜索对照

| 方法 | 数量 | 平均评分 | 标准差 |
|---|---:|---:|---:|
| 普通基准编辑 | 16 | 4.3413 | 3.4137 |
| 独立随机噪声 | 16 | 4.8393 | 3.2048 |
| 共享区域噪声 | 16 | 4.7430 | 3.1103 |
| 官方评分筛选的早期搜索 | 16 | 5.4299 | 3.2057 |

官方评分筛选的早期搜索比普通基准高约 1.09 分，满足进入 LoRA 阶段的弱正向门禁。但样本量小、方差大，结论仍属初步趋势。

## P2：LoRA

训练参数：rank=4、dropout=0、学习率 5e-5、250 步，仅在有效编辑 token 上计算速度均方误差。

- 正式教师缓存：16/16，winner 均来自官方 EditScore 两阶段 K=4 搜索。
- 最终 checkpoint：`lora_train_reward_selected/adapter_step_0250.pt`。
- 验证 masked velocity MSE：0.01147。
- 禁用 adapter 后重复输出最大误差：0.0。

## P3：LoRA 强度探测

| LoRA 强度 | 普通编辑平均评分 | 早期搜索平均评分 | 搜索提升 |
|---:|---:|---:|---:|
| 0.00 | 4.7490 | 5.1430 | +0.3940 |
| 0.25 | 5.7593 | 5.4523 | -0.3071 |
| 0.50 | 4.7264 | 5.6958 | +0.9693 |
| 1.00 | 5.3757 | 5.3307 | -0.0451 |

未编辑区域的平均区域像素误差约为 0.237 到 0.242，各组没有出现明显失控。当前最有希望的强度是 0.5，但不能排除随机种子和样本组成的影响。

## 失败与边界

早期版本把 Flux-Kontext 原生速度输出直接代入另一套同步随机微分方程漂移公式，导致候选变成抽象噪声图，官方评分为 0。修复后改为原生欧拉均值加小幅区域噪声，并保留原公式只作诊断参考。

本轮 P3 runner 已记录官方评分、区域像素误差和搜索轨迹。LPIPS、DINO 等独立感知指标尚未在本轮 P3 自动汇总，因此不能声称这些指标已经验证。P3 每个强度只有 8 个测试样本，也不支持强统计显著性结论。

## 产物位置

- P1 汇总：`/data15/hyp/experiments/flux_kontext_early_edit_reward_distill/p1_results_merged.csv`
- P3 汇总：`/data15/hyp/experiments/flux_kontext_early_edit_reward_distill/final_summary.json`
- 正式教师缓存：`/data15/hyp/experiments/flux_kontext_early_edit_reward_distill/teacher_cache_reward_selected/`
- 正式 LoRA：`/data15/hyp/experiments/flux_kontext_early_edit_reward_distill/lora_train_reward_selected/`
- P3 分片：`/data15/hyp/experiments/flux_kontext_early_edit_reward_distill/p3_scale_*`

