# 实验日志

> 基线：`run.sh` 原始配置 (commit `d8ad263` 之前)
> AUC 基线 = 0.86

---

## 实验 1：InfoNCE 修复 + 训练稳定性（核心止血）

**日期**：

### 修改内容

| 项 | 改前 | 改后 | 文件 |
|----|------|------|------|
| InfoNCE 模式 | label-pair（同标签=正样本对） | uniformity（自己=唯正样本，logsumexp） | `utils.py` |
| InfoNCE weight | 0.5 | 0.05 | `run.sh` |
| InfoNCE tau | 0.07 | 0.15 | `run.sh` |
| LR schedule | `--no_scheduler`（关闭） | 开启，warmup=1ep，cosine=10ep | `run.sh` |
| cold restart | `--reinit_cardinality_threshold 100000` | 移除（默认 0，数据无高基数特征） | `run.sh` |
| dropout | 0.01（默认） | 0.05 | `run.sh` |
| weight decay | 0.0001 | 0.01 | `run.sh` |
| infer.py bug | `ModelInput` 缺 `seq_time_diffs` → 推理 crash | 补全 | `infer.py` |

### 预期结果

| 指标 | 改前 | 预期 |
|------|------|------|
| AUC | 0.83，持续下降 | **≥ 0.86**，趋势平稳 |
| LogLoss/Validation | 持续上涨 | 平稳或下降 |
| InfoNCE | 0.1–0.3 波动 | 6–7 稳定 |
| Calibration | — | ~1.0 |

### 实际结果

| Epoch | 步数 | AUC | LogLoss | InfoNCE | Calibration | 备注 |
|-------|------|-----|---------|---------|-------------|------|
| 1 | ~4k | 0.835 | ~0.26 | ~10→8 | — | 极速收敛 |
| 3 | ~12k | 0.839 | ~0.245 | ~7.5 | — | 平稳 |
| 5 | ~20k | **0.841** | **0.244** | ~7.1 | — | 拐点，AUC 最高 |
| 7 | ~28k | 0.838 | 0.275 | ~7.1 | — | LogLoss 抬头，过拟合 |
| best | 20487 | **0.8407** | **0.244** | 7.1 | — | |

**结论**：InfoNCE 修复成功（11→7.1 平滑）。AUC 止血（0.83→0.84）但比原基线 0.86 差 2 个点。
后期 LogLoss 抬头说明 cosine 退火太早+dropout/wd 偏重，模型在有限容量下过拟合。

---

## 实验 A：补回 target_attention（消融：AUC 能否回到 0.86）

**日期**：

### 修改内容

| 项 | 实验 1 | 实验 A |
|----|--------|--------|
| use_target_attention | 无 | **启用** |

> 原 baseline (AUC 0.86) 有 `--use_target_attention`，DIN 式 item-aware 序列池化对 CVR 通常有 1-2 点提升。

### 预期结果

- AUC 从 0.841 回升到 **0.85-0.86**
- 不改其他参数（InfoNCE/scheduler/LongerEncoder 保留）

### 实际结果

| Epoch | 步数 | AUC | LogLoss | InfoNCE | 备注 |
|-------|------|-----|---------|---------|------|
| 1 | | | | | |
| 2 | | | | | |
| best | | | | | |

---

## 实验 B：回退 baseline 结构（关时间衰减/LongerEncoder/target_attention/interaction/RoPE）

**日期**：

### 修改内容

| 项 | 实验 1 | 实验 B |
|----|--------|--------|
| seq_encoder_type | longer | **transformer**（=baseline） |
| seq_max_lens | 1024-2048 | **默认 256-512**（=baseline） |
| time-decay gating | 启用 | **禁用**（model.py 注释掉） |
| use_target_attention | 无→实验A有 | **关闭**（=baseline） |
| use_feature_interaction | True | **False**（train.py 改默认） |
| use_rope | True | **False**（=baseline） |
| dropout_rate | 0.05 | **0.01**（=baseline） |
| dense_weight_decay | 0.01 | **0**（=baseline） |
| InfoNCE uniformity | ✅ 保留 | ✅ 保留 |
| warmup+cosine | ✅ 保留 | ✅ 保留 |

> 仅保留两个已验证有效的改进：InfoNCE uniformity + LR scheduler。其他全部回退到 baseline_code 等价。

### 预期结果

| 指标 | 实验 1 val | 实验 1 test | 预期实验 B |
|------|-----------|------------|-----------|
| AUC | 0.841 | 0.782 | **≥ 0.86** |
| LogLoss | 0.244→0.275↑ | — | 平稳 |
| InfoNCE | 7.1 | — | 7-8 稳定 |

### 实际结果

| Epoch | 步数 | AUC | LogLoss | InfoNCE | 备注 |
|-------|------|-----|---------|---------|------|
| 1 | | | | | |
| 2 | | | | | |
| best | | | | | |

---

## 实验 3（可选）：Focal Loss

**日期**：

### 修改内容

```bash
# run.sh 中替换
--loss_type bce+info    →  --loss_type focal+info
# 或混合
--loss_type bce+focal+info --bce_weight 0.5 --focal_weight 0.5
```

### 预期结果

- 1:9 正负比下，Focal Loss 自动降权简单负样本
- LogLoss 可能略涨（Focal 不直接优化 logloss），但 AUC 应提升

### 实际结果

| 配置 | AUC | LogLoss | 备注 |
|------|-----|---------|------|
| focal+info | | | |
| bce+focal+info | | | |

---

## 实验 4（可选）：关闭时间衰减门控

**日期**：

### 修改内容

需在 `model.py` 加 flag 或在 dataset 中将 `time_diff` 全置 0。

### 预期结果

- 若时间衰减过度压制远期数据，关闭后 AUC 可能恢复
- 若门控有效，AUC 可能下降

### 实际结果

| 配置 | AUC | 备注 |
|------|-----|------|
| 关闭 time_decay | | |

---

## 实验 5（可选）：关闭 RoPE / user-item interaction

### 修改内容

```bash
# 删 --use_rope
# 加 --no_feature_interaction
```

### 预期结果

逐个消融，确认每个新特性对 AUC 的贡献方向和量级。

### 实际结果

| 配置 | AUC | 备注 |
|------|-----|------|
| 无 RoPE | | |
| 无 interaction | | |
| 两者都无 | | |

---

## 最佳配置记录

> 实验结束后填写

```bash
# 最优 run.sh 配置
python3 -u train.py \
    ...
```
