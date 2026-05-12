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

## 实验 C：Focal Loss（不改模型，只换 loss）

**日期**：

### 修改

```bash
# run.sh 改 1 行
--loss_type bce+info  →  --loss_type focal+info
```

### 原理

正负比 1:9，BCE 同等对待所有样本。Focal Loss 自动降权简单负样本（gamma=2），让模型专注困难样本。

### 预期

| 指标 | 实验 B | 预期实验 C |
|------|--------|-----------|
| AUC | B 的基线 | **+0.5–1.0** |
| LogLoss | — | 可能微涨（Focal 不直接优化 logloss） |

### 实际结果

| Epoch | AUC | LogLoss | 备注 |
|-------|-----|---------|------|
| 1 | | | |
| best | | | |

---

## 实验 D：LongerEncoder 单独加（无时间衰减）

**日期**：

### 修改

```bash
# run.sh 加 3 行（在实验 C 基础上）
--seq_encoder_type longer \
--seq_top_k 256 \
--seq_max_lens seq_a:1024,seq_b:1024,seq_c:1024,seq_d:2048 \
```

### 原理

模型结构回 baseline 后，序列截断又回到了 13% 利用率。LongerEncoder 的 top-K cross-attention 允许 4x 长序列而不 O(n²)。这次**不带时间衰减**，避免旧数据被压死。

### 预期

| 指标 | 实验 C | 预期实验 D |
|------|--------|-----------|
| AUC | C 的基线 | **+1.0–2.0** |
| 序列利用率 | 13% | **~60%** |
| 显存 | — | +2x（若 OOM 先降 seq_d 到 1024） |

### 实际结果

| Epoch | AUC | LogLoss | 显存峰值 | 备注 |
|-------|-----|---------|---------|------|
| 1 | | | | |
| best | | | | |

---

## 实验 E：ns_groups.json + GroupNSTokenizer

**日期**：

### 修改

```bash
# run.sh 改 2 行
--ns_tokenizer_type group \
--ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
--num_queries 1 \
```

### 原理

ns_groups.json 将 46 个 user fids 按语义分为 7 组（用户属性/行为/上下文）、14 个 item fids 分为 4 组（商品类型/质量/属性）。GroupNSTokenizer 每组投影到 1 个 token（共 12 个），比 RankMixer 的 5+2 个 token 信息损失更小，且保留了分组语义。

注意：`num_queries` 需改为 1 满足 `d_model % T == 0`（T = 1*4 + 12 = 16）。

### 预期

| 指标 | 实验 D | 预期实验 E |
|------|--------|-----------|
| AUC | D 的基线 | **±0.5**（不确定方向） |

### 实际结果

| Epoch | AUC | 备注 |
|-------|-----|------|
| 1 | | |
| best | | |

---

## 实验 F：RankMixer 压缩比降低

**日期**：

### 修改

```bash
# 在实验 D 基础上，保持 rankmixer 但提高 token 数
--user_ns_tokens 8 \
--item_ns_tokens 4 \
```

### 原理

当前 RankMixer 把 46 user fids + 1 dense = 47 个 embedding 压缩到 5 个 token，压缩比 9:1。加大到 8+4=12 token 后压缩比降到 ~4:1，信息损失减小。需要验证 `d_model % T == 0`（T = num_queries*4 + 8+1+4+0，实际值依 num_queries 而定）。

### 预期

| 指标 | 实验 D | 预期实验 F |
|------|--------|-----------|
| AUC | D 的基线 | **+0.3–0.5** |

### 实际结果

| Epoch | AUC | 备注 |
|-------|-----|------|
| 1 | | |
| best | | |

---

## 最佳配置记录

> 实验结束后填写

```bash
# 最优 run.sh 配置
python3 -u train.py \
    ...
```
