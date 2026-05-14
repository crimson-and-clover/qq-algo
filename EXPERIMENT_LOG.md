# 实验日志

> 基线：`run.sh` 原始配置 (commit `d8ad263` 之前)
> AUC 基线 = 0.857 (实验 B 第3次)

---

## 5天冲 SOTA 路线图（目标 AUC ≥ 0.87）

### 教训总结（来自 git 历史 + 实验 B 回归）

| 教训 | 来源 | 对策 |
|------|------|------|
| **绝不丢弃训练数据** | `_flush_buffer` break 无条件丢尾批 → -5% 数据 | 已修复，后续不改 `_flush_buffer` |
| **采样策略是 AUC 的第一大杠杆** | log-spaced vs 头部截断差 4 个点 | 单独拉回 log-spaced 采样（不改模型） |
| **InfoNCE uniformity 被数据 bug 连坐了** | 数据丢弃时 val 还能 0.84，无 InfoNCE 时 val 只有 0.76 | 在干净 baseline 上重新验证 InfoNCE |
| **一次只改一个变量** | `d8ad263` 混了采样/衰减/RoPE/interaction 全锅端 | 每个实验只动一个维度 |

### 优先级排序（按预期收益/风险比）

```
Tier 1 ─ 高收益零风险（不改模型结构）
  C:   Focal Loss           [已提交→失败 0.80]  放弃
  C4:  BCE+InfoNCE+Scheduler [已编码]  预期 +0.3-1.0  (InfoNCE + warmup-cosine)

Tier 2 ─ 数据覆盖（改采样不改模型）
  D2:  log-spaced 采样      [待编码]  预期 +0.5-1.5  (P95=1969, 当前只用 13%)
  D:   序列翻倍 256→512     [待编码]  预期 +0.3-0.8  (OOM 风险)

Tier 3 ─ 模型增强（有风险）
  E:   LongerEncoder        [待编码]  预期 +0.5-1.0  (结构变化)
  E2:  RoPE                 [待编码]  预期 +0.2-0.5  (位置编码)
```

### 5天时间线

| 天 | 提交 | 内容 | 积累 AUC |
|----|------|------|----------|
| 1 | ~~C (失败)~~ | Focal Loss → 0.80，放弃 | 0.857 |
| 2 | C4 (已编码) | BCE + InfoNCE + Scheduler | 0.857 → ? |
| 2-3 | D2 | log-spaced 采样（只捡 `d8ad263` 的采样逻辑，不带衰减/RoPE/interaction） | ? |
| 3-4 | D | 序列翻倍 | ? |
| 4-5 | E | LongerEncoder | ? |

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

| Epoch | AUC (val) | AUC (test) | LogLoss | InfoNCE | 备注 |
|-------|-----------|------------|---------|---------|------|
| 1 | 0.837 | — | 0.25 | 7.3 | |
| 3 | 0.840 | — | 0.244 | 7.0 | |
| best | 0.840 | **0.782** | 0.244 | 7.0 | val-test 缺口 6 个点，不生效 |

> **结论**：单独加 target_attention 没用。AUC 保持在 0.84，test 仅 0.782。
> 核心问题不在"少一个 attention"，而在序列信息整体丢失（时间衰减 + 采样偏差）。

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

### 实际结果（Job: exp_b_revert / 第一次提交，含 log-spaced 采样）

| Epoch | Step | AUC (val) | LogLoss | InfoNCE | 备注 |
|-------|------|-----------|---------|---------|------|
| 1 | 3412 | **0.818** | — | — | baseline 同期 0.857，差 4 个点 |

> **失败**。Step 3412 时 AUC 仅 0.818，远低于 baseline 0.857。
> **根因：dataset.py 的 log-spaced 采样未被回退**——seq_a P95=1969，baseline 取 0..255 位置，log-spaced 取 0..127 + 稀疏 128..1968，数据分布完全不同。

---

### 实际结果（Job: exp_b_pure_baseline / 第二次提交，dataset 回退到头部截断）

| Epoch | AUC (val) | AUC (test) | LogLoss | 备注 |
|-------|-----------|------------|---------|------|
| best | **0.769** | — | — | 比第一次 0.818 还差，更远离 baseline 0.857 |

> **失败。AUC 0.769，反而更差。**
> **二次根因分析**（opus 协助排查）：排除 log-spaced 采样后，仍有两处 silent 行为差异：
> 1. `_flush_buffer` 尾批无条件丢弃 —— `overlap=0` 时 `break` 仍生效，每次 flush 最多丢 255 行，累积每 epoch ~5% 数据被截断
> 2. `__iter__` 每 epoch 随机化 row-group 顺序 —— baseline 是静态 `rg_list`
>
> 修复：commit `64f9d67`（break 条件化）+ `68c7162`（shuffle 禁用）。

---

### 实际结果（Job: exp_b_zero_diff / 第三次提交，全部修复）

| Step | AUC (val) | LogLoss | 备注 |
|------|-----------|---------|------|
| 3624 | **0.857** | — | 符合 baseline 预期 |

> **成功！回归 baseline AUC 0.857。**
> 修复清单：头部截断 + 不丢尾批 + 静态 rg_list + 移除 loss_mask。 |

---

## 实验 C：Focal Loss（不改模型，只换 loss）[已提交]

**Job**: `exp_c_focal`
**日期**：

### 修改

```bash
# run.sh：baseline BCE → Focal
--loss_type bce    →    --loss_type focal --focal_alpha 0.75 --focal_gamma 2.0
# 其余完全同 baseline：no_scheduler, no infoNCE, no target_attention
```

### 原理

正负比 1:9，BCE 同等对待所有样本。Focal Loss (alpha=0.75, gamma=2) 自动降权简单负样本，让模型专注困难样本。

### 预期

| 指标 | 实验 B | 预期实验 C |
|------|--------|-----------|
| AUC | 0.857 | **0.862–0.867** |
| LogLoss | — | 可能微涨（Focal 不直接优化 logloss） |

### 实际结果

| Epoch | AUC | 备注 |
|-------|-----|------|
| best | **0.80** | 远低于 baseline 0.857，失败 |

> **失败。根因**：Focal Loss 在初始化时梯度量级仅为 BCE 的 1/13。
> ```
> alpha=0.75, gamma=2, p≈0.5:
>   Focal/BCE = (0.1×0.75×0.25 + 0.9×0.25×0.25) × 0.693 / 0.693 = 0.052/0.693 ≈ 0.075
> ```
> 同 LR 下实际学习距离只有 BCE 的 1/13。调整 alpha/gamma 需多轮试错，5天时间成本太高。
> **决策：放弃 Focal，直接推进 InfoNCE + Scheduler。**

---

## 实验 C4：BCE + InfoNCE uniformity + Scheduler [已编码]

**Job**: `exp_c4_combo`
**日期**：

### 修改

```bash
# baseline BCE + InfoNCE uniformity + warmup-cosine scheduler
--loss_type bce+info --info_weight 0.05 --info_tau 0.15
# 删除 --no_scheduler，恢复默认 warmup=2ep + cosine=999ep（慢衰减）
# 不改模型结构、不改采样策略
```

### 原理

两个互不冲突的已验证改进合入：
- **InfoNCE uniformity**（commit `2a009a6`）：logsumexp 推远 batch 内其他样本，防表征坍缩。在数据丢弃 bug 下 val 仍有 0.84（同期无 InfoNCE 仅 0.76）
- **warmup + cosine**：warmup 防初期震荡，cosine 慢衰减防过拟合

### 预期

| 指标 | baseline (B) | 预期 C4 |
|------|-------------|---------|
| AUC | 0.857 | **0.86–0.87+** |
| InfoNCE | — | 6–8 稳定 |

### 实际结果

| Epoch | AUC | LogLoss | InfoNCE | 备注 |
|-------|-----|---------|---------|------|
| 1 | | | | |
| best | | | | |

---

## 实验 D2：log-spaced 序列采样（只捡采样，不带模型改动）[待编码]

**Job**: `exp_d2_logspace`
**日期**：

### 修改

从 commit `d8ad263` 只提取 `dataset.py` 中的 log-spaced 采样逻辑。**不引入**：时间衰减门控、RoPE、feature_interaction、cold restart、weight_decay。

```python
# dataset.py：头部截断 → log-spaced 采样
# dense_k = max_len // 2   (前一半密集：位置 0..127)
# sparse_k = max_len - dense_k  (后一半对数间隔：128..1968 均匀步长)
# 短序列 (rl <= max_len)：全保留
# 长序列：前半密集 + 后半等步长稀疏
```

### 原理

P95 序列长度 = 1969，当前头部截断只用 256（13%）。log-spaced 在 256 窗口内覆盖头尾：前半 128 个步（最新行为）+ 后半 128 个对数间隔步（历史关键节点）。不改模型结构、不改 encoder、不改 loss。

### 预期

| 指标 | C4 基线 | 预期实验 D2 |
|------|---------|-----------|
| AUC | C4 的基线 | **+0.5–1.5** |
| 序列覆盖率 | 13% | **~25%**（256 窗口内头尾兼顾） |

### 实际结果

| Epoch | AUC | LogLoss | 备注 |
|-------|-----|---------|------|
| 1 | | | |
| best | | | |

---

## 实验 D：序列翻倍（256→512，纯扩窗口）[待编码]

**Job**: `exp_d_seq2x`
**日期**：

### 修改

```bash
# 在最佳配置基础上
--seq_max_lens seq_a:512,seq_b:512,seq_c:1024,seq_d:1024
```

### 原理

当前 256-512 仅用 13% 序列。翻倍到 512-1024 覆盖 ~25%。纯扩大窗口，不改采样策略。若 OOM 先降 seq_d 回 512。

### 预期

| 指标 | 前步基线 | 预期 |
|------|---------|------|
| AUC | 前步基线 | **+0.3–0.8** |
| 显存 | — | +1.5–2x |

### 实际结果

| Epoch | AUC | LogLoss | 显存峰值 | 备注 |
|-------|-----|---------|---------|------|
| 1 | | | | |
| best | | | | |

---

## 实验 E：LongerEncoder（结构升级）[待编码]

**Job**: `exp_e_longer`
**日期**：

### 修改

```bash
--seq_encoder_type longer --seq_top_k 256
# 序列长度保持 D/D2 最优值
```

### 原理

LongerEncoder 的 top-K cross-attention 支持长序列而不 O(n²)。不改采样、不改 loss、不带时间衰减。

### 预期

| 指标 | 前步基线 | 预期 |
|------|---------|------|
| AUC | 前步基线 | **+0.5–1.0** |

### 实际结果

| Epoch | AUC | LogLoss | 备注 |
|-------|-----|---------|------|
| 1 | | | |
| best | | | |

---

## 实验 E2：RoPE 位置编码 [待编码]

**Job**: `exp_e2_rope`
**日期**：

### 修改

```bash
--use_rope true
```

### 原理

RoPE 提供相对位置编码，对长序列建模有帮助。从 `d8ad263` 单独提取。

### 预期

| 指标 | 前步基线 | 预期 |
|------|---------|------|
| AUC | 前步基线 | **+0.2–0.5** |

---

## 实验 F1/F2（低优先级，视时间）：ns_groups / RankMixer 压缩比

搁置到 Tier 3 之后，优先级低。

---

## 最佳配置记录

> 实验结束后填写

```bash
# 最优 run.sh 配置
python3 -u train.py \
    ...
```
