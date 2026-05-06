# CTR 预估损失函数调研与选型

> 基于 KDD'2024 "Understanding the Ranking Loss for Recommendation with Sparse User Feedback" 的调研总结。
> 针对本项目（PCVRHyFormer）的现状，分析各类损失函数的适用性与接入成本。

---

## 1. 项目现状

当前 `src/` 已支持的损失函数：

| Loss | 状态 | CLI 参数 |
|------|------|----------|
| BCE (`binary_cross_entropy_with_logits`) | ✅ 已内置 | `--loss_type=bce` (默认) |
| Focal Loss | ✅ 已内置 | `--loss_type=focal --focal_alpha=0.75 --focal_gamma=2.0` |
| BCE + Pairwise Hinge | ✅ 已内置 | `--loss_type=bce+pair --bce_weight=0.5 --pair_weight=0.5` |

训练代码位置：`src/trainer.py::_train_step()`
Loss 实现位置：`src/utils.py::sigmoid_focal_loss()`

当前训练模式为 **pointwise**：每个样本独立前向，logits 维度 `(B,)`，label 维度 `(B,)`。

---

## 2. 论文核心观点

**问题**：在正样本稀疏（CTR < 2%）场景下，BCE loss 的负样本存在**梯度消失**。

**原因**：
- 负样本 BCE 梯度 = `p = sigmoid(logit)` ≈ 全局 CTR
- 正样本稀疏 → CTR 极小 → 负样本梯度极小
- 正样本梯度 = `1 - p` ≈ 1，不受稀疏性影响

**结论**：在 BCE 基础上引入辅助 **ranking loss**，可以：
1. 增大负样本的梯度幅值
2. 提升模型的排序能力（AUC）
3. 同时改善分类能力（BCE loss 本身也会更低）

---

## 3. 各类损失函数分析

### 3.1 Pointwise — BCE Loss（已有）

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log\sigma(z_i) + (1-y_i)\log(1-\sigma(z_i))\right]$$

- **优点**：直接优化概率估计，校准性好，实现简单
- **缺点**：正样本稀疏时负样本梯度消失
- **适用**：基线必备，不可移除
- **接入成本**：0（已有）

---

### 3.2 Pointwise — Focal Loss（已有）

$$\mathcal{L}_{\text{Focal}} = -\frac{1}{N}\sum_{i=1}^{N} \alpha_t (1 - p_t)^{\gamma} \cdot \text{BCE}(z_i, y_i)$$

其中 $p_t = p$ 当 $y=1$，否则 $p_t = 1-p$。

- **优点**：通过 `(1-p_t)^gamma` 自动降低易分样本权重，关注难分样本
- **缺点**：论文指出，对于负样本 $p_t = 1-p \approx 1$，因此 $(1-p_t)^\gamma$ 会**进一步缩小**负样本梯度，**加剧**梯度消失（与论文目标相反）
- **适用**：样本类别不平衡，但需谨慎调参
- **接入成本**：0（已有）

> ⚠️ **注意**：论文实验表明 Focal Loss 在正样本极稀疏场景下效果不如 Combined-Pair，因为它进一步压缩了负样本梯度。

---

### 3.3 Pairwise — RankNet / Hinge Loss（⭐ 推荐接入）

**Combined-Pair 的核心**：在同一个 batch / query 内，确保正样本的 logit > 负样本的 logit。

**Hinge Loss 形式**（论文采用）：

$$\mathcal{L}_{\text{rank}} = \frac{1}{N^{+} N^{-}} \sum_{i \in \mathcal{P}} \sum_{j \in \mathcal{N}} \max(0, m - (z_i - z_j))$$

其中：
- $\mathcal{P}$ 为正样本集合，$\mathcal{N}$ 为负样本集合
- $m$ 为 margin（通常取 1.0）
- $N^{+}, N^{-}$ 为正负样本数量

**Combined-Pair**（BCE + ranking）：

$$\mathcal{L} = w_{\text{BCE}} \cdot \mathcal{L}_{\text{BCE}} + w_{\text{rank}} \cdot \mathcal{L}_{\text{rank}}$$

- **优点**：
  - 负样本 ranking loss 梯度 = 1（当 $z_i - z_j < m$ 时），远大于 BCE 的 `~CTR`
  - 有效缓解负样本梯度消失
  - 论文实验：AUC 提升 0.02%~0.10%，BCE loss 降低 0.05%~0.17%
- **缺点**：
  - 需要在一个 batch 内同时存在正负样本才能构造 pairs
  - 如果 batch 全为正或全为负，ranking loss = 0（无信号）
- **适用**：本项目 batch_size=256/512，正负样本共存概率高，**非常适合**
- **接入成本**：低（约 20 行代码，纯 tensor 运算）

---

### 3.4 Pairwise — RankNet（概率形式）

$$\mathcal{L}_{\text{RankNet}} = -\sum_{(i,j) \in \mathcal{P} \times \mathcal{N}} \log \sigma(z_i - z_j)$$

- **与 Hinge 的区别**：使用概率形式的交叉熵，无 margin 超参
- **梯度**：负样本梯度 = $\sigma(z_j - z_i)$，当正样本稀疏且 $z_i < z_j$ 时梯度 ≈ 1，同样很大
- **适用**：可作为 Hinge 的替代，无需调 margin
- **接入成本**：低

---

### 3.5 Listwise — ListNet

$$\mathcal{L}_{\text{ListNet}} = -\sum_{i \in \mathcal{P}} \log \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}$$

其中 $K$ 为一个 list/query 内的样本数。

- **优点**：从列表整体优化，排序目标更直接
- **缺点**：
  - 需要按 `query_id` / `user_id` 分组构造 list
  - 当前项目数据流是 pointwise shuffle，没有保留 query 分组信息
- **适用**：需要数据层改造（引入 `group_by` 或 `query_id`）
- **接入成本**：**高**（需改 data pipeline）

---

### 3.6 Listwise — Google RCR (Rank-Calibration Regression)

$$\mathcal{L}_{\text{RCR}} = \frac{1}{K} \sum_{i=1}^{K} \left| \sigma(z_i) - \frac{y_i}{\bar{y}} \right|$$

其中 $\bar{y}$ 为 list 内平均 label。

- **优点**：同时优化排序和校准
- **缺点**：同样需要 list 分组；且公式依赖 list 级别统计量
- **适用**：有 query/list 结构的场景
- **接入成本**：**高**

---

### 3.7 Contrastive — Combined-Contrastive

受 Supervised Contrastive Learning 启发：

$$\mathcal{L}_{\text{contrast}} = -\frac{1}{N} \sum_{i=1}^{N} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(e_i, e_p) / \tau)}{\sum_{a \in A(i)} \exp(\text{sim}(e_i, e_a) / \tau)}$$

其中：
- $e_i$ 为第 $i$ 个样本的 embedding（通常是 DNN 最后一层隐藏层输出）
- $P(i)$ 为与 $i$ 同标签的样本集合
- $A(i)$ 为 batch 内除 $i$ 外的全部样本
- $\tau$ 为温度系数

- **优点**：
  - 让同类 embedding 靠近，异类远离
  - 不直接依赖 logits，对梯度消失不敏感
- **缺点**：
  - 需要模型输出 **embedding**（而非仅 logits），当前 `forward()` 返回 logits
  - 需要改造模型结构，增加 `get_embedding()` 接口
  - batch 内正负样本比例影响对比学习效果
- **适用**：有高质量 embedding 需求的场景
- **接入成本**：**中高**（需改模型输出 + loss 计算）

---

## 4. 接入建议（按优先级）

| 优先级 | Loss | 理由 | 预估工作量 | 风险 |
|--------|------|------|-----------|------|
| **P0** | **Combined-Pair (BCE + Hinge/RankNet)** | 论文核心方法，实现简单，效果明确，与本项目 pointwise batch 天然兼容 | 低（~30行） | 低 |
| **P1** | **Focal Loss 调优** | 已有实现，但需注意论文指出其在极稀疏场景可能适得其反；建议与 Combined-Pair 做 A/B | 0 | 中 |
| **P2** | **Listwise (ListNet / RCR)** | 效果可能更好，但需数据层改造（query 分组） | 高 | 中 |
| **P3** | **Combined-Contrastive** | 概念先进，但需改模型结构输出 embedding | 中高 | 高（可能不稳定） |

---

## 5. Composite Loss 实现（供参考）

`trainer.py::_train_step()` 中通过 `parsed_losses` set 累加各 loss 项：

```python
loss = torch.tensor(0.0, device=logits.device)

if 'bce' in self.parsed_losses:
    bce_loss = F.binary_cross_entropy_with_logits(logits, label)
    loss = loss + self.bce_weight * bce_loss

if 'focal' in self.parsed_losses:
    focal_loss = sigmoid_focal_loss(logits, label, alpha=..., gamma=...)
    loss = loss + self.focal_weight * focal_loss

if 'pair' in self.parsed_losses:
    pos_mask = label == 1
    neg_mask = label == 0
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        diff = logits[pos_mask].unsqueeze(1) - logits[neg_mask].unsqueeze(0)
        rank_loss = F.relu(self.rank_margin - diff).mean()
    else:
        rank_loss = torch.tensor(0.0, device=logits.device)
    loss = loss + self.pair_weight * rank_loss
```

---

## 6. 关键超参建议

基于论文实验结论：

| 超参 | 建议范围 | 说明 |
|------|----------|------|
| `bce_weight` | 0.5 ~ 1.0 | BCE 项权重；与 `pair_weight` 独立调节 |
| `pair_weight` | 0.3 ~ 0.7 | Pairwise 项权重；论文 sweet spot 等价于 bce:pair ≈ 0.7:0.3 ~ 0.5:0.5 |
| `rank_margin` | 1.0 | Hinge loss 的标准取值 |
| `batch_size` | ≥ 256 | 需要足够大的 batch 保证正负样本共存，否则 ranking loss = 0 |

---

## 7. 与本项目约束的兼容性

| 约束 | Combined-Pair | Listwise | Contrastive |
|------|---------------|----------|-------------|
| `run.sh` 单一入口 | ✅ 兼容 | ✅ 兼容 | ✅ 兼容 |
| 环境变量控制路径 | ✅ 兼容 | ✅ 兼容 | ✅ 兼容 |
| IterableDataset 流式读取 | ✅ 兼容（pointwise batch 内构造 pairs） | ❌ 需分组 | ✅ 兼容 |
| 平台 Eval 1000 行 stdout 限制 | ✅ loss 单一标量 | ✅ loss 单一标量 | ✅ loss 单一标量 |
| `baseline_code/` baseline 保护 | ✅ 新增独立 loss 函数 | ❌ 需改 data | ❌ 需改 model |

---

## 8. 下一步行动建议

1. **短期（本周）**：实现 Combined-Pair（BCE + Hinge），作为 `--loss_type=bce+pair` 选项加入 `trainer.py` 和 `train.py`
2. **中期**：提交平台实验，对比 `--loss_type=bce` vs `--loss_type=bce+pair --bce_weight=0.5 --pair_weight=0.5`
3. **长期**：若效果正向，可尝试 ListNet（需评估 data pipeline 改造成本）

---

*文档生成时间：2026-05-06*
*参考文献：KDD'2024 "Understanding the Ranking Loss for Recommendation with Sparse User Feedback"*
