# PCVRHyFormer 涨点 TODO

## 当前实验（已提交，等结果）

- [ ] **Dense-only LR Scheduler 修复版**
  - 目标：验证 Warmup(2ep) + Cosine(20ep) 在 dense 优化器上的效果
  - 关键修复：
    - `warmup_ratio` → `warmup_epochs=2`，避免 `num_epochs=999` 导致 warmup 永远走不完
    - `cosine_t_max_epochs=20`，按固定 epoch 数计算 cosine 退火
    - sparse 优化器**无任何 scheduler**，与 baseline 100% 一致
    - `dense_weight_decay=0.0`，先回滚到 baseline，控制单一变量
  - 对比基线：baseline AUC=0.812654，上次失败版 AUC=0.795328
  - 预期：若 AUC 回到 0.81x 附近，说明 scheduler 修复有效
  - 若有效 → 进入「下一步：Target-Attention」
  - 若无效 → 回滚 scheduler，直接做模型结构改进

---

## 下一步（最优方向）：Target-Attention（DIN）

- [ ] **QueryGenerator 引入 Target-Attention 替代 MeanPool**
  - 当前问题：`seq_pooled = seq_sum / seq_count` 做 MeanPool，与目标 item 无关，1年前的牙刷和1分钟前的iPhone权重相同
  - 方案：用 `item_ns_tokens` 作为 Query，与序列每个位置 token 做 Attention，加权求和替代 MeanPool
  - 优势：
    - 纯模型侧改动，不改数据 pipeline，实验可控
    - CTR/CVR 领域最经典涨点手段（阿里 DIN）
    - 改动点集中，只改 `MultiSeqQueryGenerator.forward`
    - 与 time_bucket_emb 兼容（attention 同时建模兴趣和时序）
  - 实现要点：
    - `item_ns_tokens` 从 `PCVRHyFormer.forward` 经 `item_ns_tokenizer` 生成后传入 `MultiSeqQueryGenerator`
    - 计算 `attn_score = softmax(Q @ K^T / sqrt(d))`
    - `seq_pooled = sum(attn_score * seq_tokens)` 替代 MeanPool
    - 新 attention projection 用 `xavier_uniform_` 初始化
  - 风险：Attention 计算 O(L)，L 为截断后长度，开销可控

---

## 备选快速实验（零模型改动）

- [ ] **序列截断调优**
  - 真实数据 P95 长度：domain_a=1969, domain_b=1972, domain_c=1418, domain_d=3913
  - 当前截断 256/512 仅保留 **13%-36%** 信息，丢失 64%-87%
  - 只需改 `run.sh` 参数：`--seq_max_lens seq_a:2048,seq_b:2048,seq_c:1536,seq_d:4096`
  - 若显存 OOM → fallback：`--seq_encoder_type=longer --seq_top_k=200`
  - 这是**信息损失最严重的瓶颈**，零代码改动，收益直接

---

## 后续优化路线图（按优先级排序）

### P1：信息侧（数据不损失）

- [ ] **Target Statistics 特征工程**
  - 紧急原因：user_id 重复率 **0%**，Embedding 完全学不到
  - 加入 ad_id / advertiser_id 历史点击率、转化率
  - 数据侧改动，不改模型结构

- [ ] **严格时间滑窗验证切分**
  - 当前按物理文件尾部切分，可能存在 Future Leakage
  - 改为按 `timestamp` 字段排序后切分：训练时间 < 验证时间

### P2：模型结构改进

- [ ] **多任务学习 (CTR + CVR / ESMM)**
  - `action_num=2`，共享 HyFormer + 独立 Tower
  - 联合训练：`pCTCVR = pCTR × pCVR`
  - 解决样本选择偏差 (SSB) 和数据稀疏性 (DS)

- [ ] **序列间 Q 交互**
  - 当前各序列独立生成 Q，缺少跨域兴趣相关性
  - 在 QueryGenerator 后加 Self-Attention 让各序列 Q 互相交互

### P3：Loss & 训练策略

#### Focal Loss（已有实现，默认关闭）

**公式：**
$$
\mathcal{L}_{\text{Focal}} = -\frac{1}{N}\sum_{i=1}^{N} \alpha_t (1 - p_t)^{\gamma} \cdot \text{BCE}(z_i, y_i)
$$
其中 $p_t = p$ 当 $y=1$，否则 $p_t = 1-p$；$\alpha$ 控制正负样本权重，$\gamma$ 控制对易分样本的抑制程度。

**当前默认参数：**
- `--loss_type=bce`（默认）/ `--loss_type=focal`（显式开启）
- `--focal_alpha=0.75`：正样本权重（$y=1$ 时 $\alpha_t=0.75$，$y=0$ 时 $\alpha_t=0.25$）
- `--focal_gamma=2.0`：标准值，$\gamma=0$ 退化为 BCE

**为什么当前不推荐它作为主武器：**
- KDD'2024 论文指出：正样本稀疏场景（CTR<2%）下，负样本的 $p_t = 1-p \approx 1$，$(1-p_t)^\gamma \approx 0$
- **结果**：Focal 会进一步压缩负样本梯度，加剧梯度消失，与「增大负样本梯度」的目标背道而驰
- 真实数据正样本率 9.66%，虽然不算极稀疏，但仍需警惕

**若要用，建议参数重调：**
- Grid search：`alpha ∈ {0.25, 0.5, 0.75}`，`gamma ∈ {0.5, 1.0, 2.0}`
- 论文改造版：让负样本 $\gamma < 1$（甚至为 0），避免压缩负样本梯度

---

#### Combined-Pair（⭐ 推荐优先接入）

**来源**：KDD'2024 "Understanding the Ranking Loss for Recommendation with Sparse User Feedback"

**核心思想**：在 BCE 基础上加入 pairwise ranking loss（Hinge 形式），在同一个 batch 内确保正样本 logit > 负样本 logit。

**公式：**
$$
\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{BCE}} + (1-\lambda) \cdot \underbrace{\frac{1}{N^{+} N^{-}} \sum_{i \in \mathcal{P}} \sum_{j \in \mathcal{N}} \max(0, m - (z_i - z_j))}_{\text{pairwise hinge ranking loss}}
$$

**为什么适合本项目：**
1. **数据流兼容**：当前 pointwise batch（B=256/512）内几乎必然同时有正负样本，可直接 broadcast 构造 $(N^+, N^-)$ pairs
2. **零架构改动**：不改模型结构、不改 data pipeline，只改 `trainer.py::_train_step()`
3. **效果明确**：论文在 CTR≈3.3% 场景下，AUC 提升 0.02%~0.10%，BCE loss 本身也降低
4. **缓解梯度消失**：ranking loss 负样本梯度 = 1（当 $z_i - z_j < m$ 时），远大于 BCE 的 `~CTR`

**建议超参：**
- `bce_weight` ($\lambda$)：0.6 ~ 0.8（论文 sweet spot = 0.7）
- `rank_margin`：1.0（标准 hinge margin）
- `batch_size`：≥ 256（需保证 batch 内正负共存，否则 ranking loss = 0）

**接入成本**：低（~30 行代码，纯 tensor 运算）

**实现位置**：`src/trainer.py::_train_step()` 中增加 `--loss_type=combined_pair` 分支

---

#### 其他 Loss（暂不推荐，原因如下）

| Loss | 状态 | 不推荐理由 |
|------|------|-----------|
| **ListNet (Listwise)** | 未实现 | 需要按 `query_id` / `user_id` 分组构造 list，当前数据流是全局 shuffle 的 pointwise，需改 data pipeline（架构级改造） |
| **Google RCR** | 未实现 | 同样需要 list 分组；且依赖 list 级别统计量 $\bar{y}$ |
| **Combined-Contrastive** | 未实现 | 需要模型输出 **embedding**（当前 `forward()` 只返回 logits），需改模型结构增加 `get_embedding()` 接口；且 batch 内正样本太少（2~8个），contrastive 分子项退化 |
| **RankNet（概率形式）** | 未实现 | 可用但次优：当 $z_i \gg z_j$ 时梯度 $\sigma(z_j - z_i) \approx 0$，不如 Hinge 在 $z_i - z_j < m$ 时梯度恒为 1 稳定 |
| **PolyLoss** | 未实现 | `L = CE + ε1(1-pt) + ε2(1-pt)^2`，比 Focal 更灵活，但仍属于 pointwise 改良，不如 Combined-Pair 直接解决梯度消失 |

---

#### Loss 选型决策树

```
当前数据流 = pointwise batch shuffle（无 query 分组）
│
├─ 是否有 batch 内正负样本共存？  →  Combined-Pair（P0，零架构改动）
│
├─ 是否只想调现有参数？          →  Focal Loss 重调 alpha/gamma（P1）
│
├─ 是否能改 data pipeline（query 分组）？ → ListNet / RCR（P2，高成本）
│
└─ 是否能改模型结构（输出 embedding）？   → Combined-Contrastive（P3，高风险）
```

### P4：工程 & 细节

- [ ] **Dense 特征缺失值处理**
  - `user_dense_feats_91` 缺失率 **48.2%**
  - 建议：中位数/均值填充，或学一个 missing indicator

- [ ] **模块级差异化学习率（Layer-wise LR Decay）**
  - 底层 Seq Encoder / RankMixerBlock 用稍大 LR
  - 顶层 Classifier 用更小 LR

---

## 已实现

- [x] **双优化器**：Embedding 用 Adagrad，Dense 用 AdamW
- [x] **Focal Loss**（`--loss_type=focal`，默认值 `--focal_alpha=0.75`）
- [x] **时间分桶编码**（time_bucket_embedding）
- [x] **高基数 Embedding 过滤**（emb_skip_threshold）
- [x] **学习率调度（第一版）**：`--warmup_ratio=0.05`（因 num_epochs=999 导致失败，已废弃）
- [x] **学习率调度（修复版）**：`--warmup_epochs=2 --cosine_t_max_epochs=20`（sparse 无 scheduler）
- [x] **AdamW weight_decay 回滚**：`--dense_weight_decay=0.0`
- [x] **独立数据探查脚本**（`data_explorer.py`，采样防 OOM）

---

## 数据探查关键结论（来自全量真实数据）

| 维度 | 发现 | 对策略的影响 |
|:---|:---|:---|
| **正样本率** | **9.66%**（不是 demo 的 12.4%）| Focal Loss `alpha` 从 0.75 调低到 0.5 尝试 |
| **序列截断** | 仅保留 **13%-36%** 信息 | 最紧急的涨点来源，需扩 seq_max_lens 或换 LongerEncoder |
| **user_id 重复率** | **0%** | user_id Embedding 完全无效，必须转 Target Statistics |
| **item_id 重复率** | **26.2%** | item_id Embedding 正常可用 |
| **时间跨度** | **3.4 天** | 支持时序特征工程 + 严格时间滑窗验证 |
| **特征基数** | 无 >5000 的高基数特征 | Embedding 表内存可控 |
| **Dense 缺失** | `user_dense_feats_91` 缺失 **48.2%** | 需做缺失值填充 |

---

## 失败实验记录（避免重复踩坑）

| 时间 | 改动 | AUC | 失败原因 |
|:---|:---|:---|:---|
| 2026-05-03 | `warmup_ratio=0.05` + `dense_weight_decay=0.01` + sparse scheduler | **0.795328** | `num_epochs=999` 导致 warmup 永远走不完；sparse scheduler 每个 epoch 被重置归零 |
