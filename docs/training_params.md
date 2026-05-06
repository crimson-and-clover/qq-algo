# 训练与数据相关启动参数

## 数据路径（平台环境变量注入）

| 环境变量 | 路径 |
|---|---|
| `TRAIN_DATA_PATH` | `/data_ams/academic_training_data` |
| `TRAIN_CKPT_PATH` | `.../95d1b1c49dca22ef019dd010f6340bab/ckpt` |
| `TRAIN_LOG_PATH` | `.../95d1b1c49dca22ef019dd010f6340bab/log` |
| `TRAIN_TF_EVENTS_PATH` | `.../95d1b1c49dca22ef019dd010f6340bab/events` |

## 数据集

| 项目 | 值 |
|---|---|
| 数据格式 | Parquet（IterableDataset） |
| 文件数量 | 1000 |
| 总行数 | 1,010,000 |
| 训练集 | 900 Row Groups，907,381 行 |
| 验证集 | 100 Row Groups，102,619 行 |
| schema 路径 | `<data_dir>/schema.json` |

## 数据加载参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--batch_size` | 256 | 训练和验证的 batch size |
| `--num_workers` | 8 | DataLoader worker 数量 |
| `--buffer_batches` | 20 | Shuffle buffer 大小（以 batch 为单位） |
| `--train_ratio` | 1.0 | 使用全部训练数据 |
| `--valid_ratio` | 0.1 | 10% Row Groups 用作验证集 |
| `--seq_max_lens` | seq_a:256,seq_b:256,seq_c:512,seq_d:512 | 各序列域截断长度 |

## 训练参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--lr` | 0.0001 | AdamW 学习率（dense 参数） |
| `--sparse_lr` | 0.05 | Adagrad 学习率（sparse Embedding） |
| `--sparse_weight_decay` | 0.0 | Sparse 参数权重衰减 |
| `--num_epochs` | 999 | 最大训练轮数（由 early stopping 终止） |
| `--patience` | 5 | 早停耐心值（验证无提升次数） |
| `--seed` | 42 | 随机种子 |
| `--device` | cuda | 训练设备 |
| `--loss_type` | bce | 损失函数：BCEWithLogitsLoss |
| `--focal_alpha` | 0.1 | Focal Loss alpha（仅 loss_type=focal 生效） |
| `--focal_gamma` | 2.0 | Focal Loss gamma（仅 loss_type=focal 生效） |
| `--eval_every_n_steps` | 0 | 每 N 步验证（0=每 epoch 结束验证） |
| `--reinit_sparse_after_epoch` | 1 | 从第 1 个 epoch 起，每 epoch 结束重建高基数 Embedding |
| `--reinit_cardinality_threshold` | 0 | 重建阈值（0=不重建任何 Embedding） |

## 模型参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--d_model` | 64 | 骨干隐藏维度 |
| `--emb_dim` | 64 | Embedding 维度 |
| `--num_queries` | 2 | 每序列域 Query token 数 |
| `--num_hyformer_blocks` | 2 | HyFormer Block 层数 |
| `--num_heads` | 4 | 注意力头数 |
| `--seq_encoder_type` | transformer | 序列编码器类型 |
| `--hidden_mult` | 4 | FFN 内部维度倍数 |
| `--dropout_rate` | 0.01 | Dropout 率 |
| `--action_num` | 1 | 分类器输出维度（1=二分类） |
| `--use_time_buckets` | True | 启用时间桶 Embedding |
| `--rank_mixer_mode` | full | RankMixer 模式（token mixing + FFN） |
| `--use_rope` | False | 不使用 RoPE 位置编码 |
| `--rope_base` | 10000.0 | RoPE 基频 |
| `--seq_top_k` | 50 | LongerEncoder 保留 token 数 |
| `--seq_causal` | False | 不使用因果 mask |
| `--emb_skip_threshold` | 1000000 | vocab 超此值的特征不建 Embedding |
| `--seq_id_threshold` | 10000 | vocab 超此值的序列特征视为 id 特征 |

## NS Tokenizer 参数（run.sh 指定）

| 参数 | 值 | 说明 |
|---|---|---|
| `--ns_tokenizer_type` | rankmixer | NS tokenizer 模式 |
| `--user_ns_tokens` | 5 | 用户侧 NS token 数 |
| `--item_ns_tokens` | 2 | 物品侧 NS token 数 |
| `--ns_groups_json` | 空 | 不使用 ns_groups.json |

## 模型规模

| 项目 | 值 |
|---|---|
| 总参数量 | 239,931,649（约 2.4 亿） |
| Sparse 参数 | 237,435,776（98 个张量，Adagrad） |
| Dense 参数 | 2,495,873（380 个张量，AdamW） |
| num_ns | 8（5 user + 2 item + 1 user_dense） |
| T | 16（num_queries×4 序列域 + num_ns） |
| seq_b 跳过特征 | 1/13 |
| seq_c 跳过特征 | 3/11 |
