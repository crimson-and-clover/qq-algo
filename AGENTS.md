# 腾讯广告算法竞赛开发规范（PCVRHyFormer）

## 1. 平台约束

### 1.1 唯一入口：`run.sh`
- 平台只执行 `run.sh`，所有启动逻辑必须收敛到这里
- `run.sh` 默认跑训练，explore/调试模式用注释切换，**不能用环境变量控制分支**（平台不支持配置自定义 env）
- 保留原始训练命令的注释，方便回溯和对比

### 1.2 环境变量（平台注入）
| 变量 | 用途 | 优先级 |
|---|---|---|
| `TRAIN_DATA_PATH` | 训练数据目录 | 高于 CLI `--data_dir` |
| `TRAIN_CKPT_PATH` | 检查点输出目录 | 高于 CLI `--ckpt_dir` |
| `TRAIN_LOG_PATH` | 日志输出目录 | 高于 CLI `--log_dir` |
| `TRAIN_TF_EVENTS_PATH` | TensorBoard 事件目录 | 高于 CLI `--tf_events_dir` |

- 所有独立脚本（包括数据探查）必须对齐这个优先级逻辑
- 本地开发 fallback：`os.environ.get('TRAIN_DATA_PATH') or 'datasets/demo_1000.parquet'`

### 1.3 输出限制
- **Eval 阶段**：最多打印 1000 行 stdout，超长的报告会被截断
- **Train 阶段**：可以大量打印，也可以用 TensorBoard
- **产物文件**：平台可能无法下载或查看，重要信息必须**打印到 log**
- 因此：数据探查报告必须同时输出到 `train.log` 和 stderr

### 1.4 代码提交
- 平台通过 `run.sh` 打包上传 `src/` 目录
- 不要依赖 `PROJECT_DIR`、`..` 等相对上级目录的写法
- 所有脚本必须在 `src/` 目录内自包含

## 2. 开发规范

### 2.1 Baseline 保护
- `baseline_code/` 是原始 baseline，**绝不直接修改**
- 所有开发在 `src/` 进行，`src/` 初始为 `baseline_code/` 的完整复制
- 修改前必须确认 `src/` 和 `baseline_code/` 的 diff，确保改动最小化

### 2.2 最小改动原则
- 新增功能用**独立脚本**（如 `data_explorer.py`），不耦合到训练代码
- 修改现有代码时，**注释保留旧逻辑**，不要直接删除
- CLI 参数新增默认值不要改变原有行为

### 2.3 日志规范
- 统一使用 `utils.create_logger(log_path)` 初始化
- 日志同时输出到文件和 stderr
- 关键统计信息用 `logging.info()` 而非 `print()`，确保进入 log 文件

### 2.4 数据探查规范
- 数据探查必须是**独立可执行脚本**，不依赖训练 pipeline
- 必须复用训练入口的环境变量读取逻辑（`TRAIN_DATA_PATH`）
- 报告必须同时输出到：
  1. `train.log`（平台可下载查看）
  2. stderr（实时查看）
- 不要写产物文件到平台外目录

### 2.5 训练-推理参数一致性
- `infer.py` 通过 `_FALLBACK_MODEL_CFG` **白名单**解析 `train_config.json`：只有列入该字典的 key 才会被读取，其余字段即使存在于 `train_config.json` 也会被丢弃。
- 因此，**任何新增的模型 CLI 参数（尤其是影响模型结构的布尔/枚举值，如 `use_target_attention`）必须同步追加到 `infer.py` 的 `_FALLBACK_MODEL_CFG`**，并确保默认值与 `train.py` 的 argparse default 一致。
- 若遗漏，infer 会以默认值构建模型，导致 `load_state_dict(strict=True)` 时因结构不匹配（Unexpected / Missing keys）而直接崩溃。

## 3. 数据策略

### 3.1 Demo 数据的可信度分级
| 信息 | 可信度 | 原因 |
|---|---|---|
| Schema 结构 | ⭐⭐⭐⭐⭐ | 与全量数据完全一致 |
| 序列长度分布 | ⭐⭐⭐⭐ | 形态可信，绝对值可能有偏 |
| 特征类型（int/list/float） | ⭐⭐⭐⭐⭐ | 由 schema 决定 |
| Label 分布 | ⭐⭐ | Demo 明显上采样 |
| 时间跨度 | ⭐⭐ | Demo 仅 13 分钟 |
| 用户重复率 | ⭐⭐ | Demo 无重复用户 |

### 3.2 统计脚本设计
- 必须能处理 list-type 列（`pyarrow.ListArray`）
- 必须处理 NaN/null/空列表
- 大数据量时支持 `--max_sample_rows` 采样

## 4. 训练策略决策链

基于平台约束，建议的实验顺序：

1. **P0：学习率调度**（Warm-up + Cosine）
   - 纯训练策略优化，零模型改动
   - 风险为零，回滚只需注释 `scheduler.step()`

2. **P0：数据探查**（提交 explore 模式）
   - 获取真实 label 分布、时间跨度
   - 决定 Focal Loss 参数、截断长度

3. **P1：序列截断调优**
   - 基于真实数据 P95 长度调整 `seq_max_lens`
   - 注意 GPU OOM 风险

4. **P1：Focal Loss / 样本加权**
   - 依赖真实正负样本比

5. **P2：特征工程 / MTL**
   - 工程量大，收益高

## 5. 常见陷阱

| 陷阱 | 后果 | 规避方式 |
|---|---|---|
| 用 `PROJECT_DIR` 或 `..` 定位文件 | 平台路径不一致导致文件找不到 | 只用 `SCRIPT_DIR` 和 `os.environ` |
| 环境变量控制分支逻辑 | 平台不支持自定义 env | 用注释切换，或 CLI 参数默认值 |
| 只写文件不打印 | 平台无法查看产物 | 所有关键信息 `logging.info()` |
| `infer.py` 的 `_FALLBACK_MODEL_CFG` 漏加新模型参数 | 推理时模型结构不匹配，`load_state_dict` 报错崩溃 | 新增 CLI 参数时同步更新 `infer.py` 的 fallback 白名单 |
| 直接修改 `baseline_code/` | Baseline 丢失，无法对比 | 只在 `src/` 开发 |
| Demo 数据的 label 比调参 | 上线即崩 | 必须提交 explore 获取真实分布 |
