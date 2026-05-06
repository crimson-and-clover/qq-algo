# Demo Data Schema

数据来源: `demo_1000.parquet`，共 120 列、1000 行，用于 PCVR（点击后转化率）预估。

---

## ID & Label（5列）

| 列名 | 类型 | 含义 |
|------|------|------|
| `user_id` | int64 | 用户唯一ID |
| `item_id` | int64 | 内容唯一ID |
| `label_type` | int32 | 标签：1=点击未转化，2=点击且转化 |
| `label_time` | int64 | 标签事件时间（unix秒）|
| `timestamp` | int64 | 请求/曝光时间（unix秒）|

---

## User Int Features（46列）

### 单值特征（35列）

| 列名 | nunique | range | 含义 |
|------|---------|-------|------|
| `user_int_feats_1` | 3 | [1, 4] | 性别 |
| `user_int_feats_3` | 341 | [9, 1839] | 可能：年龄/年龄段编码 |
| `user_int_feats_4` | 268 | [1, 986] | 可能：城市/地区编码 |
| `user_int_feats_48` | 52 | [3, 99] | 可能：会员/VIP等级 |
| `user_int_feats_49` | 2 | [1, 2] | 可能：是否付费会员 |
| `user_int_feats_50` | 2 | [0, 1] | 可能：是否新用户 |
| `user_int_feats_51` | 5 | [40, 150] | 可能：年龄段分箱 |
| `user_int_feats_52` | 36 | [5, 174] | 可能：活跃等级 |
| `user_int_feats_53` | 264 | [3, 557] | 可能：注册时间区间编码 |
| `user_int_feats_54` | 462 | [3, 2843] | 可能：最近登录时间区间编码 |
| `user_int_feats_55` | 13 | [8, 41] | 可能：设备类型 |
| `user_int_feats_56` | 405 | [1, 1434] | 可能：消费力指数区间 |
| `user_int_feats_57` | 105 | [2, 250] | 可能：内容偏好类目编码 |
| `user_int_feats_58` | 2 | [1, 2] | 可能：布尔标记（实名认证等）|
| `user_int_feats_59` | 8 | [1, 14] | 可能：网络类型 |
| `user_int_feats_82` | 23 | [1, 23] | 可能：注册渠道 |
| `user_int_feats_86` | 61 | [2, 245] | 可能：用户画像标签 |
| `user_int_feats_92` | 2 | [1, 2] | 布尔/小枚举特征 |
| `user_int_feats_93` | 36 | [1, 37] | 可能：使用频次区间 |
| `user_int_feats_94` | 6 | [1, 6] | 小枚举特征 |
| `user_int_feats_95` | 3 | [1, 3] | 小枚举特征 |
| `user_int_feats_96` | 3 | [1, 3] | 小枚举特征 |
| `user_int_feats_97` | 3 | [1, 3] | 小枚举特征 |
| `user_int_feats_98` | 2 | [1, 3] | 布尔/小枚举特征 |
| `user_int_feats_99` | 2 | [1, 2] | 布尔特征 |
| `user_int_feats_100` | 2 | [1, 2] | 布尔特征 |
| `user_int_feats_101` | 2 | [2, 3] | 布尔特征 |
| `user_int_feats_102` | 2 | [1, 3] | 布尔特征 |
| `user_int_feats_103` | 3 | [1, 3] | 小枚举特征 |
| `user_int_feats_104` | 3 | [1, 3] | 小枚举特征 |
| `user_int_feats_105` | 3 | [1, 3] | 小枚举特征 |
| `user_int_feats_106` | 3 | [1, 3] | 小枚举特征 |
| `user_int_feats_107` | 2 | [1, 2] | 布尔特征 |
| `user_int_feats_108` | 6 | [2, 7] | 可能：近期活跃天数 |
| `user_int_feats_109` | 7 | [1, 7] | 可能：近期转化次数 |

> `user_int_feats_92` ~ `user_int_feats_109` 为一组布尔/小枚举特征，可能为设备环境、功能开关、行为偏好等标记。

### 多值特征（11列）

| 列名 | list_len | nunique | 含义 |
|------|----------|---------|------|
| `user_int_feats_15` | [0, 13] | 447 | 可能：兴趣标签列表 |
| `user_int_feats_60` | [0, 2] | 1 | 常量特征（可能：性别扩展）|
| `user_int_feats_62` | [0, 5] | 7 | 多值特征，与 `user_dense_feats_62` 对齐 |
| `user_int_feats_63` | [0, 11] | 33 | 多值特征，与 `user_dense_feats_63` 对齐 |
| `user_int_feats_64` | [0, 18] | 45 | 多值特征，与 `user_dense_feats_64` 对齐 |
| `user_int_feats_65` | [0, 49] | 218 | 多值特征，与 `user_dense_feats_65` 对齐 |
| `user_int_feats_66` | [0, 66] | 533 | 多值特征，与 `user_dense_feats_66` 对齐 |
| `user_int_feats_80` | [0, 5] | 12 | 多值特征，可能：设备能力标签 |
| `user_int_feats_89` | [0, 10] | 5 | 多值特征，与 `user_dense_feats_89` 对齐 |
| `user_int_feats_90` | [0, 10] | 6 | 多值特征，与 `user_dense_feats_90` 对齐 |
| `user_int_feats_91` | [0, 10] | 7 | 多值特征，与 `user_dense_feats_91` 对齐 |

### 用户 Embedding 说明

| 名称 | 列名 | 维度 | 全称 | 说明 |
|------|------|------|------|------|
| SUM | `user_dense_feats_61` | 256 | Social User Modeling | 基于用户社交关系和行为训练的用户 embedding，侧重刻画社交属性和兴趣偏好。值域窄，可能经过归一化处理。 |
| LMF4Ads | `user_dense_feats_87` | 320 | Latent Matrix Factorization for Ads | 基于广告场景的隐语义模型产出的用户 embedding，侧重刻画用户在商业化/广告场景中的意图和偏好。值域较宽，典型 embedding 输出范围。 |

两者均为上游预训练模型的产出，作为预提取特征直接输入模型，不需端到端训练。SUM 偏社交/内容侧表征，LMF4Ads 偏商业化/广告侧表征，互补提供先验知识。

> fid 62-66, 89-91 的 int 列与对应 dense 列逐元素对齐：int 存实体/类目ID，dense 存该实体的统计量（可能：停留时长、互动频次、偏好得分等）。

---

## User Dense Features（10列）

| 列名 | 维度 | 值域 | 含义 |
|------|------|------|------|
| `user_dense_feats_61` | 256 | [-0.25, 0.20] | SUM 用户 embedding（见下方说明）|
| `user_dense_feats_87` | 320 | [-0.68, 0.68] | LMF4Ads 用户 embedding（见下方说明）|
| `user_dense_feats_62` | 2 | [6K, 32K] | 对齐 `user_int_feats_62`，可能：停留时长/互动统计 |
| `user_dense_feats_63` | 4 | [713, 60K] | 对齐 `user_int_feats_63`，可能：互动频次/得分 |
| `user_dense_feats_64` | 3 | [713, 32K] | 对齐 `user_int_feats_64`，可能：消费时长统计 |
| `user_dense_feats_65` | 4 | [298, 17K] | 对齐 `user_int_feats_65`，可能：偏好得分 |
| `user_dense_feats_66` | 4 | [298, 17K] | 对齐 `user_int_feats_66`，可能：点击/消费统计 |
| `user_dense_feats_89` | 10 | [-0.55, 0.70] | 对齐 `user_int_feats_89`，可能：embedding 或归一化统计量 |
| `user_dense_feats_90` | 10 | [-0.56, 0.70] | 对齐 `user_int_feats_90`，可能：embedding 或归一化统计量 |
| `user_dense_feats_91` | 10 | [-0.40, 0.71] | 对齐 `user_int_feats_91`，可能：embedding 或归一化统计量 |

---

## Item Int Features（14列）

### 单值特征（13列）

| 列名 | nunique | range | 含义 |
|------|---------|-------|------|
| `item_int_feats_5` | 82 | [4, 325] | 可能：一级类目 |
| `item_int_feats_6` | 216 | [0, 977] | 可能：二级类目 |
| `item_int_feats_7` | 349 | [0, 2806] | 可能：三级类目/子分类 |
| `item_int_feats_8` | 226 | [-1, 2431] | 可能：创作者/作者ID编码 |
| `item_int_feats_9` | 24 | [3, 37] | 可能：内容格式/类型 |
| `item_int_feats_10` | 110 | [2, 309] | 可能：内容来源/渠道 |
| `item_int_feats_12` | 352 | [0, 2777] | 可能：发布时间编码 |
| `item_int_feats_13` | 8 | [1, 8] | 可能：质量等级 |
| `item_int_feats_16` | 662 | [2, 35259] | 可能：内容ID编码 |
| `item_int_feats_81` | 3 | [0, 2] | 可能：内容状态 |
| `item_int_feats_83` | 22 | [1, 31] | 可能：内容时长分档 |
| `item_int_feats_84` | 66 | [3, 226] | 可能：互动热度区间 |
| `item_int_feats_85` | 103 | [4, 1001] | 可能：消费完成率区间 |

### 多值特征（1列）

| 列名 | list_len | nunique | 含义 |
|------|----------|---------|------|
| `item_int_feats_11` | [0, 20] | 924 | 内容标签列表 |

---

## Domain Sequence Features（45列）

每个域为用户在该场景下的历史行为序列，每列是一个变长 int 数组，同一域内各列等长对齐。

### domain_a — 可能：信息流/推荐浏览（9列，seq_len ≤ 1888）

| 列名 | vocab | 含义 |
|------|-------|------|
| `domain_a_seq_38` | 120万 | 可能：内容ID |
| `domain_a_seq_39` | 3.2亿 | **时间戳** |
| `domain_a_seq_40` | 17 | 可能：行为类型（曝光/点击/停留等）|
| `domain_a_seq_41` | 9 | 可能：展示位置/场景 |
| `domain_a_seq_42` | 1017 | 可能：一级类目 |
| `domain_a_seq_43` | 3449 | 可能：二级类目 |
| `domain_a_seq_44` | 15146 | 可能：作者ID |
| `domain_a_seq_45` | 9212 | 可能：来源渠道 |
| `domain_a_seq_46` | 17 | 可能：交互方式 |

### domain_b — 可能：搜索行为（14列，seq_len ≤ 1952）

| 列名 | vocab | 含义 |
|------|-------|------|
| `domain_b_seq_67` | 4.2亿 | **时间戳** |
| `domain_b_seq_68` | 23 | 可能：行为类型（搜索/点击/翻页/筛选等）|
| `domain_b_seq_69` | 1.4亿 | 可能：内容ID |
| `domain_b_seq_70` | 429 | 可能：一级类目 |
| `domain_b_seq_71` | 1344 | 可能：二级类目 |
| `domain_b_seq_72` | 3663 | 可能：作者/来源 |
| `domain_b_seq_73` | 2580 | 可能：搜索结果位置 |
| `domain_b_seq_74` | 63万 | 可能：搜索query ID |
| `domain_b_seq_75` | 20 | 可能：搜索类型 |
| `domain_b_seq_76` | 16万 | 可能：搜索意图/实体ID |
| `domain_b_seq_77` | 127 | 可能：搜索类目 |
| `domain_b_seq_78` | 2522 | 可能：点击结果类目 |
| `domain_b_seq_79` | 6368 | 可能：搜索来源 |
| `domain_b_seq_88` | 28万 | 可能：关联推荐内容ID |

### domain_c — 可能：内容深度消费（12列，seq_len ≤ 3894）

| 列名 | vocab | 含义 |
|------|-------|------|
| `domain_c_seq_27` | 4.4亿 | **时间戳** |
| `domain_c_seq_28` | 59 | 可能：行为类型（播放/暂停/快进/完播/收藏等）|
| `domain_c_seq_29` | 822万 | 可能：内容ID |
| `domain_c_seq_30` | 509 | 可能：一级类目 |
| `domain_c_seq_31` | 2948 | 可能：二级类目 |
| `domain_c_seq_32` | 6 | 可能：内容时长分档 |
| `domain_c_seq_33` | 3 | 可能：内容格式 |
| `domain_c_seq_34` | 198万 | 可能：创作者ID |
| `domain_c_seq_35` | 1587 | 可能：内容来源 |
| `domain_c_seq_36` | 151万 | 可能：内容标签/实体ID |
| `domain_c_seq_37` | 9974 | 可能：播放场景 |
| `domain_c_seq_47` | 2.7亿 | 可能：曝光/推荐链路ID |

### domain_d — 可能：社交互动（10列，seq_len ≤ 3951）

| 列名 | vocab | 含义 |
|------|-------|------|
| `domain_d_seq_17` | 4 | 可能：互动类型（点赞/评论/转发/关注）|
| `domain_d_seq_18` | 422 | 可能：一级类目 |
| `domain_d_seq_19` | 1466 | 可能：二级类目 |
| `domain_d_seq_20` | 3842 | 可能：作者ID |
| `domain_d_seq_21` | 2451 | 可能：内容来源 |
| `domain_d_seq_22` | 51万 | 可能：内容ID |
| `domain_d_seq_23` | 67万 | 可能：互动对象ID |
| `domain_d_seq_24` | 24 | 可能：互动场景 |
| `domain_d_seq_25` | 11 | 可能：互动深度 |
| `domain_d_seq_26` | 5万 | **时间戳**（nunique 较少，可能精度为分钟级）|

---

## 备注

- 未标注"可能"的列为确定性较高的推断（如时间戳、embedding、官方说明项）。
- `user_int_feats_92` ~ `user_int_feats_109` 等 2~3 值布尔/小枚举特征，具体语义无法从数据统计推断，需业务方确认。
- 序列域的 side-info 列（类目、作者等）仅按 vocab 大小推测层级关系，具体含义不确定。
