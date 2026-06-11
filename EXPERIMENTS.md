# 实验归档

最后更新：2026-06-10

## 当前最佳方案

当前主线模型流程：

```text
完整对话历史 Query
  -> 包含 tag_list 的 BM25 召回前 400 个候选
  -> 构造以当前请求优先的聚焦 Query
  -> 在标签候选上继续训练的 Listwise MiniLM-L6 Cross-Encoder 重排序
  -> 选取前 20 首歌曲
  -> 豆包生成自然语言回复
```

Blind A 当前最佳排序结果：

| 指标 | 分数 |
|---|---:|
| ndcg@20 | **0.1946** |
| catalog_diversity | **0.0318** |
| lexical_diversity | 0.7300 |
| llm_judge_score | **4.7500** |
| composite_score | **0.4547** |

当前最高综合分为 `0.4559`，来自“混合训练模型 + Dense Hybrid 候选”。
但该提交的 `ndcg@20` 仅为 `0.1806`，明显低于最佳排序主线的 `0.1946`；
综合分新高主要来自更高的回复评分，因此暂不替换当前排序主线。

当前保留的标准产物：

```text
exp/reranker/minilm_bm25_tags_top400_e1/
exp/inference/blindset_A/tags_top400_e1_empty.json
exp/inference/blindset_A/tags_top400_e1_prediction.json
exp/inference/blindset_A/tags_top400_e1_submission.zip
```

`tags_top400_e1_submission.zip` 中的 `prediction.json` 与单独保留的
`tags_top400_e1_prediction.json` 完全一致。

## 分数历史

下表记录了实验过程中所有已知的提交或评测结果。部分早期提交没有使用唯一
文件名保存，因此对应的原始输出文件已经无法逐一恢复。

| 实验 | ndcg@20 | 目录多样性 | 词汇多样性 | LLM 评分 | 综合分 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 早期 Baseline A | 0.1357 | 0.0214 | 0.5473 | 1.5000 | 0.1622 | 排序相对较好，但回复质量较弱 |
| 早期 Baseline B | 0.0250 | 0.0072 | 0.5397 | 1.9500 | 召回能力较弱 |
| 部分数据 Reranker 实验 | 0.1274 | 0.0296 | 0.6126 | 3.4000 | 有价值的中间结果 |
| 全量 Reranker 实验 | 0.1488 | 0.0299 | 0.6218 | 3.2500 | 排序能力提升 |
| 较差的重新生成回复 | 0.1488 | 0.0299 | 0.3399 | 1.4000 | 排序不变，但回复质量严重下降 |
| 较好的重新生成回复 | 0.1488 | 0.0299 | 0.7489 | 4.1500 | 上一阶段最佳，综合分 0.3885 |
| Hybrid 召回快速实验 | 0.0222 | 0.0248 | 0.7275 | 4.5000 | Hybrid 召回严重损害排序效果 |
| Hybrid MiniLM-L12 实验 | 0.0351 | 0.0229 | 0.7238 | 3.8000 | 仍明显弱于 BM25 主线 |
| Listwise 聚焦式 Reranker，训练 3 轮 | 0.1842 | 0.0299 | 0.7365 | 4.3500 | 排序能力大幅提升 |
| Listwise 聚焦式 Reranker，继续训练 2 轮 | 0.1877 | 0.0297 | **0.7485** | 4.6500 | 上一阶段最佳，综合分 0.4454 |
| 标签 BM25 Top-400 + 候选匹配重训 1 轮 | **0.1946** | **0.0318** | 0.7300 | **4.7500** | 当前最佳，综合分 **0.4547** |
| 标签 Top-400 + 模型困难负例训练 1 轮 | 0.1823 | 0.0316 | 0.7378 | 4.4000 | Dev 提升但 Blind A 退化，综合分 0.4231 |
| 混合负例训练 + Dense Hybrid 候选 | 0.1806 | 0.0316 | 0.7369 | **4.8500** | 综合分 **0.4559** 创新高，但排序明显退化 |
| 混合负例训练 + 纯 BM25 Top-400 对照 | 0.1806 | 0.0316 | 0.7369 | **4.8500** | 与 Dense Hybrid 五项显示分数一致，退化来自混合训练模型 |

## 有效改进

1. BM25 召回使用包含完整历史的 `legacy` Query。离线比较表明，它的召回率
   高于仅使用聚焦 Query 的方案。
2. Reranker 使用去噪后的 `focused` Query。它将当前用户请求放在最前面，
   仅保留有用的近期用户反馈和历史歌曲信息。
3. 使用 Listwise Softmax Loss 训练，每组包含 1 个正样本和 19 个清洗后的
   负样本。
4. 过滤可能是假负样本的歌曲，包括与正样本歌手相同、专辑相同或标签高度
   相似的歌曲。
5. 在 Listwise 模型基础上，使用更低的 `3e-6` 学习率继续训练 2 轮。
6. 排序推理和回复生成分开执行，确保回复实验不会改变歌曲排序结果。
7. 将 `tag_list` 加入 BM25，并扩大候选池至 Top-400；随后在新候选分布上
   继续训练 1 轮，使 Blind A `ndcg@20` 从 `0.1877` 提升至 `0.1946`。

## 无效或效果较差的尝试

1. Hybrid 召回实验使 `ndcg@20` 大幅下降至 `0.0222-0.0351`。
2. 回复生成会显著影响词汇多样性和 LLM 评分，但不会改变 `ndcg@20`。
3. 仅训练 Reranker 无法找回未进入 BM25 前 200 个候选的相关歌曲。
4. 继续训练的排序收益开始递减：最后 2 轮仅将 `ndcg@20` 从 `0.1842`
   提升至 `0.1877`。
5. 模型困难负例训练在完整 Dev 上将 `nDCG@20` 从 `0.1112` 提升至
   `0.1152`，但 Blind A 从 `0.1946` 下降至 `0.1823`，表现出明显的
   分布过拟合，不能作为当前主线。
6. 混合负例训练 + Dense Hybrid 候选在固定 Dev 子集上略有提升，但 Blind A
   `ndcg@20` 从 `0.1946` 降至 `0.1806`。Dense 候选的离线收益没有泛化到
   Blind A，暂不作为排序主线。
7. 将同一个混合负例训练模型的候选集从 Dense Hybrid 改回纯 BM25 Top-400
   后，Blind A 五项指标在日志显示精度下完全不变。因此本轮 `ndcg@20` 退化
   的主要原因是混合负例继续训练改变了 reranker，而不是推理阶段加入 Dense
   候选。这条训练路线停止，不再追加轮数。

## 当前最佳模型的训练过程

当前最佳模型以旧 BM25 Top-200 主线的最佳模型为初始权重，在标签 BM25
Top-400 候选分布上继续训练 1 轮：

```powershell
.\.venv\Scripts\python.exe train_reranker.py `
  --model_name ./exp/reranker/minilm_bm25_listwise_focused_continue_e2 `
  --output_dir ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --corpus_types track_name artist_name album_name release_date tag_list `
  --retrieval_topk 400 `
  --negatives_per_positive 19 `
  --max_turns 50000 `
  --epochs 1 `
  --batch_size 2 `
  --gradient_accumulation_steps 8 `
  --max_length 384 `
  --learning_rate 2e-6 `
  --retrieval_query_mode legacy `
  --reranker_query_mode focused `
  --history_turns 3 `
  --seed 13 `
  --device cuda
```

标签候选重训后的 checkpoint 包含完整且可独立加载的模型权重。

## 复现当前最佳提交

运行排序推理：

```powershell
.\.venv\Scripts\python.exe run_inference_rerank_blindset.py `
  --tid bm25_tags_doubao_blindset_A `
  --reranker_dir ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --output_name tags_top400_e1_empty `
  --candidate_topk 400 `
  --rerank_batch_size 64 `
  --max_length 384 `
  --retrieval_query_mode legacy `
  --reranker_query_mode focused `
  --history_turns 3 `
  --response_mode empty `
  --device cuda
```

生成自然语言回复：

```powershell
.\.venv\Scripts\python.exe regenerate_responses_blindset.py `
  --tid bm25_tags_doubao_blindset_A `
  --input_path exp/inference/blindset_A/tags_top400_e1_empty.json `
  --output_path exp/inference/blindset_A/tags_top400_e1_prediction.json
```

打包提交文件：

```powershell
.\.venv\Scripts\python.exe -c "import zipfile,pathlib; d=pathlib.Path('exp/inference/blindset_A'); z=d/'tags_top400_e1_submission.zip'; z.unlink(missing_ok=True); f=zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED); f.write(d/'tags_top400_e1_prediction.json','prediction.json'); f.close(); print(z.resolve())"
```

## 清理记录

归档后已删除：

```text
exp/reranker/minilm_bm25_full/
exp/reranker/minilm_bm25_listwise_focused/
exp/inference/blindset_A/bm25_listwise_focused_empty.json
exp/inference/blindset_A/bm25_rerank_doubao_blindset_A.json
exp/inference/blindset_A/submission1.zip
源码目录中的 __pycache__
```

保留 Hugging Face/BM25 的 `cache/` 目录和项目 `.venv/`，因为它们能够避免
重复下载依赖和数据，便于后续高效复现实验。

## 标签 BM25 Top-400 实验

当前 BM25 只索引歌名、歌手、专辑和发行日期。完整 Dev 的召回实验表明，
加入 `tag_list` 是目前收益最明确的改进：

| 召回器 | Recall@20 | Recall@100 | Recall@200 |
|---|---:|---:|---:|
| 当前 BM25 | 0.1886 | 0.3151 | 0.3604 |
| 加入 `tag_list` | **0.2725** | **0.4421** | **0.4968** |

标签 BM25 的候选数量曲线：

| 候选数量 | Recall |
|---:|---:|
| 200 | 0.4968 |
| 300 | 0.5350 |
| 400 | **0.5598** |
| 500 | 0.5809 |

因此下一轮使用 `tag_list` BM25 Top-400，并基于新候选重新训练 Reranker。
不能只在推理时替换候选，因为新候选的困难负样本分布与旧模型训练时不同。

使用固定随机种子抽取 10 个 Dev 会话进行兼容性检查：

| 候选与模型 | Candidate Recall | Final nDCG@20 | 已召回样本条件 nDCG@20 |
|---|---:|---:|---:|
| 旧 BM25 Top-200 + 当前模型 | 0.2500 | 0.1051 | **0.4202** |
| 标签 BM25 Top-400 + 当前模型 | **0.4375** | **0.1077** | 0.2461 |

该小样本结果不能用于判断最终增益，但清楚显示了两个现象：标签候选显著扩大
召回范围；旧 Reranker 对新候选的排序能力下降。因此必须在标签候选上重新训练。

标签候选重训一轮后，在固定随机种子抽取的 100 个 Dev 会话上进行严格对照：

| 指标 | 旧 BM25 Top-200 + 当前最佳模型 | 标签 BM25 Top-400 + 重训模型 | 绝对变化 |
|---|---:|---:|---:|
| Candidate Recall@400 | 0.3713 | **0.5400** | **+0.1688** |
| Final Recall@20 | 0.2338 | **0.2800** | **+0.0463** |
| Final nDCG@10 | 0.0896 | **0.1059** | **+0.0163** |
| Final nDCG@20 | 0.1053 | **0.1267** | **+0.0214** |
| Final MRR | 0.0727 | **0.0883** | **+0.0156** |
| 条件 nDCG@20 | **0.2837** | 0.2347 | -0.0490 |

最终 `nDCG@20` 相对提升约 20.3%，且 Turn 1 至 Turn 8 的 `nDCG@20`
均未下降，已明显超过进入完整 Dev 评估所要求的 `+0.005` 门槛。

条件 nDCG@20 下降说明，标签 Top-400 引入了更多难以区分的候选；但扩大候选
召回带来的收益显著高于排序精度损失。下一步先跑完整 Dev，暂不继续训练。

完整官方 Dev（1000 个会话、8000 个推荐 turn）结果：

| 指标 | 标签 BM25 Top-400 | 标签 BM25 Top-400 + 重训 Reranker | 变化 |
|---|---:|---:|---:|
| Candidate Recall@20 | 0.2726 | 0.2726 | 不变 |
| Candidate Recall@200 | 0.4968 | 0.4968 | 不变 |
| Candidate Recall@400 | 0.5598 | 0.5598 | 不变 |
| Final Recall@20 | 0.2726 | 0.2624 | -0.0103 |
| Final nDCG@20 | 0.0988 | **0.1112** | **+0.0123** |
| Final MRR | 0.0557 | **0.0749** | **+0.0192** |

Reranker 将少量正确歌曲移出 Top-20，但显著改善了正确歌曲在 Top-20 内的
位置，使完整 Dev `nDCG@20` 相对标签 BM25 排序提升约 12.5%。训练集已出现
用户与未出现用户的 `nDCG@20` 分别为 `0.1107` 和 `0.1124`，没有明显的用户
分段退化。

完整 Dev 中 Turn 7 和 Turn 8 相对较弱，`nDCG@20` 分别为 `0.0871` 和
`0.0990`。后续困难负例训练和 Query 优化应重点分析后期对话，但当前模型已经
满足进入 Blind A 测试的条件。

Blind A 提交结果确认标签主线有效：

| 指标 | 上一阶段最佳 | 标签 Top-400 当前最佳 | 绝对变化 |
|---|---:|---:|---:|
| ndcg@20 | 0.1877 | **0.1946** | **+0.0069** |
| catalog_diversity | 0.0297 | **0.0318** | **+0.0021** |
| lexical_diversity | **0.7485** | 0.7300 | -0.0185 |
| llm_judge_score | 4.6500 | **4.7500** | **+0.1000** |
| composite_score | 0.4454 | **0.4547** | **+0.0093** |

`ndcg@20` 相对提升约 3.7%，综合分相对提升约 2.1%。标签候选同时增加了排序
相关性与目录覆盖率。下一步的主要瓶颈不再是是否使用标签，而是标签 Top-400
内部的精细排序；应优先进行基于当前 Reranker 高分错误候选的困难负例训练。

## 实验记录：模型困难负例训练

普通标签候选训练直接使用 BM25 排名靠前的清洗负例。困难负例训练改为先使用
当前最佳 Reranker 对候选打分，再选择模型最容易误判的错误歌曲。为了控制计算
量，第一轮只重排每条样本前 100 个清洗后的 BM25 候选：

```text
标签 BM25 Top-400
  -> 取前 100 个清洗后的错误候选
  -> 当前最佳 Reranker 打分
  -> 保存最高分的 12 个困难负例
  -> 训练时补充 7 个普通 BM25 负例
```

挖掘结果保存为 JSONL，可以中断后用相同命令续跑。元数据和统计信息保存在同名
`.meta.json` 文件中。

### 1. 挖掘并缓存困难负例

```powershell
.\.venv\Scripts\python.exe mine_hard_negatives.py `
  --model_name ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --output_path ./exp/hard_negatives/tags_top400_e1_pool100.jsonl `
  --corpus_types track_name artist_name album_name release_date tag_list `
  --retrieval_topk 400 `
  --candidate_pool_size 100 `
  --hard_negatives_per_positive 12 `
  --max_turns 50000 `
  --rerank_batch_size 64 `
  --max_length 384 `
  --retrieval_query_mode legacy `
  --reranker_query_mode focused `
  --history_turns 3 `
  --seed 13 `
  --device cuda
```

完成后预计缓存约 33,000 条已召回训练样本。重新运行相同命令会跳过已缓存样本，
继续完成剩余挖掘。

### 2. 使用困难负例继续训练一轮

```powershell
.\.venv\Scripts\python.exe train_reranker.py `
  --model_name ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --output_dir ./exp/reranker/minilm_bm25_tags_top400_hardneg_e1 `
  --corpus_types track_name artist_name album_name release_date tag_list `
  --retrieval_topk 400 `
  --negatives_per_positive 19 `
  --hard_negative_cache ./exp/hard_negatives/tags_top400_e1_pool100.jsonl `
  --hard_negatives_per_positive 12 `
  --require_hard_negative_cache_hit `
  --max_turns 50000 `
  --epochs 1 `
  --batch_size 2 `
  --gradient_accumulation_steps 8 `
  --max_length 384 `
  --learning_rate 1e-6 `
  --retrieval_query_mode legacy `
  --reranker_query_mode focused `
  --history_turns 3 `
  --seed 13 `
  --device cuda
```

`--require_hard_negative_cache_hit` 会在缓存不完整时终止训练，避免无意中混入大量
旧式负例。训练日志中的 `hard_negative_cache_hits` 应约等于缓存记录数，
`hard_negative_cache_misses` 和 `stale_hard_negatives` 应为 0。

### 3. 固定 100 个 Dev 会话验证

```powershell
.\.venv\Scripts\python.exe evaluate_devset.py `
  --tid bm25_tags_doubao_blindset_A `
  --reranker_dir ./exp/reranker/minilm_bm25_tags_top400_hardneg_e1 `
  --candidate_topk 400 `
  --rerank_batch_size 64 `
  --max_sessions 100 `
  --seed 13 `
  --skip_user_segments `
  --output_path exp/evaluation/dev_tags_top400_hardneg_100.json `
  --device cuda
```

重点观察 `conditional_final_ndcg@20` 是否从 `0.2347` 回升，同时要求固定 100
会话的 `final_ndcg@20` 高于当前模型的 `0.1267`。至少提升 `0.005` 后再运行
完整 Dev；若没有提升，则保留当前 `minilm_bm25_tags_top400_e1`。

困难负例训练完成后的固定 100 会话对照结果：

| 指标 | 标签 Top-400 当前最佳 | 困难负例训练 1 轮 | 绝对变化 |
|---|---:|---:|---:|
| Final Recall@10 | 0.1988 | **0.2250** | **+0.0263** |
| Final Recall@20 | 0.2800 | **0.2950** | **+0.0150** |
| Final nDCG@10 | 0.1059 | **0.1149** | **+0.0090** |
| Final nDCG@20 | 0.1267 | **0.1326** | **+0.0059** |
| Final MRR | 0.0883 | **0.0912** | **+0.0029** |
| 条件 nDCG@20 | 0.2347 | **0.2456** | **+0.0109** |

困难负例训练达到了 `+0.005` 的完整 Dev 评估门槛，并且同时提升了 Recall@20
和条件 nDCG@20，说明它确实改善了候选池内部排序。Turn 3、Turn 4 和 Turn 8
存在约 `0.002-0.005` 的轻微回落，因此必须通过完整 Dev 确认稳定性。

完整 Dev 评估命令：

```powershell
.\.venv\Scripts\python.exe evaluate_devset.py `
  --tid bm25_tags_doubao_blindset_A `
  --reranker_dir ./exp/reranker/minilm_bm25_tags_top400_hardneg_e1 `
  --candidate_topk 400 `
  --rerank_batch_size 64 `
  --output_path exp/evaluation/dev_tags_top400_hardneg_full.json `
  --device cuda
```

完整官方 Dev 的困难负例模型结果：

| 指标 | 标签 Top-400 当前最佳 | 困难负例训练 1 轮 | 绝对变化 |
|---|---:|---:|---:|
| Final Recall@1 | **0.0278** | 0.0273 | -0.0005 |
| Final Recall@10 | 0.1744 | **0.1880** | **+0.0136** |
| Final Recall@20 | 0.2624 | **0.2746** | **+0.0123** |
| Final nDCG@10 | 0.0889 | **0.0933** | **+0.0044** |
| Final nDCG@20 | 0.1112 | **0.1152** | **+0.0040** |
| Final MRR | 0.0749 | **0.0763** | **+0.0015** |
| 条件 nDCG@20 | 0.1986 | **0.2058** | **+0.0072** |

完整 Dev `nDCG@20` 相对提升约 3.6%，且训练集已出现用户和未出现用户分别从
`0.1107`、`0.1124` 提升至 `0.1154`、`0.1145`。Turn 1、2、3、4、6、7
均提升，Turn 5 和 Turn 8 分别轻微下降约 `0.0011` 和 `0.0025`。整体结果
稳定，困难负例模型满足进入 Blind A 测试的条件。

Blind A 提交结果未能复现 Dev 提升：

| 指标 | 标签 Top-400 当前最佳 | 困难负例模型 | 绝对变化 |
|---|---:|---:|---:|
| ndcg@20 | **0.1946** | 0.1823 | **-0.0123** |
| catalog_diversity | **0.0318** | 0.0316 | -0.0002 |
| lexical_diversity | 0.7300 | **0.7378** | +0.0078 |
| llm_judge_score | **4.7500** | 4.4000 | -0.3500 |
| composite_score | **0.4547** | 0.4231 | **-0.0316** |

困难负例模型的 Blind A `ndcg@20` 相对下降约 6.3%。由于 `ndcg@20` 与回复
生成无关，该退化确认来自排序模型，而不是豆包回复的随机性。困难负例来自训练
数据中当前模型的高分错误候选，比例和强度较高，使模型更适应 Train/Dev 分布，
但削弱了对 Blind A 的泛化能力。因此停止继续训练该模型，恢复
`minilm_bm25_tags_top400_e1` 为当前主线。

保留的失败实验产物：

```text
exp/reranker/minilm_bm25_tags_top400_hardneg_e1/
exp/hard_negatives/tags_top400_e1_pool100.jsonl
exp/inference/blindset_A/tags_top400_hardneg_e1_empty.json
exp/inference/blindset_A/tags_top400_hardneg_e1_prediction.json
exp/inference/blindset_A/tags_top400_hardneg_e1_submission.zip
```

### 完整 Dev 评估

`evaluate_devset.py` 使用官方 Dev 的 1000 个完整会话、8000 个推荐 turn，
评估完整候选池，而不是只评估训练时抽样出的候选组。报告包含：

```text
候选 Recall@20/100/200/400
最终 Recall、MRR、nDCG@1/10/20
正确歌曲已召回时的条件 nDCG@20
每个 turn 的召回率和 nDCG@20
训练用户中已出现/未出现用户的分段指标
```

先只评估标签 BM25 召回：

```powershell
.\.venv\Scripts\python.exe evaluate_devset.py `
  --tid bm25_tags_doubao_blindset_A `
  --candidate_topk 400 `
  --output_path exp/evaluation/dev_tags_top400_retrieval.json
```

评估当前最佳 Reranker 在标签候选上的表现。先运行 100 个 Dev 会话快速判断，
确认没有严重下降后再删除 `--max_sessions 100` 跑完整 Dev：

```powershell
.\.venv\Scripts\python.exe evaluate_devset.py `
  --tid bm25_tags_doubao_blindset_A `
  --reranker_dir ./exp/reranker/minilm_bm25_listwise_focused_continue_e2 `
  --candidate_topk 400 `
  --rerank_batch_size 64 `
  --max_sessions 100 `
  --output_path exp/evaluation/dev_tags_top400_current_reranker_100.json `
  --device cuda
```

### 基于标签候选继续训练

```powershell
.\.venv\Scripts\python.exe train_reranker.py `
  --model_name ./exp/reranker/minilm_bm25_listwise_focused_continue_e2 `
  --output_dir ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --corpus_types track_name artist_name album_name release_date tag_list `
  --retrieval_topk 400 `
  --negatives_per_positive 19 `
  --max_turns 50000 `
  --epochs 1 `
  --batch_size 2 `
  --gradient_accumulation_steps 8 `
  --max_length 384 `
  --learning_rate 2e-6 `
  --retrieval_query_mode legacy `
  --reranker_query_mode focused `
  --history_turns 3 `
  --seed 13 `
  --device cuda
```

训练完成后，使用相同标签候选运行 Dev 评估：

```powershell
.\.venv\Scripts\python.exe evaluate_devset.py `
  --tid bm25_tags_doubao_blindset_A `
  --reranker_dir ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --candidate_topk 400 `
  --rerank_batch_size 64 `
  --output_path exp/evaluation/dev_tags_top400_retrained.json `
  --device cuda
```

只有完整 Dev `final_ndcg@20` 提升时，才运行 Blind A 推理并提交。建议将
`+0.005` 绝对 nDCG@20 提升作为进入盲测的最低门槛。

通过 Dev 门槛后的 Blind A 排序推理：

```powershell
.\.venv\Scripts\python.exe run_inference_rerank_blindset.py `
  --tid bm25_tags_doubao_blindset_A `
  --reranker_dir ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --output_name tags_top400_e1_empty `
  --candidate_topk 400 `
  --rerank_batch_size 64 `
  --max_length 384 `
  --retrieval_query_mode legacy `
  --reranker_query_mode focused `
  --history_turns 3 `
  --response_mode empty `
  --device cuda
```

生成回复与打包时，将输入文件改为
`exp/inference/blindset_A/tags_top400_e1_empty.json`，其余命令与当前最佳提交
的复现流程相同。

## 语义向量补召回实验

### 目的与实现

本实验不替换当前最佳标签 BM25，而是增加一条独立语义召回通道：

```text
标签 BM25 Top-400（legacy 查询）
           +
Qwen3 metadata Dense Top-K（focused 查询）
           ↓
去重并集候选
           ↓
当前最佳 minilm_bm25_tags_top400_e1 reranker
```

Dense 曲库向量直接使用官方
`talkpl-ai/TalkPlayData-Challenge-Track-Embeddings/all_tracks` 中的
`metadata-qwen3_embedding_0.6b`。查询使用 `Qwen/Qwen3-Embedding-0.6B`
编码，并按官方格式添加音乐检索任务指令。首次运行会把 47,071 首歌曲向量缓存到：

```text
cache/dense/all_tracks__metadata-qwen3_embedding_0.6b/
```

官方数据中有 492 首歌的 metadata 向量为空。实现会保留这些 track ID，并用
1024 维零向量占位，避免索引错位；这些歌曲无法由 metadata Dense 通道召回。

### 候选召回对照

固定随机种子 `13`，抽取 100 个 Dev 会话，共 800 个推荐 turn：

| 候选方案 | 候选召回率 | 相比 BM25 额外找回目标 | 平均候选数 |
|---|---:|---:|---:|
| BM25 Top-400 | 0.5400 | - | 400.0 |
| BM25 Top-400 + Dense Top-50 | 0.55875 | 15 | 435.45 |
| BM25 Top-400 + Dense Top-100 | 0.56750 | 22 | 478.44 |
| BM25 Top-400 + Dense Top-200 | **0.58500** | **36** | 569.93 |

Dense 通道确实能够找回 BM25 漏掉的歌曲，但直接把所有候选交给旧 reranker
会产生分布偏移。Dense Top-200 无惩罚排序中，36 个新增正确目标只有 6 个进入
最终 Top-20，而且错误 Dense 候选会挤掉原有 BM25 候选：

| 方案 | Final Recall@20 | Final nDCG@20 |
|---|---:|---:|
| BM25 Top-400 当前最佳 | 0.28000 | 0.126738 |
| BM25 Top-400 + Dense Top-200，无惩罚 | 0.27875 | 0.122145 |

### 通道惩罚

为保护当前 BM25 主线，对仅由 Dense 召回的候选应用固定 reranker 分数惩罚：

```text
adjusted_score = reranker_score - dense_penalty
```

Dense Top-100 的惩罚扫描结果：

| dense_penalty | Final Recall@20 | Final nDCG@20 |
|---:|---:|---:|
| 0.00 | 0.28125 | 0.123238 |
| 0.50 | 0.28125 | 0.124432 |
| 1.00 | 0.28375 | 0.126694 |
| 1.50 | 0.28375 | 0.126970 |
| 2.00 | **0.28375** | **0.127228** |

相对同样本 BM25 baseline，`dense_penalty=2.0` 的 nDCG@20 绝对提升仅
`+0.00049`。这说明融合方式方向正确，但提升远低于进入盲测的 `+0.005`
门槛，因此当前最佳提交仍保持标签 BM25 Top-400 主线。

### 复现命令

只评估候选补召回：

```powershell
.\.venv\Scripts\python.exe evaluate_hybrid_recall_devset.py `
  --max_sessions 100 `
  --bm25_topk 400 `
  --dense_topk 100 `
  --dense_batch_size 16 `
  --output_path exp/evaluation/dev_hybrid_recall_100sessions_dense100.json `
  --device cuda
```

扫描 Dense 通道惩罚并评估最终排序：

```powershell
.\.venv\Scripts\python.exe evaluate_hybrid_devset.py `
  --max_sessions 100 `
  --bm25_topk 400 `
  --dense_topk 100 `
  --dense_penalties 0 0.25 0.5 0.75 1.0 1.5 2.0 `
  --dense_batch_size 16 `
  --rerank_batch_size 64 `
  --output_path exp/evaluation/dev_hybrid_dense100_penalty_scan_100sessions.json `
  --device cuda
```

生成 Dense Top-100、惩罚 2.0 的 Blind A 空回复排序文件：

```powershell
.\.venv\Scripts\python.exe run_inference_hybrid_dense_blindset.py `
  --reranker_dir ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --bm25_topk 400 `
  --dense_topk 100 `
  --dense_penalty 2.0 `
  --dense_batch_size 16 `
  --rerank_batch_size 64 `
  --output_name tags_top400_dense100_penalty2_empty_blindset_A `
  --device cuda
```

实验文件：

```text
exp/inference/blindset_A/tags_top400_dense100_penalty2_empty_blindset_A.json
```

下一步若继续该方向，不应盲目扩大 Dense Top-K；应使用 BM25 与 Dense 混合候选
重新训练一个能识别召回通道分布的 reranker，或训练轻量 gating 模型判断何时
启用 Dense 补召回。

## BM25 + Dense 混合候选训练

### 训练设计

为让 reranker 真正适配 Dense 独有候选，新增离线 Dense 训练候选缓存，并从
当前最佳 `minilm_bm25_tags_top400_e1` 继续训练一轮。

每个 listwise 训练组仍包含一个正样本和 19 个负样本：

```text
4 个 Dense Top-100 独有负例
+ 15 个 BM25 Top-400 负例
+ 1 个正样本
```

Dense 负例只选取不在 BM25 Top-400 中的歌曲，并继续应用同歌手、同专辑和
标签重叠过滤。这样既让模型接触新召回通道，又避免 Dense 候选主导训练。

正式配置：

```text
初始模型：minilm_bm25_tags_top400_e1
Dense 候选：metadata-qwen3_embedding_0.6b Top-100
训练 turn：50,000
Dense 独有负例：每组 4 个
总负例：每组 19 个
学习率：5e-7
训练轮数：1
```

### 训练候选统计

50,000 个训练 turn 的候选情况：

| 指标 | 数值 |
|---|---:|
| BM25 Top-400 正样本命中 | 33,131 |
| BM25 Recall@400 | 0.66262 |
| Dense Top-100 正样本命中 | 17,536 |
| Dense Recall@100 | 0.35072 |
| Dense 单独补回正样本 | 670 |
| BM25 + Dense 并集命中 | 33,801 |
| 并集召回率 | 0.67602 |
| 实际使用 Dense 独有负例 | 134,994 |

Dense 使训练候选召回绝对提升 `+0.01340`，并增加了 670 个原本无法用于训练的
正样本。正式模型保存于：

```text
exp/reranker/minilm_tags_top400_dense100_mix4_e1/
```

### 固定 100 Dev 会话对照

相同随机种子 `13`，共 800 个推荐 turn：

| 模型与候选方案 | Final Recall@20 | Final nDCG@20 |
|---|---:|---:|
| 当前最佳旧模型 + BM25 Top-400 | 0.28000 | 0.126738 |
| 混合训练模型 + BM25 Top-400 | 0.28250 | 0.127715 |
| 旧模型 + Hybrid Dense-100 penalty 2.0 | 0.28375 | 0.127228 |
| 混合训练模型 + Hybrid Dense-100 penalty 0.5 | **0.28875** | 0.127771 |
| 混合训练模型 + Hybrid Dense-100 penalty 2.0 | 0.28625 | **0.128802** |

结论：

- 混合训练没有损害纯 BM25 主线，纯 BM25 nDCG@20 提升约 `+0.00098`。
- 使用 Hybrid Dense-100、惩罚 2.0 后，相对旧主线 nDCG@20 提升约
  `+0.00206`，Recall@20 提升 `+0.00625`。
- 惩罚 0.5 的 Recall@20 最高，但惩罚 2.0 的 nDCG@20 最高，说明 Dense 独有
  候选仍需保守准入。
- 提升方向成立，但仍低于 `+0.005` 的正式替换门槛，因此不覆盖当前最佳提交。

### 复现命令

缓存 50k Dense Top-100 训练候选，支持断点续跑：

```powershell
.\.venv\Scripts\python.exe cache_dense_candidates.py `
  --output_path ./exp/dense_candidates/train_dense_top100_focused_50k.jsonl `
  --dense_topk 100 `
  --max_turns 50000 `
  --write_batch_size 1024 `
  --dense_batch_size 16 `
  --device cuda
```

从当前最佳模型进行混合候选训练：

```powershell
.\.venv\Scripts\python.exe train_reranker.py `
  --model_name ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --output_dir ./exp/reranker/minilm_tags_top400_dense100_mix4_e1 `
  --corpus_types track_name artist_name album_name release_date tag_list `
  --retrieval_topk 400 `
  --negatives_per_positive 19 `
  --dense_candidate_cache ./exp/dense_candidates/train_dense_top100_focused_50k.jsonl `
  --dense_candidate_pool_size 100 `
  --dense_negatives_per_positive 4 `
  --require_dense_candidate_cache_hit `
  --max_turns 50000 `
  --epochs 1 `
  --batch_size 2 `
  --gradient_accumulation_steps 8 `
  --max_length 384 `
  --learning_rate 5e-7 `
  --retrieval_query_mode legacy `
  --reranker_query_mode focused `
  --history_turns 3 `
  --seed 13 `
  --device cuda
```

评估混合模型在 Hybrid 候选上的通道惩罚：

```powershell
.\.venv\Scripts\python.exe evaluate_hybrid_devset.py `
  --reranker_dir ./exp/reranker/minilm_tags_top400_dense100_mix4_e1 `
  --max_sessions 100 `
  --bm25_topk 400 `
  --dense_topk 100 `
  --dense_penalties 0 0.25 0.5 0.75 1.0 1.5 2.0 `
  --dense_batch_size 16 `
  --rerank_batch_size 64 `
  --output_path exp/evaluation/dev_dense_mix4_hybrid_dense100_penalty_scan_100sessions.json `
  --device cuda
```

生成最佳 nDCG 参数对应的 Blind A 空回复排序文件：

```powershell
.\.venv\Scripts\python.exe run_inference_hybrid_dense_blindset.py `
  --reranker_dir ./exp/reranker/minilm_tags_top400_dense100_mix4_e1 `
  --bm25_topk 400 `
  --dense_topk 100 `
  --dense_penalty 2.0 `
  --output_name tags_top400_dense100_mix4_penalty2_empty_blindset_A `
  --device cuda
```

生成文件：

```text
exp/inference/blindset_A/tags_top400_dense100_mix4_penalty2_empty_blindset_A.json
```

### Blind A 正式结果与纯 BM25 对照

混合训练模型 + Dense Hybrid 候选的正式 Blind A 结果：

| 指标 | 分数 | 相对最佳排序主线 |
|---|---:|---:|
| ndcg@20 | 0.1806 | -0.0140 |
| catalog_diversity | 0.0316 | -0.0002 |
| lexical_diversity | 0.7369 | +0.0069 |
| llm_judge_score | 4.8500 | +0.1000 |
| composite_score | **0.4559** | +0.0012 |

该结果表明，`composite_score` 的小幅提升来自回复指标，而非歌曲排序。为了
隔离 Dense 候选和混合训练模型的影响，已使用同一个混合训练 checkpoint，
仅保留纯 BM25 Top-400 候选重新推理：

```powershell
.\.venv\Scripts\python.exe run_inference_rerank_blindset.py `
  --tid bm25_tags_doubao_blindset_A `
  --reranker_dir ./exp/reranker/minilm_tags_top400_dense100_mix4_e1 `
  --output_name tags_top400_dense_mix4_bm25only_empty `
  --candidate_topk 400 `
  --rerank_batch_size 64 `
  --max_length 384 `
  --retrieval_query_mode legacy `
  --reranker_query_mode focused `
  --history_turns 3 `
  --response_mode empty `
  --device cuda
```

已生成并校验：

```text
exp/inference/blindset_A/tags_top400_dense_mix4_bm25only_empty.json
```

该文件包含 80 条记录，每条恰好 20 首歌曲，回复均为空。与最佳排序主线相比，
80 条记录的完整顺序全部发生变化，平均每条 Top-20 仍有 `18.35` 首歌曲重合。
与 Dense Hybrid 结果相比，70 条记录完全相同，平均 Top-20 重合 `19.80` 首，
说明通道惩罚 2.0 已经让 Dense 候选只影响少量样本。

纯 BM25 对照与 Dense Hybrid 提交的 80 条 Top-1 推荐全部相同。由于
`regenerate_responses_blindset.py` 只根据 Top-1 歌曲生成回复，因此对照提交
直接复用了 Dense Hybrid 提交的相同回复，使下一次评测的回复文本保持不变，
分数差异只来自 Top-2 至 Top-20 排序和 Dense 候选。

已生成并校验以下提交产物：

```text
exp/inference/blindset_A/tags_top400_dense_mix4_bm25only_prediction.json
exp/inference/blindset_A/tags_top400_dense_mix4_bm25only_submission.zip
```

ZIP 根目录仅包含 `prediction.json`，压缩包完整性检查通过。

纯 BM25 对照的正式 Blind A 结果：

| 指标 | 纯 BM25 对照 | Dense Hybrid | 差值 |
|---|---:|---:|---:|
| ndcg@20 | 0.1806 | 0.1806 | 0.0000 |
| catalog_diversity | 0.0316 | 0.0316 | 0.0000 |
| lexical_diversity | 0.7369 | 0.7369 | 0.0000 |
| llm_judge_score | 4.8500 | 4.8500 | 0.0000 |
| composite_score | 0.4559 | 0.4559 | 0.0000 |

结论：Dense Hybrid 只改变了少量列表中的低位歌曲，没有影响日志显示精度下
的官方聚合指标；相对最佳主线的 `-0.0140` nDCG 损失主要来自混合负例训练
checkpoint。后续排序实验恢复使用 `exp/reranker/minilm_bm25_tags_top400_e1/`
作为唯一主线起点。

## 研究断点：发现榜单差距的结构性原因

记录时间：2026-06-10。用户反馈其他参赛者的 `nDCG@20` 已达到约 `0.5`。
该数字尚未通过本地或公开榜单接口复核，但它说明当前 `0.1946` 主线不能再靠
增加 MiniLM 训练轮数追赶，需要改变推荐架构。

### 已确认的核心问题

当前主线基本只使用：

```text
对话文本
  -> BM25 文本召回
  -> MiniLM 文本 Cross-Encoder
```

但官方同时提供歌曲音频、封面、歌词、属性、元数据、CF-BPR embedding 和
用户 CF-BPR embedding。当前主线没有使用其中绝大多数信号，也没有显式使用
历史歌曲之间的歌手、专辑和相似度关系。

官方评估以 session 和 turn 宏平均 nDCG；Blind A 每个 session 只需预测最后
一个可见 user turn。此前本地模型选择主要观察完整 Dev 的全部 8 个 turn，
与 Blind A 的实际推理形态存在偏差。后续模型选择必须增加“Dev 每个 session
最后一轮”的伪 Blind 评估。

### 官方向量与数据覆盖检查

官方 embedding 资源已下载并可从本地 Hugging Face cache 加载：

```text
talkpl-ai/TalkPlayData-Challenge-Track-Embeddings
talkpl-ai/TalkPlayData-Challenge-User-Embeddings
```

检查结果：

| 项目 | 结果 |
|---|---:|
| 歌曲总数 | 47,071 |
| 有 128 维歌曲 CF-BPR 的歌曲 | 46,455 |
| 缺失歌曲 CF-BPR | 616 |
| Dev session / user | 1,000 / 500 |
| Blind A session / user | 80 / 58 |
| Blind A 中有用户 CF-BPR 的用户 | 25 |
| Blind A 中无用户 CF-BPR 的用户 | 33 |

纯用户 CF-BPR 与歌曲 CF-BPR 点积在完整 Dev 上表现很弱：

```text
nDCG@20 = 0.0066
Recall@20 = 0.0180
```

结论：官方 CF-BPR 不能直接作为最终排序，但可作为学习排序特征。

### 伪 Blind Query 实验

在 Dev 每个 session 的第 8 轮进行评估，只测试 BM25 Top-400：

| Query 构造 | Recall@20 | Recall@100 | Recall@400 | nDCG@20 |
|---|---:|---:|---:|---:|
| 仅当前请求 | 0.105 | 0.222 | 0.303 | 0.0385 |
| listener goal + 当前请求 | 0.109 | 0.264 | 0.361 | 0.0419 |
| user thought + goal + 当前请求 | 0.155 | 0.309 | 0.424 | 0.0573 |
| 历史推荐理由 + 历史歌曲 + thought + goal | **0.256** | **0.452** | **0.579** | **0.0807** |
| 当前 focused Query | 0.231 | 0.392 | 0.519 | 0.0791 |
| 当前 legacy Query | 0.239 | 0.417 | 0.540 | 0.0747 |

`user.thought` 和历史 `assistant.thought` 含有强偏好、反馈和目标细节。当前 Query
构造丢弃这些字段，并把历史推荐不加区分地当作正向偏好，这是明确缺口。

### 历史歌曲结构信号

Dev 最后一轮目标与已知历史歌曲的关系：

| 关系 | 比例 |
|---|---:|
| 与最后一首历史歌曲同歌手 | 32.0% |
| 与任意历史歌曲同歌手 | **51.6%** |
| 与最后一首历史歌曲同专辑 | 19.0% |
| 与任意历史歌曲同专辑 | **39.8%** |

仅使用同歌手、同专辑、当前请求显式匹配等简单规则，在 Dev 最后一轮已达到：

```text
Recall@20 = 0.3770
nDCG@20 = 0.1897
```

这说明历史歌曲的结构关系本身就接近当前完整文本主线，必须显式进入候选召回
和排序特征，不能只依赖 Cross-Encoder 从文本中隐式学习。

### 官方歌曲 Embedding 续推实验

在 200 个 Dev session 的最后一轮，使用已知历史歌曲 embedding 检索目标歌曲：

| Embedding / 聚合方式 | Recall@20 | nDCG@20 |
|---|---:|---:|
| metadata Qwen，最后一首 | 0.220 | 0.0790 |
| attributes Qwen，历史均值 | 0.110 | 0.0435 |
| lyrics Qwen，最后一首 | 0.050 | 0.0208 |
| audio CLAP，历史均值 | 0.040 | 0.0124 |
| image SigLIP2，最后一首 | 0.265 | **0.2179** |
| image SigLIP2，历史均值 | **0.290** | 0.2129 |
| CF-BPR，历史均值 | 0.120 | 0.0328 |

在完整 1,000 个 Dev session 最后一轮上：

| 召回方式 | Recall@20 | Recall@100 | Recall@400 | nDCG@20 |
|---|---:|---:|---:|---:|
| legacy BM25 | 0.239 | 0.417 | 0.540 | 0.0747 |
| 历史歌曲 image SigLIP2 均值 | **0.295** | 0.373 | 0.438 | **0.2075** |

BM25 Top-400 与 image SigLIP2 候选取并集后，候选命中率从 `0.540` 提升至
`0.604`。封面向量单路排序已经高于当前 Blind A 最佳 `0.1946`，说明多模态
歌曲续推是下一阶段最重要的信号。

### Blind A 推理形态

Blind A 的 80 个 session 最后可见 turn 分布：

| Turn | Session 数 |
|---:|---:|
| 1 | 20 |
| 2 | 15 |
| 3 | 10 |
| 4 | 5 |
| 5 | 8 |
| 6 | 9 |
| 7 | 8 |
| 8 | 5 |

因此系统需要分两种场景：

1. Turn 1 没有历史歌曲，主要依赖当前请求、goal、thought、文本与内容向量。
2. Turn 2-8 有历史歌曲，应重点使用歌手/专辑关系和多模态歌曲续推。

## 恢复对话后的执行计划

### 第一阶段：建立正确评估协议

1. 新增“Dev 最后一轮伪 Blind”评估脚本，每个 Dev session 只评估最后一轮。
2. 保留完整 8-turn Dev 作为泛化检查，但不再作为 Blind A 唯一选模依据。
3. 报告 Turn 1 与 Turn 2-8 两个分组，避免历史信号掩盖冷启动表现。

### 第二阶段：实现多路候选召回

每个请求分别生成并保留通道 rank/score：

1. feedback-rich BM25 Top-400：加入当前 `user.thought`、listener goal、近期
   `assistant.thought` 和历史歌曲。
2. 历史歌曲同歌手候选。
3. 历史歌曲同专辑候选。
4. 历史歌曲 image SigLIP2 最后一首相似候选和历史均值相似候选。
5. metadata Qwen embedding 候选。
6. 当前最佳 legacy BM25 Top-400，作为稳定保底通道。

对所有通道取并集，不直接使用固定规则决定最终 Top-20。

### 第三阶段：训练 Learning-to-Rank 融合模型

优先使用 LightGBM LambdaRank 或 XGBoost ranking，而不是继续训练文本
Cross-Encoder。每个 query-track 候选至少包含：

```text
各召回通道 rank 与 score
当前 MiniLM Cross-Encoder score
是否与最后/任意历史歌曲同歌手
是否与最后/任意历史歌曲同专辑
image SigLIP2 最大、均值、最后一首相似度
metadata / attributes / lyrics / audio 相似度
当前请求是否显式匹配歌名、歌手、专辑
歌曲 popularity 与 release_date
用户 CF-BPR × 歌曲 CF-BPR 分数及缺失标记
turn_number、goal category、specificity
历史反馈与 goal_progress_assessment
```

训练使用 train session；验证必须使用 Dev 最后一轮伪 Blind，并按 session 分组
计算 nDCG@20。目标不是先追求复杂神经网络，而是确认多路信号融合能否将 Dev
最后一轮 `nDCG@20` 显著提升。

### 第四阶段：实验准入标准

只有同时满足以下条件才生成 Blind A 提交：

```text
Dev 最后一轮整体 nDCG@20 明显高于当前方案
Turn 1 不显著退化
Turn 2-8 从历史结构和多模态信号中获得稳定提升
完整 8-turn Dev 没有灾难性下降
```

### 暂停的方向

以下路线已经验证收益不足或容易过拟合，恢复后不要优先继续：

```text
增加 MiniLM 训练轮数
继续混合 Dense 困难负例训练
单独使用用户 CF-BPR 点积排序
仅使用当前 Qwen metadata Dense 召回
只调 Dense channel penalty
```

### 尚未落地的代码

本节的伪 Blind Query、Embedding 续推和简单结构规则结果来自临时诊断命令，
尚未整理成正式脚本。恢复后的第一项代码工作应是将这些实验固化为可复现的：

```text
evaluate_final_turn_recall.py
build_multichannel_candidates.py
train_ltr_ranker.py
run_inference_ltr_blindset.py
```

参考官方资源：

- https://github.com/nlp4musa/music-crs-evaluator
- https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge
- https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Embeddings
- https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-User-Embeddings

## 2026-06-11：多通道召回与 LambdaRank 已落地

上一节计划中的正式代码已经完成：

```text
evaluate_final_turn_recall.py
train_ltr_ranker.py
run_inference_ltr_blindset.py
mcrs/retrieval_modules/precomputed_embeddings.py
```

当前多通道候选包含：

```text
legacy BM25
feedback-rich BM25
历史歌曲同歌手/同专辑结构候选
image SigLIP2：最后一首与历史均值
metadata Qwen embedding：最后一首与历史均值
```

LambdaRank 使用 46 维特征，包括各通道 rank/presence、Dense 相似度、同歌手/
同专辑关系、文本显式匹配、流行度、发行年份、turn、goal、specificity，以及
用户 CF-BPR 与歌曲 CF-BPR 的余弦分数。

### 多通道召回上限

完整 8-turn Dev，共 8,000 个预测任务：

| 方法 | Recall@20 | Recall@400 | nDCG@20 |
|---|---:|---:|---:|
| legacy BM25 | 0.2903 | 0.5601 | 0.1388 |
| 结构通道 | 0.3056 | 0.4056 | 0.1536 |
| 全通道 RRF | 0.3338 | 0.5643 | 0.1629 |
| 多通道候选并集 | 0.4570 | **0.6544** | - |

Dev 最后一轮，共 1,000 个预测任务：

| 方法 | Recall@20 | Recall@400 | nDCG@20 |
|---|---:|---:|---:|
| legacy BM25 | 0.2710 | 0.5420 | 0.1285 |
| 结构通道 | 0.3780 | 0.5240 | **0.1909** |
| image SigLIP2 历史均值 | 0.2950 | 0.4390 | 0.1540 |
| 全通道 RRF | 0.3560 | 0.5960 | 0.1719 |
| 多通道候选并集 | 0.4570 | **0.6890** | - |

这里最重要的结论不是某个单通道分数，而是候选并集相对 BM25 Top-400
多找回了约 14.7 个百分点的目标歌曲。后续瓶颈已经从“完全召回不到”部分转向
“如何把并集中的正确歌曲排进 Top-20”。

正式召回报告：

```text
exp/evaluation/dev_multichannel_all_turns_full.json
```

### 第一版 LambdaRank

第一版模型使用 10,000 个 Train turn、每通道 Top-100 候选训练：

```text
exp/ltr/multichannel_v1_10k_top100/model.txt
```

Dev 最后一轮：

| 方法 | 候选命中率 | Recall@20 | nDCG@20 |
|---|---:|---:|---:|
| 结构通道 | - | 0.3780 | 0.1909 |
| 全通道 RRF | - | 0.3560 | 0.1695 |
| LambdaRank | 0.5900 | **0.3790** | **0.1977** |

该模型的最佳迭代只有 5 轮。高重要度特征主要是结构通道 rank、legacy BM25
rank、用户-歌曲 CF 相似度、feedback BM25 rank、歌曲流行度和 image SigLIP2
rank。说明树模型确实在融合异构信号，但 10,000 个训练任务仍偏少。

完整 8-turn Dev 复评结果：

| 方法 | Recall@20 | nDCG@20 | 按 Blind A 轮次加权 nDCG@20 |
|---|---:|---:|---:|
| legacy BM25 | 0.2904 | 0.1389 | 0.1415 |
| 结构通道 | 0.3056 | 0.1536 | 0.1328 |
| 全通道 RRF | 0.3330 | 0.1622 | 0.1611 |
| LambdaRank | **0.3544** | **0.1799** | **0.1795** |

LambdaRank 分轮结果：

| Turn | Recall@20 | nDCG@20 |
|---:|---:|---:|
| 1 | 0.294 | 0.1687 |
| 2 | 0.410 | 0.1979 |
| 3 | 0.358 | 0.1757 |
| 4 | 0.353 | 0.1668 |
| 5 | 0.341 | 0.1757 |
| 6 | 0.349 | 0.1761 |
| 7 | 0.351 | 0.1806 |
| 8 | 0.379 | 0.1977 |

Turn 1 没有历史歌曲，仍然明显高于 BM25 的 `0.1272`，因此当前没有必要做
Turn 1 回退门控。全轮表现也没有灾难性下降，满足生成 Blind A 实验提交的条件。

完整复评报告：

```text
exp/ltr/multichannel_v1_10k_top100_allturn_eval/report.json
```

复现命令：

```powershell
.\.venv\Scripts\python.exe train_ltr_ranker.py `
  --model_path exp/ltr/multichannel_v1_10k_top100/model.txt `
  --dev_turn_mode all `
  --channel_topk 100 `
  --output_dir exp/ltr/multichannel_v1_10k_top100_allturn_eval `
  --device cuda
```

### Blind A 排序产物

已使用上述模型完成 80 个 Blind A 会话推理：

```text
exp/inference/blindset_A/multichannel_ltr_top100_empty.json
exp/inference/blindset_A/multichannel_ltr_top100_prediction.json
exp/inference/blindset_A/multichannel_ltr_top100_submission.zip
```

格式校验结果：

```text
记录数：80
每条歌曲数：20
每条歌曲 ID 唯一：是
全部歌曲 ID 存在于官方曲库：是
空排序文件回复状态：全部为空
正式提交文件回复状态：80 条全部非空且互不重复
模型迭代数：5
候选特征行数：18,346
ZIP 根目录：仅包含 prediction.json
```

推理命令：

```powershell
.\.venv\Scripts\python.exe run_inference_ltr_blindset.py `
  --model_path exp/ltr/multichannel_v1_10k_top100/model.txt `
  --output_name multichannel_ltr_top100_empty `
  --channel_topk 100 `
  --history_turns 0 `
  --device cuda
```

新排序与此前最高 nDCG 的 MiniLM 主线差异很大：80 个会话仅 4 个 Top-1 相同，
每条 Top-20 平均重合约 5.7 首。因此这不是低风险微调，而是一次独立架构实验。
Dev 结果支持提交验证，但应保留当前 `nDCG@20=0.1946` 的提交作为稳定对照。

生成回复：

```powershell
$env:DOUBAO_API_KEY = "你的 API Key"
.\.venv\Scripts\python.exe regenerate_responses_blindset.py `
  --tid bm25_tags_doubao_blindset_A `
  --input_path exp/inference/blindset_A/multichannel_ltr_top100_empty.json `
  --output_path exp/inference/blindset_A/multichannel_ltr_top100_prediction.json
```

打包时直接指定 ZIP 内文件名为 `prediction.json`，避免 PowerShell
`Compress-Archive` 的进度条异常：

```powershell
.\.venv\Scripts\python.exe -c "import zipfile; p=r'exp/inference/blindset_A/multichannel_ltr_top100_prediction.json'; z=zipfile.ZipFile(r'exp/inference/blindset_A/multichannel_ltr_top100_submission.zip','w',zipfile.ZIP_DEFLATED); z.write(p,'prediction.json'); z.close()"
```

### Blind A 官方结果

提交时间：2026-06-11。官方评分：

| 指标 | 多通道 LambdaRank | 此前历史最高 | 变化 |
|---|---:|---:|---:|
| nDCG@20 | **0.5071** | 0.1946 | **+0.3125** |
| catalog_diversity | 0.0300 | 0.0318 | -0.0018 |
| lexical_diversity | **0.7493** | 0.7489 | +0.0004 |
| llm_judge_score | 4.5500 | 4.8500 | -0.3000 |
| composite_score | **0.5977** | 0.4559 | **+0.1418** |

与此前最高 nDCG 相比，当前 nDCG 提升约 `160.6%`，达到原来的 `2.61` 倍；
综合分提升约 `31.1%`。这正式确认了此前约 `0.19` 的瓶颈来自单路 BM25 +
MiniLM 架构，而不是训练轮数不足。多通道候选与显式结构特征是本次跃升的核心。

官方结果文件：

```text
exp/inference/blindset_A/multichannel_ltr_top100_scores.json
```

当前唯一排序主线冻结为：

```text
模型：exp/ltr/multichannel_v1_10k_top100/model.txt
预测：exp/inference/blindset_A/multichannel_ltr_top100_prediction.json
提交：exp/inference/blindset_A/multichannel_ltr_top100_submission.zip
```

需要注意：本地完整 Dev 的 `nDCG@20=0.1799`，而 Blind A 官方结果为
`0.5071`，两者绝对值不可直接互相换算。本地 Dev 仍可用于拒绝明显退化的方案，
但新实验不能仅凭很小的 Dev 增益就认定会提高官方成绩。

### 更新后的下一步

1. 冻结当前模型、预测和回复，不覆盖原文件；所有新实验使用新的目录名。
2. 优先做单变量消融：分别去掉结构、图像、metadata、CF 和 feedback BM25，
   确认 `0.5071` 的主要来源，避免在未知贡献上盲目扩模。
3. 保存 Train/Dev 候选与 46 维特征缓存，使 LambdaRank 调参不再重复召回。
4. 在缓存上比较 10k、30k、50k 训练任务，并使用多个随机种子；只有 Dev
   各 turn 稳定提升才生成新的 Blind 提交。
5. 将 MiniLM Cross-Encoder score 作为额外特征做一次受控实验，不再让
   MiniLM 单独决定最终排序。
6. 回复侧单独优化。当前 `llm_judge_score=4.55`，低于历史最高 `4.85`；
   保持歌曲排序不变，比较低温度、多候选生成和回复规则检查，目标是恢复约
   0.2 到 0.3 的 judge 分，同时保持 lexical diversity。

## 2026-06-11：缓存、消融与精简 v2

### 特征缓存

已将与冠军模型完全一致的 Train/Dev 候选特征保存为：

```text
cache/ltr/train10k_seed13_top100_v1/
cache/ltr/dev_all_top100_v1/
```

缓存规模：

| 数据 | 原始任务 | 可训练/可评估 group | 候选行 | 特征数 |
|---|---:|---:|---:|---:|
| Train | 10,000 | 6,549 | 2,288,005 | 46 |
| Dev | 8,000 | 4,363 | 1,607,953 | 46 |

缓存基线重新训练后，最佳迭代仍为 5，完整 Dev `nDCG@20=0.179895`、
Blind 轮次加权 `nDCG@20=0.179458`，与冠军模型逐位一致，证明缓存协议没有
改变样本或排序逻辑。

### 固定模型屏蔽诊断

先在冠军模型上屏蔽特征，并对召回通道同步删除仅由该通道带来的候选：

| 屏蔽组 | Blind 加权 nDCG@20 | 相对基线 |
|---|---:|---:|
| 不屏蔽 | 0.1795 | - |
| structure | 0.1553 | **-0.0242** |
| image | 0.1710 | **-0.0084** |
| query match | 0.1738 | -0.0056 |
| feedback BM25 | 0.1758 | -0.0037 |
| legacy BM25 | 0.1785 | -0.0010 |
| metadata Dense | 0.1816 | +0.0021 |
| popularity/release | 0.1828 | +0.0034 |
| CF | 0.1832 | +0.0037 |

结论：历史同歌手/同专辑结构是当前模型最重要的信号，图像续推第二；
query 显式匹配和 feedback BM25 也有稳定贡献。后三组的正增益只能说明当前
模型可能依赖过度，不能直接当作正式消融结论。

固定模型诊断报告：

```text
exp/ltr/multichannel_v1_10k_top100_ablation/report.json
```

### 重新训练消融

使用完全相同的 10k Train 缓存，以 Dev Turn 8 早停，再在完整 8-turn Dev
上评估：

| 变体 | 最佳迭代 | 完整 Dev nDCG@20 | Blind 加权 nDCG@20 | 加权变化 |
|---|---:|---:|---:|---:|
| baseline | 5 | 0.1799 | 0.1795 | - |
| no CF | 19 | 0.1795 | 0.1802 | +0.0007 |
| no popularity/release | 44 | 0.1721 | 0.1709 | -0.0086 |
| no CF + popularity/release | 56 | 0.1776 | 0.1772 | -0.0023 |
| no metadata channel | 17 | 0.1756 | 0.1747 | -0.0048 |
| no metadata + CF + popularity | **3** | **0.1905** | **0.1918** | **+0.0124** |

单独删除 metadata 或 popularity 会退化，但三组弱信号一起删除反而提升约
`6.9%`。这表明它们之间存在明显交互：在只有 10k 训练任务时，多组弱特征让
LambdaRank 产生了不稳定分裂；精简后模型更依赖结构、BM25、图像和显式
query 匹配。

精简模型在 8 个 turn 上全部提高：

| Turn | baseline | 精简 v2 | 变化 |
|---:|---:|---:|---:|
| 1 | 0.1687 | 0.1925 | +0.0238 |
| 2 | 0.1979 | 0.2096 | +0.0117 |
| 3 | 0.1757 | 0.1799 | +0.0042 |
| 4 | 0.1668 | 0.1801 | +0.0133 |
| 5 | 0.1757 | 0.1788 | +0.0031 |
| 6 | 0.1761 | 0.1818 | +0.0057 |
| 7 | 0.1806 | 0.1933 | +0.0126 |
| 8 | 0.1977 | 0.2077 | +0.0100 |

完整报告：

```text
exp/ltr/cached_ablation_10k_top100/summary.json
exp/ltr/cached_ablation_10k_top100/no_metadata_cf_popularity/report.json
```

### 精简 v2 Blind A 候选提交

精简 v2：

```text
模型：exp/ltr/cached_ablation_10k_top100/no_metadata_cf_popularity/model.txt
空回复排序：exp/inference/blindset_A/multichannel_ltr_lean_v2_empty.json
正式预测：exp/inference/blindset_A/multichannel_ltr_lean_v2_prediction.json
提交包：exp/inference/blindset_A/multichannel_ltr_lean_v2_submission.zip
```

相对 `0.5071` 冠军排序：

```text
Top-1 相同：39 / 80
Top-20 平均重合：14.55 / 20
完全相同列表：0 / 80
候选行：14,571
模型特征：36
模型迭代：3
```

回复处理采用受控策略：Top-1 未改变的 39 个会话原样复用冠军回复，只为其余
41 个会话重新生成。最终 80 条回复全部非空、互不重复、52-87 词、以问题结尾，
且不含 Markdown 标记。ZIP 根目录只包含 `prediction.json`。

校验哈希：

```text
模型 SHA256：64093D4E2756DF95CC56A5622A6E453CE19A0705D8EBD16874F99910AAC1FA1E
ZIP SHA256：DE372B5FCB21AE4C1E77375443101AD3AC351F6CAAB34072F0CBCCCED873880F
```

### 精简 v2 官方结果

提交时间：2026-06-11。官方评分：

| 指标 | 精简 v2 | v1 | 变化 |
|---|---:|---:|---:|
| nDCG@20 | **0.5555** | 0.5071 | **+0.0484** |
| catalog_diversity | **0.0307** | 0.0300 | +0.0007 |
| lexical_diversity | 0.7465 | 0.7493 | -0.0028 |
| llm_judge_score | **4.7500** | 4.5500 | +0.2000 |
| composite_score | **0.6367** | 0.5977 | **+0.0390** |

相对 v1，v2 的 nDCG 提升约 `9.54%`，综合分提升约 `6.53%`。这与本地
Blind 加权 Dev 的提升方向一致，正式验证了“同时去掉 metadata Dense 通道、
用户 CF、歌曲流行度和发行年份”能减少 10k 小样本 LambdaRank 的噪声。

官方结果文件：

```text
exp/inference/blindset_A/multichannel_ltr_lean_v2_scores.json
```

自此，精简 v2 升级为当前唯一主线；v1 继续保留为消融对照：

```text
当前冠军模型：exp/ltr/cached_ablation_10k_top100/no_metadata_cf_popularity/model.txt
当前冠军预测：exp/inference/blindset_A/multichannel_ltr_lean_v2_prediction.json
当前冠军提交：exp/inference/blindset_A/multichannel_ltr_lean_v2_submission.zip
官方 nDCG@20：0.5555
官方 composite_score：0.6367
```

下一步不重新加入已验证有害的三组弱信号。优先级调整为：

1. 在当前 36 维精简特征上扩大到 30k 训练任务，并保留相同的分轮 Dev 验证。
2. 用 3 个随机种子检查增益稳定性，避免再次依赖偶然的 3 棵树早停点。
3. 将 MiniLM Cross-Encoder score 作为单独新增特征做受控对照。
4. 保持 v2 排序不变，优化回复以尝试恢复历史最高 `llm_judge_score=4.85`。

## 2026-06-11：30k 训练与按轮次门控 v3

### 30k 特征缓存

保持 Top-100 多通道候选、36 维精简特征定义和 seed 13 的任务顺序不变，将
训练任务从 10k 扩大到 30k。文本 BM25 改为每 500 条分块执行；分块与整批
检索的排序结果已逐条验证一致，只改变内存峰值和执行方式。

```text
cache/ltr/train30k_seed13_top100_v1/
```

缓存统计：

```text
原始任务：30,000
目标进入候选并集：19,753
精简通道可训练 group：19,627
原始 46 维候选行：6,929,442
缓存大小：1,283,398,313 bytes
```

### 三个随机种子

均使用同一个 30k 缓存、相同 36 维精简特征和 Dev Turn 8 早停：

| 模型 | 最佳迭代 | 完整 Dev nDCG@20 | Blind 加权 nDCG@20 |
|---|---:|---:|---:|
| 10k v2 | 3 | 0.1905 | 0.1918 |
| 30k seed 13 | 4 | 0.1954 | **0.2013** |
| 30k seed 29 | 8 | 0.1845 | 0.1863 |
| 30k seed 47 | 3 | **0.1957** | 0.1961 |

扩大数据总体有效，但仍存在明显树模型方差。seed 13 的优势主要集中在
Turn 1 冷启动，seed 47 在 Turn 2-8 的多数轮次更稳定：

| Turn | 10k v2 | seed 13 | seed 47 |
|---:|---:|---:|---:|
| 1 | 0.1925 | **0.2260** | 0.1843 |
| 2 | 0.2096 | 0.2157 | **0.2264** |
| 3 | 0.1799 | 0.1815 | **0.1860** |
| 4 | 0.1801 | 0.1824 | **0.1883** |
| 5 | 0.1788 | 0.1821 | **0.1862** |
| 6 | 0.1818 | 0.1803 | **0.1910** |
| 7 | **0.1933** | 0.1896 | 0.1922 |
| 8 | 0.2077 | 0.2053 | **0.2109** |

### 融合与门控

测试了 raw score 均值、组内 min-max 后均值和 RRF。所有简单分数融合均低于
最佳单模型，说明不同树数和随机分裂产生的分数空间不适合直接平均。

基于 Blind 可见的 `turn_number` 使用两段门控：

```text
Turn 1：30k seed 13
Turn 2-8：30k seed 47
```

门控结果：

| 方法 | 完整 Dev nDCG@20 | Blind 加权 nDCG@20 | Blind 加权 Recall@20 |
|---|---:|---:|---:|
| 10k v2 | 0.1905 | 0.1918 | **0.3530** |
| 30k seed 13 | 0.1954 | 0.2013 | 0.3482 |
| 30k seed 47 | 0.1957 | 0.1961 | 0.3489 |
| 30k turn gate | **0.2009** | **0.2065** | 0.3501 |

门控相对 v2 的 Blind 加权 nDCG 提升约 `7.65%`。Recall@20 略低，但正确歌曲
进入 Top-20 后的位置更靠前，因此 nDCG 更高。

完整融合报告：

```text
exp/ltr/lean_30k_top100_ensemble/report.json
```

### v3 Blind A 候选提交

```text
Turn 1 模型：exp/ltr/lean_30k_top100_seed13/no_metadata_cf_popularity/model.txt
Turn 2+ 模型：exp/ltr/lean_30k_top100_seed47/no_metadata_cf_popularity/model.txt
空回复排序：exp/inference/blindset_A/multichannel_ltr_30k_turn_gate_empty.json
正式预测：exp/inference/blindset_A/multichannel_ltr_30k_turn_gate_prediction.json
提交包：exp/inference/blindset_A/multichannel_ltr_30k_turn_gate_submission.zip
```

相对当前官方冠军 v2：

```text
Top-1 相同：64 / 80
Top-20 平均重合：18.20 / 20
完全相同列表：5 / 80
```

回复处理继续采用受控复用：64 条 Top-1 未变化的回复原样保留，只重新生成
16 条。最终全部 80 条回复非空且互不重复，长度 45-87 词，无 Markdown，
均以自然问题结尾。

校验哈希：

```text
Turn 1 模型 SHA256：E4DD861D23D08D5E9ACDBEB55A37F42D393F63170ECFC1C0831CB1CAEC867320
Turn 2+ 模型 SHA256：507C4D7CB263718A9EFCA1BAF050D8B9CB731B53A51AC370F2A1FF5E7C978423
预测 JSON SHA256：904A5BE0A23BDA534C13D4244AD4B1657AB9A62E3EB2CCE7BE2EED3CABDA0EA2
提交 ZIP SHA256：D4A214459899840A856AC2662688540D5EED35A7471226D8F0B0CFF465D496FB
```

### v3 官方结果

| 指标 | v3 | v2 | 变化 |
|---|---:|---:|---:|
| nDCG@20 | 0.5270 | **0.5555** | -0.0285 |
| catalog_diversity | 0.0307 | 0.0307 | 0.0000 |
| lexical_diversity | 0.7431 | **0.7465** | -0.0034 |
| llm_judge_score | 4.3500 | **4.7500** | -0.4000 |
| composite_score | 0.5921 | **0.6367** | -0.0446 |

v3 正式拒绝。按 turn 门控在本地 Dev 上提升，但没有泛化到 Blind A，且 16 条
新回复带来了明显 judge 波动。当前主线继续保持 v2，不再依据同一 Dev 的细粒度
分轮差异做硬门控。

官方结果：

```text
exp/inference/blindset_A/multichannel_ltr_30k_turn_gate_scores.json
```

## 2026-06-11：Catalog Diversity 上限核查

官方 evaluator 的定义为：

```text
catalog_diversity = 全部预测中不同歌曲 ID 数 / 曲库歌曲总数
```

官方格式同时明确要求每条 `predicted_track_ids` 最多 20 首。Blind A 只有
80 条预测、曲库共有 47,071 首，因此合规提交的理论上限为：

```text
80 * 20 / 47,071 = 0.033991
```

榜单中出现 `catalog_diversity=1.0` 不可能来自合规的 80×20 推荐。公开 evaluator
存在一个实现缺口：nDCG 会截断到前 20，但 catalog diversity 会统计列表里的
全部歌曲；若提交超过 20 首，可能人为把 coverage 推到 1.0。这违反官方
“up to 20”格式，不作为本项目实验方向。

v2 的实际覆盖：

```text
总推荐槽位：1,600
不同歌曲：1,445
重复槽位：155
catalog_diversity：0.030698
合规理论上限：0.033991
```

即使无损消除全部跨 session 重复，catalog 对 composite 的最大贡献也只有：

```text
0.10 * (0.033991 - 0.030698) = 0.000329
```

但若替换时误删一个第 20 名的正确歌曲，按 80 个 Blind session 平均，composite
约损失：

```text
0.50 * (1 / log2(21)) / 80 = 0.001423
```

一次最低位命中损失就是全部合法 diversity 潜在收益的约 4.3 倍。因此不对
冠军 v2 做全局去重或尾部强制多样化。后续精力继续放在 nDCG 和回复质量。

复查命令：

```powershell
.\.venv\Scripts\python.exe analyze_catalog_diversity.py `
  exp/inference/blindset_A/multichannel_ltr_lean_v2_prediction.json
```

官方依据：

- https://github.com/nlp4musa/music-crs-evaluator/blob/main/metrics/metrics_diversity.py
- https://github.com/nlp4musa/music-crs-evaluator/blob/main/evaluate_devset.py
- https://www.codabench.org/competitions/15786/

## 2026-06-11：v2 排序冻结下的回复优化候选

背景：当前官方冠军仍是 `multichannel_ltr_lean_v2_prediction.json`，官方分数为：

```text
nDCG@20             0.5555
catalog_diversity   0.0307
lexical_diversity   0.7465
llm_judge_score     4.7500
composite_score     0.6367
```

由于 v3 排序实验已经证明更复杂的 turn gate 会在 Blind A 上回退，本轮不再改动
`predicted_track_ids`，只优化 `predicted_response`。新增脚本：

```text
optimize_responses_blindset.py
validate_blind_prediction.py
```

流程：

1. 对每个 session 保留 v2 原回复作为 `current` 候选。
2. 基于同一段对话、Top-1 歌曲元数据、当前回复，再生成 3 个新候选。
3. 让模型按“相关性、元数据 grounding、自然度、具体音乐解释、是否有 unsupported claim”
   选择最佳候选。
4. 只有新候选分数明显高于原回复，或原回复有明显格式/标题艺人缺失问题时才替换。
5. 全程保持 Top-20 歌曲排序不变，因此官方 nDCG 和 catalog diversity 理论上应与 v2 一致。

运行结果：

```text
输入预测：exp/inference/blindset_A/multichannel_ltr_lean_v2_prediction.json
输出预测：exp/inference/blindset_A/multichannel_ltr_lean_v2_response_opt_prediction.json
候选报告：exp/inference/blindset_A/multichannel_ltr_lean_v2_response_opt_report.json
提交包：  exp/inference/blindset_A/multichannel_ltr_lean_v2_response_opt_submission.zip

sessions: 80
changed_responses: 64
unchanged_responses: 16
```

校验结果：

```text
rows: 80
all_top20: true
empty_response: []
not_question: []
unique_recommended_tracks: 1445
catalog_diversity_if_all_tracks_47071: 0.030698306813112107
word_count_min / avg / max: 50 / 63.7 / 86
zip_entries: ["prediction.json"]
```

两个标题字面校验报警是 remaster 标题格式差异：

```text
Virgo - 2004 - Remaster -> 回复写作 Virgo (2004 Remaster)
Get Me - 2013 Remaster -> 回复写作 2013 remaster of Get Me
```

5 个 forbidden word 报警主要是误报，例如 `unapologetic` 命中了 `apolog` 子串，
不是道歉语。

### 官方结果

```text
nDCG@20             0.5555
catalog_diversity   0.0307
lexical_diversity   0.7688
llm_judge_score     4.3500
composite_score     0.6089
```

相对 v2：

| 指标 | response_opt | v2 | 变化 |
|---|---:|---:|---:|
| nDCG@20 | 0.5555 | 0.5555 | 0.0000 |
| catalog_diversity | 0.0307 | 0.0307 | 0.0000 |
| lexical_diversity | **0.7688** | 0.7465 | +0.0223 |
| llm_judge_score | 4.3500 | **4.7500** | -0.4000 |
| composite_score | 0.6089 | **0.6367** | -0.0278 |

该版本正式拒绝。排序完全未变，说明回退全部来自回复侧；更高的 lexical diversity
没有转化为更高 judge，反而被官方 judge 明显扣分。主要原因推断：

1. 大范围替换 64/80 条回复，破坏了 v2 已被官方 judge 认可的稳定风格。
2. 新回复更喜欢加入年份、album art、歌词含义、评论评价等细节，虽然更具体，
   但更容易被判定为 unsupported claim。
3. 自评 selector 与官方 judge 偏好不一致；它偏好“信息量更大”的回复，官方 judge
   更偏好稳妥、自然、少幻想的回复。

后续策略：不再做大范围回复重写。若继续碰回复侧，只允许从 v2 出发做少量微修复，
例如修正明显的引号/标题格式/事实错位；否则主攻排序侧 nDCG。

提交命令：

```powershell
# 文件已经生成，可直接提交这个 zip
exp/inference/blindset_A/multichannel_ltr_lean_v2_response_opt_submission.zip
```

## 2026-06-11：v2 回复微修复候选

由于 `response_opt` 证明大范围重写会损害官方 judge，本轮只从 v2 出发做最小修改：

```text
输入预测：exp/inference/blindset_A/multichannel_ltr_lean_v2_prediction.json
输出预测：exp/inference/blindset_A/multichannel_ltr_lean_v2_microfix_prediction.json
提交包：  exp/inference/blindset_A/multichannel_ltr_lean_v2_microfix_submission.zip
生成脚本：make_response_microfix.py
```

只改 7/80 条回复，Top-20 排序完全不变。修改目标：

1. 修正 ONE OK ROCK 场景中把 `We Are` 专辑说错的问题。
2. 补全 validator 发现的 Top-1 标题/艺人字面缺失。
3. 移除一处“虽然偏离歌词/rap”的不利表达。
4. 修正一条 Hollerado 回复里的重复口语表达。

校验结果：

```text
rows: 80
all_top20: true
empty_response: []
not_question: []
missing_title_or_artist_count: 0
unique_recommended_tracks: 1445
catalog_diversity_if_all_tracks_47071: 0.030698306813112107
word_count_min / avg / max: 52 / 63.2 / 85
zip_entries: ["prediction.json"]
```

该版本尚未官方评分。它比 `response_opt` 风险低得多，但当前正式冠军仍然是 v2：

```text
exp/inference/blindset_A/multichannel_ltr_lean_v2_submission.zip
```

### 官方结果

```text
nDCG@20             0.5555
catalog_diversity   0.0307
lexical_diversity   0.7575
llm_judge_score     4.6500
composite_score     0.6303
```

相对 v2：

| 指标 | microfix | v2 | 变化 |
|---|---:|---:|---:|
| nDCG@20 | 0.5555 | 0.5555 | 0.0000 |
| catalog_diversity | 0.0307 | 0.0307 | 0.0000 |
| lexical_diversity | **0.7575** | 0.7465 | +0.0110 |
| llm_judge_score | 4.6500 | **4.7500** | -0.1000 |
| composite_score | 0.6303 | **0.6367** | -0.0064 |

该版本也正式拒绝。它明显优于大范围 `response_opt`，说明少量微修复比大换血安全；
但仍低于 v2，说明当前 v2 回复风格已经更贴近官方 judge 偏好。后续停止回复侧优化，
除非发现致命格式错误；主线回到排序侧提升 `nDCG@20`。

## 2026-06-11：Turn1 s13 + v2 Top1 锁定候选

回复侧实验失败后，重新回到排序。已有 v3：

```text
Turn1 = 30k seed13
Turn2+ = 30k seed47
```

在本地 Dev / Blind 轮次加权上最高，但 Blind A 官方 `nDCG@20` 从 v2 的
`0.5555` 掉到 `0.5270`。因此不能继续整套替换。

重新评估所有二段门控组合：

```text
exp/ltr/turn1_gate_eval/report.json
```

本地 Dev 的关键结果：

| 方法 | Blind 加权 Dev nDCG@20 |
|---|---:|
| Turn1=s13, Turn2+=s47 | 0.2065 |
| Turn1=s13, Turn2+=s13 | 0.2013 |
| Turn1=s13, Turn2+=v2 | 0.2002 |
| v2 | 0.1918 |

考虑到 v3 已经证明 `s47` 在 Blind A 上不可靠，本轮只尝试：

```text
Turn1 = 30k seed13
Turn2+ = v2
```

直接推理后发现：

```text
ChangedLists: 20 / 80
SameTop1: 79 / 80
AvgTop20Overlap vs v2: 19.11 / 20
```

唯一 Top1 改变的样本是 Peter Doherty / Grace/Wastelands 的高特异性问题。
v2 Top1 为 `I Am The Rain`，元数据含有 `best tracks of 2009` 标签；s13 Top1
为 `New Love Grows On Trees`，反而把 `I Am The Rain` 降到第 5。该替换肉眼看
风险较高，因此最终版本锁定 v2 Top1：

```text
每条 session 的 Top1 = v2 Top1
每条 predicted_response = v2 response
Turn1 的第 2-20 位 = s13/v2 门控排序，去重后补齐
Turn2+ 完全等于 v2
```

产物：

```text
空排序：exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_empty.json
正式预测：exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_top1lock_prediction.json
提交包：  exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_top1lock_submission.zip
合并脚本：merge_locked_top1_prediction.py
```

校验结果：

```text
Rows: 80
ChangedLists: 20
ChangedTop1: 0
ResponsesSameAsV2: true
AvgTop20Overlap vs v2: 19.11
unique_recommended_tracks: 1446
catalog_diversity_if_all_tracks_47071: 0.03071955131609696
zip_entries: ["prediction.json"]
```

### 官方结果

```text
nDCG@20             0.5566
catalog_diversity   0.0307
lexical_diversity   0.7465
llm_judge_score     4.7500
composite_score     0.6373
```

相对 v2：

| 指标 | Turn1 s13 + v2 Top1 lock | v2 | 变化 |
|---|---:|---:|---:|
| nDCG@20 | **0.5566** | 0.5555 | +0.0011 |
| catalog_diversity | 0.0307 | 0.0307 | 0.0000 |
| lexical_diversity | 0.7465 | 0.7465 | 0.0000 |
| llm_judge_score | 4.7500 | 4.7500 | 0.0000 |
| composite_score | **0.6373** | 0.6367 | +0.0006 |

该版本升级为当前冠军。它验证了一个重要原则：只要 Top1 与回复不变，官方 judge
可以稳定保持；后续可以更大胆地只探索 Top2-Top20 排序。

当前冠军：

```text
预测：exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_top1lock_prediction.json
提交：exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_top1lock_submission.zip
分数：exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_top1lock_scores.json
```

## 2026-06-11：更大胆的 v3 后排锁定候选

既然 Top1/回复锁定策略有效，进一步尝试将此前失败的 v3 排序重新利用，只让它影响
后排。v3 原始提交失败的主要风险来自：

```text
nDCG@20 从 0.5555 掉到 0.5270
回复 judge 从 4.75 掉到 4.35
Top1 有 16 条变化，Top20 平均重叠仅 8.20 / 20
```

新候选继续锁定当前冠军的回复，并至少锁定 Top1：

### 候选 A：v3 Top1 lock（主推，更大胆）

```text
预测：exp/inference/blindset_A/multichannel_ltr_v3_top1lock_prediction.json
提交：exp/inference/blindset_A/multichannel_ltr_v3_top1lock_submission.zip
```

相对当前冠军：

```text
ChangedLists: 55 / 80
SameTop1: 80 / 80
SameTop5: 45 / 80
AvgTop20Overlap: 19.09 / 20
ResponsesSameAsCurrentChampion: true
unique_recommended_tracks: 1444
zip_entries: ["prediction.json"]
```

### 候选 B：v3 Top5 lock（备份，更保守）

```text
预测：exp/inference/blindset_A/multichannel_ltr_v3_top5lock_prediction.json
提交：exp/inference/blindset_A/multichannel_ltr_v3_top5lock_submission.zip
```

相对当前冠军：

```text
ChangedLists: 52 / 80
SameTop1: 80 / 80
SameTop5: 80 / 80
AvgTop20Overlap: 19.09 / 20
ResponsesSameAsCurrentChampion: true
unique_recommended_tracks: 1444
zip_entries: ["prediction.json"]
```

提交优先级：

1. 先试候选 A：`multichannel_ltr_v3_top1lock_submission.zip`。它允许 v3 重排
   第 2-20 位，潜在 nDCG 收益更大。
2. 若 A 下降，再试候选 B：`multichannel_ltr_v3_top5lock_submission.zip`。
   它只动第 6-20 位，风险小但收益上限也低。

如果 A/B 都下降，说明 v3 的后排排序也不泛化，该路线停止；继续寻找新的召回/特征
信号，而不是继续用 30k seed 模型融合。

### 官方结果

候选 A：v3 Top1 lock

```text
nDCG@20             0.5518
catalog_diversity   0.0307
lexical_diversity   0.7465
llm_judge_score     4.6500
composite_score     0.6274
```

候选 B：v3 Top5 lock

```text
nDCG@20             0.5566
catalog_diversity   0.0307
lexical_diversity   0.7465
llm_judge_score     4.7500
composite_score     0.6373
```

结论：

1. 候选 A 的 nDCG 和 Judge 同时下降。即使回复文本完全复用，只改变 Top2-5
   也可能影响官方 Judge；推测官方 Judge 会结合推荐列表前若干项评估相关性。
2. 候选 B 与当前冠军完全同分，说明 v3 对 Top6-20 的少量改动没有产生可测收益。
3. 30k seed 模型融合与 v3 后排重排路线正式停止，不再继续围绕同类模型调锁定位数。
4. 当前冠军保持不变，下一阶段转向增加具有互补性的召回和排序信号，优先研究文本
   向量召回、历史歌曲共现以及条件化流行度特征。

## 下一阶段：互补召回与特征增强

当前 LambdaRank 已包含 BM25、同歌手/同专辑结构召回、图像/元数据历史相似召回和
用户-歌曲 CF 相似度，但仍存在两个明显缺口：

1. 对首轮请求没有真正的语义文本召回通道。当前元数据 Qwen embedding 仅用于根据
   历史歌曲向量检索；Turn1 没有历史歌曲时，该通道为空。
2. CF 只作为候选级打分特征，没有作为召回通道，因此无法把 BM25 和结构召回之外
   的高 CF 候选带入 LambdaRank。

下一轮实施顺序：

```text
1. Qwen query-to-track 文本语义召回，重点提高 Turn1 候选覆盖率
2. 用户 CF 向量直接召回，补充个性化候选
3. 历史歌曲 CF last/mean 召回，补充跨歌手、跨专辑共现关系
4. 在 Dev 上分别做候选召回率与特征消融，再决定是否训练新 LambdaRank
5. Blind A 推理时继续锁定当前冠军 Top1/回复，只在新模型通过 Dev 消融后替换后排
```

## 2026-06-11：CF 召回增强 v2

### 新增通道

在原有候选池之外新增三类可选通道：

```text
query-qwen3     请求文本 -> Qwen metadata embedding 语义召回
user-cf         用户 cf-bpr 向量 -> 歌曲 cf-bpr 向量召回
cf-bpr:last     最后一首历史歌曲 -> CF 相似歌曲召回
cf-bpr:mean     历史歌曲平均向量 -> CF 相似歌曲召回
```

所有新通道默认关闭，旧冠军推理命令保持完全兼容。兼容性测试中，更新代码生成的旧冠军
排序与此前产物 `80/80` 条完全一致。

### 小样本筛选

在 100 个 Dev session、共 800 个轮次上，新通道将旧候选池的候选命中率从
`0.5400` 提升到 `0.5825`，净补回 34 个旧系统完全漏掉的正例；Turn1 候选命中率
从 `0.4300` 提升到 `0.4800`。

但是训练消融显示 `query-qwen3` 会拖累排序，因此正式全量实验只启用 CF 召回：

```text
--enable_cf_retrieval --no-enable_query_dense
```

### 全量候选与训练

```text
训练任务：10000
Dev 任务：8000
训练正例可召回组：7162
Dev 正例可召回组：4702
```

候选覆盖对比：

| 指标 | 旧候选池 | 旧候选池 + CF | 变化 |
|---|---:|---:|---:|
| overall candidate recall | 0.5454 | 0.5878 | +0.0424 |
| Turn1 candidate recall | 0.4200 | 0.4610 | +0.0410 |
| Blind 加权 candidate recall | 0.5279 | 0.5700 | +0.0421 |

CF 通道共净补回 339 个旧候选池完全漏掉的 Dev 正例。

### LambdaRank 消融结果

表现最好的版本仍采用此前验证有效的特征裁剪：

```text
保留：BM25、结构、图像、user-cf 召回、历史 cf-bpr 召回、Query 显式匹配
移除：metadata history 通道、旧 user_track_cf 特征、popularity、release_year
```

新模型：

```text
exp/ltr/cf_v2_10k_top100_ablation/no_metadata_cf_popularity/model.txt
```

与旧冠军对应模型的完整 Dev 对比：

| 范围 | 旧模型 nDCG@20 | CF v2 nDCG@20 | 变化 |
|---|---:|---:|---:|
| overall | 0.1905 | 0.2240 | +0.0336 |
| Turn1 | 0.1925 | 0.2215 | +0.0290 |
| Turn2+ | 0.1902 | 0.2244 | +0.0342 |
| Blind 轮次加权 | 0.1918 | 0.2269 | **+0.0351** |

`user-cf__score`、`cf-bpr__mean__score` 和 `cf-bpr__mean__reciprocal_rank`
均进入重要特征，说明收益确实来自新增 CF 信号。

### Blind A 待提交候选

新模型的原始 Blind A 排序相对当前冠军：

```text
ChangedLists: 78 / 80
SameTop1: 56 / 80
AvgTop20Overlap: 15.40 / 20
```

为保留当前冠军回复并控制 Judge 风险，已生成三个锁定候选：

```text
Top1 锁定： exp/inference/blindset_A/multichannel_ltr_cf_v2_top1lock_submission.zip
Top5 锁定： exp/inference/blindset_A/multichannel_ltr_cf_v2_top5lock_submission.zip
Top10 锁定：exp/inference/blindset_A/multichannel_ltr_cf_v2_top10lock_submission.zip
```

推荐提交顺序：

1. 先提交 Top5 锁定版。它保留最影响 Judge 的前五项，同时允许 CF 模型优化后排。
2. 若 Top5 有提升，再尝试 Top1 锁定版获取更高 nDCG 上限。
3. 若 Top5 下降，则提交 Top10 锁定版验证收益是否只存在于更后排位置。

三个 ZIP 均已校验为 80 行、每行 20 首歌曲、仅含 `prediction.json`，回复与当前冠军
完全一致。当前正式冠军在获得新官方结果前仍保持为：

```text
exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_top1lock_submission.zip
```

### Top5 锁定版官方结果

```text
nDCG@20             0.5566
catalog_diversity   0.0310
lexical_diversity   0.7465
llm_judge_score     4.6500
composite_score     0.6298
```

相对当前冠军：

| 指标 | CF v2 Top5 lock | 当前冠军 | 变化 |
|---|---:|---:|---:|
| nDCG@20 | 0.5566 | 0.5566 | 0.0000 |
| catalog_diversity | **0.0310** | 0.0307 | +0.0003 |
| lexical_diversity | 0.7465 | 0.7465 | 0.0000 |
| llm_judge_score | 4.6500 | **4.7500** | -0.1000 |
| composite_score | 0.6298 | **0.6373** | -0.0075 |

该候选正式拒绝。锁定 Top1-5 后，新 CF 模型只能调整 Top6-20，官方 nDCG 完全没有
变化，说明这部分调整没有产生可测收益。尽管 Top1 与回复完全相同，Judge 仍下降
`0.1`，说明官方 Judge 可能检查完整推荐列表，或者存在一定随机波动。

由于 Judge 下降 `0.1` 会损失约 `0.0075` 综合分，下一候选至少需要提升约
`0.015` nDCG 才能抵消风险。Top10 锁定版比 Top5 更保守，几乎没有提高 nDCG 的
可能，因此不提交。下一步不直接提交改动幅度很大的 Top1 锁定版，而是在 Dev 上搜索
当前冠军模型与 CF 模型的分数融合权重，再生成锁定 Top1 的平滑融合候选。

### Top1 锁定版官方结果

```text
nDCG@20             0.5501
catalog_diversity   0.0311
lexical_diversity   0.7465
llm_judge_score     4.6500
composite_score     0.6266
```

相对当前冠军：

| 指标 | CF v2 Top1 lock | 当前冠军 | 变化 |
|---|---:|---:|---:|
| nDCG@20 | 0.5501 | **0.5566** | -0.0065 |
| catalog_diversity | **0.0311** | 0.0307 | +0.0004 |
| lexical_diversity | 0.7465 | 0.7465 | 0.0000 |
| llm_judge_score | 4.6500 | **4.7500** | -0.1000 |
| composite_score | 0.6266 | **0.6373** | -0.0107 |

该候选正式拒绝。完整 Dev 上的 `+0.0351` 加权 nDCG 提升没有迁移到 Blind A，
反而在放开 Top2-5 后损失 `0.0065` nDCG。这说明用户 CF / 历史 CF 在公开 Dev
分布上有效，但对 Blind A 存在明显分布偏移；候选覆盖提升不能保证最终排序泛化。

CF 直接重排路线正式停止：

1. 不提交 Top10 锁定版，因为 Top5 已证明 Top6-20 没有收益。
2. 不再以 CF 模型单独生成 Blind 排序。
3. 跨模型融合工具仅作为诊断；除非极低 CF 权重在 Dev 上稳定、且 Blind 改动很小，
   否则不生成新的官方提交。
4. 当前正式冠军继续保持 `0.6373`。

### 跨模型融合诊断

为了确认 CF 是否只是权重过大，新增了允许不同特征集合模型共同打分的诊断工具：

```text
evaluate_cross_model_fusion.py
run_inference_ltr_fusion_blindset.py
```

在完整 Dev 候选池上搜索旧冠军模型与 CF 模型的融合权重。可直接比较的 min-max
融合结果中：

```text
CF weight = 0.0：Blind 加权 nDCG@20 = 0.18257
CF weight = 0.1：Blind 加权 nDCG@20 = 0.17970
CF weight = 0.8：Blind 加权 nDCG@20 = 0.17997
CF weight = 1.0：Blind 加权 nDCG@20 = 0.17977
```

任何非零 CF 权重都低于 `CF weight=0`。结合两个 Blind A 官方失败结果，说明 CF
并非只需要降低权重，而是其排序方向与当前稳健主线冲突。CF 路线停止，不生成融合
提交。下一阶段转向显式请求约束特征，包括标签/情绪/流派、年份/年代、热门/冷门偏好
以及同歌手/同专辑/换歌手等指令。

## 2026-06-11：显式请求约束特征与保守门控

### 目标

CF 路线在 Blind A 上发生明显分布偏移后，转向更可解释的用户明示请求信号。在
LambdaRank 候选特征中加入：

```text
query tag 短语匹配与 token overlap
精确年份、年代与年份距离
同艺人、同专辑、换艺人指令交互
instrumental、live、remix 指令交互
热门/冷门与现代/复古偏好交互
```

完整约束缓存：

```text
cache/ltr/train10k_seed13_top100_constraints_v3
cache/ltr/dev_all_top100_constraints_v3
```

### 约束模型消融

表现最好的组合为：

```text
exp/ltr/constraints_v3_champion_combos/champion_plus_constraints_no_preference/model.txt
```

它沿用冠军特征裁剪，保留标签、年份、关系和版本约束，但移除容易跨分布漂移的
热门/冷门与现代/复古偏好。

```text
完整 Dev nDCG@20             0.1981
Blind 轮次加权 nDCG@20       0.2021
旧冠军模型 Blind 加权 nDCG   0.1918
本地变化                      +0.0103
```

原始 Blind 排序相对当前冠军改动仍然过大：

```text
ChangedLists       78 / 80
SameTop1           63 / 80
AvgTop20Overlap    18.14 / 20
```

因此不直接提交原始排序。

### 高精度约束门控

新增工具：

```text
constraint_gating.py
evaluate_constraint_gating.py
merge_constraint_gated_prediction.py
```

门控只识别明确年份/年代、同艺人、换艺人、同专辑、纯音乐以及精确 live/remix
请求。先在完整 Dev 上按类别验证，只在对应类别收益为正时才允许替换冠军排序。

Dev 结果：

| 门控 | 前缀锁定 | 命中任务平均 nDCG 变化 | Blind 加权 nDCG 变化 |
|---|---:|---:|---:|
| same album（严格推荐意图） | Top1 | +0.00807 | +0.00008 |
| different artist | Top5 | +0.00145 | +0.00014 |
| year | Top1 | -0.00276 | -0.00051 |
| same artist | Top1 | -0.00060 | -0.00016 |
| instrumental | Top1 | -0.00103 | 约 0 |
| all strict | Top1 | -0.00130 | -0.00067 |

尽管 `same album` 和 `different artist` 有局部正收益，Blind 人工检查发现软特征模型
并没有稳定执行约束。例如 P3 同专辑请求中，新排序反而提升了更多非 P3 歌曲，因此
拒绝提交该候选。

### 硬约束可行性诊断

进一步统计完整 Dev 中，真实目标是否满足用户明示约束：

| 请求类型 | Dev 任务数 | 目标满足约束比例 |
|---|---:|---:|
| same album | 85 | 63.5% |
| different artist | 846 | 53.4% |
| same artist | 134 | 48.5% |
| year / decade | 888 | 68.9% |
| instrumental（按标题/标签） | 536 | 39.4% |

即使用户明确提出约束，数据集的单一标注目标也经常不满足该约束，或者元数据无法
证明其满足。因此不能用硬过滤强制执行，否则会系统性伤害官方 nDCG。

### 本阶段结论

1. 显式约束特征本身有可解释信号，但不足以支持整表重排。
2. 软特征模型没有稳定服从约束，硬过滤又与单一标注目标不完全一致。
3. 不提交本阶段生成的约束门控候选，当前正式冠军继续保持：

```text
nDCG@20             0.5566
catalog_diversity   0.0307
lexical_diversity   0.7465
llm_judge_score     4.7500
composite_score     0.6373
```

4. 下一阶段不继续扩大特征数量，转向测量候选池上限与按轮次/目标类别的误差分布，
   找出当前冠军真正丢分的子集，再针对该子集训练专门模型。

## 2026-06-11：研究断点与下一步

### 当前正式冠军

```text
提交：
exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_top1lock_submission.zip

nDCG@20             0.5566
catalog_diversity   0.0307
lexical_diversity   0.7465
llm_judge_score     4.7500
composite_score     0.6373
```

CF、跨模型融合、显式约束整表重排和约束门控均已完成诊断并停止，不应继续提交这些
候选。当前冠军在恢复研究前保持冻结。

### 冠军误差地图

新增：

```text
analyze_ltr_error_segments.py
exp/ltr/champion_error_segments/report.json
```

完整 Dev 上，当前冠军对应模型的主要指标为：

```text
candidate recall       0.5370
recall@20              0.3586
nDCG@20                0.1905
Blind 加权 nDCG@20     0.1918
```

主要弱点：

| 子集 | candidate recall | nDCG@20 |
|---|---:|---:|
| specificity LL | 0.4829 | 0.1509 |
| Goal K：宽泛器乐/配乐发现 | 0.4976 | 0.1585 |
| 视觉/封面请求 | 0.4179 | 0.1359 |
| 年份/年代请求 | 0.4865 | 0.1608 |
| mood/activity 请求 | 0.5219 | 0.1715 |

明确同专辑、同艺人和直接播放请求已经相对较强，下一阶段不优先优化这些子集。

### Top200 候选池快速实验

为了判断下一步应先扩大候选池还是直接训练监督双塔，在固定前 100 个 Dev session、
共 800 个 turn 上，将每个召回通道从 Top100 扩大到 Top200。

缓存：

```text
cache/ltr/dev_top200_allturn_100_constraints
```

使用同一个旧冠军模型直接打分：

| 指标 | Top100 | Top200 | 变化 |
|---|---:|---:|---:|
| active candidate recall | 0.5350 | 0.5775 | **+0.0425** |
| recall@5 | 0.19375 | 0.19500 | +0.00125 |
| recall@20 | 0.35125 | 0.35125 | 0 |
| nDCG@20 | 0.17877 | 0.17892 | +0.00015 |

Top200 候选池净补回 34 个正例，但旧 Top100 排序器几乎无法把新增正例排进前 20。
这说明候选池扩大有真实价值，当前瓶颈转移到了排序训练。

### 恢复后的优先执行顺序

1. 构建完整 Top200 训练与 Dev 缓存，不启用 CF/query-dense，保持当前稳健通道：

```powershell
.\.venv\Scripts\python.exe train_ltr_ranker.py `
  --cache_only `
  --max_train_tasks 10000 `
  --train_turn_mode all `
  --dev_turn_mode all `
  --seed 13 `
  --channel_topk 200 `
  --embedding_batch_size 64 `
  --text_retrieval_batch_size 5000 `
  --train_feature_cache_dir cache/ltr/train10k_seed13_top200_v1 `
  --dev_feature_cache_dir cache/ltr/dev_all_top200_v1 `
  --device cuda `
  --no-enable_query_dense `
  --no-enable_cf_retrieval
```

2. 在 Top200 缓存上重新训练冠军裁剪变体，不能直接复用 Top100 模型：

```powershell
.\.venv\Scripts\python.exe train_ltr_cached_ablation.py `
  --train_feature_cache_dir cache/ltr/train10k_seed13_top200_v1 `
  --dev_feature_cache_dir cache/ltr/dev_all_top200_v1 `
  --output_dir exp/ltr/top200_10k_ablation `
  --variants no_metadata_cf_popularity `
  --seed 13
```

3. 只有当完整 Dev 的 Blind 加权 nDCG 明显高于旧冠军 `0.1918`，并且新增候选能进入
   Top20，才生成 Blind 候选。
4. 若 Top200 重训练仍不能利用新增候选，再训练监督式对话 Query -> 目标歌曲元数据
   双塔召回器，重点补 specificity LL、Goal K、视觉和年份请求。
5. 新 Blind 候选仍必须锁定当前冠军回复，并先检查改动列表数、Top1 一致率和 Top20
   overlap；不直接覆盖正式冠军。

## 2026-06-11：Top200 重训练、稳定并列排序与新候选

### 完整 Top200 缓存

已完成完整训练集和 Dev 的 Top200 特征缓存：

```text
训练：cache/ltr/train10k_seed13_top200_v1
Dev： cache/ltr/dev_all_top200_v1

训练任务数               10000
训练可学习 group          6891
训练候选行数              5041019
Dev 可学习 group          4759
Dev 候选行数              3557266
```

相对 Top100，Dev 多找回了 396 个正例，说明扩大候选池确实提高了候选召回上限。
但是直接在完整 Top200 group 上训练会让单个正例被过多负例稀释，因此增加了训练期
hard-negative 截断：保留全部正例，再按各召回通道倒数排名之和选取最难负例。Dev
始终保留完整候选池，不做截断。

### 修正并列分数评估错误

旧评估使用：

```python
rank = count(score > positive_score) + 1
```

这会把所有与正例同分的候选都当成排在正例之后。浅层 LightGBM 树存在大量同分，
因此旧本地 nDCG 明显虚高；真实推理则使用稳定 `argsort`，同分时保持候选原顺序。

已统一修正以下脚本，使本地评估与推理完全一致：

```text
train_ltr_ranker.py
evaluate_ltr_feature_ablation.py
evaluate_ltr_seed_ensemble.py
evaluate_cross_model_fusion.py
evaluate_locked_top1_fusion.py
```

同时在 `train_ltr_cached_ablation.py` 中增加：

```text
--stable_tie_validation
--max_train_candidates
```

前者使用稳定排序的自定义 `stable_ndcg@20` 做早停，后者只截断训练 hard negatives。

旧冠军裁剪在 Top100 上的结果变化如下：

| 评估方式 | Blind 轮次加权 Dev nDCG@20 |
|---|---:|
| 旧乐观并列排名 | 0.19181 |
| 稳定推理一致排名 | **0.17081** |

因此旧报告中的绝对本地 nDCG 不再与新报告直接比较；以后只使用稳定排序结果。

### Top200 hard-negative 截断

固定 36 个旧冠军特征、seed13，并使用稳定 nDCG@20 早停：

| 训练候选上限 | 最佳迭代 | Blind 加权 Dev nDCG@20 |
|---:|---:|---:|
| Top100 基线 | 3 | **0.17081** |
| 128 | 2 | 0.09662 |
| 224 | 2 | 0.16095 |
| 256 | 3 | 0.16581 |
| 288 | 3 | 0.16509 |
| 384 | 65 | 0.17062 |

Top200 只有保留约 384 个训练候选时才能恢复到 Top100 水平。对 hard-384 增加种子：

| 模型 | 最佳迭代 | Blind 加权 Dev nDCG@20 |
|---|---:|---:|
| seed13 | 65 | 0.17062 |
| seed47 | 61 | **0.17305** |
| seed71 | 45 | **0.17300** |

对应模型：

```text
exp/ltr/top200_10k_hard384_stableval_s47/legacy_champion/model.txt
exp/ltr/top200_10k_hard384_stableval_s71/legacy_champion/model.txt
```

两个独立种子都超过 Top100 基线，说明扩大候选池在正确训练后有小幅、可复现收益。

### 轮次门控与锁定评估

纯模型门控中，最好的简单方案是：

```text
Turn1  = v2 / Top100
Turn2+ = Top200 hard-384 seed71
```

其 Blind 加权 Dev nDCG@20 为 `0.17461`，相对同缓存上的 v2 `0.17056` 提升
`0.00406`，约 `+2.38%`。

但当前官方冠军的 Turn1 已使用 30k seed13，并在 Blind A 上得到过真实正收益。
因此新候选不改动冠军 Turn1，只替换 Turn2+。按实际候选结构分别评估：

```text
Turn1：Top100 seed13，锁定 v2 Top1      nDCG@20 = 0.15416
Turn2+：Top200 seed71，锁定 v2 Top1     nDCG@20 = 0.17508
组合后的 Blind 加权 Dev nDCG@20         约 0.17089
当前冠军同构本地估计                    约 0.17005
预计本地变化                            约 +0.00084
```

前缀锁定扫描中，Top1 锁定优于 Top3、Top5 和 Top10，因此仍只锁定 Top1。

### 待官方验证候选

候选严格保持：

```text
Turn1               完全等于当前冠军
Turn2+ Top1          完全等于当前冠军
Turn2+ Top2-Top20    Top200 hard-384 seed71
predicted_response   完全等于当前冠军
```

产物：

```text
空排序：
exp/inference/blindset_A/multichannel_ltr_top200_h384_s71_empty.json

正式预测：
exp/inference/blindset_A/multichannel_ltr_champion_turn1_top200_h384_s71_top1lock_prediction.json

提交包：
exp/inference/blindset_A/multichannel_ltr_champion_turn1_top200_h384_s71_top1lock_submission.zip
```

结构校验：

```text
Rows                         80
ChangedLists                 60
ChangedTop1                  0
ResponsesSameAsChampion      80 / 80
DuplicateRows                0
AvgTop20Overlap              18.55 / 20
MinTop20Overlap              14 / 20
UniqueTracks                 1434
CatalogDiversity / 47071     0.0304646
zip entries                  ["prediction.json"]
```

该候选尚未取得官方分数，当前正式冠军仍保持不变：

```text
nDCG@20             0.5566
catalog_diversity   0.0307
lexical_diversity   0.7465
llm_judge_score     4.7500
composite_score     0.6373
```

下一步先提交上述候选。若官方 nDCG 提升，则继续围绕 Top200 hard-384 做后续轮次
种子融合或增大训练任务数；若回退，则保留稳定评估修复，但停止 Top200 软排序替换，
转向训练监督式 Query-to-track 双塔召回器。
