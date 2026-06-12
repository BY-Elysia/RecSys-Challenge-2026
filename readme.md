# Music CRS Multi-Channel LambdaRank

当前用于 RecSys Challenge 2026 Music CRS Blind A 的最佳模型流程：

```text
对话历史与用户信息
  -> legacy BM25 + feedback-rich BM25
  -> 历史歌曲同歌手/同专辑结构召回
  -> image SigLIP2 最后一首/历史均值相似召回
  -> 监督式 Qwen3 Query adapter 补充 Turn1 候选
  -> Turn1 监督稠密 LambdaRank 专家
  -> Turn2+ 使用 36 维 LambdaRank v2
  -> 选取前 20 个歌曲 ID
  -> 锁定旧冠军 Top1 与回复
  -> 豆包生成/复用自然语言回复
```

当前 Blind A 最佳分数：

```text
ndcg@20             0.5575
catalog_diversity   0.0310
lexical_diversity   0.7465
llm_judge_score     4.7500
composite_score     0.6377
```

当前冠军冻结 Qwen3 基础向量和歌曲向量，只训练轻量 Query adapter；监督稠密通道
仅用于召回缺口最大的 Turn1，并由专门的 LambdaRank 模型调整 Top2-20。旧冠军
Top1、Turn2+ 排序和全部回复保持不变。相对上一版，官方 `nDCG@20` 提升 `0.0009`，
目录多样性提升 `0.0003`，Judge 与词汇多样性不变。

完整分数历史、失败方向、保留产物和复现记录见
[EXPERIMENTS.md](EXPERIMENTS.md)。

当前冠军模型保留的核心信号：

```text
1. 两套 BM25 Query 的候选 rank
2. 历史歌曲同歌手和同专辑关系
3. image SigLIP2 歌曲续推相似度
4. 当前请求对歌名、歌手、专辑的显式匹配
5. turn、历史长度、goal category 和 specificity
6. Turn1 监督式 Qwen3 稠密召回的候选 rank 与 present 标记
```

## 环境配置

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -e .
```

训练和推理需要安装支持 CUDA 的 PyTorch。建议始终显式使用
`.\.venv\Scripts\python.exe`，避免误调用系统 Python 中的 CPU 版 PyTorch。

生成回复前，设置豆包 API 环境变量：

```powershell
$env:DOUBAO_API_KEY = "你的 API Key"
$env:DOUBAO_REASONING_EFFORT = "minimal"
$env:DOUBAO_TEMPERATURE = "0.50"
$env:DOUBAO_MAX_TOKENS = "140"
$env:DOUBAO_CONCURRENCY = "4"
```

## 运行当前最佳 Blind A 排序

```powershell
.\.venv\Scripts\python.exe run_inference_ltr_blindset.py `
  --tid bm25_tags_doubao_blindset_A `
  --model_path exp/ltr/cached_ablation_10k_top100/no_metadata_cf_popularity/model.txt `
  --turn1_model_path exp/ltr/supervised_dense_turn12_train/legacy_plus_supervised_dense_rank/model.txt `
  --output_name multichannel_ltr_turn1_supervised_dense_later_v2_empty `
  --channel_topk 100 `
  --history_turns 0 `
  --enable_supervised_dense `
  --supervised_dense_checkpoint exp/dense/supervised_qwen_query_adapter_10k_lr1e5_inbatch005 `
  --supervised_dense_query_batch_size 16 `
  --no-enable_query_dense `
  --no-enable_cf_retrieval `
  --embedding_batch_size 64 `
  --device cuda
```

将 Top1 与回复锁回旧冠军，只保留监督稠密专家的 Turn1 后排排序：

```powershell
.\.venv\Scripts\python.exe merge_locked_top1_prediction.py `
  --base_path exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_top1lock_prediction.json `
  --candidate_path exp/inference/blindset_A/multichannel_ltr_turn1_supervised_dense_later_v2_empty.json `
  --output_path exp/inference/blindset_A/multichannel_ltr_turn1_supervised_dense_later_v2_top1lock_prediction.json `
  --lock_prefix 1
```

上一版 v2 排序可用以下命令复现：

```powershell
.\.venv\Scripts\python.exe run_inference_ltr_blindset.py `
  --model_path exp/ltr/cached_ablation_10k_top100/no_metadata_cf_popularity/model.txt `
  --output_name multichannel_ltr_lean_v2_empty `
  --channel_topk 100 `
  --history_turns 0 `
  --device cuda
```

## 生成 v2 回复

```powershell
.\.venv\Scripts\python.exe regenerate_responses_blindset.py `
  --tid bm25_tags_doubao_blindset_A `
  --input_path exp/inference/blindset_A/multichannel_ltr_lean_v2_empty.json `
  --output_path exp/inference/blindset_A/multichannel_ltr_lean_v2_prediction.json
```

## 打包当前最佳提交

```powershell
.\.venv\Scripts\python.exe -c "import zipfile,pathlib; d=pathlib.Path('exp/inference/blindset_A'); dst=d/'multichannel_ltr_turn1_supervised_dense_later_v2_top1lock_submission.zip'; dst.unlink(missing_ok=True); z=zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED); z.write(d/'multichannel_ltr_turn1_supervised_dense_later_v2_top1lock_prediction.json','prediction.json'); z.close(); print(dst.resolve())"
```

## 监督稠密召回训练

训练轻量 Query adapter。Qwen3 编码器和歌曲向量保持冻结：

```powershell
.\.venv\Scripts\python.exe train_supervised_dense_retriever.py `
  --output_dir exp/dense/supervised_qwen_query_adapter_10k_lr1e5_inbatch005 `
  --learning_rate 1e-5 `
  --in_batch_weight 0.05 `
  --alignment_weight 1.0 `
  --epochs 4 `
  --device cuda
```

生成监督稠密 Turn1 专家候选，Turn2+ 继续使用 v2：

```powershell
.\.venv\Scripts\python.exe run_inference_ltr_blindset.py `
  --tid bm25_tags_doubao_blindset_A `
  --model_path exp/ltr/cached_ablation_10k_top100/no_metadata_cf_popularity/model.txt `
  --turn1_model_path exp/ltr/supervised_dense_turn12_train/legacy_plus_supervised_dense_rank/model.txt `
  --output_name multichannel_ltr_turn1_supervised_dense_later_v2_empty `
  --channel_topk 100 `
  --history_turns 0 `
  --enable_supervised_dense `
  --supervised_dense_checkpoint exp/dense/supervised_qwen_query_adapter_10k_lr1e5_inbatch005 `
  --supervised_dense_query_batch_size 16 `
  --no-enable_query_dense `
  --no-enable_cf_retrieval `
  --embedding_batch_size 64 `
  --device cuda
```

锁定正式冠军 Top1 和回复：

```powershell
.\.venv\Scripts\python.exe merge_locked_top1_prediction.py `
  --base_path exp/inference/blindset_A/multichannel_ltr_turn1_s13_later_v2_top1lock_prediction.json `
  --candidate_path exp/inference/blindset_A/multichannel_ltr_turn1_supervised_dense_later_v2_empty.json `
  --output_path exp/inference/blindset_A/multichannel_ltr_turn1_supervised_dense_later_v2_top1lock_prediction.json `
  --lock_prefix 1
```

打包：

```powershell
.\.venv\Scripts\python.exe -c "import zipfile,pathlib; d=pathlib.Path('exp/inference/blindset_A'); dst=d/'multichannel_ltr_turn1_supervised_dense_later_v2_top1lock_submission.zip'; dst.unlink(missing_ok=True); z=zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED); z.write(d/'multichannel_ltr_turn1_supervised_dense_later_v2_top1lock_prediction.json','prediction.json'); z.close(); print(dst.resolve())"
```

## CF 召回增强实验

构建 10k 训练与完整 Dev 特征缓存：

```powershell
.\.venv\Scripts\python.exe train_ltr_ranker.py `
  --train_feature_cache_dir cache/ltr/train10k_seed13_top100_cf_v2 `
  --dev_feature_cache_dir cache/ltr/dev_all_top100_cf_v2 `
  --cache_only `
  --max_train_tasks 10000 `
  --train_turn_mode all `
  --dev_turn_mode all `
  --channel_topk 100 `
  --history_turns 0 `
  --enable_cf_retrieval `
  --no-enable_query_dense `
  --embedding_batch_size 64 `
  --text_retrieval_batch_size 500 `
  --seed 13 `
  --device cuda
```

训练与消融：

```powershell
.\.venv\Scripts\python.exe train_ltr_cached_ablation.py `
  --train_feature_cache_dir cache/ltr/train10k_seed13_top100_cf_v2 `
  --dev_feature_cache_dir cache/ltr/dev_all_top100_cf_v2 `
  --output_dir exp/ltr/cf_v2_10k_top100_ablation `
  --variants baseline no_cf_retrieval no_user_cf_retrieval no_history_cf_retrieval no_metadata_cf_popularity lean_history_cf `
  --seed 13
```

运行当前最佳 CF 候选：

```powershell
.\.venv\Scripts\python.exe run_inference_ltr_blindset.py `
  --model_path exp/ltr/cf_v2_10k_top100_ablation/no_metadata_cf_popularity/model.txt `
  --output_name multichannel_ltr_cf_v2_lean_empty `
  --channel_topk 100 `
  --history_turns 0 `
  --enable_cf_retrieval `
  --no-enable_query_dense `
  --embedding_batch_size 64 `
  --device cuda
```

当前建议首先提交：

```text
exp/inference/blindset_A/multichannel_ltr_cf_v2_top5lock_submission.zip
```

## 旧版 MiniLM 实验

以下内容用于复现早期 BM25 + MiniLM 路线，已不再是当前最佳主线。

### 从头训练 Reranker

```powershell
.\.venv\Scripts\python.exe train_reranker.py `
  --model_name cross-encoder/ms-marco-MiniLM-L6-v2 `
  --output_dir ./exp/reranker/minilm_bm25_listwise_focused `
  --retrieval_topk 200 `
  --negatives_per_positive 19 `
  --max_turns 50000 `
  --epochs 3 `
  --batch_size 2 `
  --gradient_accumulation_steps 8 `
  --max_length 384 `
  --learning_rate 1e-5 `
  --retrieval_query_mode legacy `
  --reranker_query_mode focused `
  --history_turns 3 `
  --device cuda
```

训练过程中会根据验证集最高的 `ndcg@20` 保存最佳 checkpoint。

### 运行旧版 Blind A 推理

使用早期保留的 MiniLM 对照模型：

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

输出文件：

```text
exp/inference/blindset_A/tags_top400_e1_empty.json
```

### 旧版结果仅重新生成回复

该命令会保留全部 `predicted_track_ids`，仅重写 `predicted_response`：

```powershell
.\.venv\Scripts\python.exe regenerate_responses_blindset.py `
  --tid bm25_tags_doubao_blindset_A `
  --input_path exp/inference/blindset_A/tags_top400_e1_empty.json `
  --output_path exp/inference/blindset_A/tags_top400_e1_prediction.json
```

### 打包旧版提交文件

```powershell
.\.venv\Scripts\python.exe -c "import zipfile,pathlib; d=pathlib.Path('exp/inference/blindset_A'); dst=d/'tags_top400_e1_submission.zip'; dst.unlink(missing_ok=True); z=zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED); z.write(d/'tags_top400_e1_prediction.json','prediction.json'); z.close(); print(dst.resolve())"
```

校验提交文件：

```powershell
.\.venv\Scripts\python.exe -c "import json,zipfile; z=zipfile.ZipFile('exp/inference/blindset_A/tags_top400_e1_submission.zip'); d=json.loads(z.read('prediction.json').decode('utf-8')); print(z.namelist(),len(d),sorted(set(len(x['predicted_track_ids']) for x in d)))"
```

预期输出：

```text
['prediction.json'] 80 [20]
```

官方格式要求每条预测最多 20 首歌曲。Blind A 只有 80 条预测、曲库共有
47,071 首，因此合规的 `catalog_diversity` 理论上限仅为
`1600 / 47071 = 0.033991`。不要通过在 Top-20 后附加额外歌曲提高该指标；
公开 evaluator 的 nDCG 会截断到 20，但 diversity 会统计整条列表，这属于
评测实现缺口，并违反官方提交格式。

检查当前提交的合法覆盖率与多样化风险：

```powershell
.\.venv\Scripts\python.exe analyze_catalog_diversity.py `
  exp/inference/blindset_A/multichannel_ltr_lean_v2_prediction.json
```

### 标签召回实验

该历史实验使用 `tag_list` BM25 Top-400，并基于新候选继续训练 MiniLM
Reranker。先运行完整 Dev 召回评估：

```powershell
.\.venv\Scripts\python.exe evaluate_devset.py `
  --tid bm25_tags_doubao_blindset_A `
  --candidate_topk 400 `
  --output_path exp/evaluation/dev_tags_top400_retrieval.json
```

继续训练一轮：

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

训练后用官方 Dev 的完整候选池评估：

```powershell
.\.venv\Scripts\python.exe evaluate_devset.py `
  --tid bm25_tags_doubao_blindset_A `
  --reranker_dir ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --candidate_topk 400 `
  --rerank_batch_size 64 `
  --output_path exp/evaluation/dev_tags_top400_retrained.json `
  --device cuda
```

详细实验依据、候选召回率和对照方案见 [EXPERIMENTS.md](EXPERIMENTS.md)。

Dev 指标通过后运行 Blind A 排序推理：

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

## 困难负例训练

先使用当前最佳模型从每条训练样本前 100 个清洗候选中缓存 12 个高分错误歌曲：

```powershell
.\.venv\Scripts\python.exe mine_hard_negatives.py `
  --model_name ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --output_path ./exp/hard_negatives/tags_top400_e1_pool100.jsonl `
  --candidate_pool_size 100 `
  --hard_negatives_per_positive 12 `
  --max_turns 50000 `
  --device cuda
```

再使用 12 个困难负例和 7 个普通 BM25 负例继续训练一轮：

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

## 语义向量补召回

项目包含一条实验性语义召回路线：标签 BM25 Top-400 保持为主通道，使用官方
Qwen3 metadata 向量补充 BM25 漏掉的歌曲，再由当前最佳 reranker 排序。

首次运行会下载 `Qwen/Qwen3-Embedding-0.6B`，并缓存官方曲库向量。先用
100 个 Dev 会话验证候选召回：

```powershell
.\.venv\Scripts\python.exe evaluate_hybrid_recall_devset.py `
  --max_sessions 100 `
  --bm25_topk 400 `
  --dense_topk 100 `
  --dense_batch_size 16 `
  --output_path exp/evaluation/dev_hybrid_recall_100sessions_dense100.json `
  --device cuda
```

评估通道惩罚后的最终排序：

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

当前抽样 Dev 最优参数为 `dense_topk=100`、`dense_penalty=2.0`，但
`nDCG@20` 仅从 `0.126738` 提升到 `0.127228`，尚未达到替换当前最佳主线的
门槛。生成该实验路线的 Blind A 空回复排序文件：

```powershell
.\.venv\Scripts\python.exe run_inference_hybrid_dense_blindset.py `
  --reranker_dir ./exp/reranker/minilm_bm25_tags_top400_e1 `
  --bm25_topk 400 `
  --dense_topk 100 `
  --dense_penalty 2.0 `
  --output_name tags_top400_dense100_penalty2_empty_blindset_A `
  --device cuda
```

## 混合候选训练

让 reranker 适配 Dense 独有候选时，先缓存训练集 Dense Top-100：

```powershell
.\.venv\Scripts\python.exe cache_dense_candidates.py `
  --output_path ./exp/dense_candidates/train_dense_top100_focused_50k.jsonl `
  --dense_topk 100 `
  --max_turns 50000 `
  --write_batch_size 1024 `
  --dense_batch_size 16 `
  --device cuda
```

再从当前最佳模型继续训练。每组使用 4 个 Dense 独有负例和 15 个 BM25 负例：

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

该模型在固定 100 个 Dev 会话上，使用 Dense Top-100 和 `dense_penalty=2.0`
时，`nDCG@20` 从旧主线的 `0.126738` 提升到 `0.128802`，但仍未达到
`+0.005` 的正式替换门槛。详细统计见 [EXPERIMENTS.md](EXPERIMENTS.md)。

正式 Blind A 评测中，该 Hybrid 方案的 `ndcg@20` 为 `0.1806`，低于纯
BM25 排序主线的 `0.1946`；`composite_score=0.4559` 的新高主要来自回复评分。
因此 Dense Hybrid 暂不进入排序主线。已生成“混合训练模型 + 纯 BM25
Top-400”对照排序文件：

```text
exp/inference/blindset_A/tags_top400_dense_mix4_bm25only_empty.json
exp/inference/blindset_A/tags_top400_dense_mix4_bm25only_prediction.json
exp/inference/blindset_A/tags_top400_dense_mix4_bm25only_submission.zip
```

该对照与 Dense Hybrid 提交的 80 条 Top-1 推荐全部相同，因此复用了完全相同
的回复文本，下一次评测可以直接衡量去除 Dense 候选后的排序变化。

正式评测中，纯 BM25 对照与 Dense Hybrid 的五项分数在日志显示精度下完全
一致，`ndcg@20` 均为 `0.1806`。因此排序退化主要来自混合负例继续训练，
而不是 Dense 候选本身。该路线停止，后续实验继续以
`minilm_bm25_tags_top400_e1` 为主线起点。
