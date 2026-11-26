# Server16 外部资源路径

以下是原实验在 16 服务器上的路径。另一台服务器可用对应目录替换，保持固定数据、索引身份与模型版本一致。

| 类型 | 内容 | 16 服务器位置 |
| --- | --- | --- |
| 测试数据 | 固定的 2,000 题 | `/data1/zixuan/qrecc_full/subsets/test_2000.json` |
| 原始及整理后的数据 | 官方 train/test、语料原包、整理后的题库与清单 | `/data1/zixuan/qrecc_full/` |
| 训练数据 | SFT 与偏好训练两个输入 JSONL | `/data1/zixuan/RAG_Research/experiments/adaqr_compcqr/assets/` |
| 训练索引 | 基础融合奖励使用的 Pyserini/Lucene BM25 | `/data1/zixuan/qrecc_full/lucene-papers/lucene/` |
| 评测索引 | OpenSearch 的 `qrecc-official-bm25`，源服务数据目录 | `/data1/zixuan/opensearch-qrecc/` |
| 模型 | 第一阶段 Ollama `qwen3:8b` | `/data1/zixuan/ollama_models/` |
| 模型 | Mistral 基础权重 | `/data1/zixuan/models/Mistral-7B-v0.1/` |
| 模型 | CompCQR 组件生成模型 | `/data1/zixuan/models/Llama-3.1-8B-Instruct/` |
| 模型 | Evidence2 验证模型 | `/data1/zixuan/models/DeBERTa-v3-base-mnli-fever-anli/` |
| 提示词 | 原始 CompCQR 运行资产 | `/data1/zixuan/RAG_Research/prompts/compcqr/` |
| 历史缓存 | 第一阶段原始 A/B 重写的签名与检查点 | `/data1/zixuan/RAG_Research/outputs/bm25_gpu4_experiments/main/` |
| 第一阶段结果 | report.json、summary.csv、details.csv、manifest.json | `/data1/zixuan/RAG_Research/outputs/bm25_gpu4_experiments/main-test2000/` |
| 共用 SFT | 基础融合和 Evidence2 的 SFT 适配器 | `/data1/zixuan/RAG_Research/experiments/adaqr_compcqr/outputs/train4000_v1/sft/` |
| 基础融合产物 | dataset、rewards_baseline、preferences_baseline.jsonl、dpo_baseline | `/data1/zixuan/RAG_Research/experiments/adaqr_compcqr/outputs/train_half_v1/` |
| Evidence2 产物 | rewards_evidence、preferences_evidence.jsonl、dpo_evidence | `/data1/zixuan/RAG_research-evidence/experiments/adaqr_compcqr/outputs/server16_evidence/` |

基础融合奖励目录的 `manifest.json` 和 `checkpoint.jsonl` 是 Evidence2 的输入。原始服务器脚本带有环境路径和部分 GPU 默认值。迁移时查看对应参数与环境变量，再运行小样本验证。

上传 GitHub 时使用代码包根目录及其内容。逐题结果、训练样本、检查点、日志、语料、索引、安装包与模型权重位于单独的原始位置。
