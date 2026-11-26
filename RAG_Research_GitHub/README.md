# QReCC 多轮查询重写实验

本仓库收录第一阶段的 Qwen3-8B 三策略对照、AdaQR × CompCQR 基础融合及 Evidence2 的实现。两套项目保留原有模块结构。各阶段使用同一固定的 2,000 轮 QReCC 测试子集。

```text
.
├── RAG_Research/
│   ├── deploy/server16_compcqr_test2000.py  # 第一阶段固定子集入口
│   ├── evaluate_rewrite.py                  # 查询重写与检索评测
│   ├── rag/                                 # 重写、CompCQR、OpenSearch 与数据模块
│   ├── prompts/compcqr/                      # 提示词来源与恢复说明
│   ├── tests/                               # 检索和缓存相关单元测试
│   └── experiments/adaqr_compcqr/            # 第二阶段基础融合
│       ├── run_train4000_sft.sh              # SFT
│       ├── run_adaqr_half.sh                 # 抽样、组件、批量奖励和偏好对
│       ├── server16_dpo_ddp.py               # DPO 参考评分及训练
│       └── evaluate_fusion.py               # 对训练后模型做评测
├── RAG_research-evidence/
│   └── experiments/adaqr_compcqr/            # 第三阶段 Evidence2
│       ├── run_server16_evidence.sh          # 证据评分、偏好对、DPO、评测
│       ├── server16_evidence.py              # 复用基础奖励与段落
│       └── evidence_verifier.py             # DeBERTa NLI 验证
├── SERVER16_PATHS.md                         # 数据、索引、模型和结果
└── THIRD_PARTY_NOTICES.md
```

## 1. Qwen3-8B 三策略对照

直接重写、Structured 和 CompCQR 使用同一份测试集和 OpenSearch BM25 索引。CompCQR 的多条组合查询分别检索，之后用 RRF 融合并按配置执行段落选择。安装 [第一阶段依赖](RAG_Research/requirements-full.txt)，准备模型和索引，复制 `config.server16.gpu.example.json` 为 `config.server16.gpu.local.json`，然后填入本机路径。原运行入口要求 Ollama 0.9.6 可执行文件位于项目内的 `.tools/ollama-0.9.6/bin/ollama`。

`deploy/server16_compcqr_test2000.py --subset-file <路径> --gpus <卡号...>` 采用历史回收流程。它从 `outputs/bm25_gpu4_experiments/main/` 的签名清单和检查点读入直接重写、Structured，再对固定测试子集运行 CompCQR。该缓存及完整实验产物见 [Server16 路径](SERVER16_PATHS.md)。准备新的逐轮评测时，通用入口为 `python evaluate_rewrite.py --help`。

## 2. AdaQR × CompCQR 基础融合

进入 `RAG_Research/experiments/adaqr_compcqr/`。准备 [训练依赖](RAG_Research/experiments/adaqr_compcqr/requirements.txt)、外部数据与权重。SFT 的项目入口为 `run_train4000_sft.sh`。偏好训练使用固定抽样并去除无有效候选对的轮次。组件生成和批量检索的入口为 `run_adaqr_half.sh`，批量执行依赖 `parallel_adaqr_components_gpu2345.py` 与 `parallel_adaqr_half.py`。运行前按需调整 GPU 与外部路径。

训练奖励使用 Pyserini/Lucene BM25，按实际运行设置组合单查询与 CompCQR 奖励。奖励合并后，`train_fusion.py pairs` 生成偏好对，`server16_dpo_ddp.py` 的 `reference` 和 `train` 子命令依次计算 DPO 参考分并进行训练。评测入口为 `evaluate_fusion.py`，使用 OpenSearch BM25。

## 3. Evidence2

进入 `RAG_research-evidence/experiments/adaqr_compcqr/` 并配置 `ADAQR_FOUNDATION_PROJECT` 指向同级的 `RAG_Research`，用 `ADAQR_FOUNDATION_RUN` 和 `ADAQR_SFT_ADAPTER` 指向原训练产物。按 `check → smoke → rewards → pairs → dpo → evaluate` 顺序运行 `run_server16_evidence.sh` 对应的子命令。每轮候选共用固定验证问题和训练参考答案，DeBERTa 对基础融合已经检索的段落补算证据分。用新偏好对从同一 SFT 权重进行 DPO。

各阶段的输入、索引、可选提示词资产和服务器产物见 [SERVER16_PATHS.md](SERVER16_PATHS.md)。发布包包含源码和配置模板。运行需准备对应外部资产。
