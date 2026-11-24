# AdaQR × CompCQR 证据奖励

这个项目在基础融合的奖励上加入逐段证据评分，再用更新后的偏好对训练重写器。对同一轮的每条候选重写，基础融合分别做单查询 BM25 和 CompCQR 组合查询。两路各取前 5 个段落，由冻结的 Qwen3-8B 计算训练参考答案的条件对数概率。证据评分统一使用该轮固定的独立问题和训练答案。程序按段落检索权重聚合每一路奖励，然后以 50% 和 50% 合成候选总分。只在同一轮内配对，分差严格大于 0.1 时保留高分与低分候选。

## 16 服务器运行方式

这一版使用基础融合已发布的奖励记录作为输入。切换到 Qwen3-8B 后，需要重新训练 SFT adapter、采样候选并计算基础融合奖励，再将 `ADAQR_FOUNDATION_RUN` 和 `ADAQR_SFT_ADAPTER` 指向新产物。原记录提供两个检索分支的段落、排序权重和答案对数概率。冻结的 DeBERTa 验证器取本轮保存的 `components.mq` 和训练集参考答案构造一个假设，各候选及两路检索共用该假设。同一轮内一个段落的 NLI 分数只计算一次。

每个段落的支持度为 `max(0, P(蕴含) - P(矛盾))`。按原检索权重求加权对数证据项，再加到各路原奖励上。默认公式为 `原奖励 + λ × Σ(检索权重 × log(ε + (1-ε) × 支持度))`，其中 `λ=1.0`，`ε=0.1`。评测仍以查询重写器输出做检索，参考答案用于训练期的证据验证。

从项目根目录进入 `experiments/adaqr_compcqr`，按需运行对应阶段。

```bash
cd experiments/adaqr_compcqr
bash run_server16_evidence.sh check
bash run_server16_evidence.sh smoke 3
bash run_server16_evidence.sh rewards
bash run_server16_evidence.sh pairs
bash run_server16_evidence.sh dpo
bash run_server16_evidence.sh evaluate single --dry-run
bash run_server16_evidence.sh evaluate single
```

| 命令 | 用途 |
| --- | --- |
| `check` | 预检已发布的基础融合结果和证据验证器 |
| `smoke [轮数]` | 使用 CPU 验证少量已有轮次 |
| `rewards` | 通过 GPU 补算证据奖励并合并分片 |
| `pairs` | 导出新偏好对，沿用两路各一半及分差阈值 `0.1` |
| `dpo` | 从原 SFT adapter 重新开始，使用与基础融合相同的四卡训练器，独立保存参考评分再执行 DPO |
| `evaluate single` | 按固定 2,000 轮计划进行单查询推理，也可选 `compcqr`，与原对照一样向选择器提供 5 个候选段落 |
| `status` | 查看证据奖励、偏好对和模型产物 |

默认复用 `/data1/zixuan/RAG_Research/experiments/adaqr_compcqr/outputs/train_half_v1` 和同一实验目录下的 `outputs/train4000_v1/sft`，通过 `ADAQR_FOUNDATION_RUN` 与 `ADAQR_SFT_ADAPTER` 可指定其他输入。验证器默认读取 `/data1/zixuan/models/DeBERTa-v3-base-mnli-fever-anli`，可通过 `ADAQR_VERIFIER_DIR` 指定路径。Python 默认使用 `/data1/zixuan/adaqr_compcqr_env/bin/python`。

训练和验证默认使用物理 GPU `2,3,4,5`。`ADAQR_EVIDENCE_GPUS` 接受逗号分隔的物理 GPU 编号，对同一批奖励分片续跑时保持 GPU 数量一致。GPU 阶段会检查显存，并取得与基础融合共享的运行锁，适合在基础融合 GPU 任务结束后启动。权重、数值下限和每批段落数可通过 `ADAQR_EVIDENCE_WEIGHT`、`ADAQR_EVIDENCE_EPSILON`、`ADAQR_EVIDENCE_BATCH_SIZE` 设置。

结果默认保存在本项目 `experiments/adaqr_compcqr/outputs/server16_evidence/`，包括 `rewards_evidence/`、`preferences_evidence.jsonl`、`dpo_evidence_reference/`、`dpo_evidence/` 及评测报告。`ADAQR_EVIDENCE_OUTPUT` 可以指定当前实验 `outputs/` 下的新目录，用于另一组参数。基础融合结果作为只读输入。

## 本次实验设置

| 环节 | 设置 |
| --- | --- |
| SFT | Qwen3-8B（非思考模式）。实验用 4,000 条人工重写中的 3,329 条非首轮样本。 |
| 候选采样 | 从固定的 39,677 条 QReCC 训练轮次采样，每轮生成 3 条候选。 |
| 半量训练 | 随机种子 42，按轮次和可见历史字符数分层抽取 19,839 轮。候选去重后少于 2 条的轮次在组件生成前剔除。本次实验最终纳入 16,805 轮。 |
| CompCQR | Llama-3.1-8B-Instruct 生成问答组件。稀疏查询按非空子集组合，每条查询检索深度为 100，使用 RRF 聚合。 |
| 训练检索 | QReCC 全量段落索引，Pyserini/Lucene BM25，`k1=0.82`、`b=0.68`。 |
| DPO | 从同一个 SFT adapter 开始，训练 1 轮，`beta=0.1`。 |
| 评测 | 固定 2,000 轮 QReCC 测试子集，默认比较单查询推理下的重写和检索。对照评测使用原实验的 OpenSearch BM25 索引。 |

## 安装与输入

训练启动器需要 Linux、Python 3.11、CUDA GPU 和 Java。下面是原实验使用的 Python 依赖配置。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r experiments/adaqr_compcqr/requirements.txt
```

1. 准备本地模型、全量 Lucene 索引、现成的 SFT adapter，以及已采样的 `samples.jsonl`。候选文件每行需要 `id`、`split`、`turn`、`history`、`question`、`answer`、`candidates`。使用原采样文件可复用实验的候选输入。
2. 根据 [提示词资产说明](prompts/compcqr/README.md) 准备原始提示与示例。程序校验资产哈希。
3. 将 [环境变量示例](experiments/adaqr_compcqr/foundation.env.example) 复制为 `.env`，填写路径和物理 GPU 编号并运行 `source .env`。组件可用 2 或 4 张卡，基础奖励默认使用 `ADAQR_SCORE_GPUS` 的两张卡，可用 `ADAQR_REWARD_GPUS` 单独指定四张卡。

从原始资产重建训练时，实验目录里的 `build_dpo_training.py` 从官方 QReCC 和 InfoCQR 固定划分生成训练轮次和 ID，`prepare_user_sft.py` 转换用户提供的 SFT 样本，`prepare_paper_index.py` 从官方段落归档构建 Lucene。用以下命令检查输入参数和训练入口。

```bash
cd experiments/adaqr_compcqr
python build_dpo_training.py --help
python prepare_user_sft.py --help
python prepare_paper_index.py --help
python train_fusion.py sft --help
python train_fusion.py sample --help
```

`train_fusion.py` 的 SFT 和采样任务写入实验目录的 `outputs/`。重新采样后，候选重复数和可用轮次数可能随采样结果变化。

## 运行基础融合

在实验目录分步执行。上一阶段完成后，再启动下一阶段。

```bash
bash run_foundation.sh prepare
bash run_foundation.sh components
bash run_foundation.sh baseline
bash run_foundation.sh pairs
bash run_foundation.sh dpo
```

`prepare` 写出固定名单及其哈希。`components` 从两个检查点续跑，完成的轮次按原分片写回。`baseline` 记录单查询与组合查询两路奖励、段落及权重。`pairs` 导出偏好对，`dpo` 从 SFT adapter 开始训练。可随时用 `bash run_foundation.sh status` 查看组件和基础评分的累计进度。

基础奖励默认每个 GPU 启动一个评分进程，每个进程用 4 个 Lucene 线程批量执行同一轮的 BM25 查询，并按原查询顺序返回结果。已有奖励记录可继续使用，新增结果由一个进程写回原有的两份检查点。在 16 服务器上指定四张卡可用以下命令。设置 `ADAQR_BM25_BATCH_THREADS=0` 可切回串行检索。

```bash
ADAQR_REWARD_GPUS=2,3,4,5 bash run_foundation.sh baseline
```

### 复现原 2,000 轮评测

`evaluate_fusion.py` 包含原来的对照评测入口。它外置依赖原 RAG 项目、已完成且签名有效的对照结果目录、原始 QReCC 测试文件、固定的 `test_2000.json` 和原 OpenSearch 全量索引。用环境变量示例里的四个评测路径指定输入。

```bash
bash run_foundation.sh evaluate single --dry-run
bash run_foundation.sh evaluate single
```

固定子集的 SHA-256 为 `9052fa059df48f890354c34814e71746d1067eab3c75310b8d7a8d310031e010`。评测将验证 2,000 条记录的内容和顺序。用于严格对照的代码和索引仍应从原评测项目提供。公开包调整了运行入口和路径，本包生成的检查点应在本包内续跑。

## 代码结构

| 文件 | 用途 |
| --- | --- |
| `experiments/adaqr_compcqr/adaqr_half_dataset.py` | 固定半量抽样、候选去重检查和数据签名 |
| `experiments/adaqr_compcqr/adaqr_component_formats.py` | 兼容模型输出中明确可恢复的格式差异 |
| `experiments/adaqr_compcqr/parallel_components.py` | 多卡处理剩余样本，单进程持久化检查点 |
| `experiments/adaqr_compcqr/adaqr_half_rewards.py` | 两路检索、逐段似然与奖励加权 |
| `experiments/adaqr_compcqr/train_fusion.py` | SFT、候选采样、偏好对导出和 DPO |
| `experiments/adaqr_compcqr/evaluate_fusion.py` | 固定测试子集上的重写与检索评测 |

第三方方法、模型和数据来源见 [来源说明](THIRD_PARTY_NOTICES.md)。
