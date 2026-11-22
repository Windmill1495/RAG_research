# AdaQR × CompCQR 基础融合

这个项目用检索结果为多轮查询重写构建偏好对，并用 DPO 训练重写器。对同一轮的每条候选重写，分别做单查询 BM25 和 CompCQR 组合查询。两路各取前 5 个段落，由冻结的 Mistral-7B-v0.1 计算生成训练参考答案的条件对数概率。程序按段落检索权重聚合每一路奖励，然后以 50% 和 50% 合成候选总分。只在同一轮内配对，分差严格大于 0.1 时保留高分与低分候选。

## 本次实验设置

| 环节 | 设置 |
| --- | --- |
| SFT | Mistral-7B-v0.1。实验用 4,000 条人工重写中的 3,329 条非首轮样本。 |
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
3. 将 [环境变量示例](experiments/adaqr_compcqr/foundation.env.example) 复制为 `.env`，填写路径和物理 GPU 编号并运行 `source .env`。组件可用 2 或 4 张卡，基础奖励使用配置的 2 张卡。

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
