# SimpleRAG Research · 对话问题重写与混合检索

三个实验共用 **BM25 + `bge-m3` 混合检索**，用 RRF 融合词项和向量排名。实验一比较 Query 构造策略，实验二比较 Qwen3 模型规模，主实验在 8B 上比较重写质量和下游检索效果。网页可以逐题查看，命令行负责成批运行和导出。

从 [实验指南](docs/实验指南.md) 查看三个实验的命令和指标。主实验细节见 [问题重写实验说明](docs/问题重写实验说明.md)，代码结构见 [阅读顺序](docs/阅读顺序.md)。

## 运行 307 题 A/B 对照

在项目目录打开 PowerShell。全局默认模型为 `qwen3:1.7b`，主实验使用 `qwen3:8b`。运行前启动 Ollama，用 `ollama list` 确认已安装 `qwen3:8b` 和 `bge-m3`。缺少时可分别用 `ollama pull qwen3:8b` 和 `ollama pull bge-m3` 安装。

```powershell
# 先用两题离线检查数据、输出和评分流程
& 'G:\Anaconda3\python.exe' -X utf8 -B evaluate_rewrite.py --mode demo --limit 2
# 用 Qwen3-8B 跑项目全部 307 题，评估 A/B 重写与下游检索
& 'G:\Anaconda3\python.exe' -X utf8 -B evaluate_rewrite.py --mode ollama --model qwen3:8b --split all
```

`demo` 用启发式规则和 BM25 检查流程。正式报告使用 Ollama 模式的混合检索。默认读取 `data/qrecc/questions.jsonl` 的 102 道开发题和 205 道测试题，两组各 307 条，共 614 条结果。可用 `--split dev` 或 `--split test` 单独运行一组数据。批量报告保存在 `outputs/rewrite_runs/` 的新目录，分别记录重写和检索的分数、失败与耗时。

### 在网页逐题看

```powershell
& 'G:\Anaconda3\python.exe' -X utf8 -B app.py --page rewrite --mode ollama --port 8001
```

打开 [重写实验页面](http://127.0.0.1:8001/rewrite)，选一道题并运行对照。页面并排显示 A/B 输出、参考重写、ROUGE-1 F1、耗时和修复或回退信息。该页面面向逐题分析，批量报告和盲评表由上面的命令行生成。把 `--mode ollama` 换成 `--mode demo` 可离线查看流程。

### 继续做模型和数据对照

- **换模型**时，保持题目、策略和评测选项相同，在批量命令后加 `--model` 和 `ollama list` 中的模型标签。网页启动可用 `--rewrite-model` 指定重写模型。每个模型单独保存运行结果。
- **用官方 JSON 单独测重写质量**时，使用 `--data` 指向 QReCC 官方对话 JSON，指定 `--data-format qrecc-json --split test --rewrite-only` 或换成训练划分。完整下游检索还需完整语料和混合索引。
- **C 组改进方向待定**，当前运行 A/B 基线。

## 原有 RAG 网页和启动器

Windows 上双击：

| 文件 | 打开什么 |
|---|---|
| `start_qrecc.bat` | QReCC 英语知识库，自动检测本地 Qwen 和向量模型 |
| `start_qrecc_demo.bat` | QReCC 英语知识库，BM25 离线证据摘录 |
| `run_tests.bat` | 运行软件测试 |

页面地址为 **http://127.0.0.1:8001/**。两个网页启动文件请分别运行，或停止前一个后再切换；它们使用相同端口。启动器会优先查找项目虚拟环境和本机 `G:\Anaconda3\python.exe`，再查找常见 Python 安装。不需要 pip 安装运行依赖，要求 Python 3.10+。

默认生成模型为 **`qwen3:1.7b`**，混合检索向量模型为 **`bge-m3`**，服务地址是 `http://127.0.0.1:11434`。首次使用可分别运行 `ollama pull qwen3:1.7b` 和 `ollama pull bge-m3`。自动模式在模型不可用时显示 BM25 离线演示。**正式实验请显式用 `--mode ollama`**。

如终端中 `python` 不可用，可使用完整路径：

```powershell
& 'G:\Anaconda3\python.exe' -X utf8 -B app.py --dataset qrecc --mode ollama --port 8001
```

## 已实现的能力

- 内置 QReCC JSONL 语料。检索器也支持显式配置的 Markdown/纯文本语料。
- QReCC 使用完整 passage，保留原始 `passage_id`，不再次按 500 字符切分。
- 真正的 BM25 倒排索引，以及可选的原始逐块扫描路径；两路保持相同公式和分词，可做结果一致性验证。
- 正式实验默认 `hybrid`，组合 BM25 和 `bge-m3` 的排序。`bm25` 和 `dense` 可单独用于诊断。
- 四种 Query 策略：`current / history / rewrite / structured`。
- 固定角色历史与统一历史窗口、结构化输出校验、最多一次修复、明确失败/回退记录。
- 检索、上下文与生成分开计时；生成失败仍保留成功检索的指标。
- 实验一和二评估段落检索质量。主实验还评估改写质量并导出人工盲评材料。
- 逐题 JSONL 检查点、配置/源码/数据签名、严格匹配后续跑，以及人工盲评模板。

这是工程基线与可验证的启发式改进，不是原始 RAG 论文的联合训练复现，也没有进行重写模型训练。结构化检查和引用编号检查都不能保证语义正确。

## 配置怎么选

`config.json` 保留通用模型配置。`configs/qrecc.json` 设置语料、索引、评测文件、回答语言和默认预算。命令行显式参数最后覆盖。

| 项目 | QReCC 配置 |
|---|---|
| 检索材料 | `data/qrecc/passages.jsonl` |
| 切分 | 已有 passage 直接作为检索单元 |
| 混合索引缓存 | `.cache/qrecc/index.json` |
| 离线 BM25 缓存 | `.cache/qrecc/index-bm25.json` |
| 评测标签 | 一个或多个 `positive_ids` |
| 默认 Top-K | 5 |
| 历史 | 最近最多 3 个用户轮次 |
| 回答 | 英文 |

历史按真实用户消息定位窗口；缺失助手回答时不会把后面的角色交替猜错。每条历史内容最多 4000 字符；所有策略使用相同的可见历史。`history_turns` 约束的是已有历史，当前问题另行加入。

索引有语料/格式/切分/模型等指纹。更换语料或索引相关配置后会重建；切换 Query 策略不需要重建。QReCC 使用原始 passage 边界。字符分块参数用于 Markdown 和纯文本语料。实验缓存另有严格签名，真实模型模式还会读取 Ollama 当前权重 digest；修改权重、代码或提示词之后请用新的输出目录。若自行替换同名 embedding 权重，还应强制重建向量索引。

## 批量检索实验

先确认流程：

```powershell
python evaluate.py --dataset qrecc --mode demo --backend bm25 --strategies current history rewrite --split dev --top-k 3 --limit 10
```

正式模型对照：

```powershell
python evaluate.py --dataset qrecc --mode ollama --model qwen3:1.7b --backend hybrid --strategies current history rewrite --split all --top-k 3 --retrieval-only
```

实验二只运行 `rewrite`，用 `--model` 指定模型，每个模型单独保存运行结果。当前本地检索范围为 307 题和 9,975 段语料。完整官方语料仍需接入大规模混合索引。 [查看完整实验指南](docs/实验指南.md)

生成质量可选加 `--generate`；默认仅检索，但模型重写策略仍会调用 Qwen。回答提示使用相同当前问题、历史与证据格式，不额外加入重写 Query。参考答案与人工 Rewrite 不进入检索/回答调用。

输出自动放入新的 `outputs/runs/` 子目录，包含 `manifest.json`、`checkpoint.jsonl`、`report.json`、`summary.csv`、`details.csv`。如果运行中断：

```powershell
python evaluate.py --dataset qrecc --mode ollama --backend hybrid --strategies current history rewrite --split dev --top-k 3 --output-dir outputs/my_run
# 上一行首次创建目录；中断后以完全相同参数加 --resume
python evaluate.py --dataset qrecc --mode ollama --backend hybrid --strategies current history rewrite --split dev --top-k 3 --output-dir outputs/my_run --resume
```

已记录的失败题也视为完成，续跑不会悄悄重试或改变分母。配置/数据/源码不匹配会拒绝恢复；已经写完最终 report 的实验不能再次覆盖输出。

### 检索指标

QReCC 根据原始 passage ID 计分：

- `hit_at_k`：前 K 个结果至少命中一个标注正例。
- `gold_recall_at_k`：命中的不同正例数 / 该题库内标注正例总数；QReCC 的 `recall_at_k` 同此字段。
- `mrr_at_k`：首个正例的倒数排名，未命中为 0。
- `ndcg_at_k`：二元相关性与多个正例的排序质量。

名次按原始检索排名计算；重复正例只贡献一次，不在事后去重重新排名。正例标签也不能保证回答正确或包含参考答案的完全相同字符串。生成的 ROUGE/关键词覆盖都是辅助词面指标，不是准确率或幻觉率。

## 数据、来源与研究限制

QReCC 小语料来自同学提供的 v2 固定前 10000 passage；预处理明确排除 25 条空正文，保留 9975 条有效 passage。原始文件、选题清单及原 manifest 放在 `data/qrecc/source/`，来源说明见 `SOURCE_LICENSE.md` 与新 `manifest.json`。

角色历史由官方原始测试会话的**此前人工 Rewrite 与非空参考 Answer**重建，和每题 Context 核对。它是固定参考历史评测，不是系统自己生成前轮回答的滚动对话。当前答案、人工 Rewrite 和正例仅由评测层持有；先前参考回答作为历史属于明确报告的评测条件。

先固定语料，再只评测正例在该子集内的问题，仍有覆盖筛选偏差。开发/测试在原官方 test 内按会话再划分，属于项目内部划分，不能冒充官方训练/测试协议或全库成绩。本版本改变了历史处理、空段过滤、BM25 参数/分词与生成模型，必须重新报告结果，不能套用同学旧报告数字。

重新生成数据，需指定一个新的输出目录，防止覆盖冻结材料：

```powershell
python prepare_qrecc.py --passages data/qrecc/source/passages.jsonl --qrecc-test data/qrecc/source/qrecc-test.json --selected data/qrecc/source/selected_v2.jsonl --out data/qrecc_rebuilt --context-policy exclude
```

`exclude` 仅剔除无法精确核对历史的样本，并在 manifest 列明原因；不能猜测角色。实际样本数量与划分以新 manifest 为准。数据由[QReCC 官方归档](https://zenodo.org/records/5760304)和[v2 所用 passage 公开镜像](https://huggingface.co/datasets/slupart/qrecc-passages/tree/main/data)提供，保留作者归属与来源 URL。

## 从哪里读原有 RAG 代码

1. `rag/retrieval.py`：你已学习的模块；新增 JSONL 完整 passage 和倒排实现。
2. `rag/data.py` 与 `prepare_qrecc.py`：知识材料、问题、历史和评测标签如何分开。
3. `rag/rewriters.py`：独立研究的核心，查看各策略和修复/回退记录。
4. `rag/pipeline.py`：把 Query、检索、参考资料和回答连接起来。
5. `rag/metrics.py`、`evaluate.py`、`evaluate_rewrite.py`：检索与重写质量怎样分别计分。
6. `rag/run_cache.py`：索引缓存之外，实验为什么需要独立签名和检查点。

早期验证产物在 `outputs/validation/`。当前实验以本次运行目录的模型标识、数据清单和逐题报告为准。
