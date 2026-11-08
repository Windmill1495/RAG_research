# RAG Research · QReCC 对话问题重写与混合检索

项目用 QReCC 子集演示多轮 RAG 问答，并比较不同的查询构造和重写方法。网页可以试问答、查看检索证据和逐题对比 A/B/C 重写与检索，命令行负责批量评测和导出结果。正式评测默认使用 **BM25 + `bge-m3` 混合检索**，以 RRF 融合词项和向量排名。当前数据包含 **9,975 个英文段落和 307 道题**，主实验已接入 A 直接重写、B 结构化重写和 C CompCQR 组合查询。

## 快速启动

需要 Python 3.10 或以上版本。项目使用 Python 标准库，准备好 Python 即可运行离线演示。真实模型模式另需 Ollama 和对应模型。

| 入口 | 用途 |
| --- | --- |
| `start_qrecc.bat` | Windows 上双击启动 RAG 网页，自动检测 Ollama 和模型 |
| `run_tests.bat` | 双击运行程序测试 |
| `app.py` | 在命令行选择运行模式、端口或初始页面 |
| `chat.py` | 在终端进行多轮问答 |

`start_qrecc.bat` 使用 `launch.ps1` 查找 Python，尝试启动本地 Ollama，再以 `auto` 模式打开 [RAG 首页](http://127.0.0.1:8001/)。生成模型和向量模型均可用时，首页使用 Ollama 和混合检索，否则进入 BM25 离线演示。离线模式用规则构造查询，回答展示检索到的证据摘录。

在项目根目录打开 PowerShell。下文 `$py` 代表当前电脑的 Python 路径，换电脑时按实际安装位置修改。Python 已加入 PATH 时可设置 `$py = 'python'`。

```powershell
$py = 'G:\Anaconda3\python.exe'

# 强制使用离线模式
& $py -X utf8 -B app.py --mode demo --port 8001
```

运行真实模型前，在 Ollama 启动后用 `ollama list` 检查已安装的模型。需要时安装默认问答模型和向量模型。

```powershell
ollama pull qwen3:1.7b
ollama pull bge-m3
```

### 网页与索引状态

- **RAG 首页 `/`** 支持英文多轮问答，可查看检索 Query、来源段落和各阶段耗时。
- **A/B/C 重写页 `/rewrite`** 可从 307 道题中选题，比较三组输出、人工参考、ROUGE-1 F1、修复记录与适用的检索指标。已有服务可直接打开 [A/B/C 重写页](http://127.0.0.1:8001/rewrite)。
- **首次建立混合索引**时，首页先打开，程序在后台处理语料。页面显示「准备索引中」，终端打印 `Embeddings: 已完成数量/总数量`，索引就绪后页面自动更新，请保持启动窗口打开。

```powershell
& $py -X utf8 -B app.py --mode ollama --page rewrite --rewrite-model qwen3:1.7b --port 8001
```

## 查询策略与配置

| 策略 | Query 怎样构造 |
| --- | --- |
| `current` | 仅使用当前问题 |
| `history` | 拼接窗口内的历史问答和当前问题 |
| `rewrite`（主实验中的 A） | 让模型直接生成一条可以独立理解的问题，并检查输出 |
| `structured`（主实验中的 B） | 让模型输出主题、指代对象、独立问题和歧义标记，检查输出，必要时修复一次 |
| `compcqr`（主实验中的 C） | 生成查询与假设答案组件，组合多条查询检索，经 RRF 融合后由模型选择最佳段落 |

五种策略共享同一套历史处理。默认保留最近最多 **3 个用户轮次**，当前问题另行加入。完整问答把检索结果整理为编号证据，再结合原问题和历史生成回答。重写和检索各自记录耗时，模型重写失败时会记录回退来源。

配置按内置默认值、`config.json`、`configs/qrecc.json`、显式参数的顺序覆盖。下面是当前 QReCC 配置的有效值。

| 设置 | 默认值 |
| --- | --- |
| Ollama 服务 | `http://127.0.0.1:11434` |
| 生成与重写模型 | `qwen3:1.7b` |
| 检索方式 | `hybrid`，BM25 倒排 + `bge-m3` 向量排名 |
| 检索条数 | 问答默认 Top-5，主实验下游检索固定 Top-3 |
| 回答语言与证据预算 | 英文，最多 4,500 字符的证据上下文 |
| 混合索引 | `.cache/qrecc/index.json` |
| 离线索引 | `.cache/qrecc/index-bm25.json` |

QReCC 已经提供 passage 边界，索引保留完整正文和原始段落 ID。`chunk_size` 与 `overlap` 用于显式配置的 Markdown 或纯文本语料。Ollama 对超长向量输入按模型上下文窗口截断，原始段落正文仍用于 BM25 和证据整理。

索引缓存完整保存后，可在相同语料和索引配置下复用。缓存指纹包含语料、相关配置和向量模型权重摘要。切换查询策略或生成模型可继续用同一混合索引。需要手动重建时执行 `& $py -X utf8 -B rebuild_index.py --mode ollama --force`。

## 数据与评测范围

| 文件 | 内容 |
| --- | --- |
| `data/qrecc/passages.jsonl` | 固定的 10,000 条源语料去除 25 条空正文后，得到 9,975 个有效段落 |
| `data/qrecc/questions.jsonl` | 307 道有本地语料覆盖的追问，涉及 141 段对话，分为 102 道开发题与 205 道测试题 |
| `data/qrecc/source/qrecc-test.json` | 官方测试对话的 16,451 轮记录，其中 13,676 轮有历史 |
| `data/qrecc/manifest.json` | 固定语料和题目数量、划分、历史构造规则以及哈希 |

项目的 dev/test 是从官方测试对话中按会话划分的子集。历史由**此前轮次**的人工 Rewrite 和非空参考 Answer 重建，并与官方 Context 核对。**当前轮**的人工 Rewrite、参考答案和正例 ID 用于评测，模型输入使用当前问题和历史。

使用项目数据跑出的检索成绩对应 307 题和本地 9,975 个段落。官方测试 JSON 可以直接用于纯重写评测。完整下游检索还需要与官方标注匹配的完整语料索引。数据来源和许可见 [数据来源](data/qrecc/DATA_SOURCES.md)与[来源许可](data/qrecc/SOURCE_LICENSE.md)。

```powershell
& $py -X utf8 -B prepare_qrecc.py --passages data/qrecc/source/passages.jsonl --qrecc-test data/qrecc/source/qrecc-test.json --selected data/qrecc/source/selected_v2.jsonl --out data/qrecc_rebuilt --context-policy error
```

## 批量运行实验

三个实验的命令都在项目根目录运行。先在开发题上检查流程和确定参数，再用固定的题目、历史窗口、语料和指标进行对照。正式比较显式使用 `--mode ollama`，并记录实际模型标签及权重摘要。

### 实验一 · 比较 Query 构造

固定 `qwen3:1.7b`，比较 `current`、`history` 和 `rewrite`。每题取一次 Top-3，由同一个排名计算 @1 和 @3。

```powershell
& $py -X utf8 -B evaluate.py --mode demo --backend bm25 --strategies current history rewrite --split dev --top-k 3 --limit 2
& $py -X utf8 -B evaluate.py --mode ollama --model qwen3:1.7b --backend hybrid --strategies current history rewrite --split all --top-k 3 --retrieval-only
```

第一行是离线流程检查，第二行对项目 307 题运行正式模式。`evaluate.py` 默认评测检索。显式添加 `--generate` 可同时生成回答并保存辅助评分，此时省略 `--retrieval-only`。

### 实验二 · 比较 Qwen3 模型规模

固定 `rewrite`、语料、混合索引和评测设置，对比 1.7B、4B、8B、14B、32B 五个规模。每个已安装的模型单独运行一次。下面以 4B 为例，模型标签以 `ollama list` 的结果为准。

```powershell
& $py -X utf8 -B evaluate.py --mode ollama --model qwen3:4b-q4_K_M --backend hybrid --strategies rewrite --split all --top-k 3 --retrieval-only
```

### 主实验 · A/B/C 重写与检索方法对照

正式设置使用 **`qwen3:8b`**。运行前确认 Ollama 已安装 `qwen3:8b` 和 `bge-m3`，需要时执行 `ollama pull qwen3:8b`。A、B、C 三组收到同一道题和相同的可见历史。批量入口默认运行三组。C 的基础独立改写可与人工参考评分，完整 CompCQR 的效果由最终融合及选择后的检索排名评价。

```powershell
# 先检查两道题的重写评分与报告导出
& $py -X utf8 -B evaluate_rewrite.py --mode demo --split dev --limit 2 --rewrite-only

# 用已安装的 8B 模型检查两道开发题及其下游检索
& $py -X utf8 -B evaluate_rewrite.py --mode ollama --model qwen3:8b --split dev --limit 2

# 跑完项目数据，得到 307 题 × 3 组的逐题记录
& $py -X utf8 -B evaluate_rewrite.py --mode ollama --model qwen3:8b --split all
```

`--split dev` 或 `--split test` 可单独选择划分。批量入口默认在有标注且被语料覆盖的题上检索固定 Top-3。只比较重写质量时加 `--rewrite-only`，也可读取本地的官方测试对话。对 C 而言，该选项只生成组件并评价基础改写，不执行多查询检索和最佳段落选择，不能作为完整 C 的成绩。

```powershell
& $py -X utf8 -B evaluate_rewrite.py --mode ollama --model qwen3:8b --data data/qrecc/source/qrecc-test.json --data-format qrecc-json --split test --rewrite-only
```

用小索引诊断官方测试对话的检索表现时，可显式使用 `--allow-partial-corpus`，并通过 `report.json` 的 `metadata.retrieval` 核对参与检索计分的题数与覆盖范围。更多对照设置见[实验指南](docs/实验指南.md)。

## 指标、报告与续跑

| 指标 | 怎样读 |
| --- | --- |
| Gold Recall@1、@3 | 前 1 或前 3 名命中的不同正例 ID 数，除以该题评测记录中的标注正例数，再按题平均 |
| MRR@3 | 前三名中首个正例的倒数名次，未命中计 0，再按题平均 |
| ROUGE-1 Precision、Recall、F1 | 改写与人工参考的词项重合程度 |
| 耗时、失败、回退 | 对照重写和检索各阶段耗时，结合错误、修复与回退记录解释结果 |
| 人工盲评 | 核查指代与省略、原意及约束、无依据新增、能否独立理解，允许等义措辞 |

检索排名按实际返回顺序计分。项目子集的分母使用其 `positive_ids`。重写评测与检索评测分别统计参与计分的题数，检索失败时仍保留可用的重写评分。重写硬失败按空输出记零，回退按实际输出评分并单列。

| 输出 | 内容 |
| --- | --- |
| `outputs/runs/` | 检索对照的默认输出位置 |
| `outputs/rewrite_runs/` | 重写与下游检索对照的默认输出位置 |
| `summary.csv` | 每个策略的汇总指标与计数 |
| `details.csv`、`report.json` | 每题的 Query、参考、检索结果、耗时及错误记录 |
| `manifest.json`、`checkpoint.jsonl` | 运行签名与已记录的逐题结果 |
| `blind_review_reviewer1.csv`、`blind_review_reviewer2.csv`、`blind_review_key.json` | 重写实验生成的两份盲评表和单独保管的组别映射 |

需要续跑时，首次运行显式设置 `--output-dir` 指向新目录。中断后使用同一目录、相同参数并加 `--resume`。程序校验数据、配置、源码及实际模型身份，跳过已记录的题目。换模型、换数据或重新评测时，用新的输出目录。A/B/C 提示与人工盲评细节见[问题重写实验说明](docs/问题重写实验说明.md)。

## 代码从哪里读

| 文件 | 职责 |
| --- | --- |
| `prepare_qrecc.py`、`rag/data.py`、`rag/rewrite_data.py` | 语料准备、题目校验、历史和评测标签适配 |
| `rag/rewriters.py` | 查询策略注册、A/B 提示词、输出校验、修复与回退 |
| `rag/compcqr.py` | C 的组件生成、查询组合、跨查询 RRF 与最佳段落选择 |
| `rag/retrieval.py` | 分词、索引、BM25 与向量检索、RRF 融合 |
| `rag/pipeline.py`、`rag/llm.py` | 串联历史、重写、检索和回答，封装 Ollama 请求 |
| `evaluate.py`、`evaluate_rewrite.py`、`rag/metrics.py` | 批量检索、重写对照与指标计算 |
| `rag/run_cache.py` | 运行签名、检查点与续跑 |
| `app.py`、`web/` | 本地服务、RAG 首页和逐题 A/B/C 页面 |
| `tests/` | 检查数据、查询策略、检索、评测、缓存和网页行为 |

运行测试可双击 `run_tests.bat`，或执行 `& $py -X utf8 -B -m unittest discover -s tests -v`。进一步阅读可参考[代码阅读顺序](docs/阅读顺序.md)。启动入口以本页的命令为准。

## C · CompCQR 接入说明

C 使用两次模型调用生成独立改写及假设答案、后续问题及假设答案，再与原问题组合。五个非空组件最多形成 325 条有序查询；BM25 使用 31 种无序子集。每条查询默认检索 Top-100，跨查询按 RRF（常数 0）和出现频次融合，再用第三次模型调用从最多 5 个候选段落中选择一篇提升到第一名。现有混合检索的 BM25/向量 RRF（常数 60）是内部的另一层融合。假设答案只扩展查询，回答证据来自实际检索的段落。

这是使用 Qwen、bge-m3 和本地子集的 CompCQR 适配实现，模型、检索器和选择预算与论文原始实验不同。首次真实运行建议 `--limit 2`，之后固定参数再扩大。组件不完整、模型选择无效和检索错误均记录，选择失败时保留融合排名。完整流程、参数与论文差异见 [CompCQR 说明](docs/CompCQR说明.md)。

A/B/C 的混合检索内部候选深度统一为 `hybrid_candidate_depth=400`，不随最终 Top-K 或 C 每查询检索深度改变；两路 RRF 仍使用常数 60。C 选择提示的 6,000 字符预算包含指令、问题、同样历史和段落，实际候选数以运行记录为准。
