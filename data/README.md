# 数据说明

## 数据目录

`qrecc/source/` 保存原始数据和来源记录
`qrecc/` 外层保存处理后的语料和题目

| 文件 | 内容 |
| --- | --- |
| `qrecc/source/official_download.json` | 1.记录官方测试集的下载地址、文件大小、记录数和校验值 |
| `qrecc/source/passages.jsonl` | 2.保存前 10,000 个段落（与307道题有关的语料），每行包含原始段落 ID 和正文 |
| `qrecc/source/qrecc-test.json` | 1.保存官方测试集的 16,451 条对话记录，包含历史、问题、人工重写、参考答案和相关段落标注 |
| `qrecc/source/selected_v2.jsonl` | 2.保存筛出的 307 道题及其原始划分，对话历史以文本列表存储 |
| `qrecc/source/v2_manifest.json` | 2.记录语料的来源、选题规则、数据数量和文件校验值 |

| `qrecc/passages.jsonl` | 2.保存清理空正文后留下的 9,975 个有效段落（从10000筛选空正文），供 RAG 建索引和检索 |
| `qrecc/questions.jsonl` | 2.保存当前 A/B 实验使用的 307 道题，包含带角色的历史、当前问题、人工重写、参考答案和相关段落 ID |
| `qrecc/manifest.json` | 1.记录当前数据的处理规则、样本数量、开发集与测试集划分，以及文件校验值 |
| `qrecc/DATA_SOURCES.md` | 2.说明语料从哪里来、307 道题怎样筛出，以及对话历史怎样重建 |
| `qrecc/SOURCE_LICENSE.md` | 记录数据来源、作者归属与使用许可 |
| `README.md` | 本文件，介绍数据目录和评测入口 |

## 建议实验

13,676 题是官方测试集中有历史的追问总量，可用于后续扩大评测规模。当前使用项目小数据集的 307 条记录，其中 102 条为开发集 dev，205 条为测试集 test。

先在开发集选择 Top-K、历史窗口等参数，固定参数后再运行测试集。报告同时说明模型名称、模型版本、硬件、检索后端、参数、数据划分和生成设置。不要在看过测试结果后反复调参并把同一测试集当作独立验证。

```powershell
python evaluate.py --split dev --mode demo
# 开发集离线检索。默认比较 current、`history`、`rewrite` 三种 Query，`rewrite` 在这里使用规则重写
python evaluate.py --split test --mode ollama --strategies current history rewrite
# history rewrite	用测试集比较三种 Query 的检索表现。`current` 直接用当前问题，`history` 拼接历史和当前问题，`rewrite` 调用 Qwen 重写
python evaluate.py --split test --mode ollama --generate
# 默认仍比较上面三种 Query。区别是检索后还会把证据交给 Qwen 生成最终答案，并用参考答案计算答案的 ROUGE-1 F1
python evaluate.py --split dev --mode demo --top-k 1
# 开发集做离线检索，把默认的 Top-K=5 改成 1，也就是只看排第 1 的证据，和第1条对照，可以观察多少题的正确证据其实排在第 2–5 名
```

## 指标定义

`Top-K` 指检索返回的前 K 个 passage
QReCC 按原始 passage ID 匹配相关性标签。
`Hit@K` 看是否命中至少一个正例，`Gold Recall@K` 衡量找回了多少比例的标注正例
`MRR@K` 取首次命中的原始排名倒数，未命中记 0
`nDCG@K` 对靠前的正例给予更大权重

生成模式用 `answer_rouge1_f1` 记录与参考答案的词面重合
参考答案只用于评分。正式报告还应人工检查回答的事实正确性、来源支持、遗漏和无依据扩展

`latencies_ms` 保存重写、检索、上下文、生成及总耗时。总耗时不含引擎初始化和索引构建，首次模型加载、缓存、后端差异会影响结果。CLI另外记录初始化耗时。小型串行实验不能代表吞吐性能；评估效率时应固定运行条件、区分冷启动和热运行，并重复运行。BM25 不需要向量模型；dense/hybrid 实验还需记录嵌入模型与模型调用开销
