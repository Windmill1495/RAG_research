# 完整 QReCC 数据目录

`raw/qrecc-training.json` 和 `raw/qrecc-test.json` 为官方原始对话文件；`questions.jsonl` 保留两个文件的全部轮次，`manifest.json` 记录来源哈希、数量与划分。测试集保持官方划分；开发集只从训练对话中按会话哈希划分，其他训练数据仍保留。`--split all` 包括 train/dev/test，不能把它的汇总称为独立测试成绩。

默认种子和 10% 开发会话比例实际生成 79,952 题：训练 57,163、开发 6,338、测试 16,451。官方训练文件原有 63,501 轮，测试文件原有 16,451 轮。官方训练文件没有 `Passages` 标签，因此训练/开发集可计算重写指标，检索金标指标不可凭空计算；测试集中 8,282 题带有该标签，检索报告应说明有效标注分母及缺失标签数量。

题目不再要求有相关段落、非空答案或为追问；原始正例 ID 不与本地语料取交集。没有标注的题仍可评价重写，不能当作已知负例计算检索指标。历史正文逐字保留当前行的官方 `Context`；程序根据此前各轮的 Rewrite 与非空 Answer 确定角色和轮次，用户消息必须与此前 Rewrite 一致。少量 Context 中的旧回答措辞不同，保留当前 Context 的实际文字并在清单报告，不能用当前轮 Rewrite 或 Answer 补入历史。

官方部分会话的 `Turn_no` 与 Context 顺序不同，因此按 Context 长度恢复先后关系；`turn` 表示恢复后的顺序，`official_turn` 与 `official_id` 保留原标签，差异记录于清单。不能匹配的历史角色或用户锚点会报错，不会默默丢题或猜测。训练/测试 ID 使用不同命名空间，避免原始会话编号碰撞。

完整问题文件与完整检索语料是两件事。所有问题都存在，不代表已完成约5400万段的全库索引。完整语料必须在实验室服务器单独准备，并验证索引完成与覆盖情况。

## 准备命令

在项目根目录执行，Python 命令适用于 Linux 与 Windows：

```bash
python prepare_qrecc_full.py --download-dialogs
```

这只下载并校验官方训练/测试对话，保存在 `raw/`。大型语料仅在显式指定时下载，保存在数据目录根部的 `passages.zip`：

```bash
python prepare_qrecc_full.py --download-passages
```

也可传入已经下载的官方文件及外部数据目录：

```bash
python prepare_qrecc_full.py --training /data/qrecc/qrecc-training.json --test /data/qrecc/qrecc-test.json --data-dir /data/qrecc_full
```

默认下载地址是 [官方 Zenodo 5760304](https://zenodo.org/records/5760304)。训练、测试及 passage ZIP 均核对该记录发布的 MD5。已有文件不匹配时不会覆盖；准备参数与输出哈希完全相同时可复用，参数变化则要求新目录。

官方 `passages.zip` 压缩文件约27.5GB。`rag/full_data.py` 可以直接逐条读取 ZIP 内的 JSONL 分片，不需要先解压整个语料，也不会将全部段落组成 Python 列表。原始 `{id, contents}` 被适配为 `{id, text}`，保留原 ID 和全文。空正文被明确计数并跳过；题目标注不会因此被删改。还支持 JSON 数组、独立 JSONL、gzip 分片和分片目录。官方格式参考 [Apple collection 源码](https://github.com/apple-aiml-research/ml-qrecc/blob/main/collection/paragraph_chunker.py)。

请勿把完整语料、索引或模型权重推入 GitHub。项目源码和配置可迁移；服务器上将大文件保存在单独的数据磁盘。
