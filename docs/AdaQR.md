# AdaQR 接入说明

论文：[Zhang et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.746.pdf)，第 2 节和附录 Algorithm 1。

## 算法与项目接口

1. **SFT**：只从 `train` 划分、非首轮且有标注重写的样本抽取 10%，用回答 token 的负对数似然训练重写器。默认 Mistral-7B-v0.1，4-bit NF4 QLoRA，`q_proj/v_proj`、rank 8、alpha 16、dropout 0.05，AdamW 学习率 `1e-4`，10% warmup 后线性衰减。
2. **自采样**：SFT 模型每题以温度 1 生成 3 个候选，保存中间 JSONL。
3. **弱监督奖励**：每个候选调用项目的 `Retriever.search` 检索 Top-5。预训练答案模型以**原始历史、原始问题、单个段落**为条件，计算完整参考答案的 token 对数概率。对该候选的检索分数做 softmax，按论文式 (3) 对段落的对数概率加权求和。候选改写不会进入答案模型提示。
4. **偏好构造与 DPO**：枚举 3 个候选的所有不同文本配对，保留奖励差严格大于 `δ=0.1` 的配对。以 SFT adapter 为冻结参考模型和策略初值，优化论文式 (4)，`β=0.1`、AdamW `1e-5`、10% warmup 后线性衰减。
5. **推理与评估**：`adaqr` 是单独的策略；载入 DPO adapter 贪心生成一个重写，后续复用原项目检索器、回答器、ROUGE 和检索指标。首轮直接使用原问题。`current/history/rewrite/structured` 的原有实现与默认选择不变。

## 运行

训练和奖励模型需要 NVIDIA CUDA GPU。**不要在本机 Mac Air 运行以下训练、采样或奖励命令**。在具备足够显存的 CUDA 主机、对应项目环境中安装 `torch transformers peft bitsandbytes accelerate`，并准备完整语料/索引和官方 QReCC 训练文件。下列命令只是运行示例，本次没有执行。

```bash
python adaqr_train.py sft --seed-data /data/qrecc-training.json --output /runs/adaqr/sft
python adaqr_train.py sample --target-data /data/qrecc-training.json --sft-adapter /runs/adaqr/sft --output /runs/adaqr/samples.jsonl
python adaqr_train.py reward --samples /runs/adaqr/samples.jsonl --project . --backend bm25 --output /runs/adaqr/preferences.jsonl
python adaqr_train.py dpo --preferences /runs/adaqr/preferences.jsonl --sft-adapter /runs/adaqr/sft --output /runs/adaqr/dpo
```

`--seed-data` 也可传仅含 `split=train` 的 JSONL：`id, turn, history, question, rewrite, answer`。如需尽量复现论文，请将论文所用 GPT 少量重写标签放入这个文件；直接使用官方 `Rewrite` 字段会改变 SFT 监督来源。`--target-data` 必须是训练划分，不能用评测集；参考答案用于训练奖励，正例 passage ID 不参与训练。四个阶段分别保存文件，便于中断后从上一阶段继续。输出路径必须是新路径，防止覆盖旧实验。

训练后在项目环境（也需安装 `torch transformers peft accelerate` 并有可承载 7B 模型的设备）运行：

```bash
python evaluate_rewrite.py --mode ollama --strategies rewrite adaqr --adaqr-model-path /runs/adaqr/dpo --split dev --limit 10 --rewrite-only
python evaluate.py --mode ollama --strategies current rewrite adaqr --adaqr-model-path /runs/adaqr/dpo --split dev --retrieval-only
python chat.py --mode ollama --strategy adaqr --adaqr-model-path /runs/adaqr/dpo
```

也可在 `config.json` 中配置 `"adaqr_model_path": "/runs/adaqr/dpo"`，由 `RAGEngine.rewrite_query(..., strategy="adaqr")` 和 `retrieve/answer` 调用。演示模式无训练模型，会显式报错，不把启发式重写伪装成 AdaQR。

## 与论文实验的差异和限制

- 核心目标、候选数、温度、Top-K、奖励公式、阈值、DPO 参考模型与 QLoRA 参数按论文实现。论文称奖励为“边缘概率”，但正文式 (3) 实际写的是**对数概率的检索权重期望**；代码严格采用印出的公式。
- 论文使用约 3,850 条 GPT 生成的 QReCC seed rewrite、完整约 5,000 万段语料，并分别报告 BM25 和 ANCE。当前仓库仅附带 9,975 段与内部开发/测试题；不含论文 seed 标签、完整语料或 ANCE 索引。若用仓库自带数据或 BM25 实现，不能与论文指标直接比较。这里复用项目 BM25/dense/hybrid；`hybrid` 为项目 RRF，也不是论文的检索器。
- 论文没有给出答案模型的完整提示模板或全部生成长度设置。这里明确采用 `Context/Question/Passage/Answer` 格式，最长 2048 token，改写最多 128 token；输入过长时只裁剪旧提示部分，保留完整答案。可按获得的作者配置调整。
- 项目 RAG 使用最近 3 轮历史；AdaQR 走同一有界历史接口，因此长对话与论文可能不同。训练数据读取保留官方历史。当前项目没有 TopiOCQA、Doc2Dial、MultiDoc2Dial 数据接口或名为 CompCQR 的代码，未声称复现跨域数据集实验。
- 本次仅做无需下载模型的轻量算法和兼容性测试，**没有训练 adapter 或报告 AdaQR 效果数字**。
