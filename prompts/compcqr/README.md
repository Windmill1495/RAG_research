# CompCQR 原始提示词

三次调用使用原论文和其明确引用的原始模板，未再添加80词限制或简短选择理由要求。

1. 重写与假设答案使用 LLM4CS 的 `RewriteAndResponsePromptor(enable_cot=True)`。渲染器类从固定提交的 `chat_promptor.py` 原样提取，包含完整指令、尾部指令及少样本示例。输出按原始 `Rewrite: 理由. So the question should be rewritten as: 问题` 和下一行 `Response: 答案` 解析，不再要求自定义 JSON。
2. `followup.txt` 转录自 CompCQR 附录 J.2，保留输出字段和空追问规则，运行时只填入历史和当前问题。兼容原模板未包围大括号的两项输出。
3. `selection.txt` 转录自附录 J.3，保留问题分析、逐篇段落分析、选择理由和编号字段。运行时只绑定候选数量、历史、当前问题及段落。原来的自定义系统提示已移除。

正文 PDF 的排版换行被整理为可读文本，模板变量替换成实际输入。源码及示例来源、固定提交与哈希见 `SOURCES.json`。J.2/J.3 的正文以用户提供的 CompCQR.pdf 为准；官方代码中的模板存在少量排版及格式差异，本项目未把未展开的模板占位符当成实际查询发送。

来源归属：CompCQR 作者 Yunah Jang、Kang-il Lee、Joongbo Shin、Kyomin Jung；LLM4CS 作者及项目见 https://github.com/kyriemao/LLM4CS 。示例按原文件保留，当前固定版本实际含3段示例对话。此处仅作来源归属，不为上游未声明的材料另行声称许可。

缺少或损坏示例文件时，不允许静默改成零样本提示。提示词资产也纳入实验签名，修改后不能混用旧检查点。

本次只对齐提示词及对应解析。模型、hybrid 检索、历史窗口、候选数5、正文700字符和6000字符选择预算仍保持项目配置，因此不宣称已复刻论文全部实验条件。详细逐篇分析的输出可能比旧提示更长；需要时在 config.json 调整 llm_num_predict，不能把截断输出当作成功。
