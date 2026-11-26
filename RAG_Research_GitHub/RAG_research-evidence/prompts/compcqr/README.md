# 运行时提示词资产

从项目根目录执行以下命令，将固定提交的 LLM4CS 渲染器与示例下载到本目录。下载内容经过 SHA-256 校验。

```bash
python -m rag.compcqr_prompts --fetch-llm4cs
```

再从原实验的提示目录复制 `followup.txt`。它是 CompCQR 论文附录 J.2 的文本，包含原实验的历史与问题占位符。若使用原版完整 CompCQR 推理，同样准备附录 J.3 对应的 `selection.txt`。

| 资产 | 期望 SHA-256 |
| --- | --- |
| `followup.txt` | `f4bd952c483a4fcf800e42c0923c85d7df355692ec4b9452279effcf710f43c6` |
| `selection.txt` | `3a7a3a51826520f3569a9b1977318fe3f996aaa60054d7fdf3ea41a2a99b8c1c` |

保持文件字节一致，以便通过校验并复现原提示。来源提交和示例指纹见 [SOURCES.json](SOURCES.json)。 [LLM4CS 固定版本](https://github.com/kyriemao/LLM4CS/tree/a292fb88e5b9898af3e5b000ca9a2c3d2c2ea99e) · [CompCQR 论文](https://aclanthology.org/2026.findings-emnlp.1225/)
