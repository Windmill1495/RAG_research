# 冻结数据来源

- 初始 10000 个 passage：保留原始 ID、原 manifest 和来源说明。源文件 SHA256 为 `95afa05248f2aa7f644aa940df87e7604b411bd6cc7f6dd121080f26910db9f6`。本次没有重新下载完整 passage 分片。
- 实际检索库：准备程序显式排除 25 条空正文，剩余 9975 个完整 passage；空段不是选中问题的正例。处理后哈希见 manifest。
- 官方原始 QReCC test：[Zenodo 5760304](https://zenodo.org/records/5760304)。本次下载完整 19,546,617 字节、16,451 条原始记录；MD5 `7ad29296a1dfd2f75dafd653a8822269` 与官方公布值一致。下载记录见 `source/official_download.json`。
- 307 个选中问题：保留原来的 ID 与会话划分 dev102/test205，但历史从完整官方原会话重建为明确角色。全部历史展平后与官方 Context 逐字相同，无样本因历史校验被排除。
- `source/selected_v2.jsonl` 的扁平历史只作原始材料存档，不直接用于新系统模型输入

这是固定语料前缀、有正例覆盖条件的子集。该内部 dev/test 不等于官方完整基准切分；已看过的 test 案例也不能继续用于方法设计后再声称未见验证。
