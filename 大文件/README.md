# 续训状态与历史权重

本地归档中的 `大文件/training-states.tar.gz` 与 [GitHub Release 附件](https://github.com/Xyu-Chern/EQL/releases/tag/rl-archive-2026-11-18-v2) 内容相同。压缩包包含 48 个文件，覆盖 v5 与 v3 续训状态、CGPO 的 PyTorch 检查点和历史权重。常规实验记录与最终模型保存在分支的 `results/` 中。

在仓库根目录运行下列命令，可核验压缩包并解包到单独目录。解包后的路径与训练代码使用的目录结构一致。

```bash
(cd 大文件 && sha256sum -c large.sha256)
mkdir -p restored
tar -xzf 大文件/training-states.tar.gz -C restored
(cd restored && sha256sum -c ../大文件/training-states-files.sha256)
```

[压缩包校验值](large.sha256) · [按路径列出的 48 个 SHA-256](training-states-files.sha256)
