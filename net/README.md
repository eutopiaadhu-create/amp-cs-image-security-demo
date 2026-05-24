# AMP-Net 模块

该目录保留 AMP-Net 主干网络、采样矩阵读取、训练/测试入口和数据处理工具。桌面端 demo 的解密流程会通过 `core/decryptor.py` 调用 `train_AMP_Net.py` 中的 `AMPNetBasic`、`BLOCK_SIZE`、`DEFAULT_CS_RATIO` 和 `DEFAULT_PHASES`。

公开仓库不包含模型权重和采样矩阵。完整本地 demo 需要准备：

```text
models/best_model (1).pkl
dataset/sampling_matrix/60.mat
```

其他带 `_B`、`_M`、`_BM` 后缀的研究变体脚本保留在本地，不进入答辩展示版仓库。
