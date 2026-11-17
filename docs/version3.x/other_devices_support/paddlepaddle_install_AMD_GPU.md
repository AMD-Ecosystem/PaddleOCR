---
comments: true
---

<!-- Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# AMD GPU（ROCm）飞桨安装教程

本教程介绍如何通过 ROCm 在 AMD Instinct GPU 上运行 PaddleOCR。

PaddleOCR 是基于飞桨（PaddlePaddle）的 Python 工具包，在 AMD GPU 上运行无需修改源码。GPU 计算由 ROCm 版 PaddlePaddle 完成，指定 `device="gpu"` 即可在 AMD GPU 上执行。

> INFO:
> PaddleOCR-VL 模型有单独的 AMD GPU 使用方式，请参考 [PaddleOCR-VL AMD GPU 使用教程](../pipeline_usage/PaddleOCR-VL-AMD-GPU.md)。本文档介绍通用 OCR 产线。

## 已测试的配置

| 项目 | 取值 |
| --- | --- |
| GPU | AMD Instinct MI355X（`gfx950`） |
| ROCm | 10.1.0，通过公开的 ROCm Python 包安装 |
| 操作系统 | Ubuntu 24.04 |
| Python | 3.12 |
| PaddlePaddle | v3.4.0，从源码编译并启用 ROCm 支持 |
| 产线 | 通用 OCR（PP-OCRv6 文本检测与识别） |

其他 AMD Instinct GPU 以及其他 PaddleOCR 产线尚未按本流程运行验证。

## 1. 编译并安装支持 ROCm 的 PaddlePaddle

ROCm 版 PaddlePaddle 没有预编译的 wheel。请按照 [Building and Testing PaddlePaddle on AMD Instinct GPUs with ROCm](https://github.com/AMD-Ecosystem/Paddle/blob/rocm-enablement/VALIDATION_NOTES.md) 从源码编译，并安装编译得到的 wheel。

后续步骤请继续使用该指南中创建的 Python 虚拟环境和 ROCm 环境变量。PaddleOCR 会在首次使用时编译部分 GPU 内核，因此该指南中安装的 C/C++ 编译工具需要保留在系统中。

## 2. 安装 PaddleOCR

在同一个 Python 环境中，从本仓库根目录安装 PaddleOCR。

```bash
sudo apt-get install -y libgl1 libglib2.0-0 libgomp1

pip install -e ".[doc-parser]" \
  -c deploy/accelerators/amd-gpu/amd-constraints.txt "numpy<2"
```

该约束文件用于阻止 `pip` 安装 CUDA 版 PaddlePaddle 以及 CUDA 运行时 wheel，以免覆盖 ROCm 版本。

确认 PaddlePaddle 已启用 ROCm 支持并可以使用 GPU。

```bash
python -c "import paddle; print(paddle.device.is_compiled_with_rocm()); paddle.utils.run_check()"
```

该命令会输出 `True`，随后输出 PaddlePaddle 安装成功的提示。

## 3. 快速开始

下载示例图片，并通过命令行在 GPU 上运行通用 OCR 产线。

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png
paddleocr ocr -i general_ocr_002.png --device gpu
```

使用 Python 运行同一产线：

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(device="gpu")
result = ocr.predict("general_ocr_002.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

首次运行会下载模型文件。程序会打印识别出的文本，并将标注后的图片和 JSON 文件保存到 `output` 目录。

## 已知限制

以下是 ROCm 版 PaddlePaddle 的限制，与 PaddleOCR 本身无关。

- 不包含通过 MIOpen 实现的 BF16 卷积。
- ROCm 版本中未启用 FlashAttention。
