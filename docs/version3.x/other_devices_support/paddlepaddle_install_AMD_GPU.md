---
comments: true
---

<!-- Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# AMD GPU (ROCm) 飞桨安装教程

本教程介绍如何在 AMD Instinct GPU 上通过 ROCm/HIP 运行 PaddleOCR 通用套件 —— PP-OCRv6（检测 / 识别 / 分类）、PP-StructureV3、PP-DocTranslation。

GPU 计算由飞桨的 ROCm 构建（`amd-paddlepaddle` wheel）承载。PaddleOCR 是基于 `paddle` / `paddlex` 的纯 Python 套件，无需修改源码：`paddle.set_device('gpu')`（或流水线配置中的 `gpu` 设备字符串）在 ROCm 上会路由到 HIP，与在 NVIDIA 上路由到 CUDA 完全一致。

> INFO:
> PaddleOCR-VL 旗舰模型有独立的 AMD GPU 路径（使用百度托管镜像），请参考 [PaddleOCR-VL AMD GPU 使用教程](../pipeline_usage/PaddleOCR-VL-AMD-GPU.md)。本文档覆盖通用（非 VL）套件流水线。

## 支持矩阵

| 项目 | 状态 |
| --- | --- |
| GPU（已验证） | AMD Instinct MI300X (gfx942) |
| GPU（主要目标，持续验证） | AMD Instinct MI350X / MI355X (gfx950) |
| ROCm | 10.1（manylinux2.28 容器；镜像 `10.1.0a20260821`） |
| Python | 3.11 |
| 飞桨 wheel | `amd-paddlepaddle`（ROCm 构建） |
| PP-OCRv6（检测/识别/分类） | 支持 |
| PP-StructureV3 | 支持 |
| PP-DocTranslation | 尽力支持（翻译 LLM 步骤可能需要额外模型服务） |
| 自定义算子 `RoIAlignRotated`（旋转/SAST 路径） | 通过 `paddle.utils.cpp_extension.load` 经 hipcc JIT 编译 |

> INFO:
> 由于硬件多样性，目前仅在 MI300X (gfx942) 上完成精度/速度验证；MI350X/MI355X (gfx950) 为主要目标并持续验证。欢迎社区在其它 AMD GPU 上测试并分享结果。

### 已知框架级限制

以下为 ROCm 飞桨引擎（而非 PaddleOCR 套件）的限制，随飞桨 ROCm 构建一并跟踪：

- BF16 卷积路径依赖 MIOpen 对具体配置的覆盖。
- ROCm 上的 FlashAttention 依赖引擎支持；需要它的模型可能回退到较慢的注意力路径。

如遇上述情况，请优先使用 FP32/FP16 执行或默认（非 flash）注意力路径。

## 1. 环境准备

有两种方法，**强烈建议使用 Docker 镜像以减少环境问题。**

### 1.1 方法一：Docker 镜像

从随附的 Dockerfile 构建通用套件 ROCm 镜像（需要 Docker >= 19.03）：

```bash
# 在仓库根目录执行
docker build \
  -f deploy/accelerators/amd-gpu/Dockerfile \
  --build-arg AMD_PADDLE_VERSION=3.4.0.dev20260825 \
  -t paddleocr-toolkit:amd-gpu \
  deploy/accelerators/amd-gpu
```

以 AMD GPU 设备透传和 `video` 组运行容器：

```bash
docker run -it \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 64g \
  -e HIP_VISIBLE_DEVICES=0 \
  paddleocr-toolkit:amd-gpu \
  /bin/bash
# 然后在容器内调用 PaddleOCR CLI 或 Python API。
```

### 1.2 方法二：手动安装

如无法使用 Docker，可在 ROCm 10.1 主机的全新虚拟环境中安装：

```bash
# 创建并激活虚拟环境
python -m venv .venv_paddleocr
source .venv_paddleocr/bin/activate

# 从 AMD 索引安装 ROCm 飞桨 wheel
python -m pip install amd-paddlepaddle==3.4.0.dev20260825 \
  --index-url https://pypi.amd.com/rocm-10.1/simple

# 安装 PaddleOCR，并屏蔽 CUDA 飞桨 wheel，仅解析 ROCm wheel
# （见 deploy/accelerators/amd-gpu/amd-constraints.txt）
python -m pip install "paddleocr[doc-parser]" \
  -c deploy/accelerators/amd-gpu/amd-constraints.txt
```

验证 ROCm 引擎导入并枚举 GPU：

```bash
python -c "import paddle; paddle.utils.run_check(); print(paddle.device.get_device())"
```

期望输出包含：

```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
gpu:0
```

## 2. 快速开始

在 GPU 上运行通用 OCR 流水线：

```python
from paddleocr import PaddleOCR

# 设备选择是框架透明的：ROCm 上 'gpu' 会路由到 HIP。
ocr = PaddleOCR(device="gpu")
result = ocr.predict("path/to/image.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

PP-StructureV3（版面 + 表格 + 公式）：

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(device="gpu")
result = pipeline.predict("path/to/document.pdf")
for res in result:
    res.save_to_json("output")
    res.save_to_markdown("output")
```

## 3. 一键服务部署（Docker Compose）

使用随附的 Compose 文件一键部署 OCR / Structure 服务：

```bash
cd deploy/accelerators/amd-gpu
# 在 .env 中选择流水线（PIPELINE=OCR | PP-StructureV3 | PP-DocTranslation）
docker compose build
docker compose up
```

服务默认监听 `8080` 端口。如需指定某张 GPU，请在 `.env` 中设置 `HIP_VISIBLE_DEVICES`。所有可调项见 `deploy/accelerators/amd-gpu/.env`。
