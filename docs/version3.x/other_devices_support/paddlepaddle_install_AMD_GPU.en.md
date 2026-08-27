---
comments: true
---

<!-- Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# AMD GPU (ROCm) PaddlePaddle Installation Tutorial

This tutorial covers running the PaddleOCR broader toolkit -- PP-OCRv6 (detection / recognition / classification), PP-StructureV3, and PP-DocTranslation -- on AMD Instinct GPUs via ROCm/HIP.

GPU compute rides the ROCm build of PaddlePaddle (the `amd-paddlepaddle` wheel). PaddleOCR is a pure-Python toolkit over `paddle` / `paddlex`, so no source change is required: `paddle.set_device('gpu')` (or a `gpu` device string in a pipeline config) routes to HIP on ROCm exactly as it routes to CUDA on NVIDIA.

> INFO:
> The PaddleOCR-VL flagship model has its own AMD GPU path with Baidu-hosted images. See the [PaddleOCR-VL AMD GPU Usage Tutorial](../pipeline_usage/PaddleOCR-VL-AMD-GPU.en.md). This document covers the broader (non-VL) toolkit pipelines.

## Support Matrix

| Item | Status |
| --- | --- |
| GPU (validated) | AMD Instinct MI300X (gfx942) |
| GPU (primary, driven) | AMD Instinct MI350X / MI355X (gfx950) |
| ROCm | 10.1 (manylinux2.28 container; image `10.1.0a20260821`) |
| Python | 3.11 |
| PaddlePaddle wheel | `amd-paddlepaddle` (ROCm build) |
| PP-OCRv6 (det/rec/cls) | Supported |
| PP-StructureV3 | Supported |
| PP-DocTranslation | Best-effort (translation LLM step may need an additional model server) |
| Custom op `RoIAlignRotated` (rotated/SAST path) | JIT-compiled via `paddle.utils.cpp_extension.load` through hipcc |

> INFO:
> Due to hardware diversity, only MI300X (gfx942) is currently accuracy/speed-verified; MI350X/MI355X (gfx950) is the primary target and is continuously driven. We welcome the community to test on other AMD GPUs and share results.

### Known framework-level limitations

These are gaps in the ROCm PaddlePaddle engine (not in the PaddleOCR toolkit) and are tracked with the PaddlePaddle ROCm build:

- BF16 convolution paths depend on MIOpen coverage for the specific configuration.
- FlashAttention on ROCm is engine-dependent; models that require it may fall back to a slower attention path.

If a pipeline hits one of these, prefer FP32/FP16 execution or the default (non-flash) attention path.

## 1. Environment Preparation

There are two methods. **We strongly recommend the Docker image to minimize environment issues.**

### 1.1 Method 1: Docker Image

Build the broader-toolkit ROCm image from the shipped Dockerfile (requires Docker >= 19.03):

```bash
# From the repository root
docker build \
  -f deploy/accelerators/amd-gpu/Dockerfile \
  --build-arg AMD_PADDLE_VERSION=3.4.0.dev20260825 \
  -t paddleocr-toolkit:amd-gpu \
  deploy/accelerators/amd-gpu
```

Run the container with AMD GPU device passthrough and the `video` group:

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
# Then call the PaddleOCR CLI or Python API inside the container.
```

### 1.2 Method 2: Manual Install

If you cannot use Docker, install into a fresh virtual environment on a ROCm 10.1 host:

```bash
# Create and activate a virtual environment
python -m venv .venv_paddleocr
source .venv_paddleocr/bin/activate

# Install the ROCm PaddlePaddle wheel from the AMD index
python -m pip install amd-paddlepaddle==3.4.0.dev20260825 \
  --index-url https://pypi.amd.com/rocm-10.1/simple

# Install PaddleOCR, blocking the CUDA PaddlePaddle wheel so only the ROCm
# wheel resolves (see deploy/accelerators/amd-gpu/amd-constraints.txt)
python -m pip install "paddleocr[doc-parser]" \
  -c deploy/accelerators/amd-gpu/amd-constraints.txt
```

Verify the ROCm engine imports and enumerates the GPU:

```bash
python -c "import paddle; paddle.utils.run_check(); print(paddle.device.get_device())"
```

Expected output includes:

```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
gpu:0
```

## 2. Quick Start

Run a general OCR pipeline on the GPU:

```python
from paddleocr import PaddleOCR

# Device selection is framework-transparent: 'gpu' routes to HIP on ROCm.
ocr = PaddleOCR(device="gpu")
result = ocr.predict("path/to/image.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

PP-StructureV3 (layout + tables + formulas):

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(device="gpu")
result = pipeline.predict("path/to/document.pdf")
for res in result:
    res.save_to_json("output")
    res.save_to_markdown("output")
```

## 3. One-Click Service Deployment (Docker Compose)

Use the shipped Compose file for a one-click OCR / Structure service:

```bash
cd deploy/accelerators/amd-gpu
# Select the pipeline in .env (PIPELINE=OCR | PP-StructureV3 | PP-DocTranslation)
docker compose build
docker compose up
```

The service listens on port `8080` by default. To pick a specific GPU, set `HIP_VISIBLE_DEVICES` in `.env`. See `deploy/accelerators/amd-gpu/.env` for all knobs.
