---
comments: true
---

<!-- Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# AMD GPU (ROCm) PaddlePaddle Installation Tutorial

This tutorial covers running PaddleOCR on AMD Instinct GPUs through ROCm.

PaddleOCR is a Python toolkit on top of PaddlePaddle, so it needs no source changes for AMD GPUs. GPU compute goes through the ROCm build of PaddlePaddle, and `device="gpu"` runs on the AMD GPU.

> INFO:
> The PaddleOCR-VL model has its own AMD GPU path. See the [PaddleOCR-VL AMD GPU Usage Tutorial](../pipeline_usage/PaddleOCR-VL-AMD-GPU.en.md). This document covers the general OCR pipeline.

## Tested Configuration

| Item | Value |
| --- | --- |
| GPU | AMD Instinct MI355X (`gfx950`) |
| ROCm | 10.1.0, installed from the public ROCm Python packages |
| Operating system | Ubuntu 24.04 |
| Python | 3.12 |
| PaddlePaddle | v3.4.0, built from source with ROCm support |
| Pipeline | General OCR with PP-OCRv6 text detection and recognition |

Other AMD Instinct GPUs and other PaddleOCR pipelines have not been run with this procedure.

## 1. Build and Install PaddlePaddle with ROCm Support

There is no prebuilt ROCm wheel of PaddlePaddle. Build it from source by following [Building and Testing PaddlePaddle on AMD Instinct GPUs with ROCm](https://github.com/AMD-Ecosystem/Paddle/blob/rocm-enablement/VALIDATION_NOTES.md), and install the wheel it produces.

Keep the Python virtual environment and the ROCm environment variables from that guide active for the remaining steps. PaddleOCR compiles some GPU kernels the first time they are used, so the C and C++ build tools installed by that guide must stay on the system.

## 2. Install PaddleOCR

Install PaddleOCR from the root of this repository into the same Python environment.

```bash
sudo apt-get install -y libgl1 libglib2.0-0 libgomp1

pip install -e ".[doc-parser]" \
  -c deploy/accelerators/amd-gpu/amd-constraints.txt "numpy<2"
```

The constraints file stops `pip` from installing the CUDA builds of PaddlePaddle and the CUDA runtime wheels, which would replace the ROCm build.

Confirm that PaddlePaddle has ROCm support and can use the GPU.

```bash
python -c "import paddle; print(paddle.device.is_compiled_with_rocm()); paddle.utils.run_check()"
```

The command prints `True`, followed by a message that PaddlePaddle is installed successfully.

## 3. Quick Start

Download a sample image and run the general OCR pipeline on the GPU from the command line.

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png
paddleocr ocr -i general_ocr_002.png --device gpu
```

The same pipeline from Python:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(device="gpu")
result = ocr.predict("general_ocr_002.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

The first run downloads the model files. The recognized text is printed, and an annotated image and a JSON file are written to the `output` directory.

## Known Limitations

These are limits of the ROCm build of PaddlePaddle, not of PaddleOCR.

- BF16 convolution through MIOpen is not included.
- FlashAttention is disabled in ROCm builds.
