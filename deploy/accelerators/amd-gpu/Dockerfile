# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# PaddleOCR broader-toolkit image for AMD GPU (ROCm) -- PP-OCRv6, PP-StructureV3,
# PP-DocTranslation. GPU compute rides the ROCm PaddlePaddle wheel
# (amd-paddlepaddle); paddle.set_device('gpu') routes to HIP unchanged on
# Instinct (gfx942/gfx950). This is distinct from the PaddleOCR-VL flagship path
# under deploy/paddleocr_vl_docker/accelerators/amd-gpu/, which rides its own
# Baidu-hosted vLLM images.
#
# Build:
#   docker build -f deploy/accelerators/amd-gpu/Dockerfile \
#     --build-arg AMD_PADDLE_VERSION=3.4.0.dev20260825 \
#     -t paddleocr-toolkit:amd-gpu deploy/accelerators/amd-gpu
#
# Run (device passthrough + video group):
#   docker run -it --device /dev/kfd --device /dev/dri \
#     --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
#     --shm-size 64g paddleocr-toolkit:amd-gpu

ARG ROCM_BASE_IMAGE=registry.aifoundry.amd.com/rocm-manylinux228-rhel810:10.1.0a20260821
FROM ${ROCM_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ROCm floor for this port is 10.1 (matches the AMD wheel index below).
# ROCm wheel index (TheRock-aligned AMD PyPI) for the amd-paddlepaddle build.
ARG ROCM_VERSION=10.1
ARG AMD_PADDLE_INDEX_URL=https://pypi.amd.com/rocm-${ROCM_VERSION}/simple
ARG AMD_PADDLE_VERSION=3.4.0.dev20260825
ARG PADDLEOCR_EXTRAS=doc-parser

# Fonts for rendered OCR / structure visualization output.
RUN if command -v dnf >/dev/null 2>&1; then \
        dnf install -y fontconfig dejavu-sans-fonts google-noto-cjk-fonts wqy-microhei-fonts || true; \
        fc-cache -fv || true; \
    elif command -v apt-get >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends \
            fontconfig fonts-dejavu-core fonts-noto-cjk fonts-wqy-microhei && fc-cache -fv \
            && rm -rf /var/lib/apt/lists/*; \
    fi

# 1) Install the ROCm PaddlePaddle wheel (amd-paddlepaddle) from the AMD index.
#    This is the ROCm-built engine; it provides paddle + the HIP GPU kernels.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir \
        "amd-paddlepaddle==${AMD_PADDLE_VERSION}" \
        --index-url "${AMD_PADDLE_INDEX_URL}"

# 2) Install the PaddleOCR toolkit, constrained so the CUDA PaddlePaddle wheel
#    (paddlepaddle / paddlepaddle-gpu) and NVIDIA CUDA runtime wheels are blocked
#    (==0). Only the ROCm wheel installed above resolves.
COPY amd-constraints.txt /tmp/amd-constraints.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir \
        "paddleocr[${PADDLEOCR_EXTRAS}]" \
        -c /tmp/amd-constraints.txt

# Default GPU visibility; override at run time (e.g. HIP_VISIBLE_DEVICES=1).
ENV HIP_VISIBLE_DEVICES=0

# Verify the ROCm engine imports and enumerates the GPU at build time.
RUN python -c "import paddle; print('paddle', paddle.__version__); print('compiled_with_rocm', paddle.device.is_compiled_with_rocm())"

EXPOSE 8080

# Default: serve the general OCR pipeline. Override --pipeline for
# PP-StructureV3 / PP-DocTranslation (see compose.yaml / .env).
CMD ["paddlex", "--serve", "--pipeline", "OCR"]
