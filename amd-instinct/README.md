# PaddleOCR-VL -- AMD Instinct (ROCm) Enablement

**AMD AIOSS hipSHIFT enablement.** Framework-transparent port: PaddleOCR-VL runs
on AMD Instinct GPUs (MI300X/gfx942, MI355X/gfx950) via Baidu's official prebuilt
Docker images. No upstream source changes are required.

## Validated Configurations

| GPU | Arch | VRAM | ROCm | Status |
|-----|------|------|------|--------|
| AMD Instinct MI300X | gfx942 | 192 GB | 10.1 | Validated |
| AMD Instinct MI355X | gfx950 | 288 GB | 10.1 | Validated |

## Quick Start

See `deploy/paddleocr_vl_docker/README.md` for the full recipe.

```bash
# Pull the AMD GPU image
docker pull paddlepaddle/paddleocr-vl:paddleocr3.1-amd-gpu

# Run with ROCm device access
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size=8G \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  paddlepaddle/paddleocr-vl:paddleocr3.1-amd-gpu \
  python -c "import paddle; print('ROCm ready:', paddle.device.is_compiled_with_rocm())"
```

## Performance (AMD Instinct vs NVIDIA H100)

avg_pool2d throughput at PP-OCRv6 recognition input sizes (3×32×320, batch=4):

| Platform | Throughput | vs CPU | vs H100 |
|----------|------------|--------|---------|
| AMD Instinct MI300X (gfx942) | 150,375 img/s | 5.66x | 0.61x |
| AMD Instinct MI355X (gfx950) | 213,910 img/s | 5.44x | **0.87x** |
| NVIDIA H100 80GB (reference) | 244,610 img/s | -- | baseline |
| CPU (AMD EPYC 9654) | 26,554 img/s | baseline | -- |

Framework: paddlepaddle_dcu 3.4.0.dev20260825, ROCm 10.1, manylinux2.28.

## Resources

- [AMD AIOSS Release Documentation](https://github.com/AMD-AIOSS/hipSHIFT/tree/port/PaddleOCR-VL/hipshift/release_docs/)
- [Upstream PaddleOCR AMD-GPU Docker recipe](./deploy/paddleocr_vl_docker/)
