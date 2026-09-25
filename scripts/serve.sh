#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$PWD/.venv/bin:/usr/local/cuda-13.1/bin:$PATH"
export CPATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cu13/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cu13/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export SGLANG_CACHE_DIR="$PWD/.cache"
export SGLANG_DIFFUSION_CACHE_ROOT="$PWD/.cache/diffusion"
export SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/sglang serve \
  --model-path OpenVDN/vdn-minimax-h3 \
  --num-gpus 1 \
  --quantization bf16 \
  --attention-backend hybrid_window_attn_h3 \
  --performance-mode memory \
  --layerwise-offload-components dit text_encoder vae \
  --dit-offload-prefetch-size 1 \
  --dit-layerwise-resident-layers 0 \
  --layerwise-resident-layers video_vae=36 \
  --minimax-h3-adaln-online true \
  --minimax-h3-adaln-host-cache-gb 0 \
  --enable-torch-compile false \
  --warmup-mode off \
  --port 30010 \
  --log-requests \
  --log-requests-level 3
