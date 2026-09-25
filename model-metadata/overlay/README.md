---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
- OpenVDN/vdn-minimax-h3
- MiniMaxAI/MiniMax-H3
tags:
- sglang
- sglang-diffusion
- model-overlay
- minimax-h3
- video-deltanet
---

# VDN-H3 SGLang model overlay

Metadata-only overlay that lets SGLang Diffusion serve
[OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3)
(Video DeltaNet MiniMax-H3: hybrid chunked window-softmax + frame-wise linear
attention, 8-NFE DMD2 distill) as a `VDNH3Pipeline`.

There are **no weights in this repository**. SGLang downloads the VDN deltas
(`stage-dmd-step-250/`, the linear branch and two LoRA adapters) plus the
`h3-base/` Diffusers export from the source repo, links the Qwen3-VL
conditioner from `MiniMaxAI/MiniMax-H3`, and materializes a base-H3-layout
directory under `SGLANG_DIFFUSION_CACHE_ROOT/materialized_models/` with:

- both LoRA adapters prefused into the transformer weights (`W += B @ A`,
  fp32 accumulate, scale 1.0 = alpha / rank), written once as new shards;
- the linear branch attached as `transformer/sglang_vdn_linear_branch.safetensors`;
- `transformer/config.json` carrying `hybrid_attention` (the VDN transform config);
- the video VAE re-serialized in the fused source form SGLang's native
  decoder loads, exactly as the FastH3 overlay does.

Usage:

```bash
export SGLANG_DIFFUSION_CACHE_ROOT=/path/with/100GB/free
sglang serve --model-path OpenVDN/vdn-minimax-h3 --num-gpus 4 \
  --attention-backend hybrid_window_attn_h3
```

See `_overlay/materialize.py` for the exact steps and the provenance record
(`transformer/sglang_vdn_prefused.json`) it writes.

## License

VDN-H3 inherits the MiniMax H3 Community License, including its territorial
exclusions; read it before downloading or running the weights. This overlay
repository carries metadata and Apache-2.0 code only.
