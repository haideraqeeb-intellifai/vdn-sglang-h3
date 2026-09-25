# Model references (weights excluded)

No model weights, adapters, checkpoints, optimizer states, or derived parameter
files are published in this repository, Git LFS, or release assets.

| Model/source | Purpose | Exact revision |
|---|---|---|
| [OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3/tree/ec875b5bde3832ff72511b079c70ea18b277c339) | H3 base, VDN linear branch, default and turbo adapters from `stage-dmd-step-250` | `ec875b5bde3832ff72511b079c70ea18b277c339` |
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe) | Qwen3-VL text encoder, tokenizer, processor | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| [kevin-mi/VDN-H3-overlay](https://huggingface.co/kevin-mi/VDN-H3-overlay/tree/7de18275dddfe59da36a234e222bcdd274963bc3) | Metadata and materialization code | `7de18275dddfe59da36a234e222bcdd274963bc3` |

The weights inherit the [MiniMax H3 Community License](model-metadata/vdn-source/LICENSE),
including its access and territorial restrictions. Read the linked model cards
and license before downloading. If Hugging Face requires authentication or
license acceptance, complete it with your own account; no credentials are
included here. The overlay code is Apache-2.0.

After installing the captured dependencies, retrieve inputs with:

```bash
.venv/bin/python scripts/download_models.py
```

This downloads to `.cache/huggingface/hub`, pins the three revisions, and sets
their `main` aliases only in that project-local cache. `scripts/serve.sh` uses
this cache in offline mode so the upstream materializer cannot silently resolve
a newer model. The runtime then materializes/fuses the weights locally.
Allow substantial disk and host RAM: the saved materialized model alone is
approximately 139 GiB, in addition to downloaded source weights, and the
materializer requires at least 90 GiB free for conversion. First-time setup and
materialization are outside the reported generation latency.

`model-metadata/references.json` records source weight sizes and SHA-256 blob
identifiers available from the local Hugging Face cache (not freshly rehashed).
`model-metadata/materialized-weight-inventory.json` records excluded derived
weight sizes; hashes unavailable without rereading those large files are not
invented. Adapter and linear-branch checksums are also retained in
`model-metadata/materialized/transformer/sglang_vdn_prefused.json`.

Configuration, tokenizer resources, indices, and provenance are preserved in
`model-metadata/`. These metadata directories are archival references, not
complete model directories suitable for loading without the downloaded weights.
