# SPDX-License-Identifier: Apache-2.0
"""Materialize OpenVDN/vdn-minimax-h3 (VDN-H3) into the base-H3 layout.

VDN-H3 ships as *deltas* on top of the released MiniMax-H3: a 4.3 GB linear
attention branch plus two LoRA adapters (``default`` rank 64 on the attention
projections; ``turbo``, the 8-NFE DMD2 distill, on attention, feed-forward,
AdaLN and the output norm). ``h3-base/`` is a byte-identical copy of the
MiniMaxAI/MiniMax-H3 Diffusers export (transformer, video VAE, audio VAE).

This materializer produces a directory that SGLang's MiniMax-H3 loaders
consume with no VDN-specific weight handling at runtime:

1. ``text_encoder/``, ``tokenizer/``, ``processor/`` - the VDN repo does not
   carry the Qwen3-VL conditioner; they are linked in from the upstream
   MiniMaxAI/MiniMax-H3 snapshot (the same files every H3 deployment loads).
2. ``video_vae/source/model.safetensors`` and the patched VAE configs - the
   FastH3 conversion, verbatim (the same Diffusers VAE export).
3. ``transformer/`` - the 14 Diffusers shards re-written with BOTH LoRA
   adapters folded in (``W += scale * B @ A`` in fp32, rounded once to the
   stored dtype; scale = alpha / rank, which is 1.0 for every module of the
   released checkpoint). Merging in the Diffusers layout keeps the export
   valid, so SGLang's ``_diffusers_h3_checkpoint`` conversion (SwiGLU half
   swap, fused-QKV row order, ``norm_out`` bias) still applies unchanged.
4. ``transformer/sglang_vdn_linear_branch.safetensors`` - the linear branch,
   linked in and indexed; its keys (``transformer_blocks.N.attn.
   {linear_attention.*, softmax_gate.*, to_out_linear.weight}``) are mapped
   by ``MiniMaxH3DiTArchConfig.param_names_mapping``.
5. ``transformer/config.json`` - the base config plus ``hybrid_attention``
   (the linear branch's resolved transform config), which is what makes
   ``MiniMaxH3DiTConfig.update_model_arch`` build the hybrid modules.
6. ``transformer/sglang_rope_inv_freq.safetensors`` - the derivable RoPE
   buffer the Diffusers export drops (FastH3's helper).
7. ``transformer/sglang_vdn_prefused.json`` - the provenance record: adapter
   files, sha256, scale, merge order, pair counts.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

VDN_CHECKPOINT = "stage-dmd-step-250"
VDN_ADAPTER_ORDER = ("default", "turbo")
UPSTREAM_H3_REPO = "MiniMaxAI/MiniMax-H3"
UPSTREAM_COMPONENTS = ("text_encoder", "tokenizer", "processor")
LINEAR_BRANCH_FILE = "sglang_vdn_linear_branch.safetensors"
PREFUSED_RECORD_FILE = "sglang_vdn_prefused.json"
MIN_FREE_BYTES = 90 * 1024**3
EXPECTED_LINEAR_BRANCH_KEYS = 800
EXPECTED_PAIRS = {"default": 208, "turbo": 363}

_ROPE_FILE = "sglang_rope_inv_freq.safetensors"


# --------------------------------------------------------------------------
# Small file helpers (mirroring model_overlay.py's link-or-copy semantics)
# --------------------------------------------------------------------------


def _link_or_copy_file(src: str, dst: str) -> None:
    src = os.path.realpath(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def _copytree_link_or_copy(src_dir: str, dst_dir: str) -> None:
    for root, _, files in os.walk(src_dir):
        rel_root = os.path.relpath(root, src_dir)
        target_root = dst_dir if rel_root == "." else os.path.join(dst_dir, rel_root)
        os.makedirs(target_root, exist_ok=True)
        for file_name in files:
            _link_or_copy_file(
                os.path.join(root, file_name), os.path.join(target_root, file_name)
            )


def _sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _ensure_free_space(output_dir: str) -> None:
    probe = output_dir
    while not os.path.exists(probe):
        probe = os.path.dirname(probe) or "/"
    free = shutil.disk_usage(probe).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            "VDN-H3 materialization re-writes the 62 GB prefused transformer and "
            "a 10 GB VAE re-export; refusing to start with "
            f"{free / 1024**3:.1f} GB free under {probe} (need "
            f"{MIN_FREE_BYTES / 1024**3:.0f} GB). Point "
            "SGLANG_DIFFUSION_CACHE_ROOT at a larger volume."
        )


# --------------------------------------------------------------------------
# FastH3 helpers, verbatim (kevin-mi/FastH3-4step-Preview-overlay)
# --------------------------------------------------------------------------


def _write_rope_inv_freq(*, source_dir: str, output_dir: str) -> None:
    with open(os.path.join(source_dir, "transformer", "config.json")) as f:
        rope_freq_dim = int(json.load(f)["rope_freq_dim"])
    exponents = torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (
        2 * rope_freq_dim
    )
    inv_freq = 1.0 / (10000.0**exponents)
    transformer_dir = os.path.join(output_dir, "transformer")
    save_file({"rope.inv_freq": inv_freq}, os.path.join(transformer_dir, _ROPE_FILE))

    index_path = os.path.join(
        transformer_dir, "diffusion_pytorch_model.safetensors.index.json"
    )
    with open(index_path) as f:
        index = json.load(f)
    index["weight_map"]["rope.inv_freq"] = _ROPE_FILE
    if os.path.lexists(index_path):
        os.remove(index_path)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def _interleave_qkv_rows(
    qkv: torch.Tensor, *, heads: int, dim_head: int
) -> torch.Tensor:
    rest_shape = qkv.shape[1:]
    return (
        qkv.reshape(3, heads, dim_head, *rest_shape)
        .transpose(0, 1)
        .reshape(3 * heads * dim_head, *rest_shape)
    )


_DECODER_RENAMES = (
    ("decoder.proj_in.", "decoder.x_embedder."),
    (".attn.to_out.0.", ".attn.to_out."),
    (".ff.net.2.", ".ff.w2."),
)


def _rename_encoder(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "encoder" and parts[1] == "down_blocks":
        block = parts[2]
        if parts[3] == "resnets":
            tail = parts[5:]
            if tail and tail[0] == "conv_shortcut":
                tail[0] = "nin_shortcut"
            return ".".join(["encoder", "down", block, "block", parts[4], *tail])
        if parts[3] == "downsamplers":
            return ".".join(["encoder", "down", block, "downsample", *parts[5:]])
    return name


def _convert_video_vae(*, source_dir: str, output_dir: str) -> None:
    vae_dir = os.path.join(source_dir, "vae")
    with open(os.path.join(vae_dir, "config.json")) as f:
        config = json.load(f)
    heads = int(config["decoder_num_attention_heads"])
    dim_head = int(config["decoder_attention_head_dim"])

    with open(
        os.path.join(vae_dir, "diffusion_pytorch_model.safetensors.index.json")
    ) as f:
        weight_map = json.load(f)["weight_map"]

    handles: dict[str, object] = {}

    def load(name: str) -> torch.Tensor:
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(
                os.path.join(vae_dir, shard), framework="pt", device="cpu"
            )
        return handles[shard].get_tensor(name)

    converted: dict[str, torch.Tensor] = {}
    pending: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for name in weight_map:
        tensor = load(name)
        if ".attn.to_q." in name or ".attn.to_k." in name or ".attn.to_v." in name:
            slot = name.split(".attn.to_")[1][0]
            fused_name = (
                name.replace(".attn.to_q.", ".attn.to_qkv.")
                .replace(".attn.to_k.", ".attn.to_qkv.")
                .replace(".attn.to_v.", ".attn.to_qkv.")
            )
            pending[fused_name][slot] = tensor
            if len(pending[fused_name]) == 3:
                stacked = torch.cat(
                    [pending[fused_name][s] for s in ("q", "k", "v")], dim=0
                )
                converted[fused_name] = _interleave_qkv_rows(
                    stacked, heads=heads, dim_head=dim_head
                )
                del pending[fused_name]
            continue
        if ".ff.net.0.proj." in name:
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, value), dim=0)
            name = name.replace(".ff.net.0.proj.", ".ff.w1.")
        else:
            for old, new in _DECODER_RENAMES:
                name = name.replace(old, new)
            name = _rename_encoder(name)
        converted[name] = tensor

    if pending:
        raise ValueError(
            "Incomplete fused QKV groups in the Diffusers VAE export: "
            + ", ".join(sorted(pending))
        )
    converted["decoder.mask_token"] = torch.zeros(
        (1, 1, heads * dim_head), dtype=torch.float32
    )

    target_dir = os.path.join(output_dir, "video_vae", "source")
    os.makedirs(target_dir, exist_ok=True)
    source_file = os.path.join(target_dir, "model.safetensors")
    save_file(
        {name: tensor.contiguous() for name, tensor in converted.items()},
        source_file,
    )
    root_file = os.path.join(output_dir, "video_vae", "model.safetensors")
    if os.path.lexists(root_file):
        os.remove(root_file)
    os.link(source_file, root_file)


def _write_patched_component_config(
    *, source_path: str, target_path: str, class_name: str
) -> None:
    with open(source_path) as f:
        config = json.load(f)
    config["_class_name"] = class_name
    if os.path.lexists(target_path):
        os.remove(target_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


# --------------------------------------------------------------------------
# VDN-specific steps
# --------------------------------------------------------------------------


def _link_upstream_components(output_dir: str) -> str:
    """Link text_encoder/, tokenizer/, processor/ from MiniMaxAI/MiniMax-H3.

    SGLang's component loader ignores per-component
    ``pretrained_model_name_or_path`` in model_index.json, so the materializer
    resolves the upstream snapshot itself. The download is skipped when the
    hub cache already holds these files.
    """
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        UPSTREAM_H3_REPO,
        allow_patterns=[f"{component}/*" for component in UPSTREAM_COMPONENTS],
        max_workers=8,
    )
    for component in UPSTREAM_COMPONENTS:
        src = os.path.join(snapshot, component)
        if not os.path.isdir(src):
            raise ValueError(
                f"{UPSTREAM_H3_REPO} snapshot is missing {component}/ ({src})"
            )
        _copytree_link_or_copy(src, os.path.join(output_dir, component))
    return snapshot


def _lora_target(key: str) -> tuple[str, str]:
    """peft key -> (target weight name in the Diffusers export, 'A'|'B').

    ``transformer_blocks.N.attn.orig.to_q.lora_A.default.weight`` names the
    projection one level under VDN's HybridAttention wrapper; the export
    stores it at ``transformer_blocks.N.attn.to_q.weight``.
    """
    if ".lora_A." in key:
        base, kind = key.split(".lora_A.")[0], "A"
    elif ".lora_B." in key:
        base, kind = key.split(".lora_B.")[0], "B"
    else:
        raise ValueError(f"not a LoRA tensor: {key}")
    base = base.replace(".attn.orig.", ".attn.")
    return base + ".weight", kind


def _adapter_scale(adapter_config: dict, module: str) -> tuple[float, int]:
    """(alpha / rank, rank) for one target module, honoring per-module
    ``rank_pattern`` / ``alpha_pattern`` overrides."""
    rank = int(adapter_config.get("rank_pattern", {}).get(module, adapter_config["rank"]))
    alpha = float(
        adapter_config.get("alpha_pattern", {}).get(module, adapter_config["alpha"])
    )
    return alpha / rank, rank


def _load_adapter(adapter_dir: str) -> tuple[dict, dict[str, dict], str, str]:
    """Read one adapter: (config, {target: {'A': t, 'B': t, 'scale': s}}, sha256s)."""
    # OpenVDN renamed adapter_config.json to adapter_spec.json (84932c28f0, 2026-09-09).
    config_path = os.path.join(adapter_dir, "adapter_spec.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(adapter_dir, "adapter_config.json")
    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    with open(config_path) as f:
        adapter_config = json.load(f)["config"]
    pairs: dict[str, dict] = defaultdict(dict)
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            target, kind = _lora_target(key)
            pairs[target][kind] = f.get_tensor(key)
    for target, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"{adapter_dir}: incomplete LoRA pair for {target}")
        module = target[: -len(".weight")]
        scale, rank = _adapter_scale(adapter_config, module)
        if pair["A"].shape[0] != rank or pair["B"].shape[1] != rank:
            raise ValueError(
                f"{adapter_dir}: {target} has rank {pair['A'].shape[0]} but the "
                f"adapter config says {rank}"
            )
        pair["scale"] = scale
    return adapter_config, dict(pairs), _sha256(weights_path), _sha256(config_path)


def _prefuse_transformer(*, source_dir: str, output_dir: str) -> dict:
    """Re-write the transformer shards with both adapters folded in; attach
    the linear branch; write config + index. Returns the provenance record."""
    h3_transformer = os.path.join(source_dir, "h3-base", "transformer")
    ckpt_dir = os.path.join(source_dir, VDN_CHECKPOINT)
    out_transformer = os.path.join(output_dir, "transformer")
    os.makedirs(out_transformer, exist_ok=True)

    with open(os.path.join(h3_transformer, "config.json")) as f:
        base_config = json.load(f)
    with open(
        os.path.join(h3_transformer, "diffusion_pytorch_model.safetensors.index.json")
    ) as f:
        base_index = json.load(f)
    weight_map: dict[str, str] = dict(base_index["weight_map"])

    # ---- adapters -> per-target delta lists, in merge order ----------------
    deltas: dict[str, list[tuple[str, torch.Tensor, torch.Tensor, float]]] = (
        defaultdict(list)
    )
    adapter_records = []
    for adapter in VDN_ADAPTER_ORDER:
        adapter_dir = os.path.join(ckpt_dir, "adapters", adapter)
        adapter_config, pairs, weights_sha, config_sha = _load_adapter(adapter_dir)
        if len(pairs) != EXPECTED_PAIRS[adapter]:
            raise ValueError(
                f"adapter {adapter!r}: expected {EXPECTED_PAIRS[adapter]} LoRA "
                f"pairs, found {len(pairs)}"
            )
        scales = sorted({pair["scale"] for pair in pairs.values()})
        for target, pair in pairs.items():
            if target not in weight_map:
                raise ValueError(
                    f"adapter {adapter!r} targets {target}, which the base "
                    "transformer export does not contain"
                )
            deltas[target].append((adapter, pair["A"], pair["B"], pair["scale"]))
        adapter_records.append(
            {
                "name": adapter,
                "path": os.path.relpath(adapter_dir, source_dir),
                "pairs": len(pairs),
                "rank": int(adapter_config["rank"]),
                "alpha": float(adapter_config["alpha"]),
                "rank_pattern_modules": len(adapter_config.get("rank_pattern", {})),
                "scales": scales,
                "adapter_model_sha256": weights_sha,
                "adapter_config_sha256": config_sha,
            }
        )

    # ---- stream shards, fold, re-write ------------------------------------
    shards = sorted(set(weight_map.values()))
    merged_pairs = 0
    merged_tensors = 0
    for shard in shards:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(
            os.path.join(h3_transformer, shard), framework="pt", device="cpu"
        ) as f:
            metadata = f.metadata()
            for name in f.keys():
                tensor = f.get_tensor(name)
                if name in deltas:
                    merged = tensor.to(torch.float32)
                    for _, lora_a, lora_b, scale in deltas[name]:
                        delta = lora_b.to(torch.float32) @ lora_a.to(torch.float32)
                        if delta.shape != merged.shape:
                            raise ValueError(
                                f"{name}: LoRA delta {tuple(delta.shape)} does not "
                                f"match weight {tuple(merged.shape)}"
                            )
                        merged.add_(delta, alpha=scale)
                        merged_pairs += 1
                    tensor = merged.to(tensor.dtype)
                    merged_tensors += 1
                tensors[name] = tensor.contiguous()
        target = os.path.join(out_transformer, shard)
        if os.path.lexists(target):
            os.remove(target)
        save_file(tensors, target, metadata=metadata or {"format": "pt"})
        del tensors
    expected_pairs = sum(EXPECTED_PAIRS.values())
    if merged_pairs != expected_pairs:
        raise ValueError(
            f"merged {merged_pairs} LoRA pairs, expected {expected_pairs}"
        )

    # ---- linear branch: link + index --------------------------------------
    branch_src = os.path.join(ckpt_dir, "linear_branch", "model.safetensors")
    with open(os.path.join(ckpt_dir, "linear_branch", "config.json")) as f:
        branch_config = json.load(f)
    if branch_config.get("type") != "hybrid_attention" or branch_config.get(
        "version"
    ) != 2:
        raise ValueError(
            "unexpected linear_branch/config.json: "
            f"type={branch_config.get('type')!r} version={branch_config.get('version')!r}"
        )
    _link_or_copy_file(branch_src, os.path.join(out_transformer, LINEAR_BRANCH_FILE))
    with safe_open(branch_src, framework="pt", device="cpu") as f:
        branch_keys = list(f.keys())
    if len(branch_keys) != EXPECTED_LINEAR_BRANCH_KEYS:
        raise ValueError(
            f"linear branch has {len(branch_keys)} tensors, expected "
            f"{EXPECTED_LINEAR_BRANCH_KEYS}"
        )
    for key in branch_keys:
        if key in weight_map:
            raise ValueError(f"linear branch key {key} collides with the base export")
        weight_map[key] = LINEAR_BRANCH_FILE

    # ---- config + index ---------------------------------------------------
    config = dict(base_config)
    config["hybrid_attention"] = branch_config["config"]
    with open(os.path.join(out_transformer, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    index = {
        "metadata": {
            **(base_index.get("metadata") or {}),
            "total_size": int((base_index.get("metadata") or {}).get("total_size", 0))
            + os.path.getsize(branch_src),
        },
        "weight_map": weight_map,
    }
    with open(
        os.path.join(out_transformer, "diffusion_pytorch_model.safetensors.index.json"),
        "w",
    ) as f:
        json.dump(index, f, indent=2, sort_keys=True)

    with open(os.path.join(ckpt_dir, "metadata.json")) as f:
        checkpoint_metadata = json.load(f)
    record = {
        "source_model_id": "OpenVDN/vdn-minimax-h3",
        "vdn_checkpoint": VDN_CHECKPOINT,
        "checkpoint_metadata": checkpoint_metadata,
        "merge": {
            "rule": "W += (alpha / rank) * B @ A, fp32 accumulate, one rounding to the stored dtype",
            "order": list(VDN_ADAPTER_ORDER),
            "pairs_merged": merged_pairs,
            "tensors_touched": merged_tensors,
        },
        "adapters": adapter_records,
        "linear_branch": {
            "file": LINEAR_BRANCH_FILE,
            "tensors": len(branch_keys),
            "sha256": _sha256(branch_src),
            "config": branch_config,
        },
    }
    with open(os.path.join(out_transformer, PREFUSED_RECORD_FILE), "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return record


def materialize(
    *, overlay_dir: str, source_dir: str, output_dir: str, manifest: dict
) -> None:
    _ensure_free_space(output_dir)
    h3_base = os.path.join(source_dir, "h3-base")
    _link_upstream_components(output_dir)
    _convert_video_vae(source_dir=h3_base, output_dir=output_dir)
    _write_patched_component_config(
        source_path=os.path.join(h3_base, "vae", "config.json"),
        target_path=os.path.join(output_dir, "video_vae", "config.json"),
        class_name="MiniMaxH3VideoVAE",
    )
    _write_patched_component_config(
        source_path=os.path.join(h3_base, "audio_vae", "config.json"),
        target_path=os.path.join(output_dir, "audio_vae", "config.json"),
        class_name="MiniMaxH3AudioVAE",
    )
    _prefuse_transformer(source_dir=source_dir, output_dir=output_dir)
    _write_rope_inv_freq(source_dir=h3_base, output_dir=output_dir)
