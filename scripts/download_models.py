#!/usr/bin/env python3
"""Download pinned model inputs into this project's isolated Hugging Face cache."""
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    root = Path(__file__).resolve().parents[1]
    cache = root / ".cache/huggingface/hub"
    refs = json.loads((root / "model-metadata/references.json").read_text())
    patterns = {
        "OpenVDN/vdn-minimax-h3": ["h3-base/**", "stage-dmd-step-250/**", "*.json", "*.md", "LICENSE"],
        "MiniMaxAI/MiniMax-H3": ["text_encoder/*", "tokenizer/*", "processor/*"],
        "kevin-mi/VDN-H3-overlay": None,
    }
    for entry in refs:
        snapshot = Path(snapshot_download(
            entry["repo_id"], revision=entry["revision"], cache_dir=cache,
            allow_patterns=patterns[entry["repo_id"]], max_workers=8,
        ))
        # The upstream materializer resolves "main" internally. Pin that alias
        # only inside this isolated cache and launch with HF_HUB_OFFLINE=1.
        alias = snapshot.parent.parent / "refs/main"
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text(entry["revision"])
        print(f"Pinned {entry['repo_id']} at {entry['revision']}")


if __name__ == "__main__":
    main()
