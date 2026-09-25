#!/usr/bin/env python3
"""Restore the measured SGLang sources after verifying version and file hashes."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "runtime-overrides"
    manifest = json.loads((root / "manifest.json").read_text())
    dist = importlib.metadata.distribution(manifest["package"])
    if dist.version != manifest["version"]:
        raise SystemExit(f"Expected SGLang {manifest['version']}; found {dist.version}")
    pending = []
    for entry in manifest["files"]:
        source = root / entry["path"]
        target = Path(dist.locate_file(entry["path"]))
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["published_sha256"]:
            raise SystemExit(f"Override checksum mismatch: {source}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        allowed = {entry[k] for k in ("upstream_sha256", "measured_source_sha256", "published_sha256")}
        if digest not in allowed:
            raise SystemExit(f"Unexpected local modification; refusing to overwrite {target}")
        if digest != entry["published_sha256"]:
            pending.append((source, target))
    if not args.check:
        for source, target in pending:
            shutil.copyfile(source, target)
    print(f"Validated {len(manifest['files'])} files; {len(pending)} "
          + ("would be restored." if args.check else "restored."))


if __name__ == "__main__":
    main()
