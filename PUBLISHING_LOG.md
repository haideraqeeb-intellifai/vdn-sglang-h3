# Publication log

## 2026-09-25 21:38:53 UTC — initial publication package prepared

Target: https://github.com/haideraqeeb-intellifai/vdn-sglang-h3 (public, main).
This entry records package preparation; the public upload remains pending.
Benchmark performed September 18, 2026: warm-up 14:15:42–14:18:21 UTC;
measured request 14:18:52–14:21:02 UTC (128.439 seconds server generation).
Source: local `vdn-sglang` workspace; no root Git revision existed.
The nested report source revision is
`c926fb9307e612e3fb49505e28c6ec1ae900dae0`.

Included: benchmark prompt, request and submitter; warm-up and measured job
records and videos; output copies; both saved HTML report directories and their
media/assets; all five modified SGLang source files with a checksum-guarded
installer; captured dependency versions; launch/download helpers; non-weight
model metadata, tokenizers, provenance, model references, and third-party notices.
`PUBLICATION_INVENTORY.json` lists the published files and SHA-256 checksums
(excluding that inventory itself).

Intentional exclusions:

- All model/adapter/derived weights: referenced in `MODEL_REFERENCES.md`, with
  source sizes and available cache hashes in `model-metadata/references.json`.
- `.venv/`: installed dependencies; versions captured and all locally changed
  SGLang sources extracted into `runtime-overrides/`.
- `.cache/`: model weights and disposable compiler/runtime caches; all materialized
  non-weight model metadata copied into `model-metadata/materialized/`.
- `.git/`, `.agents/`, `.codex/`: local version-control and tool state.
- `report-site/.git/` and `report-site/.openai/hosting.json`: nested Git state
  and deployment-specific project binding. The complete static report is included.
- Empty `inputs/` directories: no input resources were present.

No eligible large media files were omitted. The largest published file is below
GitHub's 50 MiB warning and 100 MiB block thresholds; Git LFS is unnecessary.
Limits checked on publication day against
https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
and
https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage.

Validation: source and JSON parsing; override checks against the measured
installation; isolated restore, repeat-apply, and conflict refusal checks; shell
syntax; canonical prompt and MP4 hashes; full audio/video decoding of both saved
videos; measured ffprobe geometry/audio/duration; timing consistency; publication
inventory, credential-pattern and excluded-weight scans. A fresh dependency
installation and GPU benchmark were not performed for publication.
