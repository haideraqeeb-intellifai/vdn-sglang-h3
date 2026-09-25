# VDN + SGLang H3 — 128.44-second generation

This repository preserves the VDN-H3 configuration that generated the saved
10.125-second video in **128.439 seconds of server time** (130.249 seconds
including client polling and download), using BF16, hybrid-window attention,
four denoiser forwards, and layerwise offload on one RTX 5090 (32 GB).

- [Measured video](runs/measured/output.mp4) and [warm-up video](runs/warmup/output.mp4)
- [Measured status and timing evidence](runs/measured/status.json)
- [Self-contained HTML report](vdn-h3-four-step-hybrid-window-report/index.html)
- [Model revisions and retrieval](MODEL_REFERENCES.md)
- [Third-party licenses and notices](third_party/NOTICE.md)

## Fresh environment setup

The recorded environment used Python 3.10.12 on Linux x86_64 and the CUDA 13.1
paths shown below. Install a compatible NVIDIA driver and CUDA toolkit, Python
3.10 with venv support, and FFmpeg/ffprobe. The client itself uses only Python's
standard library. Server dependencies are captured in `requirements-runtime.txt`.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-runtime.txt
.venv/bin/python scripts/apply_runtime_overrides.py
.venv/bin/python scripts/download_models.py
bash scripts/serve.sh
```

The five runtime override files are the measured sources plus modification
notices. Their installer verifies SGLang 0.5.20 and source checksums before
writing, and refuses to overwrite unrelated edits. Use `--check` for validation
without changes. The download helper pins model revisions in an isolated cache;
the server launcher uses that cache offline. The original launch command is
also preserved below and in the report assets.

In a second terminal, use fresh labels to preserve the included reference runs:

```bash
.venv/bin/python submit.py --label reproduction-warmup
.venv/bin/python submit.py --label reproduction-measured
```

The dependency versions and runtime sources were captured from the existing
environment. The publication checks validate override checksums and installation,
Python/shell syntax, saved media, and artifact consistency; they do not constitute
a fresh dependency installation or another GPU benchmark. The 128.439-second
number is the saved measured run, not a guarantee for other environments.

## Benchmark dataset

This directory defines a small, reproducible text-to-video-with-audio benchmark
for MiniMax H3 runtimes. It is intended for agents that need to execute the
same generation case across model, attention, quantization, or runtime
configurations without accidentally changing the workload.

The current dataset contains one case. The prompt asks for a three-shot,
vertical podcast advertisement with two speaking hosts and taco B-roll. It is
long enough to exercise dialogue generation, lip synchronization, scene
transitions, identity consistency, food detail, and synchronized sound effects.

## Agent contract

Treat the following files as the canonical dataset definition:

- `prompt.txt` — exact prompt bytes, excluding an optional final newline.
- `request.json` — model request and sampling parameters. The runner replaces
  the `__PROMPT_REPLACED_BY_RUNNER__` placeholder with `prompt.txt`.
- `submit.py` — reference request, polling, download, and artifact writer.

Do not silently rewrite, summarize, translate, normalize, or “improve” the
prompt. Do not change a fixed parameter to accommodate a runtime. If a runtime
cannot execute the case exactly, stop that case and record the incompatibility.
Any intentional deviation must be named in the configuration identifier and
reported beside the result.

## Dataset identity

| Field | Value |
|---|---|
| Dataset name | `minimax-h3-podcast-tacos` |
| Dataset version | `v1` |
| Case count | `1` |
| Case ID | `podcast-tacos-seed42` |
| Task | Text-to-video with generated audio (`t2va`) |
| Prompt file | `prompt.txt` |
| Prompt SHA-256 | `3d3077f76c422a2593e079f5f7b1df8c69395b4496069328a9761eaecc223ba7` |
| Conditions | None |
| Outputs per request | `1` |

The prompt contains three timed narrative regions:

1. `0.0–3.5 s`: first podcast host speaks in the studio.
2. `3.5–6.5 s`: overhead taco-plating B-roll while the second host begins
   speaking off camera.
3. `6.5–10.0 s`: second host finishes the sentence in the same studio.

These descriptions are context only. `prompt.txt`, not this summary, is the
input that must be submitted.

## Fixed workload parameters

The following values define the benchmark case and must remain fixed across
comparable runs:

| Parameter | Required value |
|---|---|
| Seed | `42` |
| Width × height | `736 × 1280` |
| Aspect ratio | `23:40` |
| Frames | `243` |
| Frame rate | `24 FPS` |
| Exact media duration | `10.125 s` |
| Denoiser forwards | `4` |
| SGLang `num_inference_steps` | `5` |
| Guidance scale | `1.0` |
| Embedded guidance scale | `6.0` |
| Video flow shift | `12.0` |
| Audio flow shift | `3.0` |
| Negative prompt | None |
| Input image/video/audio | None |

MiniMax H3 represents a four-forward trajectory with five sigma-grid points.
For this SGLang pipeline, `num_inference_steps=5` therefore means exactly four
transformer/DiT forwards. Do not set it to `4`; that would execute only three
forwards.

The API payload has a top-level integer `seconds: 10` for request validation.
The exact target is `target.duration_seconds: 10.125`, which produces 243
frames at 24 FPS.

## Benchmark configurations

The dataset is independent of the implementation under test. Every result must
record its model, model revision, runtime revision, precision, attention
backend, hardware, device count, parallelism, memory placement, compile mode,
and any local patches.

The completed run in this directory used:

| Field | Value |
|---|---|
| Configuration ID | `vdn-h3-hybrid-4nfe-bf16-offload-1x5090` |
| Model | `OpenVDN/vdn-minimax-h3` |
| Model revision | `ec875b5bde3832ff72511b079c70ea18b277c339` |
| VDN overlay revision | `7de18275dddfe59da36a234e222bcdd274963bc3` |
| Runtime | SGLang `0.5.20` |
| Attention | `hybrid_window_attn_h3` |
| SageAttention | Not used |
| Precision | BF16 |
| Hardware | 1× NVIDIA GeForce RTX 5090, 32 GB |
| Placement | Layerwise offload for DiT, text encoder, and video VAE |
| Explicit `torch.compile` | Disabled |

The linked SGLang announcement used FP8/MXFP8 on 8×B200. That exact FP8 load
path was attempted here, but online MXFP8 conversion staged the complete
transformer before layerwise offload and exceeded the single RTX 5090’s 32 GB.
The saved output therefore is not an FP8 or 8×B200 performance reproduction.
If an agent runs the FP8 configuration on suitable hardware, it must use a
different configuration ID and must not merge its latency numbers with this
BF16 profile.

## Four-forward VDN caveat

The released VDN-H3 checkpoint metadata specifies an eight-evaluation turbo
schedule. Four denoiser forwards are deliberately tested here to match the
comparison workload. This is outside the checkpoint’s published training and
validation schedule, so no upstream quality-equivalence claim applies.

Stock SGLang also needs local compatibility changes for this exact workload:

- permit five sigma-grid points/four denoiser forwards in the VDN sample
  configuration;
- add `23:40` to the MiniMax H3 aspect-ratio allowlist;
- support online MiniMax H3 AdaLN reconstruction for the materialized Diffusers
  checkpoint layout; and
- exclude reconstructed AdaLN and final norm projection weights from normal
  loading.

The changes are published in `runtime-overrides/`, with original wheel and
measured-source checksums. Restore them with
`.venv/bin/python scripts/apply_runtime_overrides.py` after installing the
pinned runtime dependencies.

## Execution protocol

Use one server process for both requests:

1. Confirm the GPU is idle and record its name, memory capacity, driver, and
   CUDA versions.
2. Start the server with the candidate configuration.
3. Run one complete warm-up request labeled `warmup`.
4. Without restarting the server or clearing caches, run one request labeled
   `measured` with the identical prompt, seed, and request payload.
5. Retain both MP4 files and all request/status/timing metadata.
6. Stop the server, confirm the GPU is released, and validate both media files.

For the completed single-5090 profile, start the server from the repository
root with:

```bash
PATH="$PWD/.venv/bin:/usr/local/cuda-13.1/bin:$PATH" \
CPATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cu13/include" \
LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cu13/lib" \
SGLANG_CACHE_DIR="$PWD/.cache" \
SGLANG_DIFFUSION_CACHE_ROOT="$PWD/.cache/diffusion" \
SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/sglang serve \
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
```

When the server reports that it is listening on port `30010`, run:

```bash
.venv/bin/python submit.py --label warmup
.venv/bin/python submit.py --label measured
```

The runner creates this structure:

```text
runs/
├── warmup/
│   ├── request.json
│   ├── submitted.json
│   ├── status.json
│   ├── client_timing.json
│   └── output.mp4
└── measured/
    ├── request.json
    ├── submitted.json
    ├── status.json
    ├── client_timing.json
    └── output.mp4
```

Do not overwrite a valid prior run unless replacement is explicitly requested.
Use a new label or archive the existing directory first.

## Required runtime evidence

An accepted run must retain evidence for all of the following:

- resolved model and revision;
- attention backend selected by the server;
- prompt length or request record showing the canonical prompt was used;
- width, height, frames, FPS, seed, flow shifts, and inference-step count;
- progress showing exactly `4/4` denoising evaluations;
- text-encoding, denoising, decoding/output, and server end-to-end times;
- client end-to-end time;
- peak device memory reported by the runtime or an external sampler;
- completed job status and output path; and
- any warnings, fallbacks, OOM recovery, recompilation, or configuration
  deviation.

For `hybrid_window_attn_h3`, the expected geometry for this case is 72 latent
frames, 920 tokens per frame, 413 text tokens, chunk size 5, radius 1, and both
anchor frames. Treat a different resolved geometry as a configuration mismatch
until explained.

## Media validation

Both warm-up and measured files must pass complete audio/video decoding. For
the canonical case, `ffprobe` must report:

- H.264 video;
- exactly `736 × 1280`;
- exactly `243` decoded video frames;
- exactly `24/1` FPS;
- exactly `10.125000` seconds;
- AAC audio;
- exactly `32000` Hz; and
- exactly `2` audio channels.

Suggested checks:

```bash
ffprobe -v error -count_frames \
  -show_entries stream=index,codec_name,codec_type,width,height,r_frame_rate,nb_read_frames,sample_rate,channels \
  -show_entries format=duration,size \
  -of json runs/measured/output.mp4

ffmpeg -v error -i runs/measured/output.mp4 -f null -
sha256sum prompt.txt runs/warmup/output.mp4 runs/measured/output.mp4
```

Reject a run if either MP4 fails a full decode, the requested values differ, or
the runtime did not execute four denoiser forwards. A quality defect is not a
protocol failure; retain the output and score it separately.

## Metrics and reporting

Report warm-up and measured values separately. At minimum include:

| Metric | Definition |
|---|---|
| Server end-to-end | Runtime time from admitted request through saved output |
| Client end-to-end | Submission through downloaded MP4 |
| Text encoding | Runtime text-encoder stage |
| Denoising | All four DiT forwards and scheduler work |
| Decode/output | Video/audio decoding and container output as exposed by the runtime |
| Peak GPU memory | State the measurement source and units |
| Output FPS | `243 / server_end_to_end_seconds` |
| Generation ratio | `server_end_to_end_seconds / 10.125` |

Do not sum nested stage timers unless the runtime documents them as exclusive.
Do not compare timings from different hardware, precision, device counts,
parallelism, compile modes, or memory-placement policies without labeling those
differences next to the comparison.

Optional human or automated quality scoring should cover:

- dialogue transcription accuracy;
- speaker-to-shot assignment;
- lip synchronization;
- host identity and wardrobe consistency;
- shot boundary timing;
- taco detail and anatomical correctness;
- microphone and hand stability;
- unwanted text, logos, or watermarks; and
- audio continuity and sound-effect timing.

Quality scoring was not performed for the saved reference run.

## Saved reference result

The completed measured run recorded:

| Metric | Value |
|---|---|
| Server end-to-end | `128.439 s` |
| Client end-to-end | `130.249 s` |
| Text encoding | `0.808 s` |
| Four-forward denoising | `103.988 s` |
| Decode/output | `23.086 s` |
| Peak GPU memory | `27,740 MiB` |
| Output FPS | `1.892` |
| Generation ratio | `12.69×` output duration |
| Measured MP4 SHA-256 | `17882d9db644f74d1e4ea49e6ca102283cb6b2561fe8c46647804111bd9b17b7` |

The warm-up and measured decoded video hashes match. Their decoded audio hashes
do not match; deterministic CUDA algorithms were not enforced, and the audio
difference was not investigated.

## Comparison report

The published report contains the videos, configuration, timings, verification,
and downloadable machine-readable evidence:

<https://s3.renderplatform.com/agent-assets/Raqeeb/vdn-h3-four-step-hybrid-window-report/index.html>

The report intentionally uses the same high-level timing and verification
categories as the original SageAttention report, but the runtimes, checkpoints,
attention algorithms, precision, and timing instrumentation are not identical.
