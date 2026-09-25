# Third-party material

`runtime-overrides/sglang/` contains five locally modified SGLang 0.5.20
source files. They retain their upstream notices and are covered by the
included [SGLang Apache 2.0 license](SGLang-LICENSE).
Their changes enable four denoiser forwards, the 23:40 aspect ratio, and
online AdaLN reconstruction/loading for the materialized VDN checkpoint.
The manifest records original wheel hashes and measured source hashes;
published copies add only a modification notice to the measured source.

`model-metadata/overlay/` preserves the metadata and Apache-2.0 materializer
from `kevin-mi/VDN-H3-overlay` at revision
`7de18275dddfe59da36a234e222bcdd274963bc3`. The included Apache license text
also applies to that materializer, whose SPDX notice is retained.

MiniMax H3 and VDN model metadata retain their model cards and the MiniMax H3
Community License in `model-metadata/vdn-source/LICENSE`. Model weights are
not included. Consult `MODEL_REFERENCES.md` for original sources and terms.
This notice does not assign a license to the original benchmark code or media.
