# Locally modified for the VDN-H3 four-forward RTX 5090 benchmark.
# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 sampling params for this four-NFE reproduction run."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams


@dataclass
class VDNH3SamplingParams(MiniMaxH3SamplingParams):
    """VDN-H3 four-NFE experiment requested by the user.

    SGLang counts sigma-grid points, so four transformer evaluations require
    five points.  The released checkpoint was distilled for eight NFEs; this
    local override intentionally evaluates it at four and does not claim the
    upstream eight-NFE quality guarantee.
    """

    num_inference_steps: int = 5

    def _validate(self) -> None:
        super()._validate()
        if self.num_inference_steps != 5:
            raise ValueError(
                "This local VDN-H3 reproduction is fixed to five sigma grid points "
                "(four DiT "
                f"forwards); got num_inference_steps={self.num_inference_steps}. "
                "Use the upstream package for the supported eight-NFE schedule."
            )
        if self.task is not None and self.task.strip().lower() not in (
            "t2va",
            "fl2va",
        ):
            raise ValueError(
                "VDN-H3 serves t2va and fl2va; ref2va was not trained (got "
                f"task={self.task!r}). Use MiniMaxAI/MiniMax-H3 --model-variant "
                "ref2va for that task."
            )


__all__ = ["VDNH3SamplingParams"]
