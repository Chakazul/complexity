"""Compressed size as an upper-bound proxy for algorithmic (Kolmogorov) complexity.

Every measure reports bits per cell (per frame, for windows), so values are comparable
across grid sizes: about 0 for a uniform frame, 1 for incompressible binary noise and 8 for
incompressible 8-bit noise, plus each format's fixed overhead.
"""

from __future__ import annotations

import io
import lzma

import av
import numpy as np
from PIL import Image

from complexity.measures.base import Measure, to_uint8
from complexity.registry import register_measure


def _bits_per_cell(n_bytes: int, frames) -> float:
    return n_bytes * 8 / sum(f.size for f in frames)


@register_measure
class PNGSize(Measure):
    key = "png"
    name = "Image compression (PNG)"
    unit = "bits/cell"
    description = "PNG size of the current frame (DEFLATE with 2-D row filters)."

    def compute(self, frames) -> float:
        buf = io.BytesIO()
        Image.fromarray(to_uint8(frames[-1])).save(buf, format="PNG", compress_level=9)
        return _bits_per_cell(buf.tell(), frames[-1:])


@register_measure
class LZMAFrame(Measure):
    key = "lzma"
    name = "Image compression (LZMA)"
    unit = "bits/cell"
    description = "LZMA size of the current frame's raw bytes (1-D, no image model)."

    def compute(self, frames) -> float:
        return _bits_per_cell(len(lzma.compress(to_uint8(frames[-1]).tobytes())), frames[-1:])


@register_measure
class LZMAVideo(Measure):
    key = "video_lzma"
    name = "Video compression (LZMA)"
    unit = "bits/cell/frame"
    description = "LZMA size of a sliding window of frames; repeats across time compress away."
    window = 16
    temporal = "dynamic"

    def compute(self, frames) -> float:
        raw = np.stack([to_uint8(f) for f in frames]).tobytes()
        return _bits_per_cell(len(lzma.compress(raw)), frames)


@register_measure
class H264Video(Measure):
    key = "video_h264"
    name = "Video compression (H.264 lossless)"
    unit = "bits/cell/frame"
    description = (
        "Lossless x264 (qp 0) size of a sliding window of frames; motion compensation "
        "captures translating structures such as gliders."
    )
    window = 16
    temporal = "dynamic"

    def compute(self, frames) -> float:
        h, w = frames[-1].shape
        buf = io.BytesIO()
        with av.open(buf, "w", format="h264") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = w, h, "gray"
            stream.options = {"qp": "0", "preset": "medium"}
            for f in frames:
                vf = av.VideoFrame.from_ndarray(to_uint8(f), format="gray")
                container.mux(stream.encode(vf))
            container.mux(stream.encode())
        return _bits_per_cell(len(buf.getvalue()), frames)
