"""Compressed size as an upper-bound proxy for algorithmic (Kolmogorov) complexity.

Every measure reports bits per cell (per frame, for windows), so values are comparable
across grid sizes: about 0 for a uniform frame, 1 for incompressible binary noise and 8 for
incompressible 8-bit noise, plus each format's fixed overhead. Higher = less compressible.

Each compressor sees a different kind of regularity — PNG smooth 2-D variation, LZMA exact
repeats, H.264 (approximate) motion — so the composite measures combine them: the minimum
is the tightest upper bound, and video / image is the fraction of a frame that is new.
"""

from __future__ import annotations

import io
import lzma
from collections import OrderedDict

import av
import numpy as np
from PIL import Image

from complexity.measures.base import NO_CONTEXT, Measure, to_uint8
from complexity.registry import register_measure

# Compressed sizes keyed by (compressor, frame ids). The cache holds the frames themselves,
# so an id cannot be reused by a new array while its entry is alive. Measures sharing a
# compressor (and a sliding window's earlier frames) are compressed once.
_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 1024


def _cached(name: str, frames, fn) -> float:
    key = (name, *map(id, frames))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key][1]
    value = fn(frames)
    _CACHE[key] = (list(frames), value)
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return value


def _bits_per_cell(n_bytes: int, frames) -> float:
    return n_bytes * 8 / sum(f.size for f in frames)


def _png(frames) -> float:
    buf = io.BytesIO()
    Image.fromarray(to_uint8(frames[-1])).save(buf, format="PNG", compress_level=9)
    return _bits_per_cell(buf.tell(), frames[-1:])


def _lzma(frames) -> float:
    return _bits_per_cell(len(lzma.compress(to_uint8(frames[-1]).tobytes())), frames[-1:])


def _video_lzma(frames) -> float:
    raw = np.stack([to_uint8(f) for f in frames]).tobytes()
    return _bits_per_cell(len(lzma.compress(raw)), frames)


def _video_h264(frames) -> float:
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


IMAGE = {"png": _png, "lzma": _lzma}
VIDEO = {"video_lzma": _video_lzma, "video_h264": _video_h264}


def image_bits(name: str, frame) -> float:
    return _cached(name, [frame], IMAGE[name])


def video_bits(name: str, frames) -> float:
    return _cached(name, frames, VIDEO[name])


@register_measure
class LZMAFrame(Measure):
    key = "lzma"
    icon = "🖼️"
    equation = r"K(x) = \min\{|p| : U(p) = x\} \;\lesssim\; \frac{8\,|\mathrm{LZMA}(x)|}{N}"
    reference = "Kolmogorov 1965; Chaitin 1977; Zenil 2020"
    map = "tile"
    name = "Image compression (LZMA)"
    unit = "bits/cell"
    group = "Compression"
    ratio = "mass"
    ratio_unit = "bits/mass"
    description = "LZMA size of the current frame's raw bytes (1-D, no image model)."

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        return image_bits("lzma", frames[-1])


@register_measure
class PNGSize(Measure):
    key = "png"
    icon = "🖼️"
    equation = r"K(x) = \min\{|p| : U(p) = x\} \;\lesssim\; \frac{8\,|\mathrm{PNG}(x)|}{N}"
    reference = "Kolmogorov 1965; Chaitin 1977; Zenil 2020"
    map = "tile"
    name = "Image compression (PNG)"
    unit = "bits/cell"
    group = "Compression"
    ratio = "mass"
    ratio_unit = "bits/mass"
    description = "PNG size of the current frame (DEFLATE with 2-D row filters)."

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        return image_bits("png", frames[-1])


@register_measure
class LZMAVideo(Measure):
    key = "video_lzma"
    icon = "🎬"
    equation = r"\frac{8\,|\mathrm{LZMA}(x_{t-w+1}\,\cdots\,x_t)|}{N\,w}"
    reference = "Zenil 2020"
    map = "tile"
    name = "Video compression (LZMA)"
    unit = "bits/cell/frame"
    group = "Compression"
    ratio = "mass"
    ratio_unit = "bits/mass/frame"
    description = "LZMA size of a sliding window of frames; exact repeats across time compress away."
    window = 16
    params = ("window",)
    temporal = "dynamic"
    subject = "trajectory"  # a window of frames
    estimated_from = "run"

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        return video_bits("video_lzma", frames)


@register_measure
class H264Video(Measure):
    key = "video_h264"
    icon = "🎬"
    equation = r"\frac{8\,|\mathrm{H.264}(x_{t-w+1}\,\cdots\,x_t)|}{N\,w}"
    reference = "Zenil 2020"
    name = "Video compression (H.264 lossless)"
    unit = "bits/cell/frame"
    group = "Compression"
    ratio = "mass"
    ratio_unit = "bits/mass/frame"
    description = (
        "Lossless x264 (qp 0) size of a sliding window of frames; motion compensation "
        "captures translating structures, including sub-pixel motion."
    )
    window = 16
    params = ("window",)
    temporal = "dynamic"
    subject = "trajectory"  # a window of frames
    estimated_from = "run"

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        return video_bits("video_h264", frames)


@register_measure
class TemporalNovelty(Measure):
    key = "novelty"
    icon = "✨"
    equation = r"\nu = \frac{\min_{c}\, |C^{\text{video}}_c(x_{t-w+1}\cdots x_t)| \,/\, w}{\tfrac{1}{w}\sum_{i} \min_{c}\, |C^{\text{image}}_c(x_i)|}"
    name = "Temporal novelty (video / image)"
    unit = "ratio"
    group = "Compression"
    description = (
        "Best video bits per frame over best image bits per frame, across the window: "
        "the fraction of each frame that is new. ~0 for static or empty patterns, "
        "~1 for fresh noise."
    )
    window = 16
    params = ("window",)
    temporal = "dynamic"
    subject = "trajectory"  # a window of frames
    estimated_from = "run"

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        image = np.mean([min(image_bits(name, f) for name in IMAGE) for f in frames])
        video = min(video_bits(name, frames) for name in VIDEO)
        return float(video / image) if image > 1e-3 else 0.0
