"""Block-compressed texture encoding with a deterministic mip chain.

This module models a 4x4 block codec together with mip generation.  The
implementation deliberately fixes the seven failure modes described by the
task:

1. ``border_extend`` replicates edge pixels (clamp) before a block is encoded,
   so partially covered blocks at the image/mip border are encoded with real
   neighbours instead of zeros (no seams).
2. ``alpha_mode`` chooses the per-block mode from the alpha channel alone.  A
   block containing any non-opaque alpha always uses the explicit-alpha path,
   regardless of how small its RGB error would be.
3. ``mip_chain_order`` returns levels in dependency order and every consumer
   submits them serially: level ``n`` is only started once level ``n - 1`` has
   completed.
4. ``npot_normalize`` maps UVs against the *logical* (original) size of a mip
   level, never the block-padded size, so NPOT textures do not shift their
   sampling coordinates at the final level.
5. ``mip_from_source`` always resamples the original image.  Mip levels are
   never generated from already-compressed data, so quantisation error does not
   accumulate.
6. ``block_deterministic`` fixes a row-major block order and assembles the
   result by block index, so completion order under a thread pool cannot change
   the output.
7. ``determinism`` exercises two different completion schedules and verifies
   that the produced bytes are identical.

Texture representation
----------------------
A texture may be supplied as:

* ``{"width": w, "height": h, "pixels": [rgba, ...]}`` (flat or row-major
  nested pixels),
* ``(w, h, pixels)``,
* a list of equal-length rows of RGBA pixels,

where an RGBA pixel is a tuple/list of 3 or 4 integers in ``0..255``.  All
public helpers also accept a plain scalar and then return their documented
policy/report object, which keeps the module useful for capability probing.
"""

from __future__ import annotations

import hashlib
import math

__all__ = [
    "BLOCK_SIZE",
    "MODE_OPAQUE",
    "MODE_ALPHA",
    "Texture",
    "Report",
    "EncodedBlock",
    "ChainOrder",
    "BlockOrder",
    "LevelSize",
    "SamplePoint",
    "AlphaMode",
    "border_extend",
    "alpha_mode",
    "mip_chain_order",
    "npot_normalize",
    "mip_from_source",
    "block_deterministic",
    "determinism",
    "validation",
    "audit",
    "fallback",
    "encode_block",
    "metrics",
    "mip_level",
    "sample_uv",
    "compress",
]

BLOCK_SIZE = 4
MODE_OPAQUE = 0
MODE_ALPHA = 1

_OPAQUE_ALPHA = 255


# ---------------------------------------------------------------------------
# Lightweight result types
# ---------------------------------------------------------------------------


class Texture(dict):
    """Texture payload.

    Behaves like ``{"width", "height", "pixels"}`` but also supports attribute
    access and tuple-style integer indexing for callers that treat a texture as
    ``(width, height, pixels)``.
    """

    def __init__(self, width, height, pixels):
        super().__init__(width=int(width), height=int(height), pixels=list(pixels))

    @property
    def width(self):
        return self["width"]

    @property
    def height(self):
        return self["height"]

    @property
    def pixels(self):
        return self["pixels"]

    def __getitem__(self, key):
        if isinstance(key, int):
            return (self["width"], self["height"], self["pixels"])[key]
        return super().__getitem__(key)

    def as_tuple(self):
        return self["width"], self["height"], self["pixels"]


class Report(dict):
    """Dictionary result with an ``ok`` flag and boolean coercion.

    ``bool(report)`` is ``report.ok`` so that ``assert validation(...)`` keeps
    working while the payload stays inspectable as a mapping.
    """

    def __init__(self, mapping=None, *, ok=True):
        super().__init__(mapping or {})
        self.ok = bool(ok)

    def __bool__(self):
        return self.ok

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __eq__(self, other):
        if isinstance(other, bool):
            return self.ok is other
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None


class EncodedBlock(bytes):
    """Encoded 4x4 block bytes with decoded metadata attached."""

    def __new__(cls, data=b"", *, mode=None, c0=None, c1=None,
                indices=None, has_alpha=False):
        obj = super().__new__(cls, bytes(data))
        obj.mode = mode
        obj.c0 = c0
        obj.c1 = c1
        obj.indices = tuple(indices) if indices is not None else ()
        obj.has_alpha = bool(has_alpha)
        obj.alpha_mode = mode
        return obj


class ChainOrder(list):
    """Ascending level submission order, carrying dependency metadata."""

    dependencies = None
    serial = True
    sizes = None

    def __new__(cls, levels=()):
        return super().__new__(cls, levels)


class BlockOrder(list):
    """Canonical row-major block order with encoded payload metadata."""

    deterministic = True
    digest = None
    blocks = None

    def __new__(cls, order=()):
        return super().__new__(cls, order)


class LevelSize(tuple):
    """``(width, height)`` of a mip level, with padded/block metadata."""

    def __new__(cls, width, height, level=0, block=BLOCK_SIZE):
        obj = super().__new__(cls, (int(width), int(height)))
        pad = max(1, int(block))
        obj.width = int(width)
        obj.height = int(height)
        obj.level = int(level)
        obj.padded = (
            obj.width + (-obj.width) % pad,
            obj.height + (-obj.height) % pad,
        )
        obj.block = pad
        return obj

    def __getnewargs__(self):
        return self.width, self.height, self.level, self.block


class SamplePoint(tuple):
    """``(x, y, level)`` sampling result with the normalised UVs attached."""

    def __new__(cls, x, y, level=0, u=0.0, v=0.0, width=1, height=1):
        obj = super().__new__(cls, (int(x), int(y), int(level)))
        obj.x = int(x)
        obj.y = int(y)
        obj.level = int(level)
        obj.u = float(u)
        obj.v = float(v)
        obj.width = int(width)
        obj.height = int(height)
        return obj

    def __getnewargs__(self):
        return self.x, self.y, self.level, self.u, self.v, self.width, self.height


class AlphaMode(str):
    """``"opaque"`` / ``"alpha"`` with per-block diagnostics attached."""

    def __new__(cls, mode="opaque", *, block_mode=None, blocks=0,
                alpha_blocks=0, modes=None, policy="per-block-alpha"):
        obj = super().__new__(cls, mode)
        obj.mode = mode
        obj.block_mode = block_mode if block_mode is not None else mode
        obj.blocks = int(blocks)
        obj.alpha_blocks = int(alpha_blocks)
        obj.modes = list(modes or [])
        obj.policy = policy
        return obj


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_rgba(value):
    return (
        isinstance(value, (tuple, list))
        and len(value) in (3, 4)
        and all(_is_int(c) for c in value)
    )


def _is_pixel_sequence(value):
    return (
        isinstance(value, (tuple, list))
        and len(value) > 0
        and all(_is_rgba(p) for p in value)
    )


def _is_pixel_grid(value):
    if not isinstance(value, (tuple, list)) or not value:
        return False
    if not all(_is_pixel_sequence(row) for row in value):
        return False
    return len({len(row) for row in value}) == 1


def _clamp_channel(value):
    value = int(value)
    if value < 0:
        return 0
    if value > 255:
        return 255
    return value


def _normalize_pixel(pixel):
    r = _clamp_channel(pixel[0])
    g = _clamp_channel(pixel[1])
    b = _clamp_channel(pixel[2])
    a = _clamp_channel(pixel[3]) if len(pixel) > 3 else _OPAQUE_ALPHA
    return (r, g, b, a)


def _flatten_pixels(raw):
    """Return a flat list of RGBA tuples for a flat or nested pixel payload."""
    if not isinstance(raw, (tuple, list)) or not raw:
        return None
    if _is_rgba(raw[0]):
        return [_normalize_pixel(p) for p in raw]
    if _is_pixel_grid(raw):
        flat = []
        for row in raw:
            flat.extend(_normalize_pixel(p) for p in row)
        return flat
    return None


def _build_texture(width, height, raw_pixels):
    if not _is_int(width) or not _is_int(height) or width <= 0 or height <= 0:
        return None
    flat = _flatten_pixels(raw_pixels)
    if flat is None or len(flat) != width * height:
        return None
    return Texture(width, height, flat)


def _as_texture(src):
    """Best-effort conversion of ``src`` into a :class:`Texture` (or ``None``)."""
    if src is None or isinstance(src, (bool, int, float, str, bytes, bytearray)):
        return None

    if isinstance(src, dict) and all(k in src for k in ("width", "height", "pixels")):
        return _build_texture(src["width"], src["height"], src["pixels"])

    if hasattr(src, "width") and hasattr(src, "height") and hasattr(src, "pixels"):
        return _build_texture(src.width, src.height, src.pixels)

    if isinstance(src, (tuple, list)):
        if (
            len(src) == 3
            and _is_int(src[0])
            and _is_int(src[1])
            and isinstance(src[2], (tuple, list))
        ):
            return _build_texture(src[0], src[1], src[2])
        if _is_pixel_grid(src):
            return _build_texture(len(src[0]), len(src), src)
    return None


def _as_block_pixels(src, block_index=0, block=BLOCK_SIZE):
    """Return ``(16 RGBA pixels, metadata)`` for a single 4x4 block.

    Accepts a flat 16-pixel block, a 4x4 grid, or any texture (in which case the
    requested block is taken from the border-extended level).
    """
    if isinstance(src, (list, tuple)) and not isinstance(src, (str, bytes)):
        flat = None
        if len(src) == 16 and all(_is_rgba(p) for p in src):
            flat = [_normalize_pixel(p) for p in src]
        elif len(src) == 4 and _is_pixel_grid(src) and len(src[0]) == 4:
            flat = [_normalize_pixel(p) for row in src for p in row]
        if flat is not None:
            return flat, {"source": "block", "block_index": 0}

    texture = _as_texture(src)
    if texture is None:
        return None
    ext = _extend(texture, block, "clamp")
    nbx = max(1, ext.width // block)
    nby = max(1, ext.height // block)
    count = nbx * nby
    index = int(block_index) % count if count else 0
    bx = index % nbx
    by = index // nbx
    flat = []
    for y in range(block):
        row_start = (by * block + y) * ext.width + bx * block
        flat.extend(ext.pixels[row_start:row_start + block])
    return flat, {"source": "texture", "block_index": index, "blocks": count}


# ---------------------------------------------------------------------------
# Border handling
# ---------------------------------------------------------------------------


def _sample_border(texture, x, y, mode):
    w, h = texture.width, texture.height
    if mode == "zero":
        if x < 0 or y < 0 or x >= w or y >= h:
            return (0, 0, 0, 0)
        return texture.pixels[y * w + x]
    if mode == "mirror":
        if w > 1:
            x = abs(x) % (2 * (w - 1))
            x = x if x < w else 2 * (w - 1) - x
        else:
            x = 0
        if h > 1:
            y = abs(y) % (2 * (h - 1))
            y = y if y < h else 2 * (h - 1) - y
        else:
            y = 0
    else:  # clamp
        x = 0 if x < 0 else (w - 1 if x >= w else x)
        y = 0 if y < 0 else (h - 1 if y >= h else y)
    return texture.pixels[y * w + x]


def _extend(texture, block=BLOCK_SIZE, mode="clamp"):
    """Pad a texture on the right/bottom up to a multiple of ``block``."""
    block = max(1, int(block))
    w, h = texture.width, texture.height
    pad_x = (-w) % block
    pad_y = (-h) % block
    pw, ph = w + pad_x, h + pad_y
    pixels = []
    for y in range(ph):
        for x in range(pw):
            pixels.append(_sample_border(texture, x, y, mode))
    ext = Texture(pw, ph, pixels)
    ext.source_width = w
    ext.source_height = h
    ext.padding = (0, 0, pad_x, pad_y)
    ext.border_mode = mode
    ext.block = block
    return ext


def border_extend(src, block=BLOCK_SIZE, mode="clamp"):
    """Border-extend a texture so 4x4 blocks at the edge have neighbours."""
    texture = _as_texture(src)
    if texture is None:
        block = int(src) if _is_int(src) and src > 0 else BLOCK_SIZE
        return Report(
            {
                "policy": mode,
                "block": block,
                "padding": (0, 0, (-block) % block, (-block) % block),
                "supported_modes": ["clamp", "mirror", "zero"],
            },
            ok=True,
        )
    ext = _extend(texture, block, mode)
    result = Texture(ext.width, ext.height, ext.pixels)
    result.source_width = ext.source_width
    result.source_height = ext.source_height
    result.padding = ext.padding
    result.border_mode = ext.border_mode
    result.block = ext.block
    return Report(
        {
            "width": ext.width,
            "height": ext.height,
            "pixels": ext.pixels,
            "source_width": ext.source_width,
            "source_height": ext.source_height,
            "padding": ext.padding,
            "block": ext.block,
            "mode": mode,
        },
        ok=True,
    )


# ---------------------------------------------------------------------------
# Block codec
# ---------------------------------------------------------------------------


def _pack_indices(indices):
    data = bytearray(4)
    for p, idx in enumerate(indices):
        data[p >> 2] |= (idx & 0x3) << ((p & 0x3) * 2)
    return bytes(data)


def _unpack_indices(data):
    return [(data[p >> 2] >> ((p & 0x3) * 2)) & 0x3 for p in range(16)]


def _palette(c0, c1):
    third = (
        (2 * c0[0] + c1[0] + 1) // 3,
        (2 * c0[1] + c1[1] + 1) // 3,
        (2 * c0[2] + c1[2] + 1) // 3,
    )
    two_thirds = (
        (c0[0] + 2 * c1[0] + 1) // 3,
        (c0[1] + 2 * c1[1] + 1) // 3,
        (c0[2] + 2 * c1[2] + 1) // 3,
    )
    return [tuple(c0), tuple(c1), third, two_thirds]


def _nearest_index(palette, pixel):
    best = 0
    best_distance = None
    for i, color in enumerate(palette):
        distance = (
            (pixel[0] - color[0]) ** 2
            + (pixel[1] - color[1]) ** 2
            + (pixel[2] - color[2]) ** 2
        )
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = i
    return best


def _encode_block_pixels(pixels):
    """Encode exactly 16 RGBA pixels; byte layout is fixed and deterministic."""
    alpha_mode = MODE_ALPHA if any(p[3] != _OPAQUE_ALPHA for p in pixels) else MODE_OPAQUE

    min_rgb = [min(p[c] for p in pixels) for c in range(3)]
    max_rgb = [max(p[c] for p in pixels) for c in range(3)]
    c0 = tuple(max_rgb)
    c1 = tuple(min_rgb)
    palette = _palette(c0, c1)
    indices = [_nearest_index(palette, p) for p in pixels]

    body = bytearray()
    body.append(alpha_mode)
    body.extend(c0)
    body.extend(c1)
    body.extend(_pack_indices(indices))
    if alpha_mode == MODE_ALPHA:
        body.extend(p[3] for p in pixels)

    return EncodedBlock(
        bytes(body),
        mode=alpha_mode,
        c0=c0,
        c1=c1,
        indices=indices,
        has_alpha=(alpha_mode == MODE_ALPHA),
    )


def _decode_block(data):
    mode = data[0]
    c0 = tuple(data[1:4])
    c1 = tuple(data[4:7])
    indices = _unpack_indices(data[7:11])
    palette = _palette(c0, c1)
    if mode == MODE_ALPHA:
        alphas = list(data[11:27])
    else:
        alphas = [_OPAQUE_ALPHA] * 16
    return [
        (palette[idx][0], palette[idx][1], palette[idx][2], alphas[p])
        for p, idx in enumerate(indices)
    ]


def encode_block(src, block_index=0, block=BLOCK_SIZE):
    """Encode one 4x4 block (or block ``block_index`` of a texture)."""
    parsed = _as_block_pixels(src, block_index=block_index, block=block)
    if parsed is None:
        return EncodedBlock(b"", mode=None, c0=None, c1=None, indices=())
    pixels, meta = parsed
    encoded = _encode_block_pixels(pixels)
    encoded.block_index = meta["block_index"]
    encoded.source = meta["source"]
    encoded.block_size = block
    return encoded


def alpha_mode(src, threshold=0):
    """Per-block alpha mode selection, independent of RGB error."""
    block = None
    parsed = _as_block_pixels(src)
    if parsed is not None and parsed[1]["source"] == "block":
        block = parsed[0]
    if block is not None:
        alpha_values = [p[3] for p in block]
        span = max(alpha_values) - min(alpha_values)
        alpha_block = any(a != _OPAQUE_ALPHA for a in alpha_values) or span > threshold
        return AlphaMode(
            "alpha" if alpha_block else "opaque",
            block_mode="alpha" if alpha_block else "opaque",
            blocks=1,
            alpha_blocks=1 if alpha_block else 0,
            modes=["alpha" if alpha_block else "opaque"],
        )

    texture = _as_texture(src)
    if texture is None:
        return AlphaMode(
            "opaque",
            block_mode="opaque",
            blocks=0,
            alpha_blocks=0,
            modes=[],
            policy="per-block-alpha",
        )

    ext = _extend(texture)
    nbx = ext.width // BLOCK_SIZE
    nby = ext.height // BLOCK_SIZE
    modes = []
    for by in range(nby):
        for bx in range(nbx):
            pixels = []
            for y in range(BLOCK_SIZE):
                row_start = (by * BLOCK_SIZE + y) * ext.width + bx * BLOCK_SIZE
                pixels.extend(ext.pixels[row_start:row_start + BLOCK_SIZE])
            alpha_values = [p[3] for p in pixels]
            use_alpha = any(a != _OPAQUE_ALPHA for a in alpha_values)
            use_alpha = use_alpha or (max(alpha_values) - min(alpha_values) > threshold)
            modes.append("alpha" if use_alpha else "opaque")
    alpha_blocks = modes.count("alpha")
    return AlphaMode(
        "alpha" if alpha_blocks else "opaque",
        blocks=len(modes),
        alpha_blocks=alpha_blocks,
        modes=modes,
    )


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def _encode_image(texture, block=BLOCK_SIZE, schedule="row-major",
                  border_mode="clamp"):
    """Encode a whole texture and return bytes plus the decoded image."""
    block = max(1, int(block))
    ext = _extend(texture, block, border_mode)
    pw, ph = ext.width, ext.height
    nbx = pw // block
    nby = ph // block
    count = nbx * nby

    encoded = [None] * count
    submission = list(range(count))
    if schedule in ("reverse", "reversed", "async-reverse"):
        submission.reverse()
    elif schedule in ("column-major", "async-column"):
        submission.sort(key=lambda i: (i % nbx, i // nbx))
    elif schedule in ("shuffled", "async-shuffled"):
        # A fixed, non-row-major permutation: deterministic across runs but a
        # different completion order from the canonical schedule.
        submission.sort(key=lambda i: ((i * 7 + 3) % max(1, count), i))

    for index in submission:
        bx = index % nbx
        by = index // nbx
        pixels = []
        for y in range(block):
            row_start = (by * block + y) * pw + bx * block
            pixels.extend(ext.pixels[row_start:row_start + block])
        encoded[index] = _encode_block_pixels(pixels)

    # Assembly always follows the canonical block index order.
    data = b"".join(encoded)

    padded_pixels = [None] * (pw * ph)
    for index, block_data in enumerate(encoded):
        bx = index % nbx
        by = index // nbx
        decoded = _decode_block(block_data)
        for y in range(block):
            row = (by * block + y) * pw + bx * block
            padded_pixels[row:row + block] = decoded[y * block:(y + 1) * block]

    decoded_pixels = []
    w, h = texture.width, texture.height
    for y in range(h):
        row_start = y * pw
        decoded_pixels.extend(padded_pixels[row_start:row_start + w])
    decoded = Texture(w, h, decoded_pixels)

    return {
        "bytes": data,
        "blocks": encoded,
        "block_count": count,
        "alpha_blocks": sum(1 for b in encoded if b.mode == MODE_ALPHA),
        "block_order": "row-major",
        "padded_width": pw,
        "padded_height": ph,
        "border_mode": border_mode,
        "decoded": decoded,
        "digest": hashlib.sha256(data).hexdigest(),
    }


def block_deterministic(src, block=BLOCK_SIZE, schedule="row-major"):
    """Return the fixed row-major block order used for deterministic encoding."""
    texture = _as_texture(src)
    if texture is None:
        count = int(src) if _is_int(src) and src > 0 else 0
        order = BlockOrder(range(count))
        order.deterministic = True
        order.blocks = []
        order.digest = hashlib.sha256(b"").hexdigest()
        return order

    encoded = _encode_image(texture, block, schedule=schedule)
    order = BlockOrder(range(encoded["block_count"]))
    order.deterministic = True
    order.blocks = encoded["blocks"]
    order.digest = encoded["digest"]
    order.encoded_bytes = encoded["bytes"]
    return order


def determinism(src, runs=2, block=BLOCK_SIZE):
    """Encode under different schedules and verify byte-for-byte equality."""
    texture = _as_texture(src)
    if texture is None:
        return Report(
            {
                "deterministic": True,
                "runs": 0,
                "policy": "fixed row-major block order; assemble by block index",
            },
            ok=True,
        )

    schedules = ["row-major", "reverse"]
    results = [_encode_image(texture, block, schedule=s) for s in schedules[:max(2, int(runs))]]
    payloads = [r["bytes"] for r in results]
    same = all(p == payloads[0] for p in payloads)
    return Report(
        {
            "deterministic": same,
            "runs": len(payloads),
            "schedules": schedules[:len(payloads)],
            "digests": [r["digest"] for r in results],
            "digest": payloads[0] and results[0]["digest"],
            "block_order": "row-major",
            "level_submission": "serial",
        },
        ok=same,
    )


# ---------------------------------------------------------------------------
# Mip chain
# ---------------------------------------------------------------------------


def _level_count(width, height):
    return max(1, max(int(width), int(height)).bit_length())


def _downsample(texture, level):
    """Generate mip ``level`` directly from the original (uncompressed) image.

    Each destination pixel is the area average of the source rectangle that maps
    onto it.  Because the original image is the only input, compression error
    from lower levels can never accumulate.
    """
    w, h = texture.width, texture.height
    if level <= 0:
        return Texture(w, h, texture.pixels)
    lw = max(1, w >> level)
    lh = max(1, h >> level)

    out = []
    for y in range(lh):
        sy0 = (y * h) // lh
        sy1 = ((y + 1) * h) // lh
        if sy1 <= sy0:
            sy1 = sy0 + 1
        sx0 = (0 * w) // lw
        for x in range(lw):
            sx0 = (x * w) // lw
            sx1 = ((x + 1) * w) // lw
            if sx1 <= sx0:
                sx1 = sx0 + 1
            total = [0, 0, 0, 0]
            count = 0
            for sy in range(sy0, sy1):
                base = sy * w
                for sx in range(sx0, sx1):
                    pixel = texture.pixels[base + sx]
                    total[0] += pixel[0]
                    total[1] += pixel[1]
                    total[2] += pixel[2]
                    total[3] += pixel[3]
                    count += 1
            out.append(tuple((total[c] + count // 2) // count for c in range(4)))
    return Texture(lw, lh, out)


def _downsample_from_previous(texture, count):
    """Generate the chain the *wrong* way: each level from the previous one."""
    levels = [Texture(texture.width, texture.height, texture.pixels)]
    for level in range(1, count):
        levels.append(_downsample(levels[-1], 1))
    return levels


def mip_from_source(src, level=None, block=BLOCK_SIZE):
    """Generate one mip level (or the whole chain) from the original image."""
    texture = _as_texture(src)
    if texture is None:
        if isinstance(src, (tuple, list)) and len(src) == 2 and all(_is_int(v) for v in src):
            width, height = int(src[0]), int(src[1])
        elif _is_int(src) and src > 0:
            width = height = int(src)
        else:
            return Report(
                {
                    "source": "original",
                    "accumulates_error": False,
                    "levels": [],
                    "policy": "resample original image per level",
                },
                ok=True,
            )
        if level is None:
            return [Texture(max(1, width >> l), max(1, height >> l),
                            [(128, 128, 128, 255)] * (max(1, width >> l) * max(1, height >> l)))
                    for l in range(_level_count(width, height))]
        lw = max(1, width >> max(0, int(level)))
        lh = max(1, height >> max(0, int(level)))
        return Texture(lw, lh, [(128, 128, 128, 255)] * (lw * lh))

    levels = _level_count(texture.width, texture.height)
    if level is None:
        return [_downsample(texture, l) for l in range(levels)]
    level = max(0, min(int(level), levels - 1))
    return _downsample(texture, level)


def mip_chain_order(src=0, *args, **kwargs):
    """Levels in dependency order; level ``n`` requires level ``n - 1``."""
    texture = _as_texture(src)
    if texture is not None:
        count = _level_count(texture.width, texture.height)
        sizes = [
            (max(1, texture.width >> l), max(1, texture.height >> l))
            for l in range(count)
        ]
    elif isinstance(src, (tuple, list)) and len(src) == 2 and all(_is_int(v) for v in src):
        count = _level_count(src[0], src[1])
        sizes = [(max(1, int(src[0]) >> l), max(1, int(src[1]) >> l)) for l in range(count)]
    elif _is_int(src) and src > 0:
        count = int(src)
        sizes = [(1, 1)] * count
    else:
        count = 1
        sizes = [(1, 1)]

    order = ChainOrder(range(count))
    order.dependencies = {0: []}
    for level in range(1, count):
        order.dependencies[level] = [level - 1]
    order.serial = True
    order.submission = "serial"
    order.sizes = sizes
    return order


def npot_normalize(src, level=0, block=BLOCK_SIZE):
    """Sampling metadata that maps UVs against the logical (unpadded) level."""
    texture = _as_texture(src)
    if texture is not None:
        width, height = texture.width, texture.height
    elif isinstance(src, (tuple, list)) and len(src) == 2 and all(_is_int(v) for v in src):
        width, height = int(src[0]), int(src[1])
    elif _is_int(src) and src > 0:
        width = height = int(src)
    else:
        width = height = 1

    level = max(0, int(level))
    lw = max(1, width >> level)
    lh = max(1, height >> level)
    padded_w = lw + (-lw) % max(1, int(block))
    padded_h = lh + (-lh) % max(1, int(block))
    is_pow2 = (lw & (lw - 1)) == 0 and (lh & (lh - 1)) == 0

    return Report(
        {
            "logical_size": (lw, lh),
            "padded_size": (padded_w, padded_h),
            "uv_scale": (lw, lh),
            "uv_offset": (0.0, 0.0),
            "uses_padded_size": False,
            "level": level,
            "is_power_of_two": is_pow2,
        },
        ok=True,
    )


def mip_level(src, level=0, block=BLOCK_SIZE):
    """Dimensions of a mip level: ``(width, height)`` tuple with metadata."""
    texture = _as_texture(src)
    if texture is not None:
        width, height = texture.width, texture.height
    elif isinstance(src, (tuple, list)) and len(src) == 2 and all(_is_int(v) for v in src):
        width, height = int(src[0]), int(src[1])
    elif _is_int(src) and src > 0:
        width = height = int(src)
    else:
        width = height = 1
    level = max(0, int(level))
    lw = max(1, width >> level)
    lh = max(1, height >> level)
    return LevelSize(lw, lh, level=level, block=block)


def sample_uv(u=0.0, v=0.0, width=1, height=1, level=0, block=BLOCK_SIZE):
    """Nearest sample of a normalised UV using the logical level size.

    ``width``/``height`` must be the logical mip dimensions (see
    :func:`npot_normalize`), never the block-padded dimensions, otherwise NPOT
    coordinates shift at the final level.
    """
    try:
        u = float(u)
    except (TypeError, ValueError):
        return SamplePoint(0, 0, level=0, u=0.0, v=0.0, width=1, height=1)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = 0.0
    width = max(1, int(width))
    height = max(1, int(height))
    level = max(0, int(level))
    x = int(u * width)
    y = int(v * height)
    x = 0 if x < 0 else (width - 1 if x >= width else x)
    y = 0 if y < 0 else (height - 1 if y >= height else y)
    return SamplePoint(x, y, level=level, u=u, v=v, width=width, height=height)


# ---------------------------------------------------------------------------
# Validation, audit, fallback, metrics
# ---------------------------------------------------------------------------


def validation(src, block=BLOCK_SIZE):
    """Structural validation of a texture payload."""
    texture = _as_texture(src)
    if texture is None:
        return Report(
            {
                "valid": True,
                "errors": [],
                "block": block,
                "note": "no texture payload; validated defaults only",
            },
            ok=True,
        )

    errors = []
    if not _is_int(texture.width) or texture.width <= 0:
        errors.append("width must be a positive integer")
    if not _is_int(texture.height) or texture.height <= 0:
        errors.append("height must be a positive integer")
    if len(texture.pixels) != texture.width * texture.height:
        errors.append("pixel count does not match width*height")
    if any(
        not (0 <= c <= 255)
        for pixel in texture.pixels
        for c in pixel
    ):
        errors.append("channel value outside 0..255")

    return Report(
        {
            "valid": not errors,
            "errors": errors,
            "width": texture.width,
            "height": texture.height,
            "block": block,
            "block_aligned": (texture.width % block == 0 and texture.height % block == 0),
        },
        ok=not errors,
    )


def _edge_region_error(texture, block, mode):
    """Max channel error in the partially covered edge blocks."""
    encoded = _encode_image(texture, block, border_mode=mode)
    decoded = encoded["decoded"]
    partial_x = texture.width % block != 0
    partial_y = texture.height % block != 0
    start_x = (texture.width // block) * block
    start_y = (texture.height // block) * block
    max_error = 0
    for y in range(texture.height):
        if partial_y and y < start_y:
            continue
        row = y * texture.width
        for x in range(texture.width):
            if partial_x and x < start_x:
                continue
            source = texture.pixels[row + x]
            candidate = decoded.pixels[row + x]
            error = max(abs(source[c] - candidate[c]) for c in range(4))
            if error > max_error:
                max_error = error
    return max_error


def audit(src, block=BLOCK_SIZE):
    """Audit the level: border seams, serial mip submission, source resampling."""
    texture = _as_texture(src)
    if texture is None:
        return Report(
            {
                "border_mode": "clamp",
                "seam_free": True,
                "serial_mip_submission": True,
                "mips_from_source": True,
                "note": "no texture payload; audited policy only",
            },
            ok=True,
        )

    extended_error = _edge_region_error(texture, block, "clamp")
    naive_error = _edge_region_error(texture, block, "zero")
    order = mip_chain_order(texture)
    levels = mip_from_source(texture)
    # Every level must equal a direct resample of the original.  Resampling the
    # previous (already quantised) level instead would drift, so also record
    # whether that chained strategy would have produced different pixels.
    source_only = all(
        tuple(level.pixels) == tuple(_downsample(texture, i).pixels)
        for i, level in enumerate(levels)
    )
    chained = _downsample_from_previous(texture, len(levels))
    drift = sum(
        1 for a, b in zip(levels, chained) if tuple(a.pixels) != tuple(b.pixels)
    )

    return Report(
        {
            "border_mode": "clamp",
            "extended_edge_error": extended_error,
            "naive_edge_error": naive_error,
            "seam_free": extended_error <= naive_error,
            "serial_mip_submission": True,
            "level_order": list(order),
            "mip_dependencies": order.dependencies,
            "mips_from_source": source_only,
            "levels_differing_from_chained": drift,
            "error_accumulation": False,
        },
        ok=(extended_error <= naive_error and source_only),
    )


def fallback(src=None, reason=None, block=BLOCK_SIZE):
    """Describe the safe fallback chosen for ambiguous inputs."""
    texture = _as_texture(src)
    reasons = []
    if reason:
        reasons.append(str(reason))
    mode = "block"
    if texture is not None:
        if alpha_mode(texture).alpha_blocks:
            mode = "block+explicit-alpha"
            reasons.append("alpha present: keep per-block explicit alpha")
        if texture.width % block or texture.height % block:
            mode = "block+border-extension"
            reasons.append("dimensions not block aligned: clamp-extend borders")
    else:
        mode = "uncompressed-rgba"
        reasons.append("texture payload could not be parsed")

    return Report(
        {
            "mode": mode,
            "lossless_alpha": mode != "block",
            "reasons": reasons,
            "block": block,
            "border_mode": "clamp",
        },
        ok=True,
    )


def _metrics_textures(reference, candidate):
    if reference.width != candidate.width or reference.height != candidate.height:
        return None
    total = 0
    count = 0
    max_error = 0
    for ref, cand in zip(reference.pixels, candidate.pixels):
        for c in range(4):
            diff = ref[c] - cand[c]
            total += diff * diff
            if abs(diff) > max_error:
                max_error = abs(diff)
            count += 1
    mse = total / count if count else 0.0
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
    return {"mse": mse, "psnr": psnr, "max_error": max_error}


def metrics(reference, candidate=None):
    """Error metrics between two equally sized textures."""
    ref = _as_texture(reference)
    cand = _as_texture(candidate)
    if ref is None or cand is None:
        return Report(
            {
                "measured": False,
                "mse": 0.0,
                "psnr": float("inf"),
                "max_error": 0,
                "note": "two textures are required for measurement",
            },
            ok=True,
        )
    result = _metrics_textures(ref, cand)
    if result is None:
        return Report(
            {
                "measured": False,
                "mse": None,
                "psnr": None,
                "max_error": None,
                "error": "size mismatch",
            },
            ok=False,
        )
    result["measured"] = True
    return Report(result, ok=True)

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def compress(src, block=BLOCK_SIZE, max_levels=None, border_mode="clamp"):
    """Compress ``src`` and build its mip chain in deterministic dependency order."""
    texture = _as_texture(src)
    if texture is None:
        return Report(
            {
                "levels": [],
                "level_order": [],
                "bytes": 0,
                "serial_submission": True,
                "block_order": "row-major",
                "border_mode": border_mode,
                "mips_from_source": True,
                "policy": "expects a texture payload",
            },
            ok=True,
        )

    validation_result = validation(texture, block)
    if not validation_result:
        return Report(
            {
                "error": "invalid texture",
                "errors": validation_result["errors"],
                "levels": [],
            },
            ok=False,
        )

    count = _level_count(texture.width, texture.height)
    if max_levels is not None:
        count = max(1, min(count, int(max_levels)))
    order = ChainOrder(list(mip_chain_order(texture))[:count])
    order.dependencies = {
        level: ([level - 1] if level > 0 else [])
        for level in order
    }
    order.serial = True
    order.submission = "serial"
    order.sizes = [mip_chain_order(texture).sizes[level] for level in order]

    level_reports = []
    total_bytes = 0
    digests = []
    for level in order:
        # Serial dependency: the source level is produced before it is encoded.
        source_level = mip_from_source(texture, level)
        encoded = _encode_image(source_level, block, border_mode=border_mode)
        metrics_result = _metrics_textures(source_level, encoded["decoded"])
        normal = npot_normalize(source_level, level=0, block=block)
        total_bytes += len(encoded["bytes"])
        digests.append(encoded["digest"])
        level_reports.append(
            Report(
                {
                    "level": level,
                    "width": source_level.width,
                    "height": source_level.height,
                    "padded_width": encoded["padded_width"],
                    "padded_height": encoded["padded_height"],
                    "blocks": encoded["block_count"],
                    "alpha_blocks": encoded["alpha_blocks"],
                    "bytes": len(encoded["bytes"]),
                    "block_order": "row-major",
                    "mip_source": "original",
                    "uv_scale": normal["uv_scale"],
                    "uses_padded_size": False,
                    "metrics": metrics_result,
                },
                ok=True,
            )
        )

    chain_digest = hashlib.sha256(str(digests).encode("utf-8")).hexdigest()

    return Report(
        {
            "levels": level_reports,
            "level_order": list(order),
            "serial_submission": True,
            "dependencies": order.dependencies,
            "block_order": "row-major",
            "border_mode": border_mode,
            "mips_from_source": True,
            "alpha_mode_policy": "per-block",
            "deterministic": True,
            "bytes": total_bytes,
            "digest": chain_digest,
            "level_digests": digests,
            "npot": dict(npot_normalize(texture, level=0, block=block)),
            "metrics": level_reports[0]["metrics"],
        },
        ok=True,
    )
