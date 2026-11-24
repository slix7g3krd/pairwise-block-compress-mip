"""Behavioural tests for the seven block-compression / mip fixes."""

import graphics.block_compress_mip as mod


def _texture(w=6, h=4, alpha=False):
    pixels = []
    for y in range(h):
        for x in range(w):
            a = 0 if alpha and x == w - 1 else 255
            pixels.append(((x * 40) % 256, (y * 60) % 256, (x * y * 17) % 256, a))
    return mod.Texture(w, h, pixels)


def test_border_extend_pads_with_real_neighbours():
    tex = _texture()
    extended = mod.border_extend(tex)
    assert extended.padding == (0, 0, 2, 0)
    assert (extended.width, extended.height) == (8, 4)
    # Padded columns 6 and 7 repeat the last real column (clamp), not zero.
    for y in range(extended.height):
        row = extended.pixels[y * extended.width:(y + 1) * extended.width]
        assert row[6] == row[7] == row[5]
        assert row[6] != (0, 0, 0, 0)


def test_border_extension_reduces_edge_error_vs_zero_padding():
    tex = _texture()
    report = mod.audit(tex)
    assert report.extended_edge_error < report.naive_edge_error


def test_alpha_blocks_use_alpha_mode_even_when_rgb_has_no_error():
    # RGB is flat, so an RGB-only metric would choose the opaque mode and drop
    # the transparency; alpha must decide the mode per block.
    block = [(10, 10, 10, 255)] * 15 + [(10, 10, 10, 0)]
    assert str(mod.alpha_mode(block)) == "alpha"
    encoded = mod.encode_block(block)
    assert encoded.mode == mod.MODE_ALPHA
    assert len(encoded) == len(mod.encode_block([(10, 10, 10, 255)] * 16)) + 16


def test_opaque_block_stays_opaque_and_hides_no_alpha():
    encoded = mod.encode_block([(10, 10, 10, 255)] * 16)
    assert encoded.mode == mod.MODE_OPAQUE
    assert all(pixel[3] == 255 for pixel in mod._decode_block(encoded))


def test_mip_chain_declares_serial_dependencies():
    order = mod.mip_chain_order(_texture())
    assert list(order) == list(range(len(order)))
    assert order.dependencies[0] == []
    assert order.dependencies[1] == [0]
    assert order.dependencies[2] == [1]
    assert order.serial is True


def test_npot_uv_uses_logical_not_padded_size():
    tex = _texture(w=6, h=4)
    level = mod.npot_normalize(tex)
    assert level.logical_size == (6, 4)
    assert level.padded_size == (8, 4)
    assert level.uv_scale == (6, 4)
    assert level.uses_padded_size is False
    # Last texel of a 6-wide level must be x == 5, not x == 7.
    point = mod.sample_uv(0.999, 0.999, 6, 4)
    assert (point.x, point.y) == (5, 3)


def test_mips_are_resampled_from_the_original():
    tex = mod.Texture(8, 8, [(i * 4, 255 - i * 4, i, 255) for i in range(64)])
    levels = mod.mip_from_source(tex)
    assert [(level.width, level.height) for level in levels] == [
        (8, 8), (4, 4), (2, 2), (1, 1)
    ]
    for index, level in enumerate(levels):
        assert tuple(level.pixels) == tuple(mod._downsample(tex, index).pixels)


def test_no_error_accumulation_on_partial_alpha_texture():
    reported = mod.audit(_texture(alpha=True))
    assert reported.mips_from_source is True
    assert reported.error_accumulation is False


def test_block_order_is_deterministic_across_completion_schedules():
    tex = _texture()
    digests = {
        schedule: mod._encode_image(tex, schedule=schedule)["digest"]
        for schedule in ("row-major", "reverse", "column-major", "shuffled")
    }
    assert len(set(digests.values())) == 1
    assert mod.block_deterministic(tex).digest == digests["row-major"]
    assert bool(mod.determinism(tex)) is True


def test_compress_submits_levels_serially_and_reports_level_metadata():
    report = mod.compress(_texture())
    assert report.serial_submission is True
    assert report.level_order == [0, 1, 2]
    assert report.dependencies == {0: [], 1: [0], 2: [1]}
    assert report.mips_from_source is True
    assert report.deterministic is True
    for level in report.levels:
        assert level.mip_source == "original"
        assert level.uses_padded_size is False
        assert (level.padded_width, level.padded_height) == (
            level.width + (-level.width) % 4,
            level.height + (-level.height) % 4,
        )


def test_validation_accepts_all_supported_texture_forms():
    flat = [(0, 0, 0, 255)] * 16
    rows = [[(0, 0, 0, 255)] * 4 for _ in range(4)]
    assert bool(mod.validation({"width": 4, "height": 4, "pixels": flat})
                )
    assert bool(mod.validation((4, 4, flat)))
    assert bool(mod.validation(rows))
    assert bool(mod.validation(mod.Texture(4, 4, flat)))
