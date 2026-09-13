"""Component labelling and blob extraction."""

from __future__ import annotations

import numpy as np

from ocr_price_cards.ink import Ink, edge_mask, label


def _reference(mask: np.ndarray, eight: bool) -> tuple[np.ndarray, int]:
    """Flood fill, the slow way, to compare against."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=int)
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            labels[y, x] = count
            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if (dy == 0 and dx == 0) or (not eight and dy and dx):
                            continue
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and mask[ny, nx]
                            and not labels[ny, nx]
                        ):
                            labels[ny, nx] = count
                            stack.append((ny, nx))
    return labels, count


def test_label_matches_flood_fill_on_random_masks() -> None:
    rng = np.random.default_rng(7)
    for _ in range(30):
        height, width = (int(v) for v in rng.integers(1, 24, 2))
        mask = rng.random((height, width)) < rng.uniform(0.2, 0.7)
        for eight in (True, False):
            labels, count, boxes = label(mask, eight=eight)
            expected, expected_count = _reference(mask, eight)
            assert count == expected_count
            assert (labels > 0).sum() == mask.sum()
            for value in range(1, count + 1):
                members = expected[labels == value]
                assert len(set(members.tolist())) == 1
                ys, xs = np.nonzero(labels == value)
                assert tuple(boxes[value - 1]) == (
                    xs.min(),
                    ys.min(),
                    xs.max() + 1,
                    ys.max() + 1,
                    len(ys),
                )


def test_label_of_empty_mask() -> None:
    labels, count, boxes = label(np.zeros((5, 5), dtype=bool))
    assert count == 0
    assert labels.sum() == 0
    assert boxes.shape == (0, 5)


def test_edge_mask_marks_both_sides_of_a_step() -> None:
    pixels = np.full((4, 6, 3), 255, dtype=np.uint8)
    pixels[:, 3:] = 0
    edges = edge_mask(pixels)
    assert edges[:, 2].all() and edges[:, 3].all()
    assert not edges[:, :2].any() and not edges[:, 4:].any()


def _canvas(background: tuple[int, int, int], ink: tuple[int, int, int]) -> np.ndarray:
    pixels = np.full((60, 80, 3), background, dtype=np.uint8)
    pixels[20:32, 10:14] = ink
    pixels[14:17, 10:14] = ink
    pixels[20:32, 30:40] = ink
    pixels[22:30, 32:38] = background
    return pixels


def test_blobs_are_normalized_whatever_the_colours() -> None:
    dark = Ink.of(_canvas((255, 255, 255), (8, 10, 38))).blobs()
    light = Ink.of(_canvas((66, 179, 162), (255, 255, 255))).blobs()
    assert [b.box for b in dark] == [b.box for b in light]
    assert len(dark) == 3
    for a, b in zip(dark, light, strict=True):
        assert np.array_equal(a.patch >= 0.5, b.patch >= 0.5)
    stem = next(b for b in dark if b.height > 10 and b.width < 8)
    assert stem.bg == (255, 255, 255) and stem.ink == (8, 10, 38)


def test_blob_of_box_normalizes_against_the_ink_it_holds() -> None:
    ink = Ink.of(_canvas((255, 255, 255), (0, 0, 0)))
    blob = ink.blob((8, 12, 16, 34))
    assert blob is not None
    assert blob.box == (8, 12, 16, 34)
    assert blob.patch.max() == 1.0
    assert ink.blob((0, 0, 5, 5)) is None
