# Copyright (c) 2026, Renaud Allard <renaud@allard.it>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Ink: where the marks on a page are, and what each one looks like.

A page is flat colour almost everywhere, and every mark sits on one of those
flat colours. The large flat regions are the backgrounds; every other pixel
belongs to the nearest one, and how far its colour lies from that background
is how much ink it carries. Marks are the connected runs of ink, each cut out
and normalized so that its background reads 0 and its ink 1 whatever the two
colours were: dark text on white, white text on a coloured band and coloured
text on a tinted cell all come out as the same shape.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Pixels = npt.NDArray[np.uint8]
Mask = npt.NDArray[np.bool_]
Patch = npt.NDArray[np.float32]
Labels = npt.NDArray[np.int32]
Boxes = npt.NDArray[np.int64]
Colour = tuple[int, int, int]

FLAT_TOLERANCE = 3
"""Neighbouring pixels within this many levels of each other, per channel, are flat colour."""

MIN_CONTRAST = 80
"""Sum of the per-channel distances a mark needs from its background to count as ink.

The faintest text on the cards, the white title on a lilac band, is 92
away from its background; the blended edges of coloured cells are judged
by what lies on either side of them, not by how faint they are.
"""

MARGIN = 1
"""Pixels of anti-aliasing kept around a mark's core."""

BACKGROUND_AREA = 900
"""Pixels a flat region needs to be a background rather than the inside of a letter."""

BACKGROUND_SIDE = 20
"""Pixels a background region must span both ways, and the side of the solid square that tells it from a stroke."""

_LETTER_HEIGHT = 5

MAX_MARK = 400
"""Marks wider or taller than this are boxes and pictures, never glyphs, and are not cut out."""

_MIN_CORE_AREA = 2
_SMALL = 512


@dataclass(slots=True)
class Blob:
    """One mark: its pixel box on the canvas and the normalized patch inside it.

    ``x1`` and ``y1`` are exclusive. The box includes a one pixel anti-aliasing
    margin around the mark's core, so ``patch`` has shape ``(y1 - y0, x1 - x0)``.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    patch: Patch
    bg: Colour
    ink: Colour

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(slots=True)
class Ink:
    """A page measured against its backgrounds: how much ink every pixel carries.

    ``distance`` is each pixel's colour distance from its own background,
    ``background`` the label of that background and ``colours`` the colour of
    each label.
    """

    pixels: Pixels
    distance: npt.NDArray[np.int32]
    background: Labels
    colours: npt.NDArray[np.int16]

    @classmethod
    def of(cls, pixels: Pixels, *, tolerance: int = FLAT_TOLERANCE) -> Ink:
        """Measure ``pixels``: find the backgrounds and every pixel's distance from its own."""
        flat = ~edge_mask(pixels, tolerance)
        regions, count, boxes = label(flat, eight=False)
        colours = _region_colours(pixels, regions, count)
        big = np.zeros(count + 1, dtype=np.bool_)
        wide = (boxes[:, 2] - boxes[:, 0]) >= BACKGROUND_SIDE
        tall = (boxes[:, 3] - boxes[:, 1]) >= BACKGROUND_SIDE
        big[1:] = (boxes[:, 4] >= BACKGROUND_AREA) & wide & tall
        ink = cls._against(pixels, regions, colours, big)
        doubtful = np.nonzero(big & ~_solid(flat, regions, count))[0].tolist()
        letters = [region for region in doubtful if ink._letter(boxes[region - 1])]
        if letters:
            big[letters] = False
            ink = cls._against(pixels, regions, colours, big)
        return ink

    @classmethod
    def _against(
        cls,
        pixels: Pixels,
        regions: Labels,
        colours: npt.NDArray[np.int16],
        big: npt.NDArray[np.bool_],
    ) -> Ink:
        """Every pixel's distance from its background, the ``big`` regions being the backgrounds."""
        background = _background_map(regions, big)
        distance = (
            np.abs(pixels.astype(np.int16) - colours[background]).sum(axis=2).astype(np.int32)
        )
        distance[background == 0] = 0
        return cls(pixels, distance, background, colours)

    def _letter(self, box: npt.NDArray[np.int32]) -> bool:
        """Whether a background region without a solid square in it is the inside of a letter.

        The stem of a bold title letter is a flat region of many pixels,
        wide and tall by its box, and were it a background the letter would
        fall apart into the slivers of its own edges. A coloured cell of the
        same size, with no solid square in it either because it is full of
        text, carries marks that are not edges: that text. Only the pixels
        around the region are looked at, cut out with a margin so that the
        edges of the region see what lies on their far side.
        """
        x0, y0, x1, y1 = (int(v) for v in box[:4])
        pad = 2 * MARGIN
        left, top = max(x0 - pad, 0), max(y0 - pad, 0)
        crop = Ink(
            self.pixels[top : y1 + pad, left : x1 + pad],
            self.distance[top : y1 + pad, left : x1 + pad],
            self.background[top : y1 + pad, left : x1 + pad],
            self.colours,
        )
        for blob in crop.blobs():
            if (
                blob.x0 + left >= x0 - MARGIN
                and blob.y0 + top >= y0 - MARGIN
                and blob.x1 + left <= x1 + MARGIN
                and blob.y1 + top <= y1 + MARGIN
                and blob.height >= _LETTER_HEIGHT
                and not crop.between(blob)
            ):
                return False
        return True

    def between(self, blob: Blob) -> bool:
        """Whether the mark is the anti-aliased edge between two backgrounds, not a glyph.

        Where the rounded end of a coloured cell runs nearly vertical, the
        pixels blending its colour into the page around it form a sliver a
        few pixels wide, one background on its left and another on its right,
        and its own colour lies between the two. Text runs across the edge of
        a faint band now and then, and a glyph there has a background on each
        side too, but its ink is nothing like either.
        """
        height, width = self.background.shape
        sides = (
            (
                self.background[blob.y0 : blob.y1, max(blob.x0 - 1, 0)],
                self.background[blob.y0 : blob.y1, min(blob.x1, width - 1)],
            ),
            (
                self.background[max(blob.y0 - 1, 0), blob.x0 : blob.x1],
                self.background[min(blob.y1, height - 1), blob.x0 : blob.x1],
            ),
        )
        ink = np.array(blob.ink, dtype=np.int16)
        for one, other in sides:
            if (one != other).sum() * 2 < len(one):
                continue
            a = self.colours[_mode(one)]
            b = self.colours[_mode(other)]
            apart = int(np.abs(a - b).max())
            if (
                apart
                and max(np.abs(ink - a).max(), np.abs(ink - b).max()) <= apart + FLAT_TOLERANCE
            ):
                return True
        return False

    def blobs(self, *, min_contrast: int = MIN_CONTRAST, max_mark: int = MAX_MARK) -> list[Blob]:
        """Every mark on the page, each with its normalized patch.

        Ink is first gathered at half the minimum contrast, which joins a mark
        to the anti-aliasing around it, then each gathering is cut at half of
        its own contrast, which separates marks that only touch through that
        anti-aliasing.
        """
        height, width = self.distance.shape
        gathered, count, boxes = label(self.distance >= min_contrast // 2, eight=True)
        if count == 0:
            return []
        levels = np.zeros(count + 1, dtype=np.int32)
        np.maximum.at(levels, gathered.reshape(-1), self.distance.reshape(-1))
        out: list[Blob] = []
        for index, (x0, y0, x1, y1, _area) in enumerate(boxes.tolist(), start=1):
            level = int(levels[index])
            if level < min_contrast or x1 - x0 > max_mark or y1 - y0 > max_mark:
                continue
            window = self.distance[y0:y1, x0:x1]
            core = (gathered[y0:y1, x0:x1] == index) & (window * 2 >= level)
            cores, _found, core_boxes = label(core, eight=False)
            for core_index, (cx0, cy0, cx1, cy1, core_area) in enumerate(
                core_boxes.tolist(), start=1
            ):
                if core_area < _MIN_CORE_AREA:
                    continue
                own = int(window[cy0:cy1, cx0:cx1][cores[cy0:cy1, cx0:cx1] == core_index].max())
                box = (
                    max(x0 + cx0 - MARGIN, 0),
                    max(y0 + cy0 - MARGIN, 0),
                    min(x0 + cx1 + MARGIN, width),
                    min(y0 + cy1 + MARGIN, height),
                )
                blob = self._cut(box, own)
                if blob is not None:
                    out.append(blob)
        return sorted(out, key=lambda b: (b.y0, b.x0))

    def blob(
        self, box: tuple[int, int, int, int], *, min_contrast: int = MIN_CONTRAST
    ) -> Blob | None:
        """The mark inside ``box``, normalized against the ink it holds."""
        height, width = self.distance.shape
        x0, y0 = max(box[0], 0), max(box[1], 0)
        x1, y1 = min(box[2], width), min(box[3], height)
        if x1 <= x0 or y1 <= y0:
            return None
        level = int(self.distance[y0:y1, x0:x1].max())
        if level < min_contrast:
            return None
        return self._cut((x0, y0, x1, y1), level)

    def _cut(self, box: tuple[int, int, int, int], level: int) -> Blob | None:
        x0, y0, x1, y1 = box
        window = self.distance[y0:y1, x0:x1]
        patch = np.clip(window.astype(np.float32) / float(level), 0.0, 1.0)
        labels = self.background[y0:y1, x0:x1].reshape(-1)
        labels = labels[labels > 0]
        if len(labels) == 0:
            return None
        bg_label = int(np.bincount(labels).argmax())
        bg = tuple(int(v) for v in self.colours[bg_label])
        iy, ix = np.unravel_index(int(window.argmax()), window.shape)
        pixel = self.pixels[y0 + int(iy), x0 + int(ix)]
        ink = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
        return Blob(x0, y0, x1, y1, patch, (bg[0], bg[1], bg[2]), ink)


def find_blobs(pixels: Pixels) -> list[Blob]:
    """Every mark on ``pixels``, see :meth:`Ink.blobs`."""
    return Ink.of(pixels).blobs()


def edge_mask(pixels: Pixels, tolerance: int = FLAT_TOLERANCE) -> Mask:
    """True where a pixel differs from a 4-neighbour by more than ``tolerance``."""
    values = pixels.astype(np.int16)
    mask = np.zeros(pixels.shape[:2], dtype=np.bool_)
    vertical = np.abs(values[1:] - values[:-1]).max(axis=2) > tolerance
    mask[1:] |= vertical
    mask[:-1] |= vertical
    horizontal = np.abs(values[:, 1:] - values[:, :-1]).max(axis=2) > tolerance
    mask[:, 1:] |= horizontal
    mask[:, :-1] |= horizontal
    return mask


def label(mask: Mask, eight: bool = True) -> tuple[Labels, int, Boxes]:
    """Connected components of ``mask``: labels 1..n (0 for background) and per label its box.

    Works on runs of set pixels rather than pixels: every run is keyed by its
    row so that one sorted search per run finds the runs of the row above it
    overlaps, and the runs are then joined with a vectorized union-find. The
    boxes come from the runs too: x0, y0, x1, y1 (exclusive) and pixel area.
    """
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    diff = np.diff(padded, axis=1)
    rows, starts = np.nonzero(diff == 1)
    _, ends = np.nonzero(diff == -1)
    count = len(rows)
    if count == 0:
        return labels, 0, np.zeros((0, 5), dtype=np.int64)
    reach = 1 if eight else 0
    stride = width + 4
    keyed_starts = rows * stride + starts
    keyed_ends = rows * stride + ends
    current = np.nonzero(rows > 0)[0]
    row_above = (rows[current] - 1) * stride
    lo = np.searchsorted(keyed_ends, row_above + starts[current] - reach, side="right")
    hi = np.searchsorted(keyed_starts, row_above + ends[current] + reach, side="left")
    counts = np.maximum(hi - lo, 0)
    total = int(counts.sum())
    if total == 0:
        roots = np.arange(count, dtype=np.int64)
    else:
        keep = counts > 0
        lo, counts, current = lo[keep], counts[keep], current[keep]
        left = np.repeat(current, counts)
        offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        right = np.repeat(lo, counts) + offsets
        roots = _union(count, left, right)
    _, run_labels = np.unique(roots, return_inverse=True)
    lengths = ends - starts
    total_pixels = int(lengths.sum())
    flat = np.repeat(rows * width + starts, lengths) + (
        np.arange(total_pixels) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    )
    labels.reshape(-1)[flat] = np.repeat(run_labels + 1, lengths)
    found = int(run_labels.max()) + 1
    boxes = np.zeros((found, 5), dtype=np.int64)
    boxes[:, 0] = width
    boxes[:, 1] = height
    np.minimum.at(boxes[:, 0], run_labels, starts)
    np.minimum.at(boxes[:, 1], run_labels, rows)
    np.maximum.at(boxes[:, 2], run_labels, ends)
    np.maximum.at(boxes[:, 3], run_labels, rows + 1)
    np.add.at(boxes[:, 4], run_labels, lengths)
    return labels, found, boxes


def _union(count: int, u: npt.NDArray[np.int64], v: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Root of every run after joining the pairs ``u[i]``/``v[i]``."""
    parent = np.arange(count, dtype=np.int64)
    while True:
        ru, rv = parent[u], parent[v]
        differ = ru != rv
        if not differ.any():
            return parent
        ru, rv = ru[differ], rv[differ]
        np.minimum.at(parent, np.maximum(ru, rv), np.minimum(ru, rv))
        while True:
            hopped = parent[parent]
            if np.array_equal(hopped, parent):
                break
            parent = hopped


def _region_colours(pixels: Pixels, regions: Labels, count: int) -> npt.NDArray[np.int16]:
    """The mean colour of every flat region, row 0 unused."""
    flat = regions.reshape(-1)
    sums = np.zeros((count + 1, 3), dtype=np.int64)
    for channel in range(3):
        sums[:, channel] = np.bincount(
            flat, weights=pixels[:, :, channel].reshape(-1), minlength=count + 1
        )
    areas = np.maximum(np.bincount(flat, minlength=count + 1), 1)
    return np.rint(sums / areas[:, None]).astype(np.int16)


def _solid(flat: npt.NDArray[np.bool_], regions: Labels, count: int) -> npt.NDArray[np.bool_]:
    """Which regions hold a solid square of flat pixels ``BACKGROUND_SIDE`` on a side.

    The stem of a bold letter set large enough is a flat region of many
    pixels, wide and tall by its box; it is a stroke all the same, and no
    square of that size fits in it. A background has room for one.
    """
    side = BACKGROUND_SIDE
    height, width = flat.shape
    solid = np.zeros(count + 1, dtype=np.bool_)
    if height < side or width < side:
        return solid
    sums = np.zeros((height + 1, width + 1), dtype=np.int32)
    sums[1:, 1:] = flat.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
    windows = sums[side:, side:] - sums[:-side, side:] - sums[side:, :-side] + sums[:-side, :-side]
    full = windows == side * side
    solid[np.unique(regions[: height - side + 1, : width - side + 1][full])] = True
    solid[0] = False
    return solid


def _background_map(regions: Labels, big: npt.NDArray[np.bool_]) -> Labels:
    """Every pixel's background: its own flat region when that is one, else the nearest one along its row."""
    height, width = regions.shape
    assigned = np.where(big[regions], regions, 0)
    has = assigned > 0
    columns = np.arange(width)
    left = np.maximum.accumulate(np.where(has, columns, -1), axis=1)
    right = np.minimum.accumulate(np.where(has, columns, width)[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(height)[:, None]
    from_left = assigned[rows, np.clip(left, 0, width - 1)]
    from_right = assigned[rows, np.clip(right, 0, width - 1)]
    nearer_left = (left >= 0) & ((right >= width) | (columns - left <= right - columns))
    return np.where(nearer_left, from_left, np.where(right < width, from_right, 0)).astype(np.int32)


def _mode(labels: npt.NDArray[np.int32]) -> int:
    """The most common label of a strip of the background map."""
    values, counts = np.unique(labels, return_counts=True)
    return int(values[int(np.argmax(counts))])


def mode_colour(colours: npt.NDArray[np.uint8]) -> Colour:
    """The most common colour among ``colours`` (an (n, 3) array)."""
    if len(colours) == 0:
        return (255, 255, 255)
    values = colours.astype(np.int64)
    packed = (values[:, 0] << 16) | (values[:, 1] << 8) | values[:, 2]
    if len(packed) <= _SMALL:
        top = Counter(packed.tolist()).most_common(1)[0][0]
    else:
        distinct, counts = np.unique(packed, return_counts=True)
        top = int(distinct[int(counts.argmax())])
    return ((top >> 16) & 255, (top >> 8) & 255, top & 255)
