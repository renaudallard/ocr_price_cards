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

"""Build a glyph library from cards that still carry a text layer.

A card with a text layer says exactly which character sits where. Rendering
it at the library's resolution and cutting the marks out gives every glyph as
the card prints it, already labelled. Rendering it again a fraction of a pixel
to the side gives the same glyph as another rasterizer would have placed it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt

from .ink import Blob, Ink
from .layout import Glyph, build_rows, build_words
from .library import Library, Template, quantize
from .pages import DEFAULT_DPI, Card, Page, TextChar
from .reader import MAX_GLYPH, PUNCTUATION, glyph_from_char
from .specimen import Specimen

SUBPIXEL = tuple((x / 3, y / 3) for y in range(3) for x in range(3))
"""Rendering offsets, in pixels, the library learns every glyph at."""

SIZE_MIN = 3.0
SIZE_MAX = 80.0
_ZONE_ABOVE = 1.25
_ZONE_BELOW = 0.45
_HIT = 0.3
_VERTICAL = 0.6
_COVER = 0.5
_SPAN = 0.8
_THIN = 0.3
_RUN_OVERLAP = 0.3
_SAME_LINE = 0.2
_LINE_STEP = 4
_BAND_ABOVE = 0.8
_BAND_BELOW = 0.35
_WIDE = 2.2
_TALLEST = 1.6
_WIDEST = 1.2
_IGNORED = frozenset("­​‌‍﻿")

Report = Callable[[str], None]


def train(
    cards: Iterable[tuple[str, bytes]],
    *,
    dpi: float = DEFAULT_DPI,
    shifts: tuple[tuple[float, float], ...] = SUBPIXEL,
    library: Library | None = None,
    report: Report | None = None,
) -> Library:
    """Learn every glyph of every card, at every sub-pixel offset, into ``library``."""
    lib = library if library is not None else Library(dpi)
    if abs(lib.dpi - dpi) > 1e-6:
        raise ValueError(f"the library is built at {lib.dpi} dpi, not {dpi}")
    for name, payload in cards:
        with Card(payload) as card:
            for shift in shifts:
                added = 0
                for index in range(len(card)):
                    added += learn_page(card.page(index, dpi=dpi, embedded=False, shift=shift), lib)
                if report is not None:
                    report(
                        f"{name} shift {shift[0]:.2f},{shift[1]:.2f}: {added} glyphs, {len(lib)} templates"
                    )
    return lib


def train_specimens(
    items: Iterable[Specimen],
    *,
    library: Library,
    dpi: float | None = None,
    shifts: tuple[tuple[float, float], ...] = SUBPIXEL,
    report: Report | None = None,
) -> Library:
    """Learn the glyphs of specimens, whose characters are known without reading a text layer.

    ``dpi`` is checked against the library's, as :func:`train` checks it;
    left out, the specimens are set at whatever the library was built at.
    """
    if dpi is not None and abs(library.dpi - dpi) > 1e-6:
        raise ValueError(f"the library is built at {library.dpi} dpi, not {dpi}")
    for item in items:
        with Card(item.payload) as card:
            for shift in shifts:
                added = 0
                for index in range(len(card)):
                    page = card.page(index, dpi=library.dpi, embedded=False, shift=shift)
                    page.chars = item.chars[index]
                    added += learn_page(page, library)
                if report is not None:
                    report(
                        f"{item.family} shift {shift[0]:.2f},{shift[1]:.2f}: {added} glyphs, {len(library)} templates"
                    )
    return library


def harvest_words(
    cards: Iterable[tuple[str, bytes]],
    library: Library,
    *,
    report: Report | None = None,
) -> int:
    """Add to the library's lexicon the words its own reading of the cards' pixels spells.

    The text layer is what the lexicon is first built from, but where a card
    overprints text, as table headers set at an angle do, that layer comes
    out scrambled while the pixels still read cleanly. The pixels give the
    words their shape, taken row by row as they were set rather than as
    pdfplumber chains the lines of a header cell into one; each glyph the
    reading is not sure of is settled by the character the text layer puts
    at that place, and a line on which anything was refused is not taken at
    all: a refused mark leaves the text, so the word it sat in would come
    back short and be learnt that way.
    """
    from .reader import read_page

    added = 0
    for name, payload in cards:
        with Card(payload) as card:
            for index in range(len(card)):
                page = card.page(index, dpi=library.dpi, embedded=False)
                result = read_page(page, library, text_layer=False, strict=False)
                zones = _char_zones(page)
                glyphs = [
                    glyph
                    for line in result.lines
                    if not line.unread
                    for word in line.words
                    for glyph in word.glyphs
                ]
                for word in (word for row in build_rows(glyphs) for word in row.words):
                    if not all(_settled(glyph, zones) for glyph in word.glyphs):
                        continue
                    text = word.text.strip(PUNCTUATION)
                    if (
                        len(text) > 1
                        and any(c.isalpha() for c in text)
                        and text not in library.words
                    ):
                        library.words.add(text)
                        added += 1
        if report is not None:
            report(f"{name}: {added} words added, {len(library.words)} in the lexicon")
    return added


def _char_zones(page: Page) -> list[tuple[float, float, float, float, str]]:
    zones: list[tuple[float, float, float, float, str]] = []
    for char in page.chars:
        if char.text.isspace():
            continue
        em = char.size * page.transform.sx
        x0, base = page.transform.to_px(char.x0, char.baseline)
        x1, _ = page.transform.to_px(char.x1, char.baseline)
        zones.append((x0, base - em, x1, base + _ZONE_BELOW * em, char.text))
    return zones


def _settled(glyph: Glyph, zones: list[tuple[float, float, float, float, str]]) -> bool:
    """Whether a glyph reads as one thing, or as the thing the text layer puts at its place."""
    if not glyph.alternatives:
        return True
    if glyph.box is None:
        return False
    bx0, by0, bx1, by1 = glyph.box
    best: tuple[float, str] | None = None
    for x0, y0, x1, y1, text in zones:
        shared = max(0.0, min(x1, bx1) - max(x0, bx0)) * max(0.0, min(y1, by1) - max(y0, by0))
        if shared > 0 and (best is None or shared > best[0]):
            best = (shared, text)
    if best is None or best[1] not in (glyph.text, *glyph.alternatives):
        return False
    glyph.text = best[1]
    glyph.alternatives = ()
    return True


def learn_page(page: Page, library: Library) -> int:
    """Add the glyphs of one rendered page to ``library``; returns how many were learnt."""
    chars = [
        char
        for char in page.chars
        if not char.text.isspace()
        and char.text not in _IGNORED
        and not char.text.startswith("(cid:")
        and SIZE_MIN <= char.size <= SIZE_MAX
    ]
    if not chars:
        return 0
    usable = _unambiguous(chars)
    scale = page.transform.sx
    zones = np.zeros((len(chars), 4), dtype=np.float64)
    for index, char in enumerate(chars):
        em = char.size * scale
        x0, base = page.transform.to_px(char.x0, char.baseline)
        x1, _ = page.transform.to_px(char.x1, char.baseline)
        zones[index] = (x0, base - _ZONE_ABOVE * em, x1, base + _ZONE_BELOW * em)
    ink = Ink.of(page.pixels)
    owned: list[tuple[tuple[int, ...], Blob]] = []
    for blob in ink.blobs():
        if blob.width > MAX_GLYPH or blob.height > MAX_GLYPH:
            continue
        owners = _owners(blob, chars, zones, scale)
        if owners is not None and all(usable[i] for i in owners):
            owned.append((owners, blob))
    added = 0
    for owners, blobs in _grouped(owned, chars).items():
        template = _template(page, ink, chars, owners, blobs)
        if template is not None:
            library.add(template)
            added += 1
    library.words.update(
        word.text.strip(PUNCTUATION)
        for word in build_words([glyph_from_char(char) for char in page.chars])
    )
    library.words.discard("")
    return added


def _grouped(
    owned: list[tuple[tuple[int, ...], Blob]], chars: list[TextChar]
) -> dict[tuple[int, ...], list[Blob]]:
    """The marks of each glyph together, characters that share a mark folded into one glyph.

    A pair of letters that touch is one mark owned by both; the counter of one
    of them is another mark owned by that letter alone. Both belong to the
    same template, labelled with both letters.
    """
    parent: dict[int, int] = {}

    def find(i: int) -> int:
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for owners, _blob in owned:
        for other in owners[1:]:
            parent[find(other)] = find(owners[0])
    members: dict[int, set[int]] = {}
    for owners, _blob in owned:
        members.setdefault(find(owners[0]), set()).update(owners)
    groups: dict[tuple[int, ...], list[Blob]] = {}
    for owners, blob in owned:
        key = tuple(sorted(members[find(owners[0])], key=lambda i: chars[i].x0))
        groups.setdefault(key, []).append(blob)
    return groups


def _unambiguous(chars: list[TextChar]) -> list[bool]:
    """False for characters drawn over another character on the same line, as in shadowed titles."""
    usable = [True] * len(chars)
    order = sorted(range(len(chars)), key=lambda i: chars[i].x0)
    for position, i in enumerate(order):
        a = chars[i]
        for j in order[position + 1 :]:
            b = chars[j]
            if b.x0 >= a.x1:
                break
            if abs(a.baseline - b.baseline) > _SAME_LINE * max(a.size, b.size):
                continue
            overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
            if overlap > _RUN_OVERLAP * min(a.x1 - a.x0, b.x1 - b.x0):
                usable[i] = usable[j] = False
    return usable


def _owners(
    blob: Blob, chars: list[TextChar], zones: npt.NDArray[np.float64], scale: float
) -> tuple[int, ...] | None:
    """The characters a mark is part of, or None when it belongs to none or to several lines."""
    dx = np.minimum(zones[:, 2], blob.x1) - np.maximum(zones[:, 0], blob.x0)
    dy = np.minimum(zones[:, 3], blob.y1) - np.maximum(zones[:, 1], blob.y0)
    widths = zones[:, 2] - zones[:, 0]
    hit = (dx >= _HIT * np.minimum(blob.width, widths)) & (dy >= _VERTICAL * blob.height)
    candidates = [int(i) for i in np.nonzero(hit)[0]]
    if not candidates:
        return None
    candidates = _one_line(blob, chars, candidates, scale)
    coverage = {i: float(dx[i]) / max(float(widths[i]), 1.0) for i in candidates}
    primary = max(candidates, key=lambda i: coverage[i])
    covered = [i for i in candidates if coverage[i] >= _COVER]
    em = chars[primary].size * scale
    if len(covered) > 1 and primary in covered:
        span = sum(float(widths[i]) for i in covered)
        if blob.width >= _SPAN * span and blob.height >= _THIN * em:
            return tuple(sorted(covered, key=lambda i: chars[i].x0))
    if blob.width > _WIDE * max(float(widths[primary]), em / 4):
        # Far wider than the character it lies over: a rule or a cell border.
        return None
    return (primary,)


def _one_line(blob: Blob, chars: list[TextChar], candidates: list[int], scale: float) -> list[int]:
    """The candidates on the line whose core band holds most of the mark.

    A mark below a baseline, an underscore or a descender, also falls in the
    zone of the line beneath; the band from a line's ascenders to just under
    its baseline says which line the mark belongs to.
    """
    by_line: dict[int, list[int]] = {}
    for i in candidates:
        by_line.setdefault(round(chars[i].baseline * scale / _LINE_STEP), []).append(i)
    if len(by_line) == 1:
        return candidates

    def held(members: list[int]) -> float:
        char = chars[members[0]]
        em = char.size * scale
        base = char.baseline * scale
        return min(blob.y1, base + _BAND_BELOW * em) - max(blob.y0, base - _BAND_ABOVE * em)

    return max(by_line.values(), key=held)


def _template(
    page: Page, ink: Ink, chars: list[TextChar], owners: tuple[int, ...], blobs: list[Blob]
) -> Template | None:
    """One template for the marks that together draw ``owners``."""
    scale = page.transform.sx
    first, last = chars[owners[0]], chars[owners[-1]]
    em = first.size * scale
    x0 = min(b.x0 for b in blobs)
    y0 = min(b.y0 for b in blobs)
    x1 = max(b.x1 for b in blobs)
    y1 = max(b.y1 for b in blobs)
    if y1 - y0 > _TALLEST * em or x1 - x0 > _WIDEST * (len(owners) + 1) * em:
        return None
    union = ink.blob((x0, y0, x1, y1))
    if union is None:
        return None
    left, base = page.transform.to_px(first.x0, first.baseline)
    right, _ = page.transform.to_px(last.x1, last.baseline)
    label = "".join(chars[i].text for i in owners)
    return Template(
        label,
        quantize(union.patch),
        em,
        base - union.y0,
        (union.x0 - left) / em,
        (right - union.x1) / em,
        first.fontname,
        parts=len(blobs),
    )
