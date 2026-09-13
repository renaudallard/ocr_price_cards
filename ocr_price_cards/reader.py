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

"""Read a card: pixels through the glyph library, the text layer as it is.

Every mark on a page is matched against the library, but never alone: the
marks of a row are read as the cheapest sequence of glyphs that covers them,
so a letter that fell apart into two marks reads as the one letter it is, and
a percent sign that is three marks reads as one sign. A mark that could be
two glyphs is carried as ambiguous and settled from its row and its word,
because a shape alone cannot tell a lowercase l from a capital I in a font
that draws them alike. A mark that reads as nothing, sitting in a row of
text, refuses the card.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .errors import LibraryError, UnreadableError
from .ink import MARGIN as INK_MARGIN
from .ink import Blob, Ink
from .layout import Glyph, Line, Word, build_lines, text_of
from .library import Library, Match
from .pages import Page, TextChar, load_pages

ACCEPT = 0.10
"""Worst score a template may have and still be read."""

MARGIN = 0.12
"""A runner-up label whose extra difference is under this fraction of the mark's ink makes the mark ambiguous."""

GLYPH_COST = 0.02
"""Added per glyph read, so that one glyph wins over two of the same score."""

SKIP_COST = 0.6
"""Cost of leaving a mark unread, more than any reading that is accepted."""

PART_SKIP_COST = 0.3
"""Cost of leaving one part of a stack unread while the rest of it is read."""

UNION = 3
"""Longest run of neighbouring marks tried as one glyph."""

UNION_SLACK = 0.05
"""Ink a joined reading may leave unexplained beyond what its parts leave, as a fraction of its ink."""

UNION_SIZE_SLACK = 2
"""Pixels a joined or stacked reading may differ in size from its template: the parts of a stack shift on their own."""

UNION_ACCEPT = 0.06
"""Worst score a reading of several joined pieces may have: stricter than a single mark, since a wrong join reads as the wrong glyph."""

SIZE_SLACK = 0.2
"""How far a glyph's font size may sit from its word's before it is read again at the word's size."""

MAX_GLYPH = 300
"""Marks taller or wider than this many pixels are boxes and pictures, never glyphs."""

PUNCTUATION = "()[]{}.,:;!?*'\"«»"
"""Characters a word may start or end with that its spelling does not include."""

NOISE = frozenset(".,-–—'’‘`·:;\"“”*")
"""Glyphs that only mean something next to a word; alone they are the dots of a QR code."""

STROKES = frozenset("iIl|1!")
"""Glyphs that are one bar, which the modules of a QR code also form when two of them line up."""

_TEXT_ABOVE = 1.0
_TEXT_BELOW = 0.3
_COVERED = 0.3
_NEIGHBOUR_GAP = 0.6
_NEIGHBOUR_OVERLAP = 0.5
_HEIGHT_MIN = 0.1
_HEIGHT_MAX = 1.4
_SPECK = 3
_STACK_OVERLAP = 0.5
_STACK_SLACK = 0.2
_STACK_GAP = 0.35
_STACK_GAP_SMALL = 1.5
_SMALL_ALIKE = 2
_PART_RATIO = 0.5
_PART_RATIO_DOT = 0.7
_BASE_WIDTH = 10.0
_PART_WIDTH = 2.5
_NEST_OVERLAP = 0.9
_SMALL_PART = 8
_SMALL_PART_WIDTH = 12
_WORD_OVERLAP = 0.5
_ROW_OVERLAP = 0.5
_ROW_REACH = 2.0
_JOIN_GAP = 0.3
_NOISE_REACH = 1.0
_BASELINE_SLACK = 0.08
_BASELINE_SLACK_PX = 1.5
_MAX_AMBIGUOUS = 6
_SPLIT_MIN_WIDTH = 8
_SPLIT_MARGIN = 3
_SPLIT_BRIDGE = 1
_SPLIT_MIN_INK = 4


@dataclass(slots=True)
class PageText:
    """What one page reads as."""

    number: int
    width: float
    height: float
    source: str
    lines: list[Line]
    unread: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def words(self) -> list[Word]:
        return [word for line in self.lines for word in line.words]

    @property
    def text(self) -> str:
        return text_of(self.lines)


@dataclass(slots=True)
class Document:
    """What a card reads as: one page after the other, joined by a newline like pdfplumber's text."""

    pages: list[PageText]

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)


@dataclass(slots=True)
class Piece:
    """A mark, or the marks stacked into one glyph body, as one patch."""

    blobs: list[Blob]
    blob: Blob


@dataclass(slots=True)
class Read:
    """A patch and the templates it could be, best first.

    ``parts`` are the marks the patch was made of. A glyph that arrived as
    two marks, a dot over a stem, can only be a template made of two, so
    such a reading keeps no candidate made of one.
    """

    blob: Blob
    candidates: list[Match]
    parts: list[Blob]

    def __post_init__(self) -> None:
        if len(self.parts) > 1:
            kept = [match for match in self.candidates if match.template.parts == len(self.parts)]
            if kept:
                self.candidates = kept

    @property
    def match(self) -> Match:
        return self.candidates[0]

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    @property
    def baseline(self) -> float:
        return self.blob.y0 + self.match.template.base

    @property
    def cost(self) -> float:
        return self.match.score + GLYPH_COST


@dataclass(slots=True)
class Reads:
    matched: list[Read]
    unmatched: list[Blob]


@cache
def default_library() -> Library:
    """The library shipped with the package."""
    path = resources.files("ocr_price_cards").joinpath("data", "library.npz")
    with resources.as_file(path) as real:
        if not real.is_file():
            raise LibraryError("the package ships no glyph library; train one first")
        return Library.load(Path(real))


def read_pdf(
    payload: bytes,
    *,
    library: Library | None = None,
    embedded: bool = True,
    text_layer: bool = True,
    strict: bool = True,
) -> Document:
    """Read every page of a card.

    Characters the PDF states are taken as they are; everything else is read
    from the pixels through ``library``. With ``strict`` a mark that sits in a
    line of text and matches no glyph raises :class:`UnreadableError`; without
    it the mark is listed in the page's ``unread`` boxes and skipped.
    """
    lib = library or default_library()
    pages = load_pages(payload, dpi=lib.dpi, embedded=embedded)
    return Document([read_page(page, lib, text_layer=text_layer, strict=strict) for page in pages])


def read_page(
    page: Page, library: Library, *, text_layer: bool = True, strict: bool = True
) -> PageText:
    """Read one page, as :func:`read_pdf` does."""
    texts = [glyph_from_char(char) for char in page.chars] if text_layer else []
    ink = Ink.of(page.pixels)
    blobs = ink.blobs()
    if texts:
        blobs = _not_under_text(blobs, page)
    marks = [blob for blob in blobs if blob.width <= MAX_GLYPH and blob.height <= MAX_GLYPH]
    reads = recognise(marks, ink, library)
    by_box = {read.blob.box: read for read in reads.matched}
    images = [glyph_from_read(read, page) for read in reads.matched]
    lines = build_lines(texts + images)
    unread = [blob.box for blob in reads.unmatched if _in_text(blob, reads.matched, page)]
    unread.extend(settle_sizes(lines, by_box, library, page))
    unread.extend(settle(lines, library.words))
    unread.extend(_overlapping(lines))
    if unread and strict:
        box = unread[0]
        raise UnreadableError(page.number, box, _context(lines, box, page))
    return PageText(page.number, page.width, page.height, page.source, lines, unread)


def recognise(blobs: list[Blob], ink: Ink, library: Library) -> Reads:
    """Read every mark, row by row, as the cheapest sequence of glyphs covering the row."""
    matched: list[Read] = []
    unmatched: list[Blob] = []
    for row in _rows(_pieces(blobs, ink)):
        reads, skipped = _read_row(row, ink, library)
        matched.extend(reads)
        unmatched.extend(skipped)
    matched = _without_noise(_drop_inner(matched))
    unmatched = [blob for blob in unmatched if not _inside_any(blob, matched)]
    _settle_baselines(matched)
    return Reads(matched, unmatched)


def decide(matches: list[Match]) -> list[Match] | None:
    """The templates a mark can be, or None when no template is close enough.

    A runner-up label is kept as an alternative when what separates it from
    the best one is a small part of the mark's ink: the dot of an i against
    an l, the tail of a comma against a full stop.
    """
    if not matches or matches[0].score > ACCEPT:
        return None
    best = matches[0]
    mass = max(best.mass, 1.0)
    return [match for match in matches if (match.sad - best.sad) / mass < MARGIN]


def _pieces(blobs: list[Blob], ink: Ink) -> list[Piece]:
    pieces: list[Piece] = []
    for group in stack_groups(blobs):
        union = ink.blob(_union_box(group)) if len(group) > 1 else group[0]
        if union is None:
            pieces.extend(Piece([blob], blob) for blob in group)
        else:
            pieces.append(Piece(group, union))
    return pieces


def _rows(pieces: list[Piece]) -> list[list[Piece]]:
    """Pieces chained by overlapping heights and nearness, each chain one row of text."""
    order = sorted(pieces, key=lambda p: p.blob.x0)
    if not order:
        return []
    x0 = np.array([p.blob.x0 for p in order])
    x1 = np.array([p.blob.x1 for p in order])
    y0 = np.array([p.blob.y0 for p in order])
    y1 = np.array([p.blob.y1 for p in order])
    heights = y1 - y0
    parent = list(range(len(order)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(order)):
        stop = int(np.searchsorted(x0, x1[i] + _ROW_REACH * MAX_GLYPH, side="right"))
        if stop <= i + 1:
            continue
        window = slice(i + 1, stop)
        near = x0[window] - x1[i] <= _ROW_REACH * np.maximum(heights[window], heights[i])
        overlap = np.minimum(y1[window], y1[i]) - np.maximum(y0[window], y0[i])
        same = overlap >= _ROW_OVERLAP * np.minimum(heights[window], heights[i])
        for j in np.nonzero(near & same)[0].tolist():
            parent[find(i)] = find(i + 1 + j)
    rows: dict[int, list[Piece]] = {}
    for i, piece in enumerate(order):
        rows.setdefault(find(i), []).append(piece)
    return list(rows.values())


@dataclass(slots=True)
class Reading:
    """What a run of pieces reads as: glyphs, and the marks among them that read as nothing."""

    reads: list[Read]
    skipped: list[Blob]

    @property
    def cost(self) -> float:
        return sum(read.cost for read in self.reads) + PART_SKIP_COST * len(self.skipped)


def _read_row(row: list[Piece], ink: Ink, library: Library) -> tuple[list[Read], list[Blob]]:
    """The cheapest reading of a row: each piece read alone, joined with its neighbours, or skipped."""
    order = sorted(row, key=lambda p: p.blob.x0)
    count = len(order)
    cost = [0.0] + [float("inf")] * count
    choice: list[tuple[int, Reading | None]] = [(0, None)] * (count + 1)
    for i in range(1, count + 1):
        best = cost[i - 1] + SKIP_COST
        pick: tuple[int, Reading | None] = (1, None)
        for k in range(1, UNION + 1):
            if i - k < 0:
                break
            if k > 1 and not _joinable(order[i - k], order[i - k + 1]):
                break
            reading = _read_pieces(order[i - k : i], ink, library)
            if reading is None:
                continue
            total = cost[i - k] + reading.cost
            if total < best:
                best, pick = total, (k, reading)
        cost[i] = best
        choice[i] = pick
    reads: list[Read] = []
    skipped: list[Blob] = []
    i = count
    while i > 0:
        k, found = choice[i]
        if found is None:
            skipped.extend(order[i - 1].blobs)
        else:
            reads.extend(found.reads)
            skipped.extend(found.skipped)
        i -= k
    return reads, skipped


def _joinable(a: Piece, b: Piece) -> bool:
    """Whether two neighbouring pieces may be one glyph: close, side by side, and glyph-sized together."""
    if b.blob.x0 - a.blob.x1 > _JOIN_GAP * max(a.blob.height, b.blob.height):
        return False
    if min(a.blob.y1, b.blob.y1) - max(a.blob.y0, b.blob.y0) < _ROW_OVERLAP * min(
        a.blob.height, b.blob.height
    ):
        return False
    x0, y0, x1, y1 = _union_box([a.blob, b.blob])
    return x1 - x0 <= MAX_GLYPH and y1 - y0 <= MAX_GLYPH


def _read_pieces(group: list[Piece], ink: Ink, library: Library) -> Reading | None:
    """The reading of one piece, or of several neighbouring pieces as one glyph."""
    if len(group) > 1:
        parts = [blob for piece in group for blob in piece.blobs]
        union = ink.blob(_union_box(parts))
        candidates = (
            decide(library.match(union.patch, size_tolerance=UNION_SIZE_SLACK))
            if union is not None
            else None
        )
        if union is None or candidates is None:
            return None
        # Joined, the pieces must read well, and leave no more ink unexplained
        # than they do read alone, a piece that reads as nothing leaving all of
        # its ink unexplained: an l and half a quotation mark are not an r, a
        # 6 does not swallow the comma after it, and an m that fell in two
        # reads as the m it is.
        if candidates[0].score > UNION_ACCEPT:
            return None
        unexplained = 0.0
        for piece in group:
            found = decide(library.match(piece.blob.patch))
            unexplained += found[0].sad if found is not None else float(piece.blob.patch.sum())
        if candidates[0].sad > unexplained + UNION_SLACK * candidates[0].mass:
            return None
        return Reading([Read(union, candidates, parts)], [])
    piece = group[0]
    whole: Reading | None = None
    slack = UNION_SIZE_SLACK if len(piece.blobs) > 1 else 1
    candidates = decide(library.match(piece.blob.patch, size_tolerance=slack))
    if candidates is not None:
        whole = Reading([Read(piece.blob, candidates, piece.blobs)], [])
    if len(piece.blobs) == 1:
        return whole if whole is not None else _split(piece.blob, library)
    # The parts of a stack may still be glyphs of their own, but not two of
    # them over each other on one row: a dot over a stem is an i that failed,
    # never a full stop and an l.
    reads: list[Read] = []
    skipped: list[Blob] = []
    for blob in piece.blobs:
        found = decide(library.match(blob.patch))
        if found is None:
            skipped.append(blob)
        else:
            reads.append(Read(blob, found, [blob]))
    if not reads or _stack_left(reads):
        return whole
    apart = Reading(reads, skipped)
    if whole is None or apart.cost < whole.cost:
        return apart
    return whole


def _split(blob: Blob, library: Library) -> Reading | None:
    """Two glyphs that touch, read by cutting the mark where it is thinnest.

    Neighbouring letters can share a pixel of ink, an r and a w or a y and
    the hyphen after it. A mark that reads as nothing whole is cut at each
    column where its core is a single pixel tall, and read as the two marks
    on either side; the cut that reads best wins.
    """
    if blob.width < _SPLIT_MIN_WIDTH:
        return None
    core = blob.patch >= 0.5
    profile = core.sum(axis=0)
    best: Reading | None = None
    for column in range(_SPLIT_MARGIN, blob.width - _SPLIT_MARGIN):
        if profile[column] > _SPLIT_BRIDGE:
            continue
        halves = [_trimmed(blob, 0, column + 1), _trimmed(blob, column, blob.width)]
        reads: list[Read] = []
        for half in halves:
            if half is None:
                break
            found = decide(library.match(half.patch))
            if found is None:
                break
            reads.append(Read(half, found, [half]))
        if len(reads) != 2:
            continue
        reading = Reading(reads, [])
        if best is None or reading.cost < best.cost:
            best = reading
    return best


def _trimmed(blob: Blob, left: int, right: int) -> Blob | None:
    """The part of ``blob`` between two columns, its empty rows cut away, with a pixel of margin kept."""
    patch = blob.patch[:, left:right]
    rows = np.nonzero((patch >= 0.5).any(axis=1))[0]
    if len(rows) == 0:
        return None
    top, bottom = (
        max(int(rows[0]) - INK_MARGIN, 0),
        min(int(rows[-1]) + 1 + INK_MARGIN, patch.shape[0]),
    )
    patch = patch[top:bottom]
    if int((patch >= 0.5).sum()) < _SPLIT_MIN_INK:
        return None
    return Blob(
        blob.x0 + left, blob.y0 + top, blob.x0 + right, blob.y0 + bottom, patch, blob.bg, blob.ink
    )


def stack_groups(blobs: list[Blob]) -> list[list[Blob]]:
    """Marks stacked over each other (a dot over its stem, dots over a vowel) grouped as one glyph.

    A part is a mark much smaller than the base it sits on, or one of two
    small marks over each other like the dots of a colon. It attaches to the
    base whose columns it shares and that it is closest to, so the dot of an
    i joins the stem under it rather than the letter beside it whose top is a
    pixel nearer.
    """
    order = sorted(blobs, key=lambda b: (b.x0, b.y0))
    if not order:
        return []
    lefts = np.array([b.x0 for b in order])
    rights = np.array([b.x1 for b in order])
    parent = list(range(len(order)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, part in enumerate(order):
        best: tuple[int, float, int] | None = None
        slack = part.width if part.height <= _SMALL_PART else 0
        for j in np.nonzero((lefts < part.x1 + slack) & (rights > part.x0 - slack))[0].tolist():
            if i == j or not _attaches(part, order[j]):
                continue
            base = order[j]
            overlap = min(part.x1, base.x1) - max(part.x0, base.x0)
            tier = 0 if overlap >= _STACK_OVERLAP * part.width else 1
            gap = max(0, max(part.y0, base.y0) - min(part.y1, base.y1))
            distance = abs((part.x0 + part.x1) - (base.x0 + base.x1)) / 2.0
            if best is None or (tier, gap + distance) < best[:2]:
                best = (tier, gap + distance, j)
        if best is not None:
            parent[find(i)] = find(best[2])
    groups: dict[int, list[Blob]] = {}
    for i, blob in enumerate(order):
        groups.setdefault(find(i), []).append(blob)
    return [sorted(group, key=lambda b: (b.y0, b.x0)) for group in groups.values()]


def _attaches(part: Blob, base: Blob) -> bool:
    """Whether ``part`` is a piece of the glyph ``base`` is the body of.

    A part sits over or under its base, so their columns overlap; the dots of
    a diaeresis over a narrow stem only sit beside it, so a dot-sized part may
    also be within its own width of the base's columns.
    """
    overlap = min(part.x1, base.x1) - max(part.x0, base.x0)
    beside = part.height <= _SMALL_PART and overlap > -part.width
    if overlap < _STACK_OVERLAP * part.width and not beside:
        return False
    if base.width > _BASE_WIDTH * part.width or part.width > _PART_WIDTH * base.width:
        return False
    gap = max(part.y0, base.y0) - min(part.y1, base.y1)
    if gap < -_STACK_SLACK * part.height:
        return False
    ratio = _PART_RATIO_DOT if part.height <= _SMALL_PART else _PART_RATIO
    if part.height <= ratio * base.height:
        if gap <= _STACK_GAP * base.height:
            return True
        return overlap >= _NEST_OVERLAP * part.width and gap < 0
    # Two small marks over each other, the dots of a colon or the bars of an
    # equals sign, are alike and aligned, and sit a little further apart than
    # a dot sits from its stem.
    small = (
        max(part.height, base.height) <= _SMALL_PART
        and max(part.width, base.width) <= _SMALL_PART_WIDTH
    )
    alike = abs(part.height - base.height) <= _SMALL_ALIKE and overlap >= _STACK_OVERLAP * max(
        part.width, base.width
    )
    return small and alike and gap <= _STACK_GAP_SMALL * max(part.height, base.height)


def _union_box(blobs: list[Blob]) -> tuple[int, int, int, int]:
    return (
        min(b.x0 for b in blobs),
        min(b.y0 for b in blobs),
        max(b.x1 for b in blobs),
        max(b.y1 for b in blobs),
    )


def _stack_left(reads: list[Read]) -> bool:
    """Whether two readings sit over each other: a dot over a stem is never a full stop and an l."""
    for a, b in itertools.combinations(reads, 2):
        overlap = min(a.blob.x1, b.blob.x1) - max(a.blob.x0, b.blob.x0)
        if overlap >= _STACK_OVERLAP * min(a.blob.width, b.blob.width):
            return True
    return False


def _drop_inner(matched: list[Read]) -> list[Read]:
    """A reading that lies inside another reading is the hole of a letter, not a glyph."""
    if not matched:
        return matched
    x0 = np.array([r.blob.x0 for r in matched])
    y0 = np.array([r.blob.y0 for r in matched])
    x1 = np.array([r.blob.x1 for r in matched])
    y1 = np.array([r.blob.y1 for r in matched])
    keep: list[Read] = []
    for i, read in enumerate(matched):
        covers = (x0 <= x0[i]) & (y0 <= y0[i]) & (x1 >= x1[i]) & (y1 >= y1[i])
        covers &= (x1 - x0 > x1[i] - x0[i]) | (y1 - y0 > y1[i] - y0[i])
        if not covers.any():
            keep.append(read)
    return keep


def _without_noise(matched: list[Read]) -> list[Read]:
    """Drop punctuation and bare strokes with no word beside them: the modules of a QR code read as dots, dashes and bars."""
    order = sorted(matched, key=lambda read: read.blob.x0)
    keep: list[Read] = []
    for i, read in enumerate(order):
        if read.match.label not in NOISE and read.match.label not in STROKES:
            keep.append(read)
            continue
        reach = _NOISE_REACH * read.match.template.em
        near = False
        for j in itertools.chain(range(i - 1, -1, -1), range(i + 1, len(order))):
            other = order[j]
            if other.match.label in NOISE or other.match.label in STROKES:
                continue
            gap = max(0, other.blob.x0 - read.blob.x1, read.blob.x0 - other.blob.x1)
            if gap > reach:
                if j > i:
                    break
                continue
            if min(read.blob.y1, other.blob.y1) - max(read.blob.y0, other.blob.y0) > 0:
                near = True
                break
        if near:
            keep.append(read)
    return keep


def _inside(inner: Blob, outer: Blob) -> bool:
    return (
        inner.x0 >= outer.x0
        and inner.y0 >= outer.y0
        and inner.x1 <= outer.x1
        and inner.y1 <= outer.y1
        and (inner.width < outer.width or inner.height < outer.height)
    )


def _inside_any(blob: Blob, matched: list[Read]) -> bool:
    return any(_inside(blob, read.blob) for read in matched)


def _settle_baselines(matched: list[Read]) -> None:
    """Keep, of an ambiguous mark's candidates, those whose baseline agrees with the row.

    A comma and an apostrophe are one shape at two heights: the row's other
    glyphs say where the baseline runs, and only the candidate that puts the
    mark on it survives.
    """
    sure = [read for read in matched if not read.ambiguous]
    for read in matched:
        if not read.ambiguous:
            continue
        neighbours = [
            other
            for other in sure
            if min(read.blob.y1, other.blob.y1) - max(read.blob.y0, other.blob.y0)
            >= _NEIGHBOUR_OVERLAP * min(read.blob.height, other.blob.height)
            and max(0, other.blob.x0 - read.blob.x1, read.blob.x0 - other.blob.x1)
            <= _ROW_REACH * other.match.template.em
        ]
        if not neighbours:
            continue
        baseline = statistics.median(other.baseline for other in neighbours)
        kept = [
            match
            for match in read.candidates
            if abs(read.blob.y0 + match.template.base - baseline)
            <= max(_BASELINE_SLACK * match.template.em, _BASELINE_SLACK_PX)
        ]
        if kept:
            read.candidates = kept


def settle_sizes(
    lines: list[Line], reads: dict[tuple[int, int, int, int], Read], library: Library, page: Page
) -> list[tuple[int, int, int, int]]:
    """Read again, at the word's own font size, every glyph whose size disagrees with its word.

    A bare stem is an l at a smaller size and an i at the word's size; the
    word says which. A glyph that reads as nothing at the word's size is
    refused, and its box returned.
    """
    boxes: list[tuple[int, int, int, int]] = []
    scale = page.transform.sx
    for line in lines:
        for word in line.words:
            images = [glyph for glyph in word.glyphs if glyph.box is not None]
            if len(images) < 2:
                continue
            median = statistics.median(glyph.size for glyph in images)
            em = median * scale
            for glyph in images:
                if glyph.box is None:
                    continue
                read = reads[glyph.box]
                if abs(glyph.size - median) > SIZE_SLACK * median:
                    again = decide(
                        library.match(
                            read.blob.patch, em=(em * (1 - SIZE_SLACK), em * (1 + SIZE_SLACK))
                        )
                    )
                    if again is None:
                        boxes.append(glyph.box)
                        continue
                    read.candidates = again
                elif read.ambiguous:
                    # An alternative set in another size is another word's size, not this one's.
                    fitting = [
                        m for m in read.candidates if abs(m.template.em - em) <= SIZE_SLACK * em
                    ]
                    if fitting:
                        read.candidates = fitting
                glyph_from_read(read, page, into=glyph)
    return boxes


def settle(lines: list[Line], words: set[str]) -> list[tuple[int, int, int, int]]:
    """Settle the glyphs a word could spell two ways; the boxes of those that stay open.

    The spellings are tried against the words the training cards use, then
    against the case of the rest of the word: a capital among capitals, a
    lowercase letter inside a lowercase word. What neither settles is refused.
    """
    open_boxes: list[tuple[int, int, int, int]] = []
    for line in lines:
        for word in line.words:
            open_boxes.extend(_settle_word(word, words))
    return open_boxes


def _settle_word(word: Word, words: set[str]) -> list[tuple[int, int, int, int]]:
    positions = [i for i, glyph in enumerate(word.glyphs) if glyph.alternatives]
    if not positions:
        return []
    if len(positions) <= _MAX_AMBIGUOUS:
        choices = [(word.glyphs[i].text, *word.glyphs[i].alternatives) for i in positions]
        known = [
            picks
            for picks in itertools.product(*choices)
            if _spelling(word, positions, picks).strip(PUNCTUATION) in words
        ]
        if len(known) == 1:
            for i, text in zip(positions, known[0], strict=True):
                word.glyphs[i].text = text
                word.glyphs[i].alternatives = ()
            return []
    boxes: list[tuple[int, int, int, int]] = []
    letters = [j for j, g in enumerate(word.glyphs) if g.text.isalpha()]
    for i in positions:
        glyph = word.glyphs[i]
        # The rest of the word decides the case: a word set in capitals, or a
        # word whose letters after the first are lowercase, as most are.
        others = [
            word.glyphs[j].text for j in letters if j != i and not word.glyphs[j].alternatives
        ]
        inside = [
            word.glyphs[j].text for j in letters[1:] if j != i and not word.glyphs[j].alternatives
        ]
        options = (glyph.text, *glyph.alternatives)
        picked: str | None = None
        if others and all(text.isupper() for text in others):
            upper = [text for text in options if text.isupper()]
            picked = upper[0] if len(upper) == 1 else None
        elif inside and all(text.islower() for text in inside) and i != letters[0]:
            lower = [text for text in options if text.islower()]
            picked = lower[0] if len(lower) == 1 else None
        if picked is None:
            if glyph.box is not None:
                boxes.append(glyph.box)
            continue
        glyph.text = picked
        glyph.alternatives = ()
    return boxes


def _spelling(word: Word, positions: list[int], picks: tuple[str, ...]) -> str:
    texts = [glyph.text for glyph in word.glyphs]
    for i, text in zip(positions, picks, strict=True):
        texts[i] = text
    return "".join(texts)


def _not_under_text(blobs: list[Blob], page: Page) -> list[Blob]:
    """Marks lying under characters the PDF states; those characters are exact and win."""
    zones = _text_zones(page.chars, page)
    if len(zones) == 0:
        return blobs
    keep: list[Blob] = []
    for blob in blobs:
        dx = np.minimum(zones[:, 2], blob.x1) - np.maximum(zones[:, 0], blob.x0)
        dy = np.minimum(zones[:, 3], blob.y1) - np.maximum(zones[:, 1], blob.y0)
        covered = float((np.clip(dx, 0, None) * np.clip(dy, 0, None)).sum())
        if covered < _COVERED * blob.width * blob.height:
            keep.append(blob)
    return keep


def _text_zones(chars: list[TextChar], page: Page) -> npt.NDArray[np.float64]:
    zones = np.zeros((len(chars), 4), dtype=np.float64)
    for index, char in enumerate(chars):
        em = char.size * page.transform.sy
        x0, base = page.transform.to_px(char.x0, char.baseline)
        x1, _ = page.transform.to_px(char.x1, char.baseline)
        zones[index] = (x0, base - _TEXT_ABOVE * em, x1, base + _TEXT_BELOW * em)
    return zones


def _in_text(blob: Blob, matched: list[Read], page: Page) -> bool:
    """Whether an unread mark sits in a row of read glyphs, which makes it a glyph that failed.

    A mark whose core is a few pixels, or a single pixel thin, is a speck of
    anti-aliasing at the corner or the edge of a box, not a glyph, wherever
    it sits.
    """
    core = blob.patch >= 0.5
    if (
        int(core.sum()) < _SPECK
        or int(core.any(axis=0).sum()) < 2
        or int(core.any(axis=1).sum()) < 2
    ):
        return False
    for read in matched:
        other = read.blob
        if min(blob.y1, other.y1) - max(blob.y0, other.y0) < _NEIGHBOUR_OVERLAP * min(
            blob.height, other.height
        ):
            continue
        em = read.match.template.em
        gap = max(0, other.x0 - blob.x1, blob.x0 - other.x1)
        if gap > _NEIGHBOUR_GAP * em:
            continue
        if not _HEIGHT_MIN * em <= blob.height <= _HEIGHT_MAX * em:
            continue
        if blob.bg != other.bg:
            continue
        return True
    for x0, y0, x1, y1 in _text_zones(page.chars, page).tolist():
        if min(blob.y1, y1) - max(blob.y0, y0) < _NEIGHBOUR_OVERLAP * blob.height:
            continue
        em = (y1 - y0) / (_TEXT_ABOVE + _TEXT_BELOW)
        gap = max(0, x0 - blob.x1, blob.x0 - x1)
        if gap <= _NEIGHBOUR_GAP * em and _HEIGHT_MIN * em <= blob.height <= _HEIGHT_MAX * em:
            return True
    return False


def _overlapping(lines: list[Line]) -> list[tuple[int, int, int, int]]:
    """Two glyphs of one word on top of each other: a glyph read as two, refused rather than emitted."""
    out: list[tuple[int, int, int, int]] = []
    for line in lines:
        for word in line.words:
            for i, a in enumerate(word.glyphs):
                for b in word.glyphs[i + 1 :]:
                    if a.box is None or b.box is None:
                        continue
                    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
                    if overlap > _WORD_OVERLAP * min(a.x1 - a.x0, b.x1 - b.x0):
                        out.append(a.box if a.box[1] <= b.box[1] else b.box)
    return out


def _context(lines: list[Line], box: tuple[int, int, int, int], page: Page) -> str:
    """The text of the line the box falls in, for the error message."""
    _x, y = page.transform.to_pt(box[0], (box[1] + box[3]) / 2)
    closest: Line | None = None
    distance = float("inf")
    for line in lines:
        gap = 0.0 if line.top <= y <= line.bottom else min(abs(y - line.top), abs(y - line.bottom))
        if gap < distance:
            distance, closest = gap, line
    return closest.text[:80] if closest is not None else ""


def glyph_from_char(char: TextChar) -> Glyph:
    return Glyph(char.text, char.x0, char.x1, char.baseline, char.size, "text")


def glyph_from_read(read: Read, page: Page, *, into: Glyph | None = None) -> Glyph:
    """A glyph in page points from a patch and the template it matched, written ``into`` one if given."""
    template = read.match.template
    blob = read.blob
    size = template.em / page.transform.sx
    x0, _ = page.transform.to_pt(blob.x0, 0.0)
    x1, _ = page.transform.to_pt(blob.x1, 0.0)
    _, baseline = page.transform.to_pt(0.0, blob.y0 + template.base)
    glyph = into if into is not None else Glyph("", 0.0, 0.0, 0.0, 0.0, "image")
    glyph.text = template.label
    glyph.x0 = x0 - template.lsb * size
    glyph.x1 = x1 + template.rsb * size
    glyph.baseline = baseline
    glyph.size = size
    glyph.source = "image"
    glyph.score = read.match.score
    glyph.box = blob.box
    glyph.alternatives = tuple(match.label for match in read.candidates[1:])
    return glyph
