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

"""Lines and words, assembled the way pdfplumber assembles them.

The extractors that consume a card were written against ``extract_text()``:
characters clustered into rows within three points, words split where two
characters sit more than three points apart, one line per row, one space
between words. Glyphs read from pixels go through the same rules, with one
difference: pixels carry no space characters, so between two such glyphs a
word ends at a gap wider than a fraction of the font size, which is what a
space is.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

X_TOLERANCE = 3.0
Y_TOLERANCE = 3.0
ASCENT = 0.8
DESCENT = 0.2
SPACE = 0.12
"""Gap between two pixel glyphs, as a fraction of the em, that reads as a space."""

STEP = 0.75
"""Points between two consecutive baselines that still belong to one line."""


@dataclass(slots=True)
class Glyph:
    """One character on the page, in points from the top left corner.

    ``x0``/``x1`` span the advance box, ``top``/``bottom`` the font box built
    from the baseline and the size. ``source`` is ``text`` for a character the
    PDF states and ``image`` for one read from pixels, whose ink box in canvas
    pixels is then in ``box``. ``alternatives`` are the other characters a
    pixel glyph could be when its shape alone does not say.
    """

    text: str
    x0: float
    x1: float
    baseline: float
    size: float
    source: str
    score: float = 0.0
    box: tuple[int, int, int, int] | None = None
    alternatives: tuple[str, ...] = ()

    @property
    def top(self) -> float:
        return self.baseline - ASCENT * self.size

    @property
    def bottom(self) -> float:
        return self.baseline + DESCENT * self.size


@dataclass(slots=True)
class Word:
    """Consecutive glyphs on one row with no gap wide enough to be a space."""

    glyphs: list[Glyph]

    @property
    def text(self) -> str:
        return "".join(glyph.text for glyph in self.glyphs)

    @property
    def x0(self) -> float:
        return min(glyph.x0 for glyph in self.glyphs)

    @property
    def x1(self) -> float:
        return max(glyph.x1 for glyph in self.glyphs)

    @property
    def top(self) -> float:
        return min(glyph.top for glyph in self.glyphs)

    @property
    def bottom(self) -> float:
        return max(glyph.bottom for glyph in self.glyphs)

    @property
    def baseline(self) -> float:
        return max(glyph.baseline for glyph in self.glyphs)

    @property
    def source(self) -> str:
        sources = {glyph.source for glyph in self.glyphs}
        return sources.pop() if len(sources) == 1 else "mixed"


@dataclass(slots=True)
class Line:
    """Words whose baselines cluster within the tolerance, left to right."""

    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    @property
    def top(self) -> float:
        return min(word.top for word in self.words)

    @property
    def bottom(self) -> float:
        return max(word.bottom for word in self.words)


def cluster(values: Iterable[float], tolerance: float) -> dict[float, int]:
    """Group sorted distinct values into runs of close neighbours no wider than ``tolerance``.

    The baselines of one line differ by a fraction of a point, so a run is
    chained on a step well under a point; a step wider than that is the next
    line. pdfplumber chains on the whole tolerance instead, and on a table
    whose small header cells each carry their own lines a few points apart
    that chain runs across the cells and interleaves them.
    """
    out: dict[float, int] = {}
    group = -1
    first: float | None = None
    last: float | None = None
    for value in sorted(set(values)):
        if first is None or last is None or value - last > STEP or value - first > tolerance:
            group += 1
            first = value
        out[value] = group
        last = value
    return out


def build_words(glyphs: list[Glyph]) -> list[Word]:
    """Words as pdfplumber's ``extract_words`` forms them, rows clustered by baseline."""
    rows = cluster((glyph.baseline for glyph in glyphs), Y_TOLERANCE)
    by_row: dict[int, list[Glyph]] = {}
    for glyph in glyphs:
        by_row.setdefault(rows[glyph.baseline], []).append(glyph)
    words: list[Word] = []
    for row in sorted(by_row):
        current: list[Glyph] = []
        for glyph in sorted(by_row[row], key=lambda g: g.x0):
            if glyph.text.isspace():
                if current:
                    words.append(Word(current))
                current = []
                continue
            if current and begins_word(current[-1], glyph):
                words.append(Word(current))
                current = []
            current.append(glyph)
        if current:
            words.append(Word(current))
    return words


def begins_word(previous: Glyph, glyph: Glyph) -> bool:
    """Whether ``glyph`` starts a new word after ``previous`` on the same row."""
    if glyph.x0 < previous.x0 or glyph.baseline > previous.baseline + Y_TOLERANCE:
        return True
    gap = glyph.x0 - previous.x1
    if previous.source == "text" and glyph.source == "text":
        return gap > X_TOLERANCE
    return gap > SPACE * min(previous.size, glyph.size)


def build_lines(glyphs: list[Glyph]) -> list[Line]:
    """Lines as pdfplumber's ``extract_text`` clusters words, words left to right."""
    words = build_words(glyphs)
    rows = cluster((word.baseline for word in words), Y_TOLERANCE)
    by_row: dict[int, list[Word]] = {}
    for word in words:
        by_row.setdefault(rows[word.baseline], []).append(word)
    return [Line(sorted(by_row[row], key=lambda w: w.x0)) for row in sorted(by_row)]


def text_of(lines: list[Line]) -> str:
    return "\n".join(line.text for line in lines)
