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
ASCENT = 0.75
"""Where pdfplumber puts the top of a character above its baseline, in ems, for the fonts of the cards."""

DESCENT = 0.2
SPACE = 0.12
"""Gap between two pixel glyphs, as a fraction of the em, that reads as a space."""

STEP = 0.75
"""Points between two consecutive baselines that still belong to one row of glyphs."""


@dataclass(slots=True)
class Glyph:
    """One character on the page, in points from the top left corner.

    ``x0``/``x1`` span the advance box, ``top``/``bottom`` the font box built
    from the baseline and the size as pdfplumber builds it. ``source`` is ``text`` for a character the
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
    """Words whose tops cluster within the tolerance, left to right.

    ``unread`` counts the marks on the line that were refused; a line with
    any is not to be trusted, but the lines around it still are.
    """

    words: list[Word]
    unread: int = 0

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
    """Group sorted distinct values into runs where each is within ``tolerance`` of the one before.

    This is pdfplumber's ``cluster_objects``: the chain runs as long as the
    steps are small, however far it gets from where it started.
    """
    out: dict[float, int] = {}
    group = -1
    last: float | None = None
    for value in sorted(set(values)):
        if last is None or value - last > tolerance:
            group += 1
        out[value] = group
        last = value
    return out


def build_words(glyphs: list[Glyph]) -> list[Word]:
    """Words as pdfplumber's ``extract_words`` forms them, rows clustered by top."""
    rows = cluster((glyph.top for glyph in glyphs), Y_TOLERANCE)
    by_row: dict[int, list[Glyph]] = {}
    for glyph in glyphs:
        by_row.setdefault(rows[glyph.top], []).append(glyph)
    return [word for row in sorted(by_row) for word in _words(by_row[row])]


def _words(row: list[Glyph]) -> list[Word]:
    """The words of one row, left to right."""
    words: list[Word] = []
    current: list[Glyph] = []
    for glyph in sorted(row, key=lambda g: g.x0):
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


def build_rows(glyphs: list[Glyph]) -> list[Line]:
    """The rows of glyphs as they were set: one baseline each, to within a fraction of a point.

    pdfplumber chains lines while each top is within three points of the one
    before, and on a table whose small header cells each carry two lines it
    chains the two and interleaves their characters. The text is given that
    way, because that is what the consumers of a card were written against;
    what a glyph is, though, is settled on the word it was set in, and that
    word sits on one baseline. Within each of pdfplumber's lines the
    baselines are chained again on a step of a fraction of a point.
    """
    lines = cluster((glyph.top for glyph in glyphs), Y_TOLERANCE)
    by_line: dict[int, list[Glyph]] = {}
    for glyph in glyphs:
        by_line.setdefault(lines[glyph.top], []).append(glyph)
    rows: list[Line] = []
    for line in sorted(by_line):
        runs = cluster((glyph.baseline for glyph in by_line[line]), STEP)
        by_run: dict[int, list[Glyph]] = {}
        for glyph in by_line[line]:
            by_run.setdefault(runs[glyph.baseline], []).append(glyph)
        for run in sorted(by_run):
            # A run of nothing but spaces makes no words, and a line with
            # no words has no top to be asked for.
            words = _words(by_run[run])
            if words:
                rows.append(Line(words))
    return rows


def begins_word(previous: Glyph, glyph: Glyph) -> bool:
    """Whether ``glyph`` starts a new word after ``previous`` on the same row."""
    if glyph.x0 < previous.x0 or abs(glyph.top - previous.top) > Y_TOLERANCE:
        return True
    gap = glyph.x0 - previous.x1
    if previous.source == "text" and glyph.source == "text":
        return gap > X_TOLERANCE
    return gap > SPACE * min(previous.size, glyph.size)


def build_lines(glyphs: list[Glyph]) -> list[Line]:
    """Lines as pdfplumber's ``extract_text`` clusters words, words left to right."""
    words = build_words(glyphs)
    rows = cluster((word.top for word in words), Y_TOLERANCE)
    by_row: dict[int, list[Word]] = {}
    for word in words:
        by_row.setdefault(rows[word.top], []).append(word)
    return [Line(sorted(by_row[row], key=lambda w: w.x0)) for row in sorted(by_row)]


def text_of(lines: list[Line]) -> str:
    return "\n".join(line.text for line in lines)
