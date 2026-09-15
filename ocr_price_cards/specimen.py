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

"""Specimens: every glyph of a card's fonts at any size, rendered the way the cards are.

A card only prints its fonts at the sizes it happens to use. The fonts
themselves are embedded in it, so their glyphs can be set again at every
size a future card might use and rendered through the same rasterizer, which
gives the library templates for text no card has printed yet.
"""

from __future__ import annotations

import ctypes
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import resolve1

from .pages import TextChar

CHARACTERS: frozenset[str] = frozenset(
    "".join(chr(code) for code in range(0x21, 0x7F))
    + "¡¢£¥§©«®°±²³µ¶¿×÷"
    + "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
    + "ŒœŠšŸŽž€–—‘’“”…‰•"
)
"""Characters a specimen sets: the Latin text a Belgian card can carry.

A full font also carries Greek, Cyrillic and other scripts whose glyphs look
exactly like Latin ones, and a few Latin marks that look exactly like others
(a cedilla like a comma, a low quotation mark like a comma); setting those
would make every word ambiguous for no card that ever prints them.
"""

SIZES: tuple[float, ...] = tuple(
    [4.0 + 0.25 * i for i in range(33)]
    + [12.5 + 0.5 * i for i in range(24)]
    + [25.0 + 1.0 * i for i in range(12)]
)
"""Point sizes a specimen is set in: fine steps where cards set their small print, coarser above."""

_PAGE_WIDTH = 595.0
_PAGE_HEIGHT = 842.0
_MARGIN = 30.0
_LEADING = 1.6
_PITCH = 1.5


@dataclass(frozen=True, slots=True)
class EmbeddedFont:
    """A TrueType font program taken out of a PDF, with the characters it draws and their advances.

    ``advances`` maps each character the font's cmap covers to the width of
    its glyph, as a fraction of the em; a character whose glyph the subset
    stripped is left out.
    """

    name: str
    data: bytes
    advances: dict[str, float]

    @property
    def characters(self) -> str:
        return "".join(sorted(self.advances))


@dataclass(frozen=True, slots=True)
class Specimen:
    """A specimen PDF and, per page, the characters set on it, in pdfplumber's coordinates."""

    family: str
    payload: bytes
    chars: list[list[TextChar]]


def embedded_fonts(payload: bytes) -> list[EmbeddedFont]:
    """The distinct TrueType font programs embedded in ``payload``.

    Every subset a PDF embeds is returned, even two of one family, since each
    covers only the characters the page that uses it prints.
    """
    parser = PDFParser(BytesIO(payload))
    document = PDFDocument(parser)
    manager = PDFResourceManager()
    interpreter = PDFPageInterpreter(manager, PDFDevice(manager))
    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)
    out: list[EmbeddedFont] = []
    seen: set[bytes] = set()
    for font in manager._cached_fonts.values():
        descriptor = getattr(font, "descriptor", None) or {}
        if "FontFile2" not in descriptor:
            continue
        data = bytes(resolve1(descriptor["FontFile2"]).get_data())
        if data in seen:
            continue
        seen.add(data)
        name = str(getattr(font, "basefont", "")).split("+")[-1]
        advances = glyph_advances(data)
        if advances:
            out.append(EmbeddedFont(name, data, advances))
    return out


def families(fonts: Iterable[EmbeddedFont]) -> dict[str, list[EmbeddedFont]]:
    """The fonts grouped by family name, subsets of one family together."""
    out: dict[str, list[EmbeddedFont]] = {}
    for font in fonts:
        out.setdefault(font.name, []).append(font)
    return out


def _read(data: bytes, offset: int, fmt: str) -> tuple[int, ...] | None:
    """Unpack ``fmt`` at ``offset``, or None where the font does not reach.

    A font program is read out of a PDF, so every offset and count in it is
    whatever the file says. Nothing is taken on trust: a field that lies
    outside the data reads as None and the font is left alone.
    """
    if offset < 0 or offset + struct.calcsize(fmt) > len(data):
        return None
    return struct.unpack_from(fmt, data, offset)


def _tables(data: bytes) -> dict[bytes, tuple[int, int]]:
    """Each table's offset and length, by tag; empty when the directory does not hold up."""
    count = _read(data, 4, ">H")
    if count is None:
        return {}
    out: dict[bytes, tuple[int, int]] = {}
    for i in range(count[0]):
        # A record is the four byte tag, a checksum, then the offset and
        # the length; reading the last two covers the tag that precedes them.
        entry = _read(data, 20 + 16 * i, ">II")
        if entry is None:
            return {}
        offset, length = entry
        if offset + length > len(data):
            return {}
        out[data[12 + 16 * i : 16 + 16 * i]] = (offset, length)
    return out


def glyph_advances(data: bytes) -> dict[str, float]:
    """Advance width per printable character of a TrueType font, in em, for glyphs that have an outline."""
    tables = _tables(data)
    if any(name not in tables for name in (b"cmap", b"head", b"loca", b"hmtx", b"hhea", b"maxp")):
        return {}
    head = tables[b"head"][0]
    em = _read(data, head + 18, ">H")
    indexing = _read(data, head + 50, ">h")
    hhea = _read(data, tables[b"hhea"][0] + 34, ">H")
    if em is None or indexing is None or hhea is None or em[0] == 0 or hhea[0] == 0:
        return {}
    units, metrics = em[0], hhea[0]
    loca_offset, loca_length = tables[b"loca"]
    wide = indexing[0] == 1
    step = 4 if wide else 2
    entries = loca_length // step
    packed = _read(data, loca_offset, f">{entries}{'I' if wide else 'H'}")
    if packed is None:
        return {}
    loca = packed if wide else tuple(2 * v for v in packed)
    hmtx = tables[b"hmtx"][0]
    out: dict[str, float] = {}
    for character, glyph in _cmap(data, tables).items():
        if glyph + 1 >= len(loca) or loca[glyph + 1] <= loca[glyph]:
            continue
        advance = _read(data, hmtx + 4 * min(glyph, metrics - 1), ">H")
        if advance is not None:
            out[character] = advance[0] / units
    return out


def _cmap(data: bytes, tables: dict[bytes, tuple[int, int]]) -> dict[str, int]:
    """Printable character to glyph index, from the font's format 4 Unicode cmap."""
    offset, _length = tables[b"cmap"]
    header = _read(data, offset + 2, ">H")
    if header is None:
        return {}
    found: dict[str, int] = {}
    for i in range(header[0]):
        record = _read(data, offset + 4 + 8 * i, ">HHI")
        if record is None:
            return found
        platform, encoding, start = record
        if (platform, encoding) not in ((3, 1), (0, 3), (0, 4), (3, 10)):
            continue
        table = offset + start
        fmt = _read(data, table, ">H")
        count = _read(data, table + 6, ">H")
        if fmt is None or count is None or fmt[0] != 4:
            continue
        segments = count[0] // 2
        ends = _read(data, table + 14, f">{segments}H")
        starts_at = table + 16 + 2 * segments
        starts = _read(data, starts_at, f">{segments}H")
        deltas_at = starts_at + 2 * segments
        deltas = _read(data, deltas_at, f">{segments}h")
        ranges_at = deltas_at + 2 * segments
        ranges = _read(data, ranges_at, f">{segments}H")
        if ends is None or starts is None or deltas is None or ranges is None:
            return found
        for k in range(segments):
            for code in range(starts[k], min(ends[k], 0xFFFE) + 1):
                if ranges[k] == 0:
                    glyph = (code + deltas[k]) & 0xFFFF
                else:
                    at = _read(data, ranges_at + 2 * k + ranges[k] + 2 * (code - starts[k]), ">H")
                    if at is None:
                        continue
                    glyph = at[0]
                    if glyph:
                        glyph = (glyph + deltas[k]) & 0xFFFF
                if glyph and chr(code).isprintable() and not chr(code).isspace():
                    found[chr(code)] = glyph
        break
    return found


def specimen(
    family: str, fonts: list[EmbeddedFont], sizes: Iterable[float] = SIZES
) -> list[Specimen]:
    """PDFs setting every character the ``fonts`` cover at every size, characters spaced apart.

    The fonts are the subsets of one family, and pdfium keeps one font per
    name in a document, so each subset used gets a document of its own. A
    character is set once, with the subset carrying the most characters among
    those that carry its outline. The characters set come back with each PDF,
    in pdfplumber's coordinates, so no text layer has to be read to know what
    is where.
    """
    order = sorted(fonts, key=lambda font: -len(font.advances))
    assigned: dict[int, list[str]] = {}
    covered: set[str] = set()
    for index, font in enumerate(order):
        mine = sorted((set(font.advances) & CHARACTERS) - covered)
        if mine:
            assigned[index] = mine
            covered.update(mine)
    return [
        _document(family, order[index], characters, sizes) for index, characters in assigned.items()
    ]


def _document(
    family: str, font: EmbeddedFont, characters: list[str], sizes: Iterable[float]
) -> Specimen:
    document = pdfium.PdfDocument.new()
    buffer = (ctypes.c_uint8 * len(font.data)).from_buffer_copy(font.data)
    handle = pdfium_c.FPDFText_LoadFont(
        document, buffer, len(font.data), pdfium_c.FPDF_FONT_TRUETYPE, 1
    )
    if not handle:
        raise ValueError(f"{font.name}: pdfium cannot load the font")
    # Every page stays referenced until the document is written: a page
    # handle that goes away takes the objects placed on it along.
    pages: list[pdfium.PdfPage] = []
    chars: list[list[TextChar]] = []
    out = BytesIO()
    y = 0.0
    try:
        for size in sizes:
            step = _PITCH * size
            per_line = max(1, int((_PAGE_WIDTH - 2 * _MARGIN) / step))
            for start in range(0, len(characters), per_line):
                if not pages or y - _LEADING * size < _MARGIN:
                    pages.append(document.new_page(_PAGE_WIDTH, _PAGE_HEIGHT))
                    chars.append([])
                    y = _PAGE_HEIGHT - _MARGIN
                y -= _LEADING * size
                for column, character in enumerate(characters[start : start + per_line]):
                    obj = pdfium_c.FPDFPageObj_CreateTextObj(document, handle, size)
                    wide = (ctypes.c_ushort * 2)(ord(character), 0)
                    pdfium_c.FPDFText_SetText(obj, wide)
                    x = _MARGIN + column * step
                    pdfium_c.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x, y)
                    pdfium_c.FPDFPage_InsertObject(pages[-1], obj)
                    chars[-1].append(
                        TextChar(
                            character,
                            x,
                            x + font.advances[character] * size,
                            _PAGE_HEIGHT - y,
                            size,
                            family,
                        )
                    )
        for page in pages:
            pdfium_c.FPDFPage_GenerateContent(page)
        document.save(out)
    finally:
        # The text objects carry the font themselves, so the handle the load
        # took out is ours to give back once the document is written.
        pdfium_c.FPDFFont_Close(handle)
    return Specimen(family, out.getvalue(), chars)


def specimens(cards: Iterable[bytes], sizes: Iterable[float] = SIZES) -> list[Specimen]:
    """One specimen per font family embedded in ``cards``."""
    fonts: list[EmbeddedFont] = []
    for payload in cards:
        fonts.extend(embedded_fonts(payload))
    return [
        item
        for family, subsets in families(fonts).items()
        for item in specimen(family, subsets, sizes)
    ]
