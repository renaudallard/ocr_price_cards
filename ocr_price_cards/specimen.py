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


def _tables(data: bytes) -> dict[bytes, tuple[int, int]]:
    count = struct.unpack(">H", data[4:6])[0]
    return {
        data[12 + 16 * i : 16 + 16 * i]: struct.unpack(">II", data[20 + 16 * i : 28 + 16 * i])
        for i in range(count)
    }


def glyph_advances(data: bytes) -> dict[str, float]:
    """Advance width per printable character of a TrueType font, in em, for glyphs that have an outline."""
    tables = _tables(data)
    if any(name not in tables for name in (b"cmap", b"head", b"loca", b"hmtx", b"hhea", b"maxp")):
        return {}
    head = tables[b"head"][0]
    units = struct.unpack(">H", data[head + 18 : head + 20])[0]
    long_offsets = struct.unpack(">h", data[head + 50 : head + 52])[0] == 1
    loca_offset, loca_length = tables[b"loca"]
    if long_offsets:
        loca = struct.unpack(f">{loca_length // 4}I", data[loca_offset : loca_offset + loca_length])
    else:
        loca = tuple(
            2 * v
            for v in struct.unpack(
                f">{loca_length // 2}H", data[loca_offset : loca_offset + loca_length]
            )
        )
    hhea = tables[b"hhea"][0]
    metrics = struct.unpack(">H", data[hhea + 34 : hhea + 36])[0]
    hmtx = tables[b"hmtx"][0]
    out: dict[str, float] = {}
    for character, glyph in _cmap(data, tables).items():
        if glyph + 1 >= len(loca) or loca[glyph + 1] <= loca[glyph]:
            continue
        row = min(glyph, metrics - 1)
        advance = struct.unpack(">H", data[hmtx + 4 * row : hmtx + 4 * row + 2])[0]
        out[character] = advance / units
    return out


def _cmap(data: bytes, tables: dict[bytes, tuple[int, int]]) -> dict[str, int]:
    """Printable character to glyph index, from the font's format 4 Unicode cmap."""
    offset, _length = tables[b"cmap"]
    subtables = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
    found: dict[str, int] = {}
    for i in range(subtables):
        platform, encoding, start = struct.unpack(
            ">HHI", data[offset + 4 + 8 * i : offset + 12 + 8 * i]
        )
        if (platform, encoding) not in ((3, 1), (0, 3), (0, 4), (3, 10)):
            continue
        table = offset + start
        if struct.unpack(">H", data[table : table + 2])[0] != 4:
            continue
        segments = struct.unpack(">H", data[table + 6 : table + 8])[0] // 2
        ends = struct.unpack(f">{segments}H", data[table + 14 : table + 14 + 2 * segments])
        starts_at = table + 16 + 2 * segments
        starts = struct.unpack(f">{segments}H", data[starts_at : starts_at + 2 * segments])
        deltas_at = starts_at + 2 * segments
        deltas = struct.unpack(f">{segments}h", data[deltas_at : deltas_at + 2 * segments])
        ranges_at = deltas_at + 2 * segments
        ranges = struct.unpack(f">{segments}H", data[ranges_at : ranges_at + 2 * segments])
        for k in range(segments):
            for code in range(starts[k], min(ends[k], 0xFFFE) + 1):
                if ranges[k] == 0:
                    glyph = (code + deltas[k]) & 0xFFFF
                else:
                    at = ranges_at + 2 * k + ranges[k] + 2 * (code - starts[k])
                    glyph = struct.unpack(">H", data[at : at + 2])[0]
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
