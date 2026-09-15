"""Specimens: the fonts embedded in a card, set again at any size, read like the card."""

from __future__ import annotations

import ctypes
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest
from conftest import pages_of

from ocr_price_cards.library import Library
from ocr_price_cards.reader import read_page
from ocr_price_cards.specimen import CHARACTERS, embedded_fonts, glyph_advances, specimen, specimens
from ocr_price_cards.train import train_specimens

FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
TEXT = "Fluvius Antwerpen 52,3679 kWh 18,92"


def _card_with_embedded_font(font: bytes, text: str, size: float) -> bytes:
    """A one line card set in ``font``, which pdfium embeds as a TrueType program."""
    document = pdfium.PdfDocument.new()
    page = document.new_page(595.0, 200.0)
    buffer = (ctypes.c_uint8 * len(font)).from_buffer_copy(font)
    handle = pdfium_c.FPDFText_LoadFont(document, buffer, len(font), pdfium_c.FPDF_FONT_TRUETYPE, 1)
    obj = pdfium_c.FPDFPageObj_CreateTextObj(document, handle, size)
    wide = (ctypes.c_ushort * (len(text) + 1))(*([ord(c) for c in text] + [0]))
    pdfium_c.FPDFText_SetText(obj, wide)
    pdfium_c.FPDFPageObj_Transform(obj, 1, 0, 0, 1, 40.0, 100.0)
    pdfium_c.FPDFPage_InsertObject(page, obj)
    pdfium_c.FPDFPage_GenerateContent(page)
    out = BytesIO()
    document.save(out)
    return out.getvalue()


@pytest.fixture(scope="module")
def dejavu() -> bytes:
    if not FONT.is_file():
        pytest.skip(f"{FONT} is not installed")
    return FONT.read_bytes()


def test_glyph_advances_cover_the_printable_characters(dejavu: bytes) -> None:
    advances = glyph_advances(dejavu)
    assert set("Fluvius0123456789,kWh") <= set(advances)
    assert " " not in advances
    assert 0.2 < advances["i"] < advances["W"] < 1.5


def test_a_standard_font_card_embeds_nothing(text_card: bytes) -> None:
    assert embedded_fonts(text_card) == []


def test_specimens_set_every_character_at_every_size(dejavu: bytes) -> None:
    card = _card_with_embedded_font(dejavu, TEXT, 9.0)
    fonts = embedded_fonts(card)
    assert len(fonts) == 1
    assert set(TEXT.replace(" ", "")) <= set(fonts[0].characters)
    items = specimen(fonts[0].name, fonts, sizes=(8.0, 9.0))
    assert len(items) == 1
    placed = [char for page in items[0].chars for char in page]
    assert {char.size for char in placed} == {8.0, 9.0}
    assert len(placed) == 2 * len(set(fonts[0].characters) & CHARACTERS)
    assert not {char.text for char in placed} - CHARACTERS
    assert len(pages_of(items[0].payload, embedded=False)) == len(items[0].chars)


def test_a_card_reads_through_specimens_of_its_own_font_alone(dejavu: bytes) -> None:
    size = 9.0
    card = _card_with_embedded_font(dejavu, TEXT, size)
    library = train_specimens(specimens([card], sizes=(size,)), library=Library(216.0))
    page = pages_of(card, embedded=False)[0]
    result = read_page(page, library, text_layer=False)
    assert result.text == TEXT
    assert not result.unread


def test_a_font_that_does_not_hold_up_is_left_alone(dejavu: bytes) -> None:
    import struct

    from ocr_price_cards.specimen import _tables

    assert len(glyph_advances(dejavu)) > 1000
    tables = _tables(dejavu)
    for field, offset in (
        ("numberOfHMetrics", tables[b"hhea"][0] + 34),
        ("unitsPerEm", tables[b"head"][0] + 18),
    ):
        zeroed = bytearray(dejavu)
        struct.pack_into(">H", zeroed, offset, 0)
        # Zero used to read the wrong bytes and hand back every advance as
        # nought, or divide by it.
        assert glyph_advances(bytes(zeroed)) == {}, field
    for cut in (b"", b"\x00\x01\x00\x00", dejavu[:12], dejavu[: len(dejavu) // 2], dejavu[:-1]):
        assert glyph_advances(cut) == {}
