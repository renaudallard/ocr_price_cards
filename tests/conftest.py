"""Synthetic cards built with pdfium's own Helvetica, so the suite needs no card and no proprietary font."""

from __future__ import annotations

import ctypes
from io import BytesIO
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest

from ocr_price_cards.library import Library
from ocr_price_cards.pages import DEFAULT_DPI, Card
from ocr_price_cards.train import train

LINES: tuple[tuple[float, float, str], ...] = (
    (60.0, 760.0, "Fluvius Antwerpen 52,3679 5,35329 4,81301 18,92 18,92"),
    (60.0, 740.0, "Verbruik tussen 0 & 3.000 kWh 5,03288"),
    (60.0, 720.0, "Injectie: (BELPEX-SPP-M * 0,0884) - 0,5000"),
    (60.0, 690.0, "Maandprijs: 11,81 11,81 11,81 11,81"),
    (60.0, 660.0, "Digitale meter, exclusief nacht: 0,20417 c/kWh"),
)
SIZES = (7.0, 8.0, 9.0, 10.0)


def _text_card(lines: tuple[tuple[float, float, str], ...], sizes: tuple[float, ...]) -> bytes:
    """A one page card with ``lines`` set in Helvetica at every size in ``sizes``, one block per size."""
    document = pdfium.PdfDocument.new()
    page = document.new_page(595.0, 842.0)
    font = pdfium_c.FPDFText_LoadStandardFont(document, b"Helvetica")
    y_shift = 0.0
    for size in sizes:
        for x, y, text in lines:
            obj = pdfium_c.FPDFPageObj_CreateTextObj(document, font, size)
            wide = (ctypes.c_ushort * (len(text) + 1))(*([ord(c) for c in text] + [0]))
            pdfium_c.FPDFText_SetText(obj, wide)
            pdfium_c.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x, y - y_shift)
            pdfium_c.FPDFPage_InsertObject(page, obj)
        y_shift += 130.0
    pdfium_c.FPDFPage_GenerateContent(page)
    out = BytesIO()
    document.save(out)
    return out.getvalue()


def _image_card(
    text_card: bytes, mask: tuple[float, float, float, float], overlay: str, size: float
) -> bytes:
    """The text card as a page image at the library's resolution, a white mask painted on it and ``overlay`` set over the mask."""
    scale = DEFAULT_DPI / 72.0
    source = pdfium.PdfDocument(text_card)
    bitmap = source[0].render(scale=scale)
    pixels = np.array(bitmap.to_pil().convert("RGB"))
    height, width = pixels.shape[:2]
    document = pdfium.PdfDocument.new()
    page = document.new_page(595.0, 842.0)
    image = pdfium_c.FPDFPageObj_NewImageObj(document)
    canvas = pdfium.PdfBitmap.new_native(
        width, height, format=pdfium_c.FPDFBitmap_BGR, rev_byteorder=False
    )
    buffer = np.array(canvas.to_numpy())
    buffer[:, :, 0] = pixels[:, :, 2]
    buffer[:, :, 1] = pixels[:, :, 1]
    buffer[:, :, 2] = pixels[:, :, 0]
    canvas.to_numpy()[:] = buffer
    pdfium_c.FPDFImageObj_SetBitmap(None, 0, image, canvas)
    pdfium_c.FPDFPageObj_Transform(image, 595.0, 0, 0, 842.0, 0, 0)
    pdfium_c.FPDFPage_InsertObject(page, image)
    x0, y0, x1, y1 = mask
    rect = pdfium_c.FPDFPageObj_CreateNewRect(x0, y0, x1 - x0, y1 - y0)
    pdfium_c.FPDFPageObj_SetFillColor(rect, 255, 255, 255, 255)
    pdfium_c.FPDFPath_SetDrawMode(rect, pdfium_c.FPDF_FILLMODE_WINDING, 0)
    pdfium_c.FPDFPage_InsertObject(page, rect)
    font = pdfium_c.FPDFText_LoadStandardFont(document, b"Helvetica")
    obj = pdfium_c.FPDFPageObj_CreateTextObj(document, font, size)
    wide = (ctypes.c_ushort * (len(overlay) + 1))(*([ord(c) for c in overlay] + [0]))
    pdfium_c.FPDFText_SetText(obj, wide)
    pdfium_c.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x0 + 1.0, y0 + 1.0 + 0.21 * size)
    pdfium_c.FPDFPage_InsertObject(page, obj)
    pdfium_c.FPDFPage_GenerateContent(page)
    out = BytesIO()
    document.save(out)
    return out.getvalue()


@pytest.fixture(scope="session")
def text_card() -> bytes:
    return _text_card(LINES, SIZES)


@pytest.fixture(scope="session")
def library(text_card: bytes) -> Library:
    """A library learnt from the synthetic card at the sub-pixel offsets a shipped library uses."""
    return train([("card", text_card)])


def _figure_box(payload: bytes, text: str, size: float) -> tuple[float, float, float, float]:
    """The box, in PDF coordinates, of the first word ``text`` set at ``size`` points."""
    import pdfplumber

    with pdfplumber.open(BytesIO(payload)) as pdf:
        page = pdf.pages[0]
        for word in page.extract_words(extra_attrs=["size"]):
            if word["text"] == text and abs(word["size"] - size) < 0.1:
                return (
                    word["x0"] - 1.0,
                    page.height - word["bottom"] - 1.0,
                    word["x1"] + 1.0,
                    page.height - word["top"] + 1.0,
                )
    raise AssertionError(f"{text!r} at {size}pt not on the card")


@pytest.fixture(scope="session")
def image_card(text_card: bytes) -> bytes:
    """The card as a page image, with the first 10pt Maandprijs figure masked and replaced by 14,32."""
    return _image_card(text_card, _figure_box(text_card, "11,81", 10.0), "14,32", 10.0)


@pytest.fixture(scope="session")
def scratch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cards")


def pages_of(payload: bytes, **options: object) -> list:
    with Card(payload) as card:
        return [card.page(i, **options) for i in range(len(card))]  # type: ignore[arg-type]
