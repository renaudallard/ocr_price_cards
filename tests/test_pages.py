"""Opening a card, and what the page a card carries has to look like to be read as it is."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

import ocr_price_cards.pages as pages
from ocr_price_cards.errors import OcrError


def test_a_card_that_opens_half_way_closes_what_it_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    class Document:
        def close(self) -> None:
            closed.append("pdfium")

    def refuse(_stream: Any) -> Any:
        raise ValueError("pdfplumber will not have this one")

    monkeypatch.setattr(pages.pdfium, "PdfDocument", lambda payload: Document())
    monkeypatch.setattr(pages.pdfplumber, "open", refuse)
    with pytest.raises(OcrError, match="cannot read the card"):
        pages.Card(b"not really a pdf")
    assert closed == ["pdfium"]


def test_a_card_neither_reader_will_take_is_refused() -> None:
    # Whatever the readers raise over a file that is not a card, the caller
    # asked for a card and did not get one.
    for payload in (b"", b"certainly not a pdf", b"%PDF-1.7\nand then nothing"):
        with pytest.raises(OcrError, match="cannot read the card"):
            pages.Card(payload)


def _image_page(width_px: int, height_px: int) -> bytes:
    """A one page PDF that is a single full page image of that many pixels."""
    from io import BytesIO

    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    document = pdfium.PdfDocument.new()
    page = document.new_page(595.0, 842.0)
    image = pdfium_c.FPDFPageObj_NewImageObj(document)
    canvas = pdfium.PdfBitmap.new_native(
        width_px, height_px, format=pdfium_c.FPDFBitmap_BGR, rev_byteorder=False
    )
    canvas.fill_rect((255, 255, 255, 255), 0, 0, width_px, height_px)
    pdfium_c.FPDFImageObj_SetBitmap(None, 0, image, canvas)
    pdfium_c.FPDFPageObj_Transform(image, 595.0, 0, 0, 842.0, 0, 0)
    pdfium_c.FPDFPage_InsertObject(page, image)
    pdfium_c.FPDFPage_GenerateContent(page)
    out = BytesIO()
    document.save(out)
    return out.getvalue()


def test_a_page_image_squashed_one_way_is_rendered_rather_than_read() -> None:
    # 595 by 842 points at 216 dpi is 1785 by 2526 pixels.
    with pages.Card(_image_page(1785, 2526)) as card:
        assert card.page(0, dpi=216.0).source == "image"
    # The same across, half as many down: taken as it is, its glyphs would
    # be matched against templates twice their height.
    with pages.Card(_image_page(1785, 1263)) as card:
        assert card.page(0, dpi=216.0).source == "render"


def test_a_card_closes_both_readers_even_if_the_first_will_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pathlib import Path as P

    closed: list[str] = []
    card = pages.Card(P("tmp/synthetic_text.pdf").read_bytes())

    def refuse() -> None:
        raise RuntimeError("pdfplumber will not close")

    monkeypatch.setattr(card._plumber, "close", refuse)
    monkeypatch.setattr(card._document, "close", lambda: closed.append("pdfium"))
    with pytest.raises(RuntimeError):
        card.close()
    assert closed == ["pdfium"]


def test_a_card_whose_readers_disagree_on_its_length_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ocr_price_cards.errors import OcrError

    closed: list[str] = []

    class Plumber:
        pages: ClassVar[list[None]] = [None, None, None]

        def close(self) -> None:
            closed.append("plumber")

    payload = Path("tmp/synthetic_text.pdf").read_bytes()
    monkeypatch.setattr(pages.pdfplumber, "open", lambda _b: Plumber())
    # The length comes from one reader and the pages from the other, so a
    # card they read differently would lose a page or ask for a missing one.
    with pytest.raises(OcrError, match="disagree on its length"):
        pages.Card(payload)
    assert closed == ["plumber"]
