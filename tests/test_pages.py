"""Opening a card, and what the page a card carries has to look like to be read as it is."""

from __future__ import annotations

from typing import Any

import pytest

import ocr_price_cards.pages as pages


def test_a_card_that_opens_half_way_closes_what_it_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    class Document:
        def close(self) -> None:
            closed.append("pdfium")

    def refuse(_stream: Any) -> Any:
        raise ValueError("pdfplumber will not have this one")

    monkeypatch.setattr(pages.pdfium, "PdfDocument", lambda payload: Document())
    monkeypatch.setattr(pages.pdfplumber, "open", refuse)
    with pytest.raises(ValueError):
        pages.Card(b"not really a pdf")
    assert closed == ["pdfium"]
