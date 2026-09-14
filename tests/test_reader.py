"""Reading whole cards: text-era cards rasterized, image-only cards, refusals."""

from __future__ import annotations

import pytest
from conftest import pages_of

from ocr_price_cards.errors import UnreadableError
from ocr_price_cards.ink import Blob
from ocr_price_cards.layout import build_lines, text_of
from ocr_price_cards.library import Library, Match, Template
from ocr_price_cards.reader import Read, glyph_from_char, read_page, read_pdf
from ocr_price_cards.train import train


def _expected(payload: bytes) -> str:
    page = pages_of(payload, embedded=False)[0]
    return text_of(build_lines([glyph_from_char(char) for char in page.chars]))


def test_a_rasterized_card_reads_exactly_like_its_text_layer(
    text_card: bytes, library: Library
) -> None:
    page = pages_of(text_card, embedded=False)[0]
    result = read_page(page, library, text_layer=False)
    assert result.text == _expected(text_card)
    assert not result.unread
    assert all(word.source == "image" for word in result.words)


def test_a_card_rendered_off_the_pixel_grid_still_reads_exactly(
    text_card: bytes, library: Library
) -> None:
    # Half a pixel lies farthest from every offset the library learnt at.
    page = pages_of(text_card, embedded=False, shift=(0.5, 0.5))[0]
    result = read_page(page, library, text_layer=False)
    assert result.text == _expected(text_card)


def test_an_image_card_reads_the_image_and_splices_the_text_layer(
    image_card: bytes, text_card: bytes, library: Library
) -> None:
    document = read_pdf(image_card, library=library)
    assert document.pages[0].source == "image"
    lines = _expected(text_card).splitlines()
    index = [i for i, line in enumerate(lines) if line.startswith("Maandprijs")][-1]
    lines[index] = lines[index].replace("11,81", "14,32", 1)
    assert document.text == "\n".join(lines)
    overlaid = next(word for word in document.pages[0].words if word.text == "14,32")
    assert overlaid.source == "text"
    assert all(word.source == "image" for word in document.pages[0].words if word is not overlaid)


def test_the_same_image_card_reads_the_same_when_rendered(
    image_card: bytes, text_card: bytes, library: Library
) -> None:
    document = read_pdf(image_card, library=library, embedded=False)
    assert document.pages[0].source == "render"
    assert document.text == read_pdf(image_card, library=library).text


def test_a_glyph_the_library_lacks_refuses_the_card(text_card: bytes) -> None:
    partial = train([("card", text_card)], shifts=((0.0, 0.0),))
    partial.templates = [t for t in partial.templates if t.label not in "&"]
    partial = Library(partial.dpi, partial.templates, partial.words)
    with pytest.raises(UnreadableError) as caught:
        read_pdf(text_card, library=partial, embedded=False, text_layer=False)
    assert caught.value.page == 1
    assert "3.000" in caught.value.context or "kWh" in caught.value.context
    lenient = read_pdf(text_card, library=partial, embedded=False, text_layer=False, strict=False)
    assert lenient.pages[0].unread


def test_read_card_returns_the_text(text_card: bytes, library: Library) -> None:
    from ocr_price_cards import read_card

    assert read_card(text_card, library=library, embedded=False) == _expected(text_card)


def test_a_reading_keeps_only_templates_made_of_as_many_marks() -> None:
    import numpy as np

    patch = np.ones((4, 2), dtype=np.float32)
    blob = Blob(0, 0, 2, 4, patch, (255, 255, 255), (0, 0, 0))
    stem = Template("I", patch, 21.0, 4.0, 0.0, 0.0, parts=1)
    dotted = Template("i", patch, 21.0, 4.0, 0.0, 0.0, parts=2)
    read = Read(blob, [Match("I", 0.04, stem), Match("i", 0.07, dotted)], [blob])
    assert [match.label for match in read.candidates] == ["I"]
    read = Read(blob, [Match("i", 0.03, dotted), Match("I", 0.05, stem)], [blob, blob])
    assert [match.label for match in read.candidates] == ["i"]
