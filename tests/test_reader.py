"""Reading whole cards: text-era cards rasterized, image-only cards, refusals."""

from __future__ import annotations

import pytest
from conftest import pages_of

from ocr_price_cards.errors import UnreadableError
from ocr_price_cards.ink import Blob
from ocr_price_cards.layout import Glyph, Line, Word, build_lines, text_of
from ocr_price_cards.library import Library, Match, Template
from ocr_price_cards.reader import (
    Read,
    _without_texture,
    glyph_from_char,
    read_page,
    read_pdf,
    settle,
)
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


def test_everything_inside_a_grid_of_dots_goes_with_the_grid() -> None:
    import numpy as np

    def blob(x0: int, y0: int, width: int, height: int) -> Blob:
        return Blob(
            x0,
            y0,
            x0 + width,
            y0 + height,
            np.ones((height, width), np.float32),
            (255,) * 3,
            (0,) * 3,
        )

    def read(b: Blob, label: str) -> Read:
        template = Template(label, b.patch, 21.0, float(b.height), 0.0, 0.0)
        return Read(b, [Match(label, 0.02, template)], [b])

    # A QR code: a grid of 4px modules, one of which pairs into a colon and
    # four of which gather into an o. Then a word well away from it.
    modules = [
        blob(100 + 8 * i, 100 + 8 * j, 4, 4) for i in range(6) for j in range(6) if (i + j) % 3
    ]
    colon = read(blob(108, 100, 4, 12), ":")
    o = read(blob(116, 116, 12, 12), "o")
    word = [read(blob(300 + 10 * i, 104, 8, 10), c) for i, c in enumerate("kWh")]
    matched, unmatched = _without_texture([colon, o, *word], modules)
    assert [r.match.label for r in matched] == ["k", "W", "h"]
    assert unmatched == []


def test_a_compound_settles_on_the_words_around_its_hyphen() -> None:
    glyphs = [
        Glyph(c, 10.0 + 4.0 * i, 14.0 + 4.0 * i, 100.0, 7.0, "image", box=(i, 0, i + 1, 1))
        for i, c in enumerate("Online-formule")
    ]
    glyphs[0].alternatives = ("0",)
    glyphs[7].alternatives = ("t",)
    line = Line([Word(glyphs)])
    assert settle([line], {"Online", "formule"}) == []
    assert line.text == "Online-formule"
    glyphs[0].alternatives = ("0",)
    assert settle([line], {"formule"}) == [(0, 0, 1, 1)]
    glyphs[0].alternatives = ("0",)
    assert settle([line], {"Online", "Flexy-formule"}) == []


def test_a_refused_mark_is_left_out_of_the_text() -> None:
    from ocr_price_cards.reader import _without_refused

    def glyph(text: str, x0: float, box: tuple[int, int, int, int] | None) -> Glyph:
        return Glyph(text, x0, x0 + 5.0, 100.0, 10.0, "image" if box else "text", box=box)

    # A word whose first mark was refused, a line every mark of which was,
    # and a character the PDF states, which carries no box and never goes.
    word = Word([glyph("R", 0.0, (1, 1, 2, 2)), glyph("e", 5.0, (3, 1, 4, 2))])
    stated = Word([glyph("A", 0.0, None)])
    whole = Line([Word([glyph("x", 0.0, (9, 9, 10, 10))])])
    lines = [Line([word, stated]), whole]
    kept = _without_refused(lines, {(1, 1, 2, 2), (9, 9, 10, 10)})
    assert [line.text for line in kept] == ["e A"]
    assert _without_refused(lines, set()) is lines


def test_a_refused_ambiguity_leaves_its_word_short(text_card: bytes) -> None:
    trained = train([("card", text_card)], shifts=((0.0, 0.0),))
    # Without a lexicon the capital I of Injectie is an I or an l and nothing
    # settles it. It is refused after it has become a glyph, which is the
    # refusal the text used to print anyway.
    bare = Library(trained.dpi, trained.templates, set())
    page = read_pdf(text_card, library=bare, embedded=False, text_layer=False, strict=False).pages[
        0
    ]
    assert page.unread
    refused = set(page.unread)
    assert [g.text for word in page.words for g in word.glyphs if g.box in refused] == []
    # The card sets the word at four sizes and the I is refused at some of
    # them, so the short spelling is there beside the whole one.
    assert "njectie:" in page.text


def test_modules_of_two_widths_are_searched_in_the_order_they_are_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    import ocr_price_cards.reader as reader

    def blob(x0: int, y0: int, width: int) -> Blob:
        return Blob(
            x0, y0, x0 + width, y0 + 3, np.ones((3, width), np.float32), (255,) * 3, (0,) * 3
        )

    sorted_input: list[bool] = []
    real = np.searchsorted

    def checked(a: object, v: object, **kw: object) -> object:
        sorted_input.append(bool(np.all(np.diff(np.asarray(a)) >= 0)))
        return real(a, v, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(reader.np, "searchsorted", checked)
    # Marks cut from a code are not all one width, and where a wide one
    # starts just before a narrow one their centres fall in the opposite
    # order to their left edges.
    grid = [
        blob(9 * c + (0 if wide else 1), 7 * r, 5 if wide else 1)
        for r in range(4)
        for c in range(4)
        for wide in (True, False)
    ]
    away = blob(900, 7, 4)
    matched, unmatched = _without_texture([], [*grid, away])
    assert matched == []
    assert unmatched == [away]
    assert sorted_input and all(sorted_input)
