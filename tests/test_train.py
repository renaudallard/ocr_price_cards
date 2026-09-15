"""Training: the lexicon and what a library learns."""

from __future__ import annotations

from ocr_price_cards.library import Library
from ocr_price_cards.train import harvest_words


def test_the_lexicon_holds_the_words_of_the_training_cards(library: Library) -> None:
    assert {"Fluvius", "Antwerpen", "Verbruik", "Injectie", "Maandprijs", "kWh"} <= library.words
    assert "5,03288" in library.words
    assert "" not in library.words


def test_harvest_reads_the_words_back_off_the_pixels(text_card: bytes, library: Library) -> None:
    bare = Library(library.dpi, list(library.templates), set())
    added = harvest_words([("card", text_card)], bare)
    assert added > 0
    assert {"Fluvius", "Antwerpen", "Verbruik", "Injectie", "Maandprijs"} <= bare.words
    assert "5,03288" not in bare.words


def test_templates_remember_how_many_marks_made_them(library: Library) -> None:
    parts = {t.label: t.parts for t in library.templates if t.label in ("i", "j", ":", "l", "n")}
    assert parts["i"] == 2 and parts["j"] == 2 and parts[":"] == 2
    assert parts["l"] == 1 and parts["n"] == 1


def test_specimens_refuse_a_library_built_at_another_dpi(library: Library) -> None:
    import pytest

    from ocr_price_cards.train import train_specimens

    assert train_specimens([], library=library) is library
    assert train_specimens([], library=library, dpi=library.dpi) is library
    with pytest.raises(ValueError, match="not 300"):
        train_specimens([], library=library, dpi=300.0)
