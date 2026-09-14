"""The glyph library: matching, ambiguity and persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ocr_price_cards.errors import LibraryError
from ocr_price_cards.library import Library, Template
from ocr_price_cards.reader import ACCEPT, decide


def _bar(height: int, width: int, gap: int | None = None) -> np.ndarray:
    patch = np.zeros((height, width), dtype=np.float32)
    patch[:, 1:-1] = 1.0
    if gap is not None:
        patch[gap : gap + 2, :] = 0.0
    return patch


def _library() -> Library:
    return Library(
        216.0,
        [
            Template("l", _bar(16, 5), 21.0, 15.0, 0.05, 0.05, "Test", parts=1),
            Template("i", _bar(16, 5, gap=3), 21.0, 15.0, 0.05, 0.05, "Test", parts=2),
            Template("I", _bar(16, 5), 21.0, 15.0, 0.05, 0.05, "Test", parts=1),
            Template(".", _bar(5, 5), 21.0, 4.0, 0.05, 0.05, "Test"),
        ],
    )


def test_identical_templates_fold_together() -> None:
    library = _library()
    assert len(library) == 4
    library.add(Template("l", _bar(16, 5), 21.0, 15.0, 0.05, 0.05, "Test"))
    assert len(library) == 4
    assert next(t for t in library.templates if t.label == "l").count == 2


def test_match_scores_the_identical_template_zero_and_a_pixel_off_nearly_zero() -> None:
    library = _library()
    exact = library.match(_bar(16, 5))
    assert exact[0].label in ("l", "I") and exact[0].score == 0.0
    shifted = np.zeros((17, 6), dtype=np.float32)
    shifted[1:, 1:] = _bar(16, 5)
    assert library.match(shifted)[0].score < 0.02


def test_a_bar_is_ambiguous_between_l_and_capital_i_but_not_with_i() -> None:
    candidates = decide(_library().match(_bar(16, 5)))
    assert candidates is not None
    assert {m.label for m in candidates} == {"l", "I"}


def test_a_mark_unlike_anything_is_refused() -> None:
    blob = np.zeros((16, 5), dtype=np.float32)
    blob[::2] = 1.0
    matches = _library().match(blob)
    assert not matches or matches[0].score > ACCEPT
    assert decide(matches) is None


def test_mass_prefilter_skips_templates_of_another_weight() -> None:
    faint = _bar(16, 5) * 0.4
    assert (
        all(m.label != "l" for m in _library().match(faint))
        or _library().match(faint)[0].score > 0.0
    )


def test_round_trip_through_a_file(tmp_path: Path) -> None:
    library = _library()
    library.words.update({"Fluvius", "Injectie"})
    path = tmp_path / "lib.npz"
    library.save(path)
    loaded = Library.load(path)
    assert loaded.dpi == 216.0
    assert len(loaded) == 4
    assert loaded.words == {"Fluvius", "Injectie"}
    original = {(t.label, t.height, t.width, t.parts) for t in library.templates}
    assert {(t.label, t.height, t.width, t.parts) for t in loaded.templates} == original
    assert loaded.match(_bar(16, 5))[0].score == 0.0


def test_an_empty_library_cannot_be_saved(tmp_path: Path) -> None:
    with pytest.raises(LibraryError):
        Library(216.0).save(tmp_path / "empty.npz")


def test_a_broken_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.npz"
    path.write_bytes(b"not a library")
    with pytest.raises(LibraryError):
        Library.load(path)


def test_the_shipped_library_loads_and_covers_the_figures() -> None:
    from ocr_price_cards import default_library

    library = default_library()
    assert library.dpi == 216.0
    assert set("0123456789,.€") <= library.labels()
    assert "Fluvius" in library.words
