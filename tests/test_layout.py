"""Lines and words."""

from __future__ import annotations

from ocr_price_cards.layout import (
    SPACE,
    X_TOLERANCE,
    Glyph,
    build_lines,
    build_rows,
    build_words,
    cluster,
)


def _glyph(
    text: str, x0: float, width: float, baseline: float, size: float = 8.0, source: str = "image"
) -> Glyph:
    return Glyph(text, x0, x0 + width, baseline, size, source)


def test_cluster_keeps_one_line_together_and_splits_the_next() -> None:
    groups = cluster([100.0, 100.3, 100.6, 100.9, 106.0, 106.2], 3.0)
    assert groups[100.0] == groups[100.9]
    assert groups[106.0] == groups[106.2]
    assert groups[100.0] != groups[106.0]


def test_cluster_chains_a_staircase_the_way_pdfplumber_does() -> None:
    groups = cluster([100.0, 102.5, 105.0, 107.5, 111.0], 3.0)
    assert len({groups[100.0], groups[102.5], groups[105.0], groups[107.5]}) == 1
    assert groups[111.0] != groups[107.5]


def test_pixel_glyphs_split_at_a_space_sized_gap() -> None:
    size = 8.0
    glyphs = [
        _glyph("a", 10.0, 4.0, 100.0, size),
        _glyph("b", 14.2, 4.0, 100.0, size),
        _glyph("c", 18.2 + SPACE * size + 0.5, 4.0, 100.0, size),
    ]
    assert [w.text for w in build_words(glyphs)] == ["ab", "c"]


def test_text_glyphs_split_only_past_pdfplumber_tolerance() -> None:
    glyphs = [
        _glyph("a", 10.0, 4.0, 100.0, source="text"),
        _glyph("b", 16.5, 4.0, 100.0, source="text"),
        _glyph("c", 20.5 + X_TOLERANCE + 0.1, 4.0, 100.0, source="text"),
    ]
    assert [w.text for w in build_words(glyphs)] == ["ab", "c"]


def test_whitespace_text_glyph_ends_a_word() -> None:
    glyphs = [
        _glyph("a", 10.0, 4.0, 100.0, source="text"),
        _glyph(" ", 14.0, 2.0, 100.0, source="text"),
        _glyph("b", 16.0, 4.0, 100.0, source="text"),
    ]
    assert [w.text for w in build_words(glyphs)] == ["a", "b"]


def test_lines_are_ordered_top_down_and_words_left_to_right() -> None:
    glyphs = [
        _glyph("b", 30.0, 4.0, 200.0),
        _glyph("a", 10.0, 4.0, 200.0),
        _glyph("x", 10.0, 4.0, 100.0),
    ]
    lines = build_lines(glyphs)
    assert [line.text for line in lines] == ["x", "a b"]
    assert lines[1].words[0].source == "image"


def test_rows_keep_the_two_lines_of_a_cell_apart_where_lines_chain_them() -> None:
    # Two lines 2.5pt apart in a header cell: pdfplumber chains them into one
    # line and interleaves the characters, a row keeps each word whole.
    size = 5.0
    upper = [_glyph(c, 10.0 + 3.0 * i, 3.0, 100.0, size) for i, c in enumerate("Totaal")]
    lower = [_glyph(c, 11.5 + 3.0 * i, 3.0, 102.5, size) for i, c in enumerate("kWh")]
    glyphs = upper + lower
    assert len(build_lines(glyphs)) == 1
    assert [word.text for row in build_rows(glyphs) for word in row.words] == ["Totaal", "kWh"]
