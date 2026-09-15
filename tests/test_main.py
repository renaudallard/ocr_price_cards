"""The command line: what a user gets wrong comes back as an error, not a traceback."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocr_price_cards.__main__ import _shift, main


def test_a_shift_that_is_not_a_pair_of_numbers_is_rejected() -> None:
    assert _shift("0,0") == (0.0, 0.0)
    assert _shift("0.33,0.67") == (0.33, 0.67)
    with pytest.raises(SystemExit) as caught:
        main(["check", "--shift", "sideways", "card.pdf"])
    assert caught.value.code == 2


def test_a_card_that_is_not_there_is_an_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["read", str(tmp_path / "absent.pdf")]) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_a_library_that_is_not_a_library_is_an_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"PK\x03\x04 and then nothing that follows")
    # The card is read before the library is, so it has to be there for the
    # library to be the thing that fails.
    card = tmp_path / "card.pdf"
    card.write_bytes(b"%PDF-1.7\n")
    assert main(["read", "--library", str(broken), str(card)]) == 2
    assert "cannot load the glyph library" in capsys.readouterr().err
