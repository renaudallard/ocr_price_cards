# Copyright (c) 2026, Renaud Allard <renaud@allard.it>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Command line: read a card, build a library, or check a library against cards with a text layer."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import OcrError
from .layout import build_lines, text_of
from .library import Library
from .pages import DEFAULT_DPI, load_pages
from .reader import Document, default_library, glyph_from_char, read_page, read_pdf
from .specimen import specimens
from .train import SUBPIXEL, harvest_words, train, train_specimens


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ocr-price-cards", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    read = commands.add_parser("read", help="print what a card says")
    read.add_argument("card", type=Path)
    read.add_argument(
        "--library", type=Path, help="glyph library to read with (default: the shipped one)"
    )
    read.add_argument(
        "--json", action="store_true", help="print words with their boxes and sources"
    )
    read.add_argument(
        "--render",
        action="store_true",
        help="rasterize every page instead of reading an embedded page image",
    )
    read.add_argument(
        "--no-text-layer", action="store_true", help="ignore the text layer and read pixels only"
    )
    read.add_argument(
        "--lenient", action="store_true", help="skip unreadable marks instead of refusing the card"
    )
    read.set_defaults(func=_read)

    build = commands.add_parser(
        "train", help="build a glyph library from cards that carry a text layer"
    )
    build.add_argument("library", type=Path, help="npz file to write")
    build.add_argument("cards", type=Path, nargs="+")
    build.add_argument("--dpi", type=float, default=DEFAULT_DPI)
    build.add_argument("--append", action="store_true", help="add to the library if it exists")
    build.add_argument(
        "--single-shift",
        action="store_true",
        help="render once instead of at every sub-pixel offset",
    )
    build.set_defaults(func=_train)

    grow = commands.add_parser(
        "specimen", help="learn the fonts embedded in cards at every size, into a library"
    )
    grow.add_argument("library", type=Path, help="npz file to write, added to if it exists")
    grow.add_argument("cards", type=Path, nargs="+", help="cards whose embedded fonts to set")
    grow.add_argument("--dpi", type=float, default=DEFAULT_DPI)
    grow.add_argument(
        "--single-shift",
        action="store_true",
        help="render once instead of at every sub-pixel offset",
    )
    grow.set_defaults(func=_specimen)

    words = commands.add_parser(
        "words", help="add the words a library reads off cards' pixels to its lexicon"
    )
    words.add_argument("library", type=Path)
    words.add_argument("cards", type=Path, nargs="+")
    words.set_defaults(func=_words)

    check = commands.add_parser(
        "check", help="read cards from pixels alone and compare with their text layer"
    )
    check.add_argument("cards", type=Path, nargs="+")
    check.add_argument("--library", type=Path)
    check.add_argument(
        "--shift", type=_shift, default=(0.0, 0.0), help="sub-pixel rendering offset, e.g. 0.5,0.5"
    )
    check.add_argument("--quiet", action="store_true", help="only print the summary")
    check.set_defaults(func=_check)

    args = parser.parse_args(argv)
    try:
        result: int = args.func(args)
    except (OcrError, OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return result


def _shift(value: str) -> tuple[float, float]:
    """A sub-pixel rendering offset, written as two numbers with a comma between them."""
    try:
        x, y = (float(v) for v in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a pair of numbers, e.g. 0.5,0.5"
        ) from None
    return x, y


def _library(path: Path | None) -> Library:
    return Library.load(path) if path is not None else default_library()


def _read(args: argparse.Namespace) -> int:
    document = read_pdf(
        args.card.read_bytes(),
        library=_library(args.library),
        embedded=not args.render,
        text_layer=not args.no_text_layer,
        strict=not args.lenient,
    )
    if args.json:
        print(json.dumps(_as_json(document), ensure_ascii=False, indent=1))
    else:
        print(document.text)
    unread = sum(len(page.unread) for page in document.pages)
    if unread:
        print(f"{unread} unreadable mark(s) skipped", file=sys.stderr)
    return 1 if unread else 0


def _as_json(document: Document) -> dict[str, object]:
    return {
        "pages": [
            {
                "number": page.number,
                "width": page.width,
                "height": page.height,
                "source": page.source,
                "unread": page.unread,
                "lines": [
                    {
                        "unread": line.unread,
                        "words": [
                            {
                                "text": word.text,
                                "x0": round(word.x0, 2),
                                "top": round(word.top, 2),
                                "x1": round(word.x1, 2),
                                "bottom": round(word.bottom, 2),
                                "source": word.source,
                            }
                            for word in line.words
                        ],
                    }
                    for line in page.lines
                ],
            }
            for page in document.pages
        ]
    }


def _train(args: argparse.Namespace) -> int:
    library = Library.load(args.library) if args.append and args.library.exists() else None
    shifts = ((0.0, 0.0),) if args.single_shift else SUBPIXEL
    library = train(
        ((str(card), card.read_bytes()) for card in args.cards),
        dpi=args.dpi,
        shifts=shifts,
        library=library,
        report=print,
    )
    library.save(args.library)
    print(f"{args.library}: {len(library)} templates, {len(library.labels())} labels")
    return 0


def _specimen(args: argparse.Namespace) -> int:
    library = Library.load(args.library) if args.library.exists() else Library(args.dpi)
    shifts = ((0.0, 0.0),) if args.single_shift else SUBPIXEL
    items = specimens(card.read_bytes() for card in args.cards)
    for item in items:
        print(
            f"{item.family}: {sum(len(page) for page in item.chars)} characters set on {len(item.chars)} page(s)"
        )
    train_specimens(items, library=library, dpi=args.dpi, shifts=shifts, report=print)
    library.save(args.library)
    print(f"{args.library}: {len(library)} templates, {len(library.labels())} labels")
    return 0


def _words(args: argparse.Namespace) -> int:
    library = Library.load(args.library)
    added = harvest_words(
        ((str(card), card.read_bytes()) for card in args.cards), library, report=print
    )
    library.save(args.library)
    print(f"{args.library}: {added} words added, {len(library.words)} in the lexicon")
    return 0


def _check(args: argparse.Namespace) -> int:
    library = _library(args.library)
    failed = 0
    for card in args.cards:
        for page in load_pages(
            card.read_bytes(), dpi=library.dpi, embedded=False, shift=args.shift
        ):
            expected = text_of(build_lines([glyph_from_char(char) for char in page.chars]))
            got = read_page(page, library, text_layer=False, strict=False)
            status = _verdict(expected, got.text, bool(got.unread))
            if status == "DIFFERS":
                failed += 1
            print(f"{card} page {page.number}: {status}, {len(got.unread)} unreadable")
            if status != "identical" and not args.quiet:
                diff = difflib.unified_diff(
                    expected.splitlines(),
                    got.text.splitlines(),
                    "text layer",
                    "pixels",
                    lineterm="",
                    n=0,
                )
                for line in diff:
                    print("   ", line)
    print(f"{failed} page(s) differ")
    return 1 if failed else 0


def _verdict(expected: str, got: str, unread: bool) -> str:
    """How a pixel reading compares with the text layer.

    ``identical`` is the same text; ``same glyphs`` the same characters with
    only the spaces and line breaks placed differently, which the two layouts
    do around overprinted or widely spaced text; anything else, or a mark
    left unread, ``DIFFERS``.
    """
    if unread:
        return "DIFFERS"
    if got == expected:
        return "identical"
    if "".join(got.split()) == "".join(expected.split()):
        return "same glyphs"
    return "DIFFERS"


if __name__ == "__main__":
    sys.exit(main())
