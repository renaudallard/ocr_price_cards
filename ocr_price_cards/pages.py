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

"""What a page looks like and what it already says.

A page is read from two sources at once: its pixels, which is where an
image-only card keeps its figures, and its text layer, which is exact wherever
it exists and takes precedence over anything read from the pixels underneath.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt
import pdfplumber
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from .errors import OcrError

Pixels = npt.NDArray[np.uint8]

DEFAULT_DPI = 216.0
"""Ecofix rasterizes its cards at 216 dpi, and the shipped library is built at that resolution."""

_PAGE_IMAGE_COVERAGE = 0.9
_MASK_MAX_COVERAGE = 0.5
_DPI_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class TextChar:
    """One character of the text layer, in points from the page's top left corner.

    ``baseline`` comes from the text matrix rather than from the character box:
    a font descriptor without metrics gives pdfplumber a box that floats well
    above the ink, while the matrix always says where the glyph was drawn.
    """

    text: str
    x0: float
    x1: float
    baseline: float
    size: float
    fontname: str


@dataclass(frozen=True, slots=True)
class Transform:
    """Maps page points (origin top left, y down) to canvas pixels and back."""

    sx: float
    sy: float
    ox: float = 0.0
    oy: float = 0.0

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.ox) * self.sx, (y - self.oy) * self.sy

    def to_pt(self, px: float, py: float) -> tuple[float, float]:
        return px / self.sx + self.ox, py / self.sy + self.oy


@dataclass(slots=True)
class Page:
    """One page: pixels to read, the text layer to splice in, and how they align.

    ``origin`` is where the page's top left corner falls in pdfplumber's
    coordinates, which keep the PDF's own units and its MediaBox offset; two
    cards laid out alike but boxed differently align through it.
    """

    number: int
    width: float
    height: float
    pixels: Pixels
    transform: Transform
    chars: list[TextChar]
    source: str
    origin: tuple[float, float] = (0.0, 0.0)


class Card:
    """A PDF opened once: its text layer per page, and any page rendered on demand."""

    def __init__(self, payload: bytes) -> None:
        # A card is a file from elsewhere, and either reader may refuse it
        # for reasons of its own. Whatever they raise, the caller asked for
        # a card and did not get one.
        try:
            self._document = pdfium.PdfDocument(payload)
        except Exception as err:
            raise OcrError(f"cannot read the card: {err}") from err
        try:
            self._plumber = pdfplumber.open(BytesIO(payload))
        except Exception as err:
            # Nothing has entered the with statement yet, so close by hand
            # what the line above opened.
            self._document.close()
            raise OcrError(f"cannot read the card: {err}") from err
        rendered, stated = len(self._document), len(self._plumber.pages)
        if rendered != stated:
            # The length is taken from one reader and the pages from the
            # other, so a card they read differently would lose a page or
            # ask for one that is not there. Say so instead.
            self.close()
            raise OcrError(
                f"the card's readers disagree on its length: "
                f"{rendered} page(s) to render, {stated} stated"
            )
        self._chars: dict[int, list[TextChar]] = {}

    def __len__(self) -> int:
        return len(self._plumber.pages)

    def close(self) -> None:
        # Both get closed even if the first will not: the card holds two
        # readers of the same bytes and neither is the other's business.
        try:
            self._plumber.close()
        finally:
            self._document.close()

    def __enter__(self) -> Card:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chars(self, index: int) -> list[TextChar]:
        """The text layer of page ``index`` (0-based), read once."""
        found = self._chars.get(index)
        if found is None:
            found = text_chars(self._plumber.pages[index])
            self._chars[index] = found
        return found

    def page(
        self,
        index: int,
        *,
        dpi: float = DEFAULT_DPI,
        embedded: bool = True,
        shift: tuple[float, float] = (0.0, 0.0),
    ) -> Page:
        """Page ``index`` (0-based) as pixels at ``dpi`` plus its text layer.

        A page that is one embedded image at that resolution is read from the
        image itself when ``embedded`` is true, so the glyphs keep the exact
        pixels the publisher rasterized; every other page is rendered.
        ``shift`` moves a rendering by a fraction of a pixel, which is how the
        library learns the same glyph at every sub-pixel position.
        """
        scale = dpi / 72.0
        plumber_page = self._plumber.pages[index]
        pdf_page = self._document[index]
        origin = _render_origin(plumber_page)
        found = _embedded_image(pdf_page, plumber_page, dpi) if embedded else None
        if found is not None:
            pixels, transform = found
            _paint_masks(pixels, transform, plumber_page)
            source = "image"
        else:
            pixels = render(pdf_page, scale, shift)
            transform = Transform(
                scale, scale, origin[0] - shift[0] / scale, origin[1] - shift[1] / scale
            )
            source = "render"
        return Page(
            index + 1,
            float(plumber_page.width),
            float(plumber_page.height),
            pixels,
            transform,
            self.chars(index),
            source,
            origin,
        )


def load_pages(
    payload: bytes,
    *,
    dpi: float = DEFAULT_DPI,
    embedded: bool = True,
    shift: tuple[float, float] = (0.0, 0.0),
) -> list[Page]:
    """Every page of a PDF, see :meth:`Card.page`."""
    with Card(payload) as card:
        return [
            card.page(index, dpi=dpi, embedded=embedded, shift=shift) for index in range(len(card))
        ]


def _render_origin(plumber_page: Any) -> tuple[float, float]:
    """Where the top left corner of a rendering falls in pdfplumber's coordinates.

    pdfplumber keeps every object in the PDF's own units, shifted so that the
    MediaBox origin stays where the file put it; a rendering starts at the
    CropBox's top left corner. Both boxes usually start at the origin, in which
    case this is (0, 0).
    """
    mediabox = plumber_page.mediabox
    cropbox = plumber_page.cropbox
    return float(cropbox[0]), float(plumber_page.height) + 2.0 * float(mediabox[1]) - float(
        cropbox[3]
    )


def render(pdf_page: Any, scale: float, shift: tuple[float, float] = (0.0, 0.0)) -> Pixels:
    """Rasterize a page at ``scale`` pixels per point, moved by ``shift`` pixels."""
    width = math.ceil(pdf_page.get_width() * scale)
    height = math.ceil(pdf_page.get_height() * scale)
    bitmap = pdfium.PdfBitmap.new_native(
        width, height, format=pdfium_c.FPDFBitmap_BGR, rev_byteorder=True
    )
    bitmap.fill_rect((255, 255, 255, 255), 0, 0, width, height)
    matrix = pdfium_c.FS_MATRIX(scale, 0, 0, scale, shift[0], shift[1])
    clip = pdfium_c.FS_RECTF(0, 0, width, height)
    pdfium_c.FPDF_RenderPageBitmapWithMatrix(
        bitmap,
        pdf_page,
        matrix,
        clip,
        pdfium_c.FPDF_ANNOT | pdfium_c.FPDF_REVERSE_BYTE_ORDER,
    )
    return np.array(bitmap.to_numpy(), dtype=np.uint8, copy=True)


def text_chars(plumber_page: Any) -> list[TextChar]:
    """The upright characters of the text layer, deduplicated as pdfplumber does."""
    out: list[TextChar] = []
    for char in plumber_page.dedupe_chars().chars:
        text = str(char.get("text") or "")
        size = float(char.get("size") or 0.0)
        matrix = char.get("matrix")
        if not text or size <= 0.0 or not matrix or not char.get("upright", True):
            continue
        baseline = float(char["bottom"]) - (float(matrix[5]) - float(char["y0"]))
        out.append(
            TextChar(
                text,
                float(char["x0"]),
                float(char["x1"]),
                baseline,
                size,
                str(char.get("fontname") or ""),
            )
        )
    return out


def _embedded_image(
    pdf_page: Any, plumber_page: Any, dpi: float
) -> tuple[Pixels, Transform] | None:
    """The page's own image, when the page is one image at the library's resolution."""
    images = plumber_page.images
    if len(images) != 1:
        return None
    info = images[0]
    width_pt = float(info["x1"]) - float(info["x0"])
    height_pt = float(info["bottom"]) - float(info["top"])
    if width_pt <= 0.0 or height_pt <= 0.0:
        return None
    page_area = float(plumber_page.width) * float(plumber_page.height)
    if width_pt * height_pt < _PAGE_IMAGE_COVERAGE * page_area:
        return None
    src_w, src_h = (int(v) for v in info["srcsize"])
    if src_w <= 0 or src_h <= 0:
        return None
    # Both ways: an image at the right resolution across and another down is
    # squashed, and its glyphs would be matched against templates the wrong
    # shape. Rendering the page instead gives them back their proportions.
    if abs(src_w / width_pt * 72.0 - dpi) > _DPI_TOLERANCE * dpi:
        return None
    if abs(src_h / height_pt * 72.0 - dpi) > _DPI_TOLERANCE * dpi:
        return None
    for obj in pdf_page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)):
        if tuple(obj.get_px_size()) != (src_w, src_h):
            continue
        a, b, c, d, _e, _f = obj.get_matrix().get()
        if b or c or a <= 0 or d <= 0:
            return None
        try:
            bitmap = obj.get_bitmap(render=False)
        except pdfium.PdfiumError:
            return None
        pixels = np.asarray(bitmap.to_pil().convert("RGB"), dtype=np.uint8).copy()
        if pixels.shape[:2] != (src_h, src_w):
            return None
        transform = Transform(
            src_w / width_pt, src_h / height_pt, float(info["x0"]), float(info["top"])
        )
        return pixels, transform
    return None


def _paint_masks(pixels: Pixels, transform: Transform, plumber_page: Any) -> None:
    """Paint the filled rectangles drawn over the image, so stale figures under them stay hidden."""
    area = float(plumber_page.width) * float(plumber_page.height)
    height, width = pixels.shape[:2]
    for rect in plumber_page.rects:
        if not rect.get("fill"):
            continue
        colour = _rgb(rect.get("non_stroking_color"))
        if colour is None:
            continue
        rect_w = float(rect["x1"]) - float(rect["x0"])
        rect_h = float(rect["bottom"]) - float(rect["top"])
        if rect_w * rect_h >= _MASK_MAX_COVERAGE * area:
            continue
        x0, y0 = transform.to_px(float(rect["x0"]), float(rect["top"]))
        x1, y1 = transform.to_px(float(rect["x1"]), float(rect["bottom"]))
        xa, ya = max(round(x0), 0), max(round(y0), 0)
        xb, yb = min(round(x1), width), min(round(y1), height)
        if xb > xa and yb > ya:
            pixels[ya:yb, xa:xb] = colour


def _rgb(value: Any) -> tuple[int, int, int] | None:
    """A pdfminer colour (gray, RGB or CMYK components in 0..1) as 8-bit RGB."""
    if isinstance(value, int | float):
        value = (value,)
    if not isinstance(value, tuple | list):
        return None
    try:
        parts = [float(v) for v in value]
    except TypeError, ValueError:
        return None
    if len(parts) == 1:
        rgb = (parts[0], parts[0], parts[0])
    elif len(parts) == 3:
        rgb = (parts[0], parts[1], parts[2])
    elif len(parts) == 4:
        c, m, y, k = parts
        rgb = ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    else:
        return None
    r, g, b = (round(max(0.0, min(1.0, v)) * 255) for v in rgb)
    return r, g, b
