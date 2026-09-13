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

"""Read Belgian tariff cards that are published as page images.

The cards a supplier prints are set in a handful of fonts at a handful of
sizes. Every glyph they can show is held in a library, exactly as the card
rasterizes it, and reading a card is matching its marks against that library:
a mark reads as the one glyph it looks like, or the card is refused.
"""

from .errors import LibraryError, OcrError, UnreadableError
from .layout import Glyph, Line, Word
from .library import Library, Template
from .reader import Document, PageText, default_library, read_pdf

__version__ = "0.1.0"


def read_card(payload: bytes, **options: object) -> str:
    """The text of a card, shaped like pdfplumber's ``extract_text()`` page after page."""
    return read_pdf(payload, **options).text  # type: ignore[arg-type]


__all__ = [
    "Document",
    "Glyph",
    "Library",
    "LibraryError",
    "Line",
    "OcrError",
    "PageText",
    "Template",
    "UnreadableError",
    "Word",
    "__version__",
    "default_library",
    "read_card",
    "read_pdf",
]
