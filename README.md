# ocr_price_cards

Reads Belgian electricity and water tariff cards that are published as page
images, for [homeassistant_be_electricity_prices](https://github.com/renaudallard/homeassistant_be_electricity_prices)
and [homeassistant_be_water_prices](https://github.com/renaudallard/homeassistant_be_water_prices).
The cards themselves are kept in [be_price_cards](https://github.com/renaudallard/be_price_cards).

Since August 2026 Ecofix generates its cards as page images: the labels, the
DSO tables and the tax block are pixels, and only the figures that change
from month to month are still text. A general OCR engine reads such a card at
about 98% per value, and every miss is a decimal comma, which is the one
error a tariff card cannot afford. This engine is not general. It knows the
fonts these cards are set in, glyph by glyph, at the resolution the cards are
rasterized at, and it reads a mark as the one glyph it looks like or it
refuses the card.

## What it does

`read_pdf()` takes a PDF and returns its text shaped like pdfplumber's
`extract_text()`: one line per visual row, words separated by one space,
pages joined by a newline. That is what the integrations' extractors already
parse, so a card without a text layer can be handed to the same parser as a
card with one.

- Characters the PDF states are taken as they are and take precedence over
  anything read from the pixels under them. On an Ecofix card that is the
  month name, the standing charge, the monthly rates and the formulas.
- Everything else is read from the pixels through a glyph library.
- A page that is one embedded image at the library's resolution is read from
  the image itself, so the glyphs keep the exact pixels the publisher
  rasterized; the rectangles the publisher paints over stale figures are
  painted onto the image first. Any other page is rendered with pdfium.
- Every word says where it came from: `text`, `image` or `mixed`.

It reads a Raspberry Pi 4 page in well under a minute; there is no neural
network, no model download and no C extension, only numpy, pypdfium2 and
pdfplumber.

## How it reads

1. **Ink.** The page is flat colour almost everywhere. The large flat
   regions are the backgrounds: those that hold a solid square, and those
   too full of text to hold one but bearing marks that are not their own
   edges. The stem of a bold title letter is large and flat too, and
   bears nothing, so it stays a letter. Every other pixel belongs to the
   nearest background along its row, and its colour distance from that
   background is how much ink it carries. Marks are the connected runs of ink, each cut out and normalized
   so that its background reads 0 and its ink 1: dark text on white, white
   text on a teal band and navy text in a green cell come out as the same
   shape. A mark with one background on its left and another on its right,
   its own colour between the two, is the anti-aliased edge of a coloured
   cell, not a glyph. A dense grid of dot-sized marks is a QR code, and
   nothing inside it is text, whatever some of its modules read as.
2. **Stacks.** A dot over a stem, dots over a vowel, the two dots of a colon
   and the parts of a percent sign are gathered into one mark before
   matching, each part attaching to the base whose columns it shares and
   that it sits closest to.
3. **Matching.** A mark is compared with every template of about its size by
   the mean absolute difference over the mark's pixels, minimized over every
   placement within a pixel, and the ink mass rules out templates of another
   weight first. The best label is read; a runner-up whose extra difference
   is a small fraction of the mark's ink makes the mark ambiguous.
4. **Rows.** The marks of a row are read as the cheapest sequence of glyphs
   covering them, so an `m` that fell into two marks reads as the one `m`
   it is, a joined reading never explains less ink than the parts it
   replaces, and a mark that reads as nothing stays unread.
5. **Ambiguity.** In Product Sans a lowercase `l` and a capital `I` are the
   same shape. An ambiguous glyph is settled from its row (a comma and an
   apostrophe are one shape at two heights), from its word's font size (a
   bare stem is an `l` at a smaller size and an `i` at the word's size), from
   the lexicon of words the training cards spell, and from the case of the
   rest of the word. What none of that settles is refused.
6. **Layout.** Glyphs become words and lines by pdfplumber's rules, so that
   the text is what `extract_text()` gave for the same card when it still
   had a text layer: characters chain into a line while each top is within
   three points of the one before, a word ends at a gap wider than three
   points between characters the PDF states and wider than an eighth of the
   em between glyphs read from pixels, which is what a space is. Where two
   lines of a small header cell sit within three points, pdfplumber chains
   them into one line and interleaves their characters, and so does the
   reader; a glyph is settled on the word it was set in, though, which sits
   on one baseline. A row of display-sized marks a quarter of which read as
   nothing is a wordmark set in a font of its own, and is skipped, not
   refused.
7. **Refusal.** A mark that sits in a row of read glyphs and matches nothing,
   an ambiguity nothing settles, or two glyphs read on top of each other
   raise `UnreadableError` with the page, the pixel box and the line it sits
   on. Nothing is guessed. The `strict=False` reading lists those boxes
   instead, counts them on the line each sits on, and gives the text of the
   other lines as `trusted_text`.

## Install

Python 3.14 or later.

    pip install .

The package ships the library for the fonts the Belgian cards use, built
from cards that still carry a text layer.

## Use

    ocr-price-cards read card.pdf            # the text, pdfplumber-shaped
    ocr-price-cards read card.pdf --json     # words with boxes and sources
    ocr-price-cards read card.pdf --lenient  # skip unreadable marks, exit 1 if any

From Python:

```python
from ocr_price_cards import read_pdf, read_card, UnreadableError

try:
    text = read_card(payload)          # str, one page after the other
    document = read_pdf(payload)       # pages, lines, words, sources
except UnreadableError as err:
    ...                                # err.page, err.box, err.context

document = read_pdf(payload, strict=False)
document.text                          # every line, refused marks left out
document.trusted_text                  # only the lines on which nothing was refused
document.pages[0].unread               # the pixel boxes refused on the page
document.pages[0].lines[3].unread      # how many of them sit on that line
```

A consumer that needs certain figures reads with `strict=False` and takes
them from `trusted_text`: a line that carries any refused mark is left out
of it, so a figure that is there was read whole, and a figure that is
missing was not read rather than misread. With `--json` each line carries
its `unread` count.

`read_pdf(payload, embedded=False)` renders every page instead of reading an
embedded page image; `text_layer=False` reads the pixels alone, which is how
a card with a text layer is checked.

## Building the library

A card with a text layer says exactly which character sits where. Rendering
it at the library's resolution and cutting the marks out gives every glyph
as the card prints it, already labelled, and rendering it again a third of a
pixel to the side gives the same glyph as another rasterizer would have
placed it. That is the whole of the training:

    ocr-price-cards train library.npz card1.pdf card2.pdf ...

A card only prints its fonts at the sizes it happens to use, and the next
month's card may use another. The fonts themselves are embedded in the PDF,
so `ocr_price_cards.specimen` takes them out, sets every character they
cover at every size from 4 to 36 points and renders that through the same
rasterizer; the specimens are trained on like cards.

    ocr-price-cards words library.npz card1.pdf ...

adds to the lexicon the words the library reads off the cards' pixels
without any doubt. The text layer is what the lexicon is first built from,
but where a card overprints its table headers the text layer comes out
scrambled while the pixels read cleanly.

## Checking a library

    ocr-price-cards check card.pdf ...

renders each page of a card that has a text layer, reads it from the pixels
alone and compares with the text layer laid out by the same rules. A page
passes when every glyph agrees; a difference is printed as a diff. Rendering
with `--shift 0.33,0.67` reads the page off the pixel grid, which is what a
card rasterized elsewhere looks like.

Some differences are in the cards, not in the reading, and the diff shows
them page after page: a title the text layer states twice on top of itself
and the pixels show once; a plus sign drawn as a shape between two boxes,
which the pixels read as `+` and the text layer does not have; two text runs
set a visible gap apart with no space character between them, which
pdfplumber joins (`inmei`, `1.Elke`) and the pixels separate; and a table
label whose top sits three points from its figures' to within a pixel, which
pdfplumber chains onto one line or not as the rounding falls. On the Ecofix
cards every other line agrees glyph for glyph, and nothing is refused.

Against the real Ecofix cards the reading is scored the other way round: the
page image the September card carries is July's card, whose DSO tables and
labels are identical to the May card's text layer, so every glyph of those
tables has a known answer. The runner-up margins measured there are what set
the ambiguity threshold.

## What it does not do

- It reads the fonts it was trained on and nothing else. A supplier that
  switches fonts needs a card with a text layer, or its embedded fonts, to
  train from.
- It reads at the resolution it was trained at, 216 dpi. A page image at
  another resolution is rendered at 216 dpi instead of being read directly.
- It does not read rotated text, photographs or text over gradients.
- A logo whose letters are set in a font the library does not hold is
  skipped as a wordmark; a heading in a known font with one glyph refused is
  refused, as it should be. The title of the Ecofix Flexy cards is such a
  heading: the export clips the descender of its `y`, and the mark that
  remains matches nothing whole.
- Lines are pdfplumber's lines, with what that entails: two lines of a small
  table cell a few points apart chain into one, exactly as `extract_text()`
  chains them on a card with a text layer.
- A page image that is a stale month's card stays stale: the reader says
  what the pixels say. Whether a figure read off the image is current is for
  the caller to decide.

## Development

    pip install -e .[dev]
    pytest
    ruff check ocr_price_cards tests && ruff format --check ocr_price_cards tests
    mypy ocr_price_cards

The tests build their own cards with pdfium's Helvetica, train a library from
them and read them back rasterized, off the pixel grid, and as an image-only
card with a masked figure and text over it, so they need no real card and no
proprietary font.
