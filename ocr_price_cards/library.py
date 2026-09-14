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

"""The glyph library: every mark a card prints, and what it reads as.

A template is one mark exactly as a card rasterizes it: the normalized patch,
the font size it was set in, where its baseline runs and how much of its
advance width lies outside the ink. Matching is a plain comparison of pixels,
so a glyph reads as what it looks like and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from .errors import LibraryError

Patch = npt.NDArray[np.float32]
_Stack = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]

FORMAT = 1

SHORTLIST = 48
"""Templates kept for the exact search once a size holds more than this many candidates."""

MASS_TOLERANCE = 0.25
"""Templates whose ink mass differs from the patch's by more than this fraction are skipped."""

CACHE_SIZE = 20000


@dataclass(slots=True)
class Template:
    """One glyph as rasterized at the library's resolution.

    ``em`` is the font size in pixels, ``base`` the number of rows from the top
    of the patch down to the baseline, ``lsb`` and ``rsb`` the gaps between the
    ink and the advance box on either side as fractions of the em, and
    ``parts`` how many separate marks the glyph was made of, two for an i.
    """

    label: str
    patch: Patch
    em: float
    base: float
    lsb: float
    rsb: float
    font: str = ""
    count: int = 1
    parts: int = 1

    @property
    def height(self) -> int:
        return int(self.patch.shape[0])

    @property
    def width(self) -> int:
        return int(self.patch.shape[1])


@dataclass(frozen=True, slots=True)
class Match:
    """How well one template fits a patch.

    ``score`` is the mean absolute difference per pixel of the padded patch, 0
    for identical; ``sad`` the same difference summed, and ``mass`` the ink
    the patch holds, so ``sad / mass`` says how much of the mark is unexplained.
    """

    label: str
    score: float
    template: Template
    sad: float = 0.0
    mass: float = 0.0


@dataclass(slots=True)
class Library:
    """Templates indexed by size, with matching against a patch, and the words the cards use.

    ``words`` is every word the training cards' text layers spell, which is
    what tells a lowercase l from a capital I in a font that draws them alike.
    """

    dpi: float
    templates: list[Template] = field(default_factory=list)
    words: set[str] = field(default_factory=set)
    _by_size: dict[tuple[int, int], list[int]] = field(default_factory=dict, repr=False)
    _stacks: dict[tuple[int, int], _Stack] = field(default_factory=dict, repr=False)
    _keys: dict[tuple[str, bytes], int] = field(default_factory=dict, repr=False)
    _bearings: dict[tuple[str, float], tuple[float, float]] = field(
        default_factory=dict, repr=False
    )
    _cache: dict[tuple[tuple[int, ...], bytes, tuple[float, float] | None], list[Match]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        templates, self.templates = self.templates, []
        for template in templates:
            self.add(template)

    def __len__(self) -> int:
        return len(self.templates)

    def add(self, template: Template) -> Template:
        """Add a template, folding it into an identical one already held."""
        quantized = quantize(template.patch)
        key = (
            template.label,
            np.array(quantized.shape, dtype=np.int32).tobytes() + quantized.tobytes(),
        )
        index = self._keys.get(key)
        if index is not None:
            held = self.templates[index]
            held.count += template.count
            return held
        template.patch = quantized.astype(np.float32) / 255.0
        self._keys[key] = len(self.templates)
        self.templates.append(template)
        size = (template.height, template.width)
        self._by_size.setdefault(size, []).append(len(self.templates) - 1)
        self._stacks.pop(size, None)
        self._cache.clear()
        return template

    def bearings(self, label: str, em: float) -> tuple[float, float]:
        """The side bearings of ``label`` at ``em``, in ems, as the mean over its templates.

        Each template's own bearings carry the rounding of the raster it was
        cut from, up to a pixel a side; the mean over the sub-pixel offsets
        it was learnt at is the font's.
        """
        key = (label, em)
        found = self._bearings.get(key)
        if found is None:
            alike = [t for t in self.templates if t.label == label and abs(t.em - em) < 0.01]
            found = (
                sum(t.lsb for t in alike) / len(alike),
                sum(t.rsb for t in alike) / len(alike),
            )
            self._bearings[key] = found
        return found

    def labels(self) -> set[str]:
        return {template.label for template in self.templates}

    def match(
        self,
        patch: Patch,
        *,
        size_tolerance: int = 1,
        em: tuple[float, float] | None = None,
    ) -> list[Match]:
        """Every label whose templates are within ``size_tolerance`` pixels of the patch's size, best first.

        The score is the mean absolute difference over the patch padded by one
        pixel, minimized over every placement of the template inside it, so a
        glyph a pixel taller or wider than its template, or a pixel off, still
        scores close to zero. Templates whose ink mass is far from the patch's
        are not compared at all. ``em`` restricts the templates to a range of
        font sizes in pixels.
        """
        key = (patch.shape, quantize(patch).tobytes(), em)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        height, width = patch.shape
        padded = np.zeros((height + 2, width + 2), dtype=np.float32)
        padded[1:-1, 1:-1] = patch
        mass = float(padded.sum())
        area = float(padded.size)
        best: dict[str, Match] = {}
        for th in range(height - size_tolerance, height + size_tolerance + 1):
            for tw in range(width - size_tolerance, width + size_tolerance + 1):
                if th <= 0 or tw <= 0 or th > height + 2 or tw > width + 2:
                    continue
                indices = self._by_size.get((th, tw))
                if not indices:
                    continue
                stack, masses, ems, blurred = self._stack((th, tw))
                near = np.abs(masses - mass) <= MASS_TOLERANCE * np.maximum(masses, mass)
                if em is not None:
                    near &= (ems >= em[0]) & (ems <= em[1])
                chosen = np.nonzero(near)[0]
                if len(chosen) == 0:
                    continue
                if len(chosen) > SHORTLIST:
                    # Sort the candidates by how they compare blurred and in
                    # one placement, which a pixel of misplacement barely
                    # moves, and keep the closest for the exact search, plus
                    # the closest of every other label: a runner-up label
                    # has to be measured, not left out, for the reading to
                    # know when it is ambiguous.
                    top, left = (height + 2 - th) // 2, (width + 2 - tw) // 2
                    soft = _blur(padded)[top : top + th, left : left + tw]
                    rough = np.abs(blurred[chosen] - soft).sum(axis=(1, 2))
                    order = np.argsort(rough)
                    keep = list(order[:SHORTLIST])
                    seen = {self.templates[indices[chosen[i]]].label for i in keep}
                    for i in order[SHORTLIST:]:
                        label = self.templates[indices[chosen[i]]].label
                        if label not in seen:
                            seen.add(label)
                            keep.append(i)
                    chosen = chosen[np.array(keep)]
                subset = stack[chosen]
                # Every placement of the templates inside the padded patch at
                # once: windows is (rows, columns, th, tw), the difference
                # (rows, columns, templates, th, tw).
                windows = sliding_window_view(padded, (th, tw))
                inside = np.abs(subset[None, None] - windows[:, :, None]).sum(axis=(3, 4))
                outside = mass - windows.sum(axis=(2, 3))
                lowest = (inside + outside[:, :, None]).min(axis=(0, 1))
                for position, choice in enumerate(chosen.tolist()):
                    template = self.templates[indices[choice]]
                    sad = float(lowest[position])
                    held = best.get(template.label)
                    if held is None or sad < held.sad:
                        best[template.label] = Match(
                            template.label, sad / area, template, sad, mass
                        )
        result = sorted(best.values(), key=lambda m: m.score)
        if len(self._cache) >= CACHE_SIZE:
            self._cache.clear()
        self._cache[key] = result
        return result

    def _stack(self, size: tuple[int, int]) -> _Stack:
        stack = self._stacks.get(size)
        if stack is None:
            patches = np.stack([self.templates[i].patch for i in self._by_size[size]])
            masses = patches.sum(axis=(1, 2)).astype(np.float32)
            ems = np.array([self.templates[i].em for i in self._by_size[size]], dtype=np.float32)
            blurred = np.stack([_blur(patch) for patch in patches])
            stack = (patches, masses, ems, blurred)
            self._stacks[size] = stack
        return stack

    def save(self, path: Path) -> None:
        """Write the library as a compressed npz file."""
        if not self.templates:
            raise LibraryError("nothing to save: the library is empty")
        heights = np.array([t.height for t in self.templates], dtype=np.int32)
        widths = np.array([t.width for t in self.templates], dtype=np.int32)
        data = np.concatenate([quantize(t.patch).reshape(-1) for t in self.templates])
        np.savez_compressed(
            path,
            format=np.int32(FORMAT),
            dpi=np.float64(self.dpi),
            heights=heights,
            widths=widths,
            data=data,
            em=np.array([t.em for t in self.templates], dtype=np.float32),
            base=np.array([t.base for t in self.templates], dtype=np.float32),
            lsb=np.array([t.lsb for t in self.templates], dtype=np.float32),
            rsb=np.array([t.rsb for t in self.templates], dtype=np.float32),
            count=np.array([t.count for t in self.templates], dtype=np.int32),
            parts=np.array([t.parts for t in self.templates], dtype=np.int32),
            labels=np.str_(json.dumps([t.label for t in self.templates])),
            fonts=np.str_(json.dumps([t.font for t in self.templates])),
            words=np.str_(json.dumps(sorted(self.words))),
        )

    @classmethod
    def load(cls, path: Path) -> Library:
        """Read a library written by :meth:`save`."""
        try:
            with np.load(path) as stored:
                if int(stored["format"]) != FORMAT:
                    raise LibraryError(
                        f"{path}: library format {int(stored['format'])} is not {FORMAT}"
                    )
                heights = stored["heights"].astype(int)
                widths = stored["widths"].astype(int)
                data = stored["data"]
                labels = json.loads(str(stored["labels"]))
                fonts = json.loads(str(stored["fonts"]))
                ems, bases = stored["em"], stored["base"]
                lsbs, rsbs, counts = stored["lsb"], stored["rsb"], stored["count"]
                parts = (
                    stored["parts"] if "parts" in stored else np.ones(len(counts), dtype=np.int32)
                )
                dpi = float(stored["dpi"])
                words = set(json.loads(str(stored["words"])))
        except (OSError, KeyError, ValueError) as err:
            raise LibraryError(f"{path}: cannot load the glyph library: {err}") from err
        templates: list[Template] = []
        offset = 0
        for index, label in enumerate(labels):
            h, w = int(heights[index]), int(widths[index])
            patch = data[offset : offset + h * w].reshape(h, w).astype(np.float32) / 255.0
            offset += h * w
            templates.append(
                Template(
                    str(label),
                    patch,
                    float(ems[index]),
                    float(bases[index]),
                    float(lsbs[index]),
                    float(rsbs[index]),
                    str(fonts[index]),
                    int(counts[index]),
                    int(parts[index]),
                )
            )
        if offset != len(data):
            raise LibraryError(f"{path}: patch data does not match the template sizes")
        return cls(dpi, templates, words)


def _blur(patch: Patch) -> Patch:
    """The patch averaged over 3 by 3 neighbourhoods, edges included."""
    padded = np.pad(patch, 1, mode="edge")
    out = np.zeros_like(patch)
    for dy in range(3):
        for dx in range(3):
            out += padded[dy : dy + patch.shape[0], dx : dx + patch.shape[1]]
    return out / 9.0


def quantize(patch: Patch) -> npt.NDArray[np.uint8]:
    quantized: npt.NDArray[np.uint8] = np.clip(np.rint(patch * 255.0), 0, 255).astype(np.uint8)
    return quantized
