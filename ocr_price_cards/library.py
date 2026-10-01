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

import hashlib
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from .errors import LibraryError

Patch = npt.NDArray[np.float32]
Stored = npt.NDArray[np.uint8]
"""A template's pixels as they are kept: quantized, which is all they ever carried."""
_Stack = tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]
_Key = tuple[tuple[int, ...], bytes, tuple[float, float] | None, int]
"""What a match is cached under: the patch, the font sizes asked for and the size window."""

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

    ``patch`` is kept quantized. It is set from a patch of any kind when the
    template is added, and it is the one the size stack is built from; holding
    a byte a pixel in four bytes bought nothing.
    """

    label: str
    patch: Stored
    em: float
    base: float
    lsb: float
    rsb: float
    font: str = ""
    count: int = 1
    parts: int = 1

    def __post_init__(self) -> None:
        self.patch = quantize(self.patch)

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
    _blocks: dict[tuple[int, int], Stored] = field(default_factory=dict, repr=False)
    _stacks: dict[tuple[int, int], _Stack] = field(default_factory=dict, repr=False)
    _blurs: dict[tuple[int, int], Patch] = field(default_factory=dict, repr=False)
    _keys: dict[tuple[str, tuple[int, ...], bytes], int] = field(default_factory=dict, repr=False)
    _indexed: bool = field(default=True, repr=False)
    _bearings: dict[tuple[str, float], tuple[float, float]] = field(
        default_factory=dict, repr=False
    )
    _cache: dict[_Key, list[Match]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        templates, self.templates = self.templates, []
        for template in templates:
            self.add(template)

    def __len__(self) -> int:
        return len(self.templates)

    def add(self, template: Template) -> Template:
        """Add a template, folding it into an identical one already held.

        Templates are looked up by a digest of their pixels rather than by the
        pixels themselves: keeping the pixels twice costs as much as the
        library does. A digest that matches is still checked against the
        patch it stands for, so the fold is on the pixels as before.

        """
        if not self._indexed:
            self._index()
        quantized = template.patch
        key = _key(template)
        index = self._keys.get(key)
        if index is not None:
            held = self.templates[index]
            if np.array_equal(held.patch, quantized):
                held.count += template.count
                return held
        template.patch = quantized
        self._keys[key] = len(self.templates)
        self.templates.append(template)
        size = (template.height, template.width)
        self._by_size.setdefault(size, []).append(len(self.templates) - 1)
        self._blocks.pop(size, None)
        self._stacks.pop(size, None)
        self._blurs.pop(size, None)
        self._bearings.clear()
        self._cache.clear()
        return template

    def _index(self) -> None:
        """Index the templates by their pixels, which only adding one asks for."""
        self._keys = {}
        for index, template in enumerate(self.templates):
            self._keys.setdefault(_key(template), index)
        self._indexed = True

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
        # The size window belongs in the key: the same mark is matched at one
        # pixel alone and at two as part of a stack or a joined run, and the
        # two searches do not hold the same templates.
        key = (patch.shape, quantize(patch).tobytes(), em, size_tolerance)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        height, width = patch.shape
        padded = np.zeros((height + 2, width + 2), dtype=np.float32)
        padded[1:-1, 1:-1] = patch
        mass = float(padded.sum())
        area = float(padded.size)
        blur: Patch | None = None
        best: dict[str, Match] = {}
        for th in range(height - size_tolerance, height + size_tolerance + 1):
            for tw in range(width - size_tolerance, width + size_tolerance + 1):
                if th <= 0 or tw <= 0 or th > height + 2 or tw > width + 2:
                    continue
                indices = self._by_size.get((th, tw))
                if not indices:
                    continue
                stack, masses, ems = self._stack((th, tw))
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
                    if blur is None:
                        blur = _blur(padded)
                    soft = blur[top : top + th, left : left + tw]
                    blurred = self._blurred((th, tw), stack)
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
                subset = stack[chosen].astype(np.float32) / 255.0
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
        """The templates of one size as bytes, with their ink masses and font sizes.

        A loaded library already holds each size as one block, and the
        templates' pixels are views into it; only a size a template was added
        to since is gathered afresh. The search converts to floats just the
        templates it compares, so a size costs its bytes and not four times
        as much.
        """
        stack = self._stacks.get(size)
        if stack is None:
            stored = self._blocks.get(size)
            if stored is None:
                stored = np.stack([self.templates[i].patch for i in self._by_size[size]])
            masses = (stored.astype(np.float32) / 255.0).sum(axis=(1, 2)).astype(np.float32)
            ems = np.array([self.templates[i].em for i in self._by_size[size]], dtype=np.float32)
            stack = (stored, masses, ems)
            self._stacks[size] = stack
        return stack

    def _blurred(self, size: tuple[int, int], stored: Stored) -> Patch:
        """The templates of one size blurred, built the first time a size shortlists.

        Only a size holding more candidates than the shortlist is ever ranked
        this way, and most sizes hold far fewer, so blurring them all as the
        stack is built is work and memory for nothing.
        """
        blurred = self._blurs.get(size)
        if blurred is None:
            blurred = np.stack([_blur(patch) for patch in stored.astype(np.float32) / 255.0])
            self._blurs[size] = blurred
        return blurred

    def save(self, path: Path) -> None:
        """Write the library as a compressed npz file."""
        if not self.templates:
            raise LibraryError("nothing to save: the library is empty")
        heights = np.array([t.height for t in self.templates], dtype=np.int32)
        widths = np.array([t.width for t in self.templates], dtype=np.int32)
        data = np.concatenate([t.patch.reshape(-1) for t in self.templates])
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
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as err:
            raise LibraryError(f"{path}: cannot load the glyph library: {err}") from err
        fields = (heights, widths, fonts, ems, bases, lsbs, rsbs, counts, parts)
        if any(len(values) != len(labels) for values in fields):
            raise LibraryError(f"{path}: the template fields do not agree on how many there are")
        offsets = np.zeros(len(labels) + 1, dtype=np.int64)
        np.cumsum(heights * widths, out=offsets[1:])
        if offsets[-1] != len(data):
            raise LibraryError(f"{path}: patch data does not match the template sizes")
        # Each size laid out as one block, in the order its templates are
        # held, so that the search reads a size where it lies.
        members: dict[tuple[int, int], list[int]] = {}
        for index in range(len(labels)):
            members.setdefault((int(heights[index]), int(widths[index])), []).append(index)
        blocks: dict[tuple[int, int], Stored] = {}
        patches: dict[int, Stored] = {}
        for (h, w), indices in members.items():
            block = np.stack([data[offsets[i] : offsets[i + 1]] for i in indices])
            block = block.reshape(len(indices), h, w)
            blocks[(h, w)] = block
            for k, i in enumerate(indices):
                patches[i] = block[k]
        library = cls(dpi, words=words)
        for index, label in enumerate(labels):
            library.templates.append(
                Template(
                    sys.intern(str(label)),
                    patches[index],
                    float(ems[index]),
                    float(bases[index]),
                    float(lsbs[index]),
                    float(rsbs[index]),
                    sys.intern(str(fonts[index])),
                    int(counts[index]),
                    int(parts[index]),
                )
            )
        # A saved library was folded as it was built, so its templates are
        # taken as they are; the index that folds them is built only if one
        # is added.
        library._by_size = {size: list(indices) for size, indices in members.items()}
        library._blocks = blocks
        library._indexed = False
        return library


def _key(template: Template) -> tuple[str, tuple[int, ...], bytes]:
    """What a template is folded under: its label, its size and a digest of its pixels."""
    patch = template.patch
    return (template.label, patch.shape, hashlib.blake2b(patch.tobytes(), digest_size=16).digest())


def _blur(patch: Patch) -> Patch:
    """The patch averaged over 3 by 3 neighbourhoods, edges included."""
    padded = np.pad(patch, 1, mode="edge")
    out = np.zeros_like(patch)
    for dy in range(3):
        for dx in range(3):
            out += padded[dy : dy + patch.shape[0], dx : dx + patch.shape[1]]
    return out / 9.0


def quantize(patch: Patch | Stored) -> Stored:
    """A patch as the bytes a template keeps; one that is already those is given back as it is."""
    if patch.dtype == np.uint8:
        return cast(Stored, patch)
    quantized: Stored = np.clip(np.rint(patch * 255.0), 0, 255).astype(np.uint8)
    return quantized
