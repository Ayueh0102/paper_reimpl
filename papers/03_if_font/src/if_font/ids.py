"""IDS (Ideographic Description Sequence) tokenizer for IF-Font — Phase 2.

Phase 2 alignment to official `iffont/data/cn.py` + `modules/encoder.IDSEncoder`:

- IDS dictionary source = **BabelStone CJK IDS** (Andrew West, Unicode 15.0,
  97058 entries) + the IF-Font team's hand-curated supplement
  (`ids_iffont.txt`, 165 entries). NOT the CHISE-derived
  `cns_unicode_ids.tsv` used in Phase 1.
- Two resolution modes: `radical` (leaf = first-level component) and
  `stroke` (leaf = atomic stroke / single-unit glyph). Official trains with
  `ids_mode: radical` (base.yaml:70), but coverage similarity always uses
  `stroke` mode (encoder.py:340-344).
- The tokenizer carries only one special token: `'pad'` (when ids_mode !=
  'all'). No BOS/EOS — the AR decoder uses the IDS embedding as a prefix
  prepended to the target VQ tokens (official `nanogpt.GPT.forward`).

The two vendored files live at:
  ``~/Char/datasets/ids/cn_mainland/babelstone_cjk_ids.txt``
  ``~/Char/datasets/ids/cn_mainland/ids_iffont.txt``
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Ideographic Description Characters, U+2FF0..U+2FFB (12 entries).
DEFAULT_IDC_CHARS: tuple[str, ...] = (
    "⿰", "⿱", "⿲", "⿳", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻",
)
# Number of components per IDC (matches official `cn.N_IDC_COMPS`).
N_IDC_COMPS: dict[str, int] = {
    "⿰": 2, "⿱": 2, "⿲": 3, "⿳": 3, "⿴": 2, "⿵": 2,
    "⿶": 2, "⿷": 2, "⿸": 2, "⿹": 2, "⿺": 2, "⿻": 2,
}

STRUCTURE_NAMES: dict[str, str] = {
    "⿰": "left_right", "⿱": "top_bottom", "⿲": "left_mid_right",
    "⿳": "top_mid_bottom", "⿴": "surround_full", "⿵": "surround_open_bottom",
    "⿶": "surround_open_top", "⿷": "surround_open_right",
    "⿸": "surround_open_TR", "⿹": "surround_open_TL",
    "⿺": "surround_open_BR", "⿻": "overlap",
}


def parse_structure_class(ids: str) -> str:
    """Return one of the 12 structure names or ``'atomic'`` / ``'unknown'``."""
    if not ids:
        return "unknown"
    return STRUCTURE_NAMES.get(ids[0], "atomic")


PAD = "[PAD]"
# Phase 2: BOS/EOS/UNK are kept for tokenizer-level compatibility but are not
# used by the AR decoder (which uses IDS as a prefix). They remain so that
# legacy tests / shared utilities that call `bos_id` / `eos_id` still work.
BOS = "[BOS]"
EOS = "[EOS]"
UNK = "[UNK]"
SPECIAL_TOKENS: tuple[str, ...] = (PAD, BOS, EOS, UNK)


# ======================================================================
# BabelStone + ids_iffont resolver (subset of official `cn.resolve_IDS_babelstone`)
# ======================================================================


_DEFAULT_BABELSTONE_PATH = Path("~/Char/datasets/ids/cn_mainland/babelstone_cjk_ids.txt")
_DEFAULT_IDS_IFFONT_PATH = Path("~/Char/datasets/ids/cn_mainland/ids_iffont.txt")


_BABELSTONE_CH_RE = re.compile(r"U\+[0-9A-Z]+\t(.).*?\t\^(.+?)\$")
_BABELSTONE_COMP_RE = re.compile(r"#\t(\{\d+\})\t.*?\t(.*)")
_IDS_IFFONT_RE = re.compile(r"^([^#\s]+)\t([^#\s]+)")
_SPLIT_IDS_RE = re.compile(r"(?<![0-9])(?![0-9])")


def _read_babelstone(path: Path) -> dict[str, str]:
    """Read the BabelStone IDS file into ``{char: raw_ids_string}``.

    Picks the first ``^...$`` IDS column for each line (BabelStone often
    lists multiple region-specific forms; the first one matches official
    `cn.read_ids` behaviour, which keeps only the first match).
    """
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = _BABELSTONE_CH_RE.match(line)
            if m is None:
                m = _BABELSTONE_COMP_RE.match(line)
            if m is None:
                continue
            ch = m.group(1)
            ids = m.group(2)
            if ch not in out:
                out[ch] = ids
    return out


def _read_ids_iffont(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            m = _IDS_IFFONT_RE.match(line)
            if m is None:
                continue
            out[m.group(1)] = m.group(2)
    return out


@dataclass
class IDSResolver:
    """Recursive IDS decomposition over BabelStone + ids_iffont.

    Public API:
      * ``IDSResolver.load(level='radical')`` — factory using the vendored
        default paths.
      * ``resolve(char)`` → tuple[str, ...] — the decomposed IDS token list.
      * ``vocab()`` → set[str] — the set of leaf tokens across the dictionary.
    """

    raw_ids: dict[str, str]
    level: str = "radical"  # 'radical' | 'stroke'
    _resolved: dict[str, tuple[str, ...]] = field(default_factory=dict)
    _idc_set: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_IDC_CHARS))

    @classmethod
    def load(
        cls,
        *,
        level: str = "radical",
        babelstone_path: Path | str | None = None,
        ids_iffont_path: Path | str | None = None,
    ) -> IDSResolver:
        bp = Path(babelstone_path or _DEFAULT_BABELSTONE_PATH).expanduser()
        ip = Path(ids_iffont_path or _DEFAULT_IDS_IFFONT_PATH).expanduser()
        raw: dict[str, str] = {}
        # 12 IDCs map to themselves.
        for c in DEFAULT_IDC_CHARS:
            raw[c] = c
        if bp.exists():
            raw |= _read_babelstone(bp)
        if ip.exists():
            raw |= _read_ids_iffont(ip)
        return cls(raw_ids=raw, level=level)

    # ------------------------------------------------------------------
    # recursive resolution
    # ------------------------------------------------------------------

    def _split(self, s: str) -> list[str]:
        """Split a raw IDS string into characters, treating `{NN}` as one token."""
        return [t for t in _SPLIT_IDS_RE.split(s) if t]

    def _resolve_stroke(self, ch: str) -> tuple[str, ...]:
        if ch in self._resolved:
            return self._resolved[ch]
        ids = self.raw_ids.get(ch, ch)
        if ids == ch:
            self._resolved[ch] = (ch,)
            return self._resolved[ch]
        parts = self._split(ids)
        out: list[str] = []
        for p in parts:
            out.extend(self._resolve_stroke(p))
        self._resolved[ch] = tuple(out)
        return self._resolved[ch]

    def _resolve_radical(self, ch: str) -> tuple[str, ...]:
        if ch in self._resolved:
            return self._resolved[ch]
        ids = self.raw_ids.get(ch, ch)
        parts = self._split(ids)
        if not parts or parts == [ch]:
            self._resolved[ch] = (ch,)
            return self._resolved[ch]
        # If every part is itself a 1-char (atomic) entry, this is a radical-level
        # decomposition — keep it as-is.
        is_radical = True
        for p in parts:
            origin = self.raw_ids.get(p, p)
            origin_parts = self._split(origin)
            if len(origin_parts) > 1:
                is_radical = False
                break
        # ⿻ overlay: always treat as radical-level (official behaviour, encoder.py:174).
        if parts and parts[0] == "⿻":
            is_radical = True
        if is_radical:
            self._resolved[ch] = tuple(parts)
            return self._resolved[ch]
        out: list[str] = []
        for p in parts:
            out.extend(self._resolve_radical(p))
        self._resolved[ch] = tuple(out)
        return self._resolved[ch]

    def resolve(self, ch: str) -> tuple[str, ...]:
        if self.level == "stroke":
            return self._resolve_stroke(ch)
        return self._resolve_radical(ch)

    def vocab(self, chars: Iterable[str] | None = None) -> set[str]:
        """Return the set of leaf tokens for ``chars`` (or all known chars)."""
        out: set[str] = set(DEFAULT_IDC_CHARS)
        it = chars if chars is not None else self.raw_ids.keys()
        for c in it:
            try:
                out.update(self.resolve(c))
            except RecursionError:
                # Fall back to leaving the raw IDS string's first level.
                out.update(self._split(self.raw_ids.get(c, c)))
        return out


# ======================================================================
# Tokenizer
# ======================================================================


@dataclass
class IDSTokenizer:
    """Token <-> id mapping for IDS strings (Phase-2 aligned).

    Vocab layout (deterministic):
      0..3   special tokens (PAD/BOS/EOS/UNK)
      4..15  12 IDC chars
      16..   leaf components added via fit_*

    The official `IDSEncoder` uses only `'pad'` (when ids_mode != 'all') as a
    special token; we keep BOS/EOS/UNK for back-compat with shared code, but
    the AR decoder ignores them (the IDS sequence is prefix-prepended whole).
    """

    vocab: list[str] = field(default_factory=list)
    token_to_id: dict[str, int] = field(default_factory=dict)
    is_frozen: bool = False

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __post_init__(self) -> None:
        if not self.vocab:
            self._reset_with_specials()
        else:
            self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}

    def _reset_with_specials(self) -> None:
        self.vocab = list(SPECIAL_TOKENS) + list(DEFAULT_IDC_CHARS)
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}

    def add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        if self.is_frozen:
            return self.unk_id
        idx = len(self.vocab)
        self.vocab.append(token)
        self.token_to_id[token] = idx
        return idx

    def fit_from_strings(self, ids_strings: Iterable[str]) -> IDSTokenizer:
        if self.is_frozen:
            return self
        for s in ids_strings:
            if not s:
                continue
            for ch in s:
                self.add_token(ch)
        return self

    def fit_from_resolver(
        self,
        resolver: IDSResolver,
        chars: Iterable[str] | None = None,
    ) -> IDSTokenizer:
        """Fit vocab from a resolver's leaf set (cleaner than scanning strings)."""
        if self.is_frozen:
            return self
        for tok in resolver.vocab(chars):
            self.add_token(tok)
        return self

    def freeze(self) -> IDSTokenizer:
        self.is_frozen = True
        return self

    @classmethod
    def from_charset(
        cls,
        chars: Iterable[str],
        ids_lookup: Callable[[str], str],
    ) -> IDSTokenizer:
        tok = cls()
        tok.fit_from_strings(ids_lookup(ch) for ch in chars)
        return tok

    @classmethod
    def from_idc_only(cls) -> IDSTokenizer:
        return cls()

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        ids: str | Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Map IDS string (or pre-tokenised tuple) → list[int].

        Phase-2 default: no BOS/EOS (matches official prefix-prepending).
        Pass ``add_bos=True`` / ``add_eos=True`` for legacy callers.
        """
        out: list[int] = []
        if add_bos:
            out.append(self.bos_id)
        if isinstance(ids, str):
            for ch in ids:
                out.append(self.token_to_id.get(ch, self.unk_id))
        else:
            for tok in ids:
                out.append(self.token_to_id.get(tok, self.unk_id))
        if add_eos:
            out.append(self.eos_id)
        return out

    def batch_encode(
        self,
        ids_strings: Sequence[str | Sequence[str]],
        *,
        max_len: int,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch encode + right-pad with PAD to ``max_len``.

        Returns (token_ids[B, max_len] long, attention_mask[B, max_len] bool).
        Mask is True at real-token positions.
        """
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        b = len(ids_strings)
        ids_t = torch.full((b, max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((b, max_len), dtype=torch.bool)
        for i, s in enumerate(ids_strings):
            enc = self.encode(s, add_bos=add_bos, add_eos=add_eos)[:max_len]
            ids_t[i, : len(enc)] = torch.tensor(enc, dtype=torch.long)
            mask[i, : len(enc)] = True
        return ids_t, mask

    def decode(self, token_ids: Sequence[int]) -> str:
        out: list[str] = []
        special_set = set(SPECIAL_TOKENS)
        for tid in token_ids:
            tok = self.vocab[int(tid)] if 0 <= int(tid) < len(self.vocab) else UNK
            if tok in special_set:
                continue
            out.append(tok)
        return "".join(out)


# Silence ruff F401 by exporting itertools so the module remains importable
# even when re-exported into older shared utilities.
_ = itertools


__all__ = [
    "BOS",
    "DEFAULT_IDC_CHARS",
    "EOS",
    "IDSResolver",
    "IDSTokenizer",
    "N_IDC_COMPS",
    "PAD",
    "SPECIAL_TOKENS",
    "STRUCTURE_NAMES",
    "UNK",
    "parse_structure_class",
]
