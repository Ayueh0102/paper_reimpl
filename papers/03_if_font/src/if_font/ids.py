"""IDS (Ideographic Description Sequence) tokenizer for IF-Font.

The paper specifies IDS as the *content* signal that replaces the source glyph:
each character is described by a sequence such as ``⿰示畐`` (for 福) using the
12 IDC structural characters plus the leaf components (themselves CJK chars).
[paper §3, p.1]

This module implements a small, self-contained tokenizer:
  * `DEFAULT_IDC_CHARS` — the 12 Unicode IDC symbols (U+2FF0..U+2FFB).
  * Special tokens — ``[PAD] [BOS] [EOS] [UNK]``.
  * `IDSTokenizer.encode(ids: str)` → list[int].
  * `IDSTokenizer.batch_encode(strs, max_len)` → (token_ids, mask) tensors.

We do **not** ship a hard-coded component vocabulary. Instead we build the
vocab on-the-fly from a corpus (e.g. all chars in the manifest) by calling
``IDSTokenizer.fit_from_chars(chars, ids_lookup)``. The IDS lookup callable
is expected to come from the user-supplied dataset under
``~/Char/datasets/ids/scripts/lookup_ids.py`` (paper does not specify the
dictionary; we use CHISE-derived ``cns_unicode_ids.tsv`` per CLAUDE.md).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import torch

# IDS Description Characters (Unicode block "Ideographic Description Characters",
# U+2FF0..U+2FFB). 12 symbols.
DEFAULT_IDC_CHARS: tuple[str, ...] = (
    "⿰",  # ⿰  left-right
    "⿱",  # ⿱  top-bottom
    "⿲",  # ⿲  left-mid-right
    "⿳",  # ⿳  top-mid-bottom
    "⿴",  # ⿴  surround-full
    "⿵",  # ⿵  surround-open-bottom
    "⿶",  # ⿶  surround-open-top
    "⿷",  # ⿷  surround-open-right
    "⿸",  # ⿸  surround-open-TR
    "⿹",  # ⿹  surround-open-TL
    "⿺",  # ⿺  surround-open-BR
    "⿻",  # ⿻  overlap
)

STRUCTURE_NAMES: dict[str, str] = {
    "⿰": "left_right",
    "⿱": "top_bottom",
    "⿲": "left_mid_right",
    "⿳": "top_mid_bottom",
    "⿴": "surround_full",
    "⿵": "surround_open_bottom",
    "⿶": "surround_open_top",
    "⿷": "surround_open_right",
    "⿸": "surround_open_TR",
    "⿹": "surround_open_TL",
    "⿺": "surround_open_BR",
    "⿻": "overlap",
}


def parse_structure_class(ids: str) -> str:
    """Return one of the 12 structure names or ``'atomic'`` / ``'unknown'``.

    Mirrors ``lookup_ids.parse_structure`` (the user-provided helper) but is
    self-contained so this package does not have a runtime dependency on the
    external IDS dataset path.
    """
    if not ids:
        return "unknown"
    return STRUCTURE_NAMES.get(ids[0], "atomic")


PAD = "[PAD]"
BOS = "[BOS]"
EOS = "[EOS]"
UNK = "[UNK]"
SPECIAL_TOKENS: tuple[str, ...] = (PAD, BOS, EOS, UNK)


@dataclass
class IDSTokenizer:
    """Token <-> id mapping for IDS strings.

    Vocab layout (deterministic):
      0..3   special tokens (PAD/BOS/EOS/UNK)
      4..15  12 IDC chars (from ``DEFAULT_IDC_CHARS``)
      16..   leaf components (CJK chars discovered during fit)
    """

    vocab: list[str] = field(default_factory=list)
    token_to_id: dict[str, int] = field(default_factory=dict)

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

    # ------------------------------------------------------------------
    # vocab construction
    # ------------------------------------------------------------------

    def add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.vocab)
        self.vocab.append(token)
        self.token_to_id[token] = idx
        return idx

    def fit_from_strings(self, ids_strings: Iterable[str]) -> IDSTokenizer:
        """Add every char in ``ids_strings`` as its own token.

        IDS is character-level: each Unicode code point is one token (this is
        the same convention as char-rnn over CJK + IDC). The 12 IDC chars are
        already pre-registered so they collide with no leaf component.
        """
        for s in ids_strings:
            if not s:
                continue
            for ch in s:
                self.add_token(ch)
        return self

    @classmethod
    def from_charset(
        cls,
        chars: Iterable[str],
        ids_lookup: Callable[[str], str],
    ) -> IDSTokenizer:
        """Convenience constructor: look up IDS for each char and fit.

        Args:
            chars: iterable of target Unicode characters in the manifest.
            ids_lookup: function mapping char → IDS string. Pass
                ``lookup_ids.get_ids`` from ``~/Char/datasets/ids/scripts``.
        """
        tok = cls()
        ids_strings = [ids_lookup(ch) for ch in chars]
        tok.fit_from_strings(ids_strings)
        return tok

    @classmethod
    def from_idc_only(cls) -> IDSTokenizer:
        """Vocab with only IDC + specials. Used by smoke tests.

        Tests can synthesise IDS strings using just the 12 IDC chars and
        small ascii digit "leaf" placeholders that will be added by encode
        via the UNK fallback (or by explicit fit_from_strings).
        """
        return cls()

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------

    def encode(self, ids: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Map IDS string → list[int]. Unknown chars → UNK."""
        out: list[int] = []
        if add_bos:
            out.append(self.bos_id)
        for ch in ids:
            out.append(self.token_to_id.get(ch, self.unk_id))
        if add_eos:
            out.append(self.eos_id)
        return out

    def batch_encode(
        self,
        ids_strings: Sequence[str],
        *,
        max_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch encode + pad to ``max_len``.

        Returns (token_ids[B, max_len] long, attention_mask[B, max_len] bool).
        Mask is True where the position is a real token, False at PAD.
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
        """Map id sequence → string. Special tokens are dropped."""
        out: list[str] = []
        special_set = set(SPECIAL_TOKENS)
        for tid in token_ids:
            tok = self.vocab[int(tid)] if 0 <= int(tid) < len(self.vocab) else UNK
            if tok in special_set:
                continue
            out.append(tok)
        return "".join(out)


__all__ = [
    "BOS",
    "DEFAULT_IDC_CHARS",
    "EOS",
    "IDSTokenizer",
    "PAD",
    "SPECIAL_TOKENS",
    "STRUCTURE_NAMES",
    "UNK",
    "parse_structure_class",
]
