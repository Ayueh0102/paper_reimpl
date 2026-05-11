"""Chinese BERT wrapper for Calliffusion text conditioning.

The paper uses Chinese BERT (default: ``bert-base-chinese``) to embed prompts
of the form ``"<character> <script> <calligrapher>"``. We expose two paths:

1. ``BertTextEncoder``: real ``transformers.BertModel`` + ``BertTokenizer``
   loaded from a local cache or the HF Hub. This is the production path.
2. ``StubTextEncoder``: a deterministic random-weight stub used by the smoke
   test so the test does not need network access. It emulates the same API
   (``encode(prompts) -> (input_ids, attention_mask, last_hidden_state)``).

Both encoders produce a ``[B, L, hidden]`` last-hidden-state and an
``[B, L]`` attention mask suitable for cross-attention.

The stub also supports the "add calligrapher names as special tokens"
strategy used in `paper_notes/06.md` §2.4 — calling ``add_special_tokens``
extends the (toy) vocabulary and grows the embedding table accordingly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TextEncoderOutput:
    """Bundled output of any text encoder used by the U-Net."""

    input_ids: torch.Tensor          # [B, L]
    attention_mask: torch.Tensor     # [B, L]
    last_hidden_state: torch.Tensor  # [B, L, hidden]


class StubTextEncoder(nn.Module):
    """Offline stand-in for Chinese BERT used by tests + synthetic dry-runs.

    Tokenisation is the simplest thing that still exercises the
    "special-token expansion" code path: split the prompt on whitespace and
    map each piece to a slot in a pre-grown vocabulary. Tokens not in the
    table get the [UNK] id (= 1). [PAD] = 0, [CLS] = 2, [SEP] = 3.

    Vocabulary growth model:
      - At construction the embedding table is sized to ``initial_vocab_size``
        (>= NUM_SPECIAL).
      - ``add_special_tokens(...)`` is the *only* allowed growth point. It
        extends ``token_to_id`` and the embedding table for every new token.
      - ``_tokenize`` is read-only — unknown tokens fall back to ``UNK_ID``.
        This matches real BERT semantics (which never grows the vocab during
        forward) and makes the encoder safe for multi-worker DataLoaders and
        non-deterministic test ordering.
    """

    PAD_ID = 0
    UNK_ID = 1
    CLS_ID = 2
    SEP_ID = 3
    NUM_SPECIAL = 4

    def __init__(
        self,
        *,
        hidden_size: int = 768,
        max_length: int = 32,
        initial_vocab_size: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_length = int(max_length)
        self.token_to_id: dict[str, int] = {}
        self._vocab_size = max(self.NUM_SPECIAL, int(initial_vocab_size))
        gen = torch.Generator().manual_seed(int(seed))
        weight = torch.randn(self._vocab_size, self.hidden_size, generator=gen) * 0.02
        self.embedding = nn.Embedding(self._vocab_size, self.hidden_size, padding_idx=self.PAD_ID)
        with torch.no_grad():
            self.embedding.weight.copy_(weight)
        # Tiny "self-attn-like" projection so different prompts give different
        # contexts even with the stub. Keeps signal flowing for smoke tests.
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def add_special_tokens(self, tokens: Iterable[str]) -> int:
        """Append new special tokens; resize the embedding table.

        Returns the number of *new* tokens added. This is the only public
        way to grow the vocabulary — once training begins, callers should
        not invoke this again, since the embedding table is reallocated
        (which would break optimizer state and any DataLoader worker
        already iterating the encoder).
        """
        added = 0
        for tok in tokens:
            cleaned = str(tok).strip()
            if not cleaned or cleaned in self.token_to_id:
                continue
            self.token_to_id[cleaned] = self._vocab_size
            self._vocab_size += 1
            added += 1
        if added:
            old = self.embedding
            new = nn.Embedding(self._vocab_size, self.hidden_size, padding_idx=self.PAD_ID)
            with torch.no_grad():
                new.weight[: old.num_embeddings].copy_(old.weight)
                new.weight[old.num_embeddings :].normal_(mean=0.0, std=0.02)
            self.embedding = new.to(old.weight.device)
        return added

    def _tokenize(self, prompt: str) -> list[int]:
        """Read-only tokenisation. Unknown pieces fall back to ``UNK_ID``.

        Crucially this method does NOT mutate ``token_to_id`` or
        ``embedding`` — preventing a data race when ``num_workers > 0`` and
        keeping the embedding table deterministic across call order.
        """
        ids: list[int] = [self.CLS_ID]
        for piece in str(prompt).split():
            ids.append(self.token_to_id.get(piece, self.UNK_ID))
        ids.append(self.SEP_ID)
        return ids[: self.max_length]

    def encode(self, prompts: list[str]) -> TextEncoderOutput:
        device = next(self.parameters()).device
        token_lists = [self._tokenize(p) for p in prompts]
        seq_len = max(1, max(len(ids) for ids in token_lists))
        seq_len = min(seq_len, self.max_length)
        batch = len(prompts)
        input_ids = torch.full((batch, seq_len), self.PAD_ID, dtype=torch.long, device=device)
        attn = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
        for b, ids in enumerate(token_lists):
            ids = ids[:seq_len]
            input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attn[b, : len(ids)] = 1
        emb = self.embedding(input_ids)
        # very light "self-mixing" so adjacent tokens influence each other.
        ctx = self.proj(emb)
        return TextEncoderOutput(input_ids=input_ids, attention_mask=attn, last_hidden_state=ctx)

    def forward(self, prompts: list[str]) -> TextEncoderOutput:  # noqa: D401
        return self.encode(prompts)


class BertTextEncoder(nn.Module):
    """Production encoder using HuggingFace ``transformers`` Chinese BERT.

    Loaded lazily so the smoke test does not require the model to be on disk.

    Usage:
        enc = BertTextEncoder("bert-base-chinese")
        enc.add_special_tokens(["顏真卿", "王羲之", ...])
        out = enc(["人 隸書 曹全碑", "山 楷書 顏真卿"])
    """

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        *,
        max_length: int = 32,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        # Imported lazily to keep the smoke test offline.
        from transformers import BertModel, BertTokenizer

        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.bert = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.hidden_size = int(self.bert.config.hidden_size)
        self.max_length = int(max_length)

    @property
    def vocab_size(self) -> int:
        return int(self.bert.config.vocab_size)

    def add_special_tokens(self, tokens: Iterable[str]) -> int:
        new = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [str(t) for t in tokens]}
        )
        if new:
            self.bert.resize_token_embeddings(len(self.tokenizer))
        return int(new)

    def freeze(self, *, embeddings_trainable: bool = False) -> None:
        """Freeze BERT. If ``embeddings_trainable`` keep the word-embedding
        table trainable so newly-added special-token rows can be learned.
        """
        for p in self.bert.parameters():
            p.requires_grad = False
        if embeddings_trainable:
            for p in self.bert.embeddings.word_embeddings.parameters():
                p.requires_grad = True

    def encode(self, prompts: list[str]) -> TextEncoderOutput:
        device = next(self.bert.parameters()).device
        toks = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.bert(**toks)
        return TextEncoderOutput(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            last_hidden_state=out.last_hidden_state,
        )

    def forward(self, prompts: list[str]) -> TextEncoderOutput:  # noqa: D401
        return self.encode(prompts)


def build_text_encoder(
    *,
    use_bert: bool = False,
    hidden_size: int = 768,
    max_length: int = 32,
    model_name: str = "bert-base-chinese",
    cache_dir: str | None = None,
) -> nn.Module:
    """Build either a real BERT encoder or the offline stub."""
    if use_bert:
        return BertTextEncoder(model_name, max_length=max_length, cache_dir=cache_dir)
    return StubTextEncoder(hidden_size=hidden_size, max_length=max_length)
