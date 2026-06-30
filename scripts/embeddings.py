"""Pluggable text-embedding provider for the JN Grader task bank.

The schema reserves `assignments.embedding vector(1024)` for semantic
similar-problem grouping/retrieval; this module is the pluggable source that
fills it. Two providers:

  - ``hash``   : deterministic, offline, no API. A hashed bag-of-tokens vector
                 (tf-weighted, L2-normalized) into `dim` buckets. Always
                 available — used for tests and as graceful degradation so
                 semantic queries work without any embedding API. Captures
                 lexical overlap (shared FEM terms), not deep semantics.
  - ``openai`` : any OpenAI-compatible `/embeddings` endpoint. For
                 text-embedding-3 models the target dimensionality is requested
                 natively via ``dimensions=``; otherwise the returned vector is
                 truncated + renormalized to `dim` (Matryoshka-style) so it fits
                 the fixed `vector(1024)` column.

Output is always exactly `dim` floats, L2-normalized — so cosine distance
(`<=>` in pgvector) is meaningful and comparable across providers.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Iterable

logger = logging.getLogger("embeddings")

DEFAULT_DIM = 1024
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def to_pgvector(vec: list[float]) -> str:
    """Format a vector as a pgvector literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity (assumes finite vectors; safe on zero vectors)."""
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Hash (offline, deterministic) embedder
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def hash_embed(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    """Deterministic hashed tf vector, L2-normalized. Offline, no dependencies.

    Each token is hashed to a bucket (with a sign hash to reduce collisions),
    accumulating tf weight. Shared vocabulary → higher cosine similarity.
    """
    vec = [0.0] * dim
    for tok in _tokenize(text):
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[idx] += sign
    return l2_normalize(vec)


# ---------------------------------------------------------------------------
# Embedder facade
# ---------------------------------------------------------------------------

class Embedder:
    """Embeds text into fixed-`dim`, L2-normalized vectors via a chosen provider."""

    def __init__(
        self,
        provider: str = "hash",
        model: str | None = None,
        dim: int = DEFAULT_DIM,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.provider = provider
        self.dim = dim
        self.model = model or (DEFAULT_OPENAI_MODEL if provider == "openai" else "hash")
        self._client = None
        if provider == "openai":
            from openai import OpenAI
            api_key = api_key or os.environ.get("LLM_API_KEY")
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        elif provider != "hash":
            raise ValueError(f"Unknown embedding provider: {provider!r} (use 'hash' or 'openai')")

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        if self.provider == "hash":
            return [hash_embed(t, self.dim) for t in texts]
        return self._embed_openai(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        # text-embedding-3 supports native dimensionality; older models don't.
        kwargs: dict = {"model": self.model, "input": texts}
        if "text-embedding-3" in (self.model or ""):
            kwargs["dimensions"] = self.dim
        try:
            resp = self._client.embeddings.create(**kwargs)
        except TypeError:
            resp = self._client.embeddings.create(model=self.model, input=texts)
        out = []
        for item in resp.data:
            v = list(item.embedding)
            out.append(l2_normalize(self._fit_dim(v)))
        return out

    def _fit_dim(self, v: list[float]) -> list[float]:
        """Truncate or zero-pad a raw embedding to exactly `self.dim` (Matryoshka
        truncation works well for text-embedding-3; padding is a harmless fallback)."""
        if len(v) == self.dim:
            return v
        if len(v) > self.dim:
            return v[: self.dim]
        return v + [0.0] * (self.dim - len(v))


def build_embedder_from_args(args) -> Embedder:
    """Construct an Embedder from argparse-style attributes (provider/model/dim/
    api_key/base_url), tolerating missing ones."""
    return Embedder(
        provider=getattr(args, "embed_provider", "hash"),
        model=getattr(args, "embed_model", None),
        dim=getattr(args, "embed_dim", DEFAULT_DIM),
        api_key=getattr(args, "api_key", None),
        base_url=getattr(args, "base_url", None),
    )
