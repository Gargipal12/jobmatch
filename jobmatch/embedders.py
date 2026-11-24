"""Turning text into vectors ("embeddings").

Two backends with the same interface: encode(list_of_texts) -> (n, dim) array
of L2-normalised vectors, so a dot product equals cosine similarity.

- SentenceTransformerEmbedder ("st"): a real neural embedding model. Captures
  meaning, e.g. "ML engineer" is close to "machine learning developer". Use
  this for real results.
- HashingEmbedder ("hash"): offline fallback built from character n-grams. It
  only captures spelling overlap (not meaning), so it is for testing the
  pipeline without downloading a model. Never report its numbers as semantic
  search results.
"""
import zlib

import numpy as np

from .bm25 import tokenize


class HashingEmbedder:
    name = "hash"

    def __init__(self, dim: int = 2048, ngram_range: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.lo, self.hi = ngram_range

    def _grams(self, token: str):
        padded = f"<{token}>"
        for n in range(self.lo, self.hi + 1):
            for i in range(len(padded) - n + 1):
                yield padded[i : i + n]

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                for g in self._grams(tok):
                    h = zlib.crc32(g.encode("utf-8"))
                    out[row, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    name = "st"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Imported lazily so the rest of the project works without it installed.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts) -> np.ndarray:
        vecs = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(kind: str):
    if kind == "hash":
        return HashingEmbedder()
    if kind == "st":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {kind!r}; use 'st' or 'hash'")
