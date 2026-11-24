"""BM25: the classic keyword-ranking formula, written from scratch.

Idea: a document scores higher when it contains the query's words, especially
rare words (high IDF), with diminishing returns for repeats (k1) and a penalty
for very long documents (b).
"""
import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "the", "to", "of", "in", "on", "for", "with", "we", "our",
    "you", "your", "is", "are", "be", "at", "as", "or", "by", "from", "that",
    "this", "it",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.doc_len = [sum(c.values()) for c in self.tf]
        self.avg_len = sum(self.doc_len) / max(len(docs), 1)
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        out = np.zeros(len(self.tf))
        for i, (tf, dl) in enumerate(zip(self.tf, self.doc_len)):
            s = 0.0
            for t in q_tokens:
                f = tf.get(t)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            out[i] = s
        return out
