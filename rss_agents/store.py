"""JSONL storage (the RAG hand-off format), chunking, and a small BM25 retriever."""

from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9$%.]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were will with "
    "what which who how why when this these those do does did about into than then there their".split()
)


def tokenize(text: str) -> list[str]:
    return [t.strip(".") for t in _TOKEN.findall(text.lower()) if t.strip(".") and t not in _STOPWORDS]


def chunk_text(text: str, max_words: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text] if text else []
    chunks, step = [], max_words - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return chunks


class TopicStore:
    """Per-topic output folder:

    articles.jsonl  one enriched article per line (full record)
    chunks.jsonl    RAG-ready chunks with metadata — point your vector DB loader here
    seen.json       article ids already ingested (dedupe across runs)
    """

    def __init__(self, root: Path, topic: str) -> None:
        self.topic = topic
        self.dir = root / topic
        self.dir.mkdir(parents=True, exist_ok=True)
        self.articles_path = self.dir / "articles.jsonl"
        self.chunks_path = self.dir / "chunks.jsonl"
        self.seen_path = self.dir / "seen.json"
        self._lock = threading.Lock()

    def seen_ids(self) -> set[str]:
        if not self.seen_path.exists():
            return set()
        return set(json.loads(self.seen_path.read_text(encoding="utf-8")))

    def add(self, article: dict, chunks: list[dict]) -> None:
        """Append one article and its chunks. Thread-safe; written immediately so
        a crash mid-run keeps everything ingested so far."""
        with self._lock:
            with self.articles_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")
            with self.chunks_path.open("a", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            seen = self.seen_ids()
            seen.add(article["id"])
            self.seen_path.write_text(json.dumps(sorted(seen)), encoding="utf-8")

    def load_chunks(self) -> list[dict]:
        if not self.chunks_path.exists():
            return []
        with self.chunks_path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class BM25:
    """Keyword retriever used by the built-in `ask` command. Your RAG portal will
    replace this with vector search, but it lets the agents answer questions today."""

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(self._doc_text(d))) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @staticmethod
    def _doc_text(d: dict) -> str:
        m = d.get("metadata", {})
        extra = " ".join(str(v) for v in m.values() if isinstance(v, (str, list)))
        return f"{d['text']} {extra}"

    def search(self, query: str, k: int = 8) -> list[tuple[float, dict]]:
        terms = tokenize(query)
        scored = []
        for doc, tf, length in zip(self.docs, self.tfs, self.lens):
            score = 0.0
            for t in terms:
                if t not in tf:
                    continue
                f = tf[t]
                norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                score += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
