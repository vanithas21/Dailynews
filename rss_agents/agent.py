"""TopicAgent: collects a topic's feeds, enriches each article with Claude,
writes RAG-ready chunks, and answers questions over what it has ingested."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import anthropic
from pydantic import BaseModel

from .feeds import FeedItem, fetch_feed
from .store import BM25, TopicStore, chunk_text

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("RSS_AGENT_MODEL", "claude-opus-5-5")

# Server-side refusal fallback: if a safety classifier declines, the API re-runs the
# request on Anthropic's recommended fallback model instead of returning a refusal.
# Set RSS_AGENT_FALLBACKS=0 to turn it off.
_FALLBACKS = os.environ.get("RSS_AGENT_FALLBACKS", "1") != "0"
_FALLBACK_KW = (
    {
        "extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01"},
        "extra_body": {"fallbacks": "default"},
    }
    if _FALLBACKS
    else {}
)

ANSWER_INSTRUCTIONS = """\
Answer the user's question using ONLY the numbered sources provided. Rules:
- Cite every factual claim with its source number in square brackets, e.g. [2] or [1][4].
- Mention publication dates when timing matters; flag when sources are old or conflict.
- If the sources don't contain the answer, say so plainly and state what is missing — do not fill gaps from memory.
- Lead with the direct answer, then supporting detail. Keep it concise."""


def format_sources(hits: list[tuple[float, dict]]) -> str:
    blocks = []
    for i, (_, c) in enumerate(hits, 1):
        m = c["metadata"]
        blocks.append(
            f"[{i}] {m['title']}\n"
            f"Source: {m['source']} | Topic: {c['topic']} | Published: {m.get('published') or 'unknown'}\n"
            f"URL: {m['url']}\n{c['text']}"
        )
    return "\n\n".join(blocks)


def answer_question(
    client: anthropic.Anthropic, model: str, system: str, question: str, hits: list[tuple[float, dict]]
) -> str:
    if not hits:
        return "No ingested articles match that question yet. Run `ingest` first or rephrase the question."
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=f"{system}\n\n{ANSWER_INSTRUCTIONS}",
        output_config={"effort": "medium"},
        messages=[
            {
                "role": "user",
                "content": f"<sources>\n{format_sources(hits)}\n</sources>\n\nQuestion: {question}",
            }
        ],
        **_FALLBACK_KW,
    )
    if response.stop_reason == "refusal":
        return "The model declined to answer this question."
    return "".join(b.text for b in response.content if b.type == "text").strip()


class TopicAgent:
    """Subclasses set `topic`, `system_prompt`, and `insights_model` (a pydantic
    schema with the topic-specific fields Claude extracts per article)."""

    topic: str
    system_prompt: str
    insights_model: type[BaseModel]

    def __init__(
        self,
        client: anthropic.Anthropic,
        feeds: list[dict],
        store: TopicStore,
        model: str = DEFAULT_MODEL,
        workers: int = 4,
    ) -> None:
        self.client = client
        self.feeds = feeds
        self.store = store
        self.model = model
        self.workers = workers

    # ---- ingestion -------------------------------------------------------

    def collect(self) -> list[FeedItem]:
        """Fetch every feed for this topic and return items not ingested before."""
        seen = self.store.seen_ids()
        new: dict[str, FeedItem] = {}
        for feed in self.feeds:
            for item in fetch_feed(feed.get("name", ""), feed["url"], self.topic):
                if item.id not in seen and item.id not in new:
                    new[item.id] = item
        return list(new.values())

    def enrich(self, item: FeedItem) -> BaseModel | None:
        """Ask Claude for structured insights on one article. Returns None on refusal."""
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=4000,
            system=self.system_prompt,
            output_config={"effort": "low"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Analyse this RSS article and extract the requested fields. "
                        "Base everything on the text below; leave a list empty rather than guessing.\n\n"
                        f"Title: {item.title}\nSource: {item.source}\nPublished: {item.published or 'unknown'}\n"
                        f"URL: {item.url}\n\n{item.content or '(no body text in feed — use the title)'}"
                    ),
                }
            ],
            output_format=self.insights_model,
            **_FALLBACK_KW,
        )
        if response.stop_reason == "refusal":
            log.warning("Refused: %s", item.title)
            return None
        return response.parsed_output

    def build_records(self, item: FeedItem, insights: BaseModel | None) -> tuple[dict, list[dict]]:
        ingested_at = datetime.now(timezone.utc).isoformat()
        data = insights.model_dump() if insights else {}
        article = {
            **item.to_dict(),
            "ingested_at": ingested_at,
            "model": self.model if insights else None,
            "insights": data,
        }

        # Chunk metadata: everything except the long-form text fields, so a vector
        # DB can filter by source, date, tickers, countries, etc.
        meta = {
            "article_id": item.id,
            "title": item.title,
            "url": item.url,
            "source": item.source,
            "published": item.published,
            "author": item.author,
            **{k: v for k, v in data.items() if k not in ("summary", "key_points")},
        }

        chunks = []
        if data.get("summary"):
            points = "\n".join(f"- {p}" for p in data.get("key_points", []))
            chunks.append(
                {
                    "id": f"{item.id}-summary",
                    "topic": self.topic,
                    "chunk_type": "summary",
                    "text": f"{item.title}\n\n{data['summary']}\n\nKey points:\n{points}".strip(),
                    "metadata": meta,
                }
            )
        for i, piece in enumerate(chunk_text(item.content)):
            chunks.append(
                {
                    "id": f"{item.id}-c{i}",
                    "topic": self.topic,
                    "chunk_type": "content",
                    "text": f"{item.title}\n\n{piece}",
                    "metadata": meta,
                }
            )
        if not chunks:  # title-only item with no enrichment
            chunks.append(
                {"id": f"{item.id}-c0", "topic": self.topic, "chunk_type": "content", "text": item.title, "metadata": meta}
            )
        return article, chunks

    def _process(self, item: FeedItem, enrich: bool) -> bool:
        insights = None
        if enrich:
            try:
                insights = self.enrich(item)
            except anthropic.APIStatusError as e:
                if e.status_code in (401, 403):
                    raise  # credentials problem — stop the run
                log.error("Enrichment failed for %r (%s): %s", item.title, e.status_code, e.message)
                return False  # not marked seen, retried next run
            except anthropic.APIConnectionError as e:
                log.error("Network error enriching %r: %s", item.title, e)
                return False
        article, chunks = self.build_records(item, insights)
        self.store.add(article, chunks)
        return True

    def ingest(self, limit: int | None = None, enrich: bool = True) -> dict:
        items = self.collect()
        items.sort(key=lambda i: i.published or "", reverse=True)
        if limit:
            items = items[:limit]
        log.info("[%s] %d new articles to process", self.topic, len(items))

        ok = failed = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._process, it, enrich) for it in items]
            for fut in as_completed(futures):
                if fut.result():
                    ok += 1
                else:
                    failed += 1
        return {"topic": self.topic, "new": len(items), "stored": ok, "failed": failed}

    # ---- question answering ---------------------------------------------

    def ask(self, question: str, k: int = 8) -> str:
        hits = BM25(self.store.load_chunks()).search(question, k)
        return answer_question(self.client, self.model, self.system_prompt, question, hits)
