"""Fetching and normalising RSS/Atom feeds."""

from __future__ import annotations

import calendar
import hashlib
import logging
import re
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser

import feedparser

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; rss-research-agents/1.0)"


@dataclass
class FeedItem:
    id: str
    topic: str
    source: str
    feed_url: str
    title: str
    url: str
    published: str | None
    author: str | None
    content: str

    def to_dict(self) -> dict:
        return asdict(self)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strip_html(raw: str | None) -> str:
    parser = _TextExtractor()
    parser.feed(raw or "")
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def _item_id(entry) -> str:
    key = entry.get("id") or entry.get("link") or entry.get("title", "")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _published(entry) -> str | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc).isoformat()


def _content(entry) -> str:
    if entry.get("content"):
        text = strip_html(entry.content[0].get("value"))
        if text:
            return text
    return strip_html(entry.get("summary") or entry.get("description"))


def fetch_feed(name: str, url: str, topic: str, timeout: float = 20.0) -> list[FeedItem]:
    """Download one feed and return its entries. Network errors are logged, not raised."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception as e:  # one bad feed shouldn't stop the run
        log.warning("Failed to fetch %s (%s): %s", name, url, e)
        return []

    parsed = feedparser.parse(raw)
    if parsed.bozo and not parsed.entries:
        log.warning("Could not parse feed %s: %s", name, parsed.bozo_exception)
        return []

    source = name or parsed.feed.get("title") or url
    items = []
    for entry in parsed.entries:
        title = strip_html(entry.get("title"))
        if not title:
            continue
        items.append(
            FeedItem(
                id=_item_id(entry),
                topic=topic,
                source=source,
                feed_url=url,
                title=title,
                url=entry.get("link", ""),
                published=_published(entry),
                author=entry.get("author"),
                content=_content(entry),
            )
        )
    log.info("%s: %d entries", source, len(items))
    return items
