# RSS Research Agents: Finance, Technology, Politics

Three topic agents, each with its own RSS feeds, its own analyst system prompt, and its own extraction schema. Each agent:

1. **Collects** new items from its feeds and skips anything it has already ingested.
2. **Enriches** each article with Claude (`claude-opus-5-5`, structured output): a summary, key points, entities, tags, tone, and fields specific to the topic.
3. **Writes RAG-ready JSONL** to `data/<topic>/`.
4. **Answers questions** with citations from what it has ingested. This is a built-in baseline until your RAG portal is ready.

| Agent | Topic-specific fields |
|---|---|
| `finance` | `tickers`, `companies`, `asset_classes`, `economic_indicators`, `market_impact`, `figures` |
| `technology` | `companies`, `products`, `technologies`, `category`, `announcement_type` |
| `politics` | `countries`, `political_figures`, `institutions`, `policy_areas`, `event_type` (neutral, attributed summaries) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set ANTHROPIC_API_KEY=sk-ant-...
```

Feeds are configured in `config/feeds.yaml`.

## Usage

```bash
python run.py ingest                                  # all three agents
python run.py ingest --topic finance --limit 20       # one agent, cap new articles
python run.py ingest --no-enrich                      # no Claude calls, raw articles only
python run.py ask "What did the Fed signal about rates?" --topic finance
python run.py ask "How is AI regulation affecting chipmakers?"   # searches all topics
```

Run `ingest` on a schedule (for example, Windows Task Scheduler every hour). Only new articles cost API calls.

Environment options: `RSS_AGENT_MODEL` overrides the model, and `RSS_AGENT_FALLBACKS=0` disables server-side refusal fallback (it is on by default).

## Output for your RAG system

```
data/
  finance/    articles.jsonl  chunks.jsonl  seen.json
  technology/ ...
  politics/   ...
```

**`chunks.jsonl`** is the file to load into your vector store. It has one chunk per line:

```json
{
  "id": "4924527dfc9a1b77-summary",
  "topic": "politics",
  "chunk_type": "summary",          // or "content"
  "text": "Title\n\nSummary...\n\nKey points:\n- ...",
  "metadata": {
    "article_id": "...", "title": "...", "url": "...", "source": "BBC Politics",
    "published": "2026-10-05T09:12:00+00:00", "author": "...",
    "entities": [...], "tags": [...], "tone": "neutral",
    "countries": [...], "event_type": "diplomacy", ...
  }
}
```

- Each article produces one `summary` chunk plus one or more `content` chunks (about 220 words each, with overlap). Chunk `id`s are stable, so you can upsert them without creating duplicates.
- Use `topic`, `source`, `published`, `tickers`, `countries`, and similar fields as **metadata filters** in the portal. For example, a "Finance only, last 7 days" filter maps directly onto these fields.
- Some vector DBs (such as Chroma) only accept scalar metadata. For those, join list fields like `", ".join(tags)` when you load the chunks.
- **`articles.jsonl`** holds the full enriched record for each article. Use it for an article detail view or to re-chunk the data with a different strategy.

### Loading example (embedding model of your choice)

```python
import json
for line in open("data/finance/chunks.jsonl", encoding="utf-8"):
    c = json.loads(line)
    vector_store.upsert(id=c["id"], text=c["text"], metadata={"topic": c["topic"], **c["metadata"]})
```

## Notes

- RSS feeds often include only a teaser rather than the full article, so answers are limited to what the feed contains. If you need more depth, add a step that fetches the full article text.
- `ask` uses BM25 keyword retrieval over the local JSONL. Your portal will replace this with vector search, but you can reuse `answer_question()` in `rss_agents/agent.py`: it already formats sources and enforces answers that cite their sources.
- A feed that fails to load is logged and skipped. An article whose enrichment fails is not marked as seen, so the next run retries it.
