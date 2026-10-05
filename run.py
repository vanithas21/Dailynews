"""CLI for the RSS research agents.

  python run.py ingest                       # all three agents
  python run.py ingest --topic finance --limit 20
  python run.py ask "What did the Fed signal about rates?" --topic finance
  python run.py ask "How is AI regulation affecting chipmakers?"   # searches every topic
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import anthropic
import yaml

from rss_agents.agent import DEFAULT_MODEL, answer_question
from rss_agents.store import BM25, TopicStore
from rss_agents.topics import AGENTS

ROOT = Path(__file__).parent
CROSS_TOPIC_SYSTEM = (
    "You are a research analyst covering finance, technology, and politics. You connect developments "
    "across these domains, stay neutral on political questions, and never give personalised investment advice."
)


def build_agents(topics: list[str], args) -> list:
    feeds = yaml.safe_load(Path(args.feeds).read_text(encoding="utf-8"))
    client = anthropic.Anthropic()
    return [
        AGENTS[t](client, feeds.get(t, []), TopicStore(Path(args.out), t), model=args.model, workers=args.workers)
        for t in topics
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Finance / Technology / Politics RSS research agents")
    p.add_argument("--feeds", default=str(ROOT / "config" / "feeds.yaml"))
    p.add_argument("--out", default=str(ROOT / "data"), help="Output folder for the RAG JSONL files")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=4, help="Parallel Claude calls per agent")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Fetch feeds, enrich new articles, write JSONL")
    ing.add_argument("--topic", choices=[*AGENTS, "all"], default="all")
    ing.add_argument("--limit", type=int, help="Max new articles per topic this run")
    ing.add_argument("--no-enrich", action="store_true", help="Skip Claude; store raw articles only")

    ask = sub.add_parser("ask", help="Answer a question from ingested articles")
    ask.add_argument("question")
    ask.add_argument("--topic", choices=[*AGENTS, "all"], default="all")
    ask.add_argument("-k", type=int, default=8, help="Number of chunks to retrieve")

    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose or args.cmd == "ingest" else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    topics = list(AGENTS) if args.topic == "all" else [args.topic]
    agents = build_agents(topics, args)

    if args.cmd == "ingest":
        for agent in agents:
            print(json.dumps(agent.ingest(limit=args.limit, enrich=not args.no_enrich)))
        return 0

    if len(agents) == 1:
        print(agents[0].ask(args.question, k=args.k))
    else:
        chunks = [c for a in agents for c in a.store.load_chunks()]
        hits = BM25(chunks).search(args.question, args.k)
        print(answer_question(agents[0].client, args.model, CROSS_TOPIC_SYSTEM, args.question, hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
