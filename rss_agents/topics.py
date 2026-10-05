"""The three topic agents. Each one has its own analyst persona and its own
extraction schema, so the RAG metadata fits the domain."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .agent import TopicAgent

Tone = Literal["positive", "negative", "neutral", "mixed"]


class BaseInsights(BaseModel):
    summary: str = Field(description="3-5 sentence factual summary of the article.")
    key_points: list[str] = Field(description="The most important facts, one per item.")
    entities: list[str] = Field(description="Named people, organisations, and places mentioned.")
    tags: list[str] = Field(description="3-8 short lowercase topical tags for search filtering.")
    tone: Tone = Field(description="Overall tone of the reported news.")


# ---- Finance ----------------------------------------------------------------


class FinanceInsights(BaseInsights):
    tickers: list[str] = Field(description="Stock/ETF/crypto tickers explicitly relevant, e.g. AAPL, BTC.")
    companies: list[str]
    asset_classes: list[str] = Field(description="e.g. equities, bonds, fx, commodities, crypto, real estate.")
    economic_indicators: list[str] = Field(description="e.g. CPI, unemployment rate, fed funds rate, GDP.")
    market_impact: Literal["bullish", "bearish", "neutral", "mixed", "unclear"]
    figures: list[str] = Field(description="Key numbers with units and context, e.g. 'Q3 revenue $94.9B (+6% YoY)'.")


class FinanceAgent(TopicAgent):
    topic = "finance"
    insights_model = FinanceInsights
    system_prompt = (
        "You are a financial markets research analyst. You read business and markets news and "
        "extract precise, verifiable information: companies, tickers, figures with units and periods, "
        "macro indicators, central-bank actions, and likely market relevance. Be exact with numbers and "
        "never invent tickers or figures that are not in the text. You do not give personalised "
        "investment advice; you report and explain what the sources say."
    )


# ---- Technology -------------------------------------------------------------


class TechnologyInsights(BaseInsights):
    companies: list[str]
    products: list[str] = Field(description="Named products, services, models, or projects.")
    technologies: list[str] = Field(description="e.g. large language models, 3nm chips, Rust, quantum computing.")
    category: Literal[
        "ai", "security", "hardware", "software", "cloud", "mobile", "science",
        "business", "policy_regulation", "consumer", "other",
    ]
    announcement_type: Literal[
        "product_launch", "research", "funding_or_mna", "security_incident",
        "regulation", "earnings", "opinion_or_analysis", "other",
    ]


class TechnologyAgent(TopicAgent):
    topic = "technology"
    insights_model = TechnologyInsights
    system_prompt = (
        "You are a technology research analyst. You read tech news and extract the companies, products, "
        "underlying technologies, and what actually happened (launch, research result, funding, breach, "
        "regulation). Distinguish shipped products from announcements and rumours, and keep technical "
        "terms accurate."
    )


# ---- Politics ---------------------------------------------------------------


class PoliticsInsights(BaseInsights):
    countries: list[str]
    political_figures: list[str]
    institutions: list[str] = Field(description="Parties, legislatures, courts, agencies, international bodies.")
    policy_areas: list[str] = Field(description="e.g. economy, immigration, healthcare, defence, climate.")
    event_type: Literal[
        "election", "legislation", "policy_announcement", "court_ruling", "diplomacy",
        "conflict", "protest", "scandal_or_investigation", "polling", "opinion", "other",
    ]


class PoliticsAgent(TopicAgent):
    topic = "politics"
    insights_model = PoliticsInsights
    system_prompt = (
        "You are a non-partisan political research analyst. You read political news and extract who did "
        "what, where, and with what policy consequence. Report positions by attributing them to the people "
        "or parties who hold them, keep summaries neutral and factual, and clearly separate reported facts "
        "from opinion or commentary in the source."
    )


AGENTS: dict[str, type[TopicAgent]] = {
    "finance": FinanceAgent,
    "technology": TechnologyAgent,
    "politics": PoliticsAgent,
}
