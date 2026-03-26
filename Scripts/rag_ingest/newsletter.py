"""
newsletter.py — Matchday newsletter module.

Scrapes football news from local-language sources via Google News RSS,
summarizes with GPT, and produces per-league matchday digests.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client (lazy, same pattern as rag_cli_v2.py)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "gpt-4o-mini")

_openai_client = None


def _get_openai_client():
    """Lazy-init OpenAI client."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        return _openai_client
    except Exception as exc:
        log.warning("Failed to initialize OpenAI client: %s", exc)
        return None


# ---------------------------------------------------------------------------
# League-specific news source configuration
# ---------------------------------------------------------------------------
LEAGUE_NEWS_CONFIG: Dict[str, Dict] = {
    "EPL": {
        "lang": "en",
        "country": "GB",
        "ceid": "GB:en",
        "sources": ["bbc.co.uk", "skysports.com", "theguardian.com"],
    },
    "LaLiga": {
        "lang": "es",
        "country": "ES",
        "ceid": "ES:es",
        "sources": ["marca.com", "as.com", "mundodeportivo.com"],
    },
    "SerieA": {
        "lang": "it",
        "country": "IT",
        "ceid": "IT:it",
        "sources": ["gazzetta.it", "corrieredellosport.it"],
    },
    "Bundesliga": {
        "lang": "de",
        "country": "DE",
        "ceid": "DE:de",
        "sources": ["kicker.de", "bild.de", "sport1.de"],
    },
    "Ligue1": {
        "lang": "fr",
        "country": "FR",
        "ceid": "FR:fr",
        "sources": ["lequipe.fr", "rmcsport.bfmtv.com"],
    },
    "UCL": {
        "lang": "en",
        "country": "GB",
        "ceid": "GB:en",
        "sources": ["uefa.com"],
    },
    "UEL": {
        "lang": "en",
        "country": "GB",
        "ceid": "GB:en",
        "sources": ["uefa.com"],
    },
    "UECL": {
        "lang": "en",
        "country": "GB",
        "ceid": "GB:en",
        "sources": ["uefa.com"],
    },
}

# Google News RSS base URL
_GNEWS_RSS_BASE = "https://news.google.com/rss/search"

# How far back (in hours) to consider articles
_ARTICLE_AGE_HOURS = 48

# Title similarity threshold for deduplication (0.0 - 1.0)
_TITLE_SIMILARITY_THRESHOLD = 0.80

# HTTP request timeout in seconds
_HTTP_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _title_similarity(a: str, b: str) -> float:
    """Return 0.0-1.0 similarity ratio between two title strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _parse_pub_date(raw: str) -> Optional[datetime]:
    """Parse RSS pubDate into a timezone-aware datetime."""
    try:
        return parsedate_to_datetime(raw)
    except Exception:
        return None


def _fetch_rss(query: str, lang: str, country: str, ceid: str) -> List[Dict]:
    """Fetch and parse a Google News RSS feed. Returns list of article dicts."""
    encoded_query = urllib.request.quote(query)
    url = (
        f"{_GNEWS_RSS_BASE}?q={encoded_query}"
        f"&hl={lang}&gl={country}&ceid={ceid}"
    )
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; MatchwiseBot/1.0)"
        })
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            data = resp.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        log.debug("RSS fetch failed for query=%r: %s", query, exc)
        return []

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        log.debug("RSS XML parse failed for query=%r: %s", query, exc)
        return []

    articles: List[Dict] = []
    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        source_el = item.find("source")
        pub_el = item.find("pubDate")
        desc_el = item.find("description")

        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        link = link_el.text.strip() if link_el is not None and link_el.text else ""
        source = source_el.text.strip() if source_el is not None and source_el.text else ""
        pub_raw = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
        desc = _strip_html(desc_el.text) if desc_el is not None and desc_el.text else ""

        if not title:
            continue

        articles.append({
            "title": title,
            "snippet": desc,
            "source": source,
            "url": link,
            "published": pub_raw,
            "_pub_dt": _parse_pub_date(pub_raw),
        })

    return articles


def _deduplicate(articles: List[Dict]) -> List[Dict]:
    """Remove articles whose titles are too similar to an already-kept article."""
    kept: List[Dict] = []
    for art in articles:
        is_dup = False
        for existing in kept:
            if _title_similarity(art["title"], existing["title"]) > _TITLE_SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            kept.append(art)
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news_for_fixture(
    home: str,
    away: str,
    league: str,
    max_articles: int = 8,
) -> List[Dict]:
    """
    Fetch recent news articles about a fixture from Google News RSS.

    Returns a list of article dicts, each with keys:
      title, snippet, source, url, published
    Sorted by recency, capped at *max_articles*. Returns [] on any failure.
    """
    try:
        config = LEAGUE_NEWS_CONFIG.get(league, LEAGUE_NEWS_CONFIG["EPL"])
        lang = config["lang"]
        country = config["country"]
        ceid = config["ceid"]

        # Two query variants for better coverage
        queries = [
            f'"{home}" "{away}"',
            f'"{home} vs {away}"',
        ]

        all_articles: List[Dict] = []
        for q in queries:
            all_articles.extend(_fetch_rss(q, lang, country, ceid))

        if not all_articles:
            return []

        # Filter to articles from the last 48 hours
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=_ARTICLE_AGE_HOURS)
        recent: List[Dict] = []
        for art in all_articles:
            pub_dt = art.get("_pub_dt")
            if pub_dt is not None:
                if pub_dt < cutoff:
                    continue
            # If we can't parse the date, keep it (benefit of the doubt)
            recent.append(art)

        # Deduplicate by title similarity
        deduped = _deduplicate(recent)

        # Sort by recency (newest first), articles without parseable dates go last
        def _sort_key(art: Dict):
            dt = art.get("_pub_dt")
            if dt is None:
                return datetime.min.replace(tzinfo=timezone.utc)
            return dt

        deduped.sort(key=_sort_key, reverse=True)

        # Strip internal fields and cap
        result: List[Dict] = []
        for art in deduped[:max_articles]:
            result.append({
                "title": art["title"],
                "snippet": art["snippet"],
                "source": art["source"],
                "url": art["url"],
                "published": art["published"],
            })

        return result

    except Exception as exc:
        log.warning("fetch_news_for_fixture(%s vs %s, %s) failed: %s", home, away, league, exc)
        return []


def summarize_fixture_news(
    home: str,
    away: str,
    league: str,
    articles: List[Dict],
) -> Optional[str]:
    """
    Summarize news articles about a fixture using GPT.

    Returns a structured summary string or None if no articles / API unavailable.
    """
    if not articles:
        return None

    client = _get_openai_client()
    if client is None:
        log.warning("summarize_fixture_news: OpenAI client unavailable (no API key?)")
        return None

    # Format articles for the prompt
    formatted_lines: List[str] = []
    for i, art in enumerate(articles, 1):
        lines = [f"[{i}] {art['title']}"]
        if art.get("snippet"):
            lines.append(f"    {art['snippet']}")
        if art.get("source"):
            lines.append(f"    Source: {art['source']}")
        formatted_lines.append("\n".join(lines))

    articles_block = "\n\n".join(formatted_lines)

    prompt = f"""\
You are a football match preview writer for a betting analysis community.
Summarize the following news articles about {home} vs {away} ({league}).

Extract and organize into these categories (skip any with no information):
- INJURIES & AVAILABILITY: Who's doubtful, ruled out, or returning
- LINEUP HINTS: Rotation signals, confirmed starters, formation changes
- TACTICAL NOTES: Style changes, pressing intensity, set piece focus
- FORM & MOTIVATION: What each team is playing for, recent results context
- HEAD-TO-HEAD: Recent meetings, historical patterns in this fixture

Rules:
- Write in English regardless of source article language
- Cite the source in parentheses after each fact, e.g. "Musiala doubtful (Kicker)"
- Be concise — max 3-4 bullet points per category
- If a category has no relevant information, omit it entirely
- Do not invent information not present in the articles

Articles:
{articles_block}"""

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
        )
        text = response.choices[0].message.content
        return text.strip() if text else None
    except Exception as exc:
        log.warning("summarize_fixture_news GPT call failed: %s", exc)
        return None


def generate_league_digest(
    league: str,
    fixtures: List[Dict],
) -> Optional[str]:
    """
    Generate a news digest for all fixtures in a league.

    *fixtures* is a list of dicts with at least ``home_team`` and ``away_team`` keys.
    Returns a combined digest string or None if no fixture produced content.
    """
    separator = "\u2501" * 30  # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    sections: List[str] = []
    for fix in fixtures:
        home = fix.get("home_team", "")
        away = fix.get("away_team", "")
        if not home or not away:
            continue

        articles = fetch_news_for_fixture(home, away, league)
        if not articles:
            log.debug("No articles found for %s vs %s (%s)", home, away, league)
            continue

        summary = summarize_fixture_news(home, away, league, articles)
        if not summary:
            continue

        sections.append(f"{home} vs {away}\n{summary}")

    if not sections:
        return None

    return f"\n{separator}\n".join(sections)


def generate_newsletter(
    leagues: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Main entry point. Generate matchday newsletter for the given leagues.

    If *leagues* is None, defaults to all 5 domestic leagues plus active European
    competitions. Fetches today's fixtures via the odds provider, then builds
    a digest per league.

    Returns ``{league: digest_text}`` for leagues that had content.
    """
    if leagues is None:
        leagues = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1",
                    "UCL", "UEL", "UECL"]

    # Lazy import to avoid circular dependencies and keep module standalone-testable
    try:
        from rag_cli_v2 import fetch_events, DEFAULT_MARKETS  # noqa: F811
    except ImportError:
        log.warning("Could not import fetch_events from rag_cli_v2; "
                     "generate_newsletter requires the RAG CLI module on sys.path.")
        return {}

    digests: Dict[str, str] = {}
    for league in leagues:
        try:
            events, _notes = fetch_events(league, set(DEFAULT_MARKETS))
        except Exception as exc:
            log.warning("Failed to fetch events for %s: %s", league, exc)
            continue

        if not events:
            log.debug("No events for %s today", league)
            continue

        # Convert odds-API events to fixture dicts
        fixtures: List[Dict] = []
        for ev in events:
            home = ev.get("home_team", "")
            away = ev.get("away_team", "")
            if home and away:
                fixtures.append({"home_team": home, "away_team": away})

        if not fixtures:
            continue

        digest = generate_league_digest(league, fixtures)
        if digest:
            digests[league] = digest

    return digests


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Matchday newsletter generator")
    parser.add_argument("--league", default="EPL", help="League code (default: EPL)")
    parser.add_argument("--home", default=None, help="Home team name (for single-fixture test)")
    parser.add_argument("--away", default=None, help="Away team name (for single-fixture test)")
    args = parser.parse_args()

    if args.home and args.away:
        # Test single fixture
        articles = fetch_news_for_fixture(args.home, args.away, args.league)
        print(f"Found {len(articles)} articles")
        for a in articles:
            print(f"  [{a['source']}] {a['title']}")
        if articles:
            summary = summarize_fixture_news(args.home, args.away, args.league, articles)
            print(f"\nSummary:\n{summary}")
    else:
        # Test full league
        digests = generate_newsletter([args.league])
        for league, text in digests.items():
            print(f"\n{'=' * 60}")
            print(f"  {league}")
            print(f"{'=' * 60}")
            print(text)
