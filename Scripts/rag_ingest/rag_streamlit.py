#!/usr/bin/env python3
"""
Streamlit UI for rag_cli_v2.
Optimized for intent-heavy prompts (parlays, totals advice, comparisons).
"""

from __future__ import annotations

import io
import os
import time
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from typing import Dict, List

import streamlit as st


def apply_runtime_secrets() -> None:
    """
    Mirror Streamlit secrets into environment variables expected by CLI/runtime.
    """
    if not hasattr(st, "secrets"):
        return
    try:
        sec = st.secrets
        # Triggers Streamlit secrets parsing once; if no secrets file exists,
        # we should silently fall back to environment variables.
        _ = list(sec.keys())
    except Exception:
        return
    mappings = [
        ("OPENAI_API_KEY", "OPENAI_API_KEY"),
        ("ODDS_API_KEY", "ODDS_API_KEY"),
        ("ODDS_API_KEY", "ODDS-API"),
        ("ODDS-API", "ODDS-API"),
        ("ODDS-API", "ODDS_API_KEY"),
        ("RAG_CHAT_MODEL", "RAG_CHAT_MODEL"),
        ("CHROMA_COLLECTION", "CHROMA_COLLECTION"),
        ("CHROMA_MODE", "CHROMA_MODE"),
        ("CHROMA_HTTP_HOST", "CHROMA_HTTP_HOST"),
        ("CHROMA_HTTP_PORT", "CHROMA_HTTP_PORT"),
        ("CHROMA_HTTP_SSL", "CHROMA_HTTP_SSL"),
        ("CHROMA_AUTH_TOKEN", "CHROMA_AUTH_TOKEN"),
        ("CHROMA_DATABASE", "CHROMA_DATABASE"),
        ("CHROMA_TENANT", "CHROMA_TENANT"),
    ]
    for src, dst in mappings:
        try:
            val = sec.get(src, None)
        except Exception:
            val = None
        if val is not None and str(val).strip() != "":
            os.environ[dst] = str(val)


apply_runtime_secrets()

import rag_cli_v2 as rag


LEAGUES = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]


def _next_seven_days() -> List[Dict]:
    """Build selectable date options for the next 7 days."""
    today = datetime.now().date()
    options: List[Dict] = []
    for i in range(7):
        d = today + timedelta(days=i)
        if i == 0:
            prefix = "Today"
        elif i == 1:
            prefix = "Tomorrow"
        else:
            prefix = d.strftime("%A")
        day_suffix = _ordinal(d.day)
        label = f"{prefix} — {d.strftime('%b')} {day_suffix}"
        phrase = f"{day_suffix} {d.strftime('%B')}"
        options.append({"label": label, "phrase": phrase, "date": d})
    return options


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_day_fixtures(league: str, iso_date: str) -> List[Dict]:
    """Fetch fixtures for a specific date (cached 5 min). iso_date = 'YYYY-MM-DD'."""
    from datetime import date as _date

    try:
        events, _ = rag.fetch_events(league, set(rag.DEFAULT_MARKETS))
    except Exception:
        return []
    if not events:
        return []
    target = _date.fromisoformat(iso_date)
    filtered = rag.filter_events_by_exact_date(events, target)
    return [
        {"home_team": ev.get("home_team", ""), "away_team": ev.get("away_team", "")}
        for ev in filtered
        if ev.get("home_team") and ev.get("away_team")
    ]


def _is_single_fixture_template(template_key: str) -> bool:
    k = template_key.lower()
    return "(single" in k


def init_state() -> None:
    if "chat" not in st.session_state:
        st.session_state.chat: List[Dict] = []
    if "league" not in st.session_state:
        st.session_state.league = "EPL"
    if "draft_prompt" not in st.session_state:
        st.session_state.draft_prompt = ""
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False


def clear_all() -> None:
    st.session_state.chat = []
    st.session_state.draft_prompt = ""
    rag.memory["history"] = []
    rag.memory["last_query"] = None
    rag.memory["last_answer"] = None
    rag.memory["last_selected_event_ids"] = []
    rag.memory["last_selected_fixtures"] = []
    rag.memory["last_leg_count"] = None
    rag.memory["last_intent"] = None
    rag.memory["last_league"] = None
    rag.memory["last_constraint_spec"] = None
    rag.memory["last_selected_legs"] = []
    rag.memory["last_market_groups_used"] = []


def run_query(prompt: str, league: str) -> str:
    buf = io.StringIO()
    start = time.perf_counter()
    with redirect_stdout(buf):
        rag.answer_once(prompt, default_league=league)
    elapsed = time.perf_counter() - start
    text = buf.getvalue().strip() or "(No output)"
    stamp = datetime.now().strftime("%H:%M:%S")
    return text, elapsed, stamp


_LEAGUE_EXAMPLE_TEAMS: Dict[str, tuple] = {
    "EPL": ("Arsenal", "Chelsea"),
    "LaLiga": ("Real Madrid", "Barcelona"),
    "SerieA": ("Juventus", "Inter Milan"),
    "Bundesliga": ("Bayern Munich", "Borussia Dortmund"),
    "Ligue1": ("Paris Saint Germain", "Marseille"),
}

def template_prompts(target_date: str, home: str = "", away: str = "", league: str = "EPL") -> Dict[str, Dict[str, str]]:
    date_text = target_date.strip() or "10th February"
    lg = league or "EPL"
    defaults = _LEAGUE_EXAMPLE_TEAMS.get(lg, _LEAGUE_EXAMPLE_TEAMS["EPL"])
    h = home or defaults[0]
    a = away or defaults[1]
    fx = f"{h} vs {a}"
    fx_at = f"{h} @ {a}"
    return {
        "Schedule & Availability": {
            "Schedule Check": f"What {lg} matches are on {date_text}?",
            "Market Availability (corners)": f"Are corner lines available for {lg} games on {date_text}?",
            "Market Availability (cards)": f"Are booking/card markets available for {lg} games on {date_text}?",
            "Market Availability (player props)": f"Are player prop markets available for {lg} games on {date_text}?",
            "Fixture + Kickoff Times": f"Show me all {lg} fixtures on {date_text} and their kickoff times.",
            "Full Day Overview": (
                f"Give me a full overview of all {lg} action on {date_text}: "
                "fixtures, kickoff times, and which market types are open for each game."
            ),
        },
        "Comparisons": {
            "Corners Winner (all fixtures)": (
                f"For all {lg} games on {date_text}, which team will get more corners in each fixture?"
            ),
            "Corners Winner (single fixture)": (
                f"Who will win the corner count in {fx} on {date_text}? "
                "Break down why with stats."
            ),
            "Cards Winner (single fixture)": (
                f"Which team will get more yellow cards in {fx} on {date_text}? "
                "Give me a solid explanation of your reasoning."
            ),
            "Cards Winner (all fixtures)": (
                f"For every {lg} game on {date_text}, which team will get more cards? "
                "Rate your confidence for each."
            ),
            "Goals Winner (single fixture)": (
                f"Who is more likely to score more goals in {fx} on {date_text}? "
                "Back it up with form and xG data."
            ),
            "SoT Winner (single fixture)": (
                f"Which team will have more shots on target in {fx} on {date_text}? "
                "Use recent SoT averages and form data."
            ),
            "SoT Winner (all fixtures)": (
                f"For every {lg} game on {date_text}, which team will have more shots on target? "
                "Rate your confidence for each."
            ),
            "Possession & Control (single fixture)": (
                f"Compare possession and control index for both teams in {fx} on {date_text}."
            ),
        },
        "Totals Lines": {
            "Goals O/U (single fixture)": (
                f"{fx} on {date_text}. What over/under goals line should I take?"
            ),
            "Goals O/U (all fixtures)": (
                f"Give me the over/under goals line I should take for each {lg} game on {date_text}."
            ),
            "Corners O/U (single fixture)": (
                f"{fx_at} on {date_text}. Give me corner totals for this game. "
                "Which line should I take?"
            ),
            "Corners O/U (all fixtures)": (
                f"For every {lg} game on {date_text}, recommend a corner totals line for each fixture."
            ),
            "Cards O/U (single fixture)": (
                f"{fx} on {date_text}. What over/under cards line should I take?"
            ),
            "Cards O/U (all fixtures)": (
                f"For each {lg} game on {date_text}, recommend a cards over/under line."
            ),
            "SoT O/U (single fixture)": (
                f"{fx} on {date_text}. What over/under shots on target line should I take? "
                "Use recent SoT averages for both teams."
            ),
            "SoT O/U (all fixtures)": (
                f"For every {lg} game on {date_text}, recommend a shots on target over/under line for each fixture."
            ),
            "BTTS (single fixture)": (
                f"{fx} on {date_text}. Should I take both teams to score yes or no? "
                "Explain your reasoning."
            ),
            "BTTS (all fixtures)": (
                f"For all {lg} games on {date_text}, which fixtures are best for BTTS Yes?"
            ),
        },
        "Parlays": {
            "4-leg Goals Totals (odds cap)": (
                f"For all {lg} games on {date_text}, make a 4-leg parlay with exactly one leg per fixture "
                "using only full-game over/under goals totals lines. Keep combined decimal odds at or below 4.3x."
            ),
            "3-leg Goals Totals (conservative)": (
                f"For {lg} games on {date_text}, build a 3-leg parlay using only goals over/under totals. "
                "Keep it safe — combined odds under 3.0x."
            ),
            "Single Fixture 3-leg (single fixture)": (
                f"It's {date_text}. I want to bet {fx}. "
                "Make me a 3-leg parlay for this game with combined odds no more than 3.0x."
            ),
            "Single Fixture 2-leg (single fixture)": (
                f"It's {date_text}. {fx}. "
                "Make me a 2-leg same-game parlay with combined odds no more than 2.2x."
            ),
            "Single Bet (single fixture)": (
                f"It's {date_text}. {fx}. "
                "Give me a 1-leg parlay (single bet) for this game, odds between 1.50 and 2.10, with reasoning."
            ),
            "Corners-only Multi-fixture": (
                f"For all {lg} games on {date_text}, make a parlay with ONLY CORNER TOTALS, "
                "one corner total per game, full-game markets only, no spreads, around 3.0x combined odds."
            ),
            "Cards-only Multi-fixture": (
                f"For all {lg} games on {date_text}, build a parlay with ONLY CARD TOTALS, "
                "one card total per game, full-game markets only, no spreads, around 3.5x combined odds."
            ),
            "SoT-only Multi-fixture": (
                f"For all {lg} games on {date_text}, build a parlay using ONLY shots on target totals, "
                "one SoT leg per game, full-game markets only, no spreads, around 3.0x combined odds."
            ),
            "Mixed Markets (3-leg)": (
                f"Build a 3-leg {lg} parlay for {date_text} around 2.8x. "
                "Include exactly one corners leg, one cards leg, and one goals totals leg."
            ),
            "Mixed Markets (4-leg)": (
                f"Build a 4-leg {lg} parlay for {date_text} around 4.0x. "
                "Mix goals, corners, and cards markets. At least two different fixtures."
            ),
            "High-odds Multi-fixture (5x+)": (
                f"For all {lg} games on {date_text}, build a parlay targeting around 5.5x combined odds. "
                "One leg per fixture, mix market types, no spreads."
            ),
        },
        "Player Props": {
            "Anytime Goalscorer (single fixture)": (
                f"It's {date_text}. Who is the best anytime goalscorer bet for {fx}? "
                "Use recent form, xG, and minutes data."
            ),
            "Anytime Goalscorer (all fixtures)": (
                f"For each {lg} game on {date_text}, give me one anytime goalscorer pick per fixture "
                "with stats-backed reasoning."
            ),
            "Shots on Target (single fixture)": (
                f"{fx} on {date_text}. Which player is most likely to go over 0.5 shots on target? "
                "Use recent SoT data and minutes played."
            ),
            "Shots on Target (best picks)": (
                f"For {lg} games on {date_text}, which players are best bets for over 0.5 shots on target? "
                "Give me your top 3 picks with reasoning."
            ),
            "Player Cards (who gets booked?)": (
                f"Which players are most likely to receive a yellow card across all {lg} games on {date_text}? "
                "Give me your top 3 picks with fouls and card history."
            ),
            "Player Assists (single fixture)": (
                f"Who is most likely to register an assist in {fx} on {date_text}? "
                "Use recent chance creation and assist data."
            ),
            "Player Parlay (2-leg)": (
                f"Build a 2-leg player props parlay from {lg} games on {date_text}. "
                "Mix goalscorer and shots markets, keep odds around 3.0x."
            ),
            "Player Parlay (3-leg)": (
                f"Build a 3-leg player props parlay from {lg} games on {date_text}. "
                "Include one goalscorer, one shots, and one cards pick. Target around 6.0x odds."
            ),
        },
        "Memory Follow-ups": {
            "Add One More Leg": "Add one more leg.",
            "Swap Weakest Leg": "Replace the weakest leg with a better alternative from the same fixture.",
            "Tighten Risk": "Keep the same fixtures, but make the parlay safer and cap combined odds at 2.6x.",
            "Increase Risk": "Same fixtures and market types, but push the combined odds up to around 5.0x.",
            "Keep Markets, New Odds Target": "Same matchup and market types, but target around 3.5x this time.",
            "Same Fixtures, Different Markets": (
                "Keep the same fixtures but switch to different market types. Surprise me."
            ),
        },
    }


def render_chat() -> None:
    for turn in st.session_state.chat:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.code(turn["assistant"], language="text")
            st.caption(f"{turn['meta']}")


def sidebar_ui() -> None:
    with st.sidebar:
        st.header("Settings")
        st.session_state.league = st.selectbox("League", options=LEAGUES, index=LEAGUES.index(st.session_state.league))
        st.session_state.show_debug = st.toggle("Show debug memory", value=st.session_state.show_debug)

        st.header("Templates")
        date_options = _next_seven_days()
        date_labels = [d["label"] for d in date_options]
        selected_date_idx = st.selectbox(
            "Match day",
            options=range(len(date_labels)),
            format_func=lambda i: date_labels[i],
        )
        target_date = date_options[selected_date_idx]["phrase"]
        target_date_obj = date_options[selected_date_idx]["date"]

        # Build templates with example teams first (for workflow/template key navigation)
        templates = template_prompts(target_date, league=st.session_state.league)
        selected_workflow = st.selectbox("Workflow", options=list(templates.keys()))
        selected_template = st.selectbox(
            "Prompt template",
            options=list(templates[selected_workflow].keys()),
        )

        # Conditional fixture picker for single-fixture templates
        home, away = "", ""
        if _is_single_fixture_template(selected_template):
            fixtures = _fetch_day_fixtures(
                st.session_state.league, target_date_obj.isoformat(),
            )
            if fixtures:
                fixture_labels = [
                    f"{f['home_team']} vs {f['away_team']}" for f in fixtures
                ]
                selected_fx = st.selectbox("Match", options=fixture_labels)
                parts = selected_fx.split(" vs ", 1)
                home, away = parts[0].strip(), parts[1].strip()
            else:
                st.caption("No fixtures found for this date yet.")

        # Rebuild templates with real fixture teams when available
        if home and away:
            templates = template_prompts(target_date, home=home, away=away, league=st.session_state.league)

        if st.button("Load template"):
            st.session_state.draft_prompt = templates[selected_workflow][selected_template]

        with st.expander("All workflow examples"):
            for workflow, prompts in templates.items():
                st.markdown(f"**{workflow}**")
                for label, text in prompts.items():
                    st.markdown(f"- `{label}`: {text}")

        st.header("Session")
        if st.button("Clear chat + memory"):
            clear_all()
            st.rerun()

        if st.session_state.show_debug:
            st.subheader("RAG memory")
            st.json(
                {
                    "last_query": rag.memory.get("last_query"),
                    "last_intent": rag.memory.get("last_intent"),
                    "last_league": rag.memory.get("last_league"),
                    "last_selected_fixtures": rag.memory.get("last_selected_fixtures"),
                    "last_leg_count": rag.memory.get("last_leg_count"),
                    "last_market_groups_used": rag.memory.get("last_market_groups_used"),
                    "last_constraint_spec": rag.memory.get("last_constraint_spec"),
                    "last_selected_legs": rag.memory.get("last_selected_legs"),
                    "history_turns": len(rag.memory.get("history") or []),
                }
            )


def main() -> None:
    st.set_page_config(page_title="Betting RAG v2", layout="wide")
    kb_ok, kb_msg = rag.kb_healthcheck()
    if not kb_ok:
        st.error(f"KB health check failed: {kb_msg}")
        st.info(
            "Set remote Chroma env vars/secrets (`CHROMA_MODE=http`, `CHROMA_HTTP_HOST`, etc.) "
            "or provide local `Index/chroma` with data."
        )
        st.stop()

    init_state()
    sidebar_ui()

    st.title("Betting RAG v2")
    st.caption("Streamlit wrapper for `rag_cli_v2` with prompt templates and session memory visibility.")
    st.caption(kb_msg)

    render_chat()

    with st.form("prompt_form", clear_on_submit=False):
        prompt = st.text_area(
            "Prompt",
            value=st.session_state.draft_prompt,
            height=110,
            placeholder="Ask for parlays, totals lines, fixture schedule, or stat comparisons...",
        )
        col1, col2 = st.columns([1, 1])
        submitted = col1.form_submit_button("Send")
        quick_followup = col2.form_submit_button("Add one more leg")

    if quick_followup:
        prompt = "Add one more leg."
        st.session_state.draft_prompt = ""

    if submitted and prompt.strip():
        clean_prompt = prompt.strip()
        st.session_state.draft_prompt = ""

        with st.chat_message("user"):
            st.write(clean_prompt)

        with st.spinner("Running RAG..."):
            answer_text, elapsed, stamp = run_query(clean_prompt, st.session_state.league)

        with st.chat_message("assistant"):
            st.code(answer_text, language="text")
            st.caption(f"{stamp} | {elapsed:.2f}s | league={st.session_state.league}")

        st.session_state.chat.append(
            {
                "user": clean_prompt,
                "assistant": answer_text,
                "meta": f"{stamp} | {elapsed:.2f}s | league={st.session_state.league}",
            }
        )


if __name__ == "__main__":
    main()
