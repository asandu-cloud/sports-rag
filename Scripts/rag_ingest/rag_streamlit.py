#!/usr/bin/env python3
"""
Streamlit UI for rag_cli_v2.
Optimized for intent-heavy prompts (parlays, totals advice, comparisons).
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from datetime import datetime
from typing import Dict, List

import streamlit as st

import rag_cli_v2 as rag


LEAGUES = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]


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


def run_query(prompt: str, league: str) -> str:
    buf = io.StringIO()
    start = time.perf_counter()
    with redirect_stdout(buf):
        rag.answer_once(prompt, default_league=league)
    elapsed = time.perf_counter() - start
    text = buf.getvalue().strip() or "(No output)"
    stamp = datetime.now().strftime("%H:%M:%S")
    return text, elapsed, stamp


def template_prompts(target_date: str) -> Dict[str, str]:
    date_text = target_date.strip() or "10th February"
    return {
        "4-leg Goals Parlay (cap odds)": (
            f"For all EPL games on {date_text}, make a 4-leg parlay with exactly one leg per fixture "
            "using only full-game over/under goals totals lines. Keep combined decimal odds at or below 4.3x."
        ),
        "Corners Total Line (single fixture)": (
            f"Manchester United @ West Ham {date_text}. Give me corner totals for this game. Which line should I take?"
        ),
        "Goals Line (single fixture)": (
            f"Chelsea vs Leeds {date_text}. What over/under goals line should I take?"
        ),
        "Cards Comparison (single fixture)": (
            f"Which team will get more yellow cards in West Ham vs Manchester United on {date_text}? "
            "Give me a solid explanation of your reasoning."
        ),
        "Schedule Check": f"What EPL matches are on {date_text}?",
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
        target_date = st.text_input("Date phrase", value="10th February")
        templates = template_prompts(target_date)
        selected_template = st.selectbox("Prompt template", options=list(templates.keys()))
        if st.button("Load template"):
            st.session_state.draft_prompt = templates[selected_template]

        st.header("Session")
        if st.button("Clear chat + memory"):
            clear_all()
            st.rerun()

        if st.session_state.show_debug:
            st.subheader("RAG memory")
            st.json(
                {
                    "last_query": rag.memory.get("last_query"),
                    "last_selected_fixtures": rag.memory.get("last_selected_fixtures"),
                    "last_leg_count": rag.memory.get("last_leg_count"),
                    "history_turns": len(rag.memory.get("history") or []),
                }
            )


def main() -> None:
    st.set_page_config(page_title="Betting RAG v2", layout="wide")
    init_state()
    sidebar_ui()

    st.title("Betting RAG v2")
    st.caption("Streamlit wrapper for `rag_cli_v2` with prompt templates and session memory visibility.")

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
