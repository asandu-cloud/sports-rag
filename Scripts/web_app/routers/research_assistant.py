"""The Research Assistant tab (tracker D94): signed-in members ask about past numbers, behind ``SPIX_STATS_QUESTIONS``.

The model only chooses one of the seven question types (``Scripts/ops/stats_questions``); the code counts the answer on
the read-only stats file, never platform.db. A free account gets 1 answered question a day and a paid one 50 (owner,
10 Nov); the whole site stops at 2,000. Every question goes in the owner's log with what it cost, which members never
see.
"""
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from Scripts.ops.stats_questions import log, model, questions, store
from web_app.auth import TIER_HIERARCHY, get_current_user

router = APIRouter(prefix="/api/research-assistant", tags=["research-assistant"])
STATS_PATH = store.DEFAULT_PATH
LOG_PATH = log.DEFAULT_PATH
SITE_DAILY_LIMIT = 2000
DAILY_LIMITS = {"free": 1, "paid": 50}
PAID_STATUSES = {"active", "trial"}
UNAVAILABLE = "The Research Assistant is unavailable right now. Please try again later."
ask_model = model.ask


class AskBody(BaseModel):
    question: str = Field(max_length=2000)
    previous: Optional[Dict[str, Any]] = None


class ChooseBody(BaseModel):
    type: str = Field(max_length=40)
    params: Dict[str, Any] = Field(default_factory=dict)


def enabled() -> bool:
    return os.environ.get("SPIX_STATS_QUESTIONS", "").strip().lower() == "on"


def plan(user: Dict[str, Any]) -> str:
    """Paid: a paid tier with an active or trial subscription. A cancelled or unpaid one counts as free."""
    paid = TIER_HIERARCHY.get(user.get("tier") or "free", 0) > 0 and (user.get("status") or "active") in PAID_STATUSES
    return "paid" if paid else "free"


def _refuse(status: int, kind: str, message: str):
    raise HTTPException(status_code=status, detail={"kind": kind, "message": message})


def _member(user):
    if not enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    if user is None:
        _refuse(401, "sign_in", "Sign in to ask the Research Assistant.")
    return user


def _stats():
    if not STATS_PATH.is_file():
        _refuse(503, "unavailable", UNAVAILABLE)
    return store.connect(STATS_PATH)


def _data_up_to():
    if not STATS_PATH.is_file():
        return None
    conn = store.connect(STATS_PATH)
    try:
        return questions.data_up_to(conn)
    finally:
        conn.close()


@router.get("/status")
def status(user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Whether the tab shows, and for a member how many questions are left today."""
    if not enabled():
        return {"enabled": False}
    out = {"enabled": True, "signed_in": user is not None, "data_up_to": _data_up_to()}
    if user is not None:
        kind, conn = plan(user), log.connect(LOG_PATH)
        try:
            asked = log.asked_today(conn, user_id=user["id"])
        finally:
            conn.close()
        limit = DAILY_LIMITS[kind]
        out.update(plan=kind, daily_limit=limit, asked_today=asked, remaining=max(0, limit - asked))
    return out


@router.post("/ask")
def ask(body: AskBody, user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    member = _member(user)
    kind, conn = plan(member), log.connect(LOG_PATH)
    try:
        limit = DAILY_LIMITS[kind]
        if log.asked_today(conn, user_id=member["id"]) >= limit:
            _refuse(429, "member_limit", f"You have asked {questions.plural(limit, 'question')} today, the most for "
                                         "your account. You can ask again tomorrow.")
        if log.asked_today(conn) >= SITE_DAILY_LIMIT:
            _refuse(429, "site_limit", "The Research Assistant has answered all it can today. Please try again "
                                       "tomorrow.")
        stats = _stats()
        try:
            out = ask_model(stats, body.question, previous=body.previous)
        except model.ModelError as exc:
            if exc.status == 400:  # empty or too long: never sent, so nothing to log
                _refuse(400, exc.kind, exc.message)
            log.record(conn, mode="ask", question=body.question[:model.MAX_SQL_CHARS], error=exc.kind,
                       user_id=member["id"], plan=kind, source="website")
            _refuse(503, "unavailable", UNAVAILABLE)
        finally:
            stats.close()
        log.record(conn, mode="ask", question=out["question"], calls=out["calls"], answers=out["answers"],
                   model=out["model"], ms=out["ms"], usage=out["usage"], cost_usd=out["cost_usd"],
                   flagged_bet=out["flagged_bet"], user_id=member["id"], plan=kind, source="website")
        remaining = max(0, limit - log.asked_today(conn, user_id=member["id"]))
    finally:
        conn.close()
    return {"question": out["question"], "calls": out["calls"], "answers": out["answers"], "remaining": remaining}


@router.post("/choose")
def choose(body: ChooseBody, user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """A pick from a "Which one?" card (a name given as "#id"). No model, and it does not use up a question."""
    member = _member(user)
    if not any(isinstance(v, str) and v.startswith("#") for v in body.params.values()):
        _refuse(400, "not_a_pick", "Pick one of the names on the card.")
    stats = _stats()
    try:
        card = questions.answer(stats, body.type, body.params)
    finally:
        stats.close()
    calls = [{"type": body.type, "params": body.params}]
    conn = log.connect(LOG_PATH)
    try:
        log.record(conn, mode="choose", calls=calls, answers=[card], user_id=member["id"], plan=plan(member),
                   source="website")
    finally:
        conn.close()
    return {"calls": calls, "answers": [card]}
