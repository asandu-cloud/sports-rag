"""Publication lineage derived from immutable deliveries, never from outcomes."""
from collections import defaultdict
from sqlalchemy import select

from .models import PublishedRecommendation, MatchRead, MatchReadDelivery, MatchReadSelection
from .publication_identity import tracking_identity


PUBLICATION_SCOPES = {"initial", "amendments", "all"}


def publication_metadata(session, prediction_ids):
    if not prediction_ids:
        return {}
    p = PublishedRecommendation
    # Project only audit fields, not every full canonical result/context and
    # system manifest. A year's release history must not load all card blobs.
    publications = session.execute(select(
        p.id, p.prediction_id, p.recommendation_key, p.released_at,
        p.input_snapshot_id, p.pipeline_version, p.model_version,
        p.decision_json["fixture"].label("fixture"),
        p.decision_json["market"].label("market"),
        p.decision_json["decision"]["quote"].label("quote"),
        p.decision_json["provenance"].label("provenance"),
    ).where(
        PublishedRecommendation.prediction_id.in_(prediction_ids))).all()
    fixture_ids = {(p.fixture or {}).get("event_id") for p in publications}
    # Include no-bet cards and unresolved originals. Filtering outcomes before
    # choosing the first card would let a settled amendment replace its origin.
    deliveries = session.execute(select(MatchRead.id, MatchRead.league,
        MatchRead.fixture_api_id, MatchRead.version, MatchRead.stage).select_from(MatchReadDelivery).join(
        MatchRead, MatchRead.id == MatchReadDelivery.match_read_id).where(
        MatchRead.fixture_api_id.in_(fixture_ids)).order_by(
        MatchReadDelivery.delivered_at, MatchReadDelivery.id)).all()
    first_read, delivered = {}, {}
    for read in deliveries:
        key = (read.league, read.fixture_api_id)
        first_read.setdefault(key, read.id)
        delivered[read.id] = read
    links = defaultdict(set)
    if publications:
        for rec_id, read_id in session.execute(select(
            MatchReadSelection.published_recommendation_id, MatchReadSelection.match_read_id
        ).where(MatchReadSelection.published_recommendation_id.in_([p.id for p in publications]))):
            if read_id in delivered:
                links[rec_id].add(read_id)
    result = {}
    for publication in publications:
        payload = {"fixture": publication.fixture, "market": publication.market,
                   "decision": {"quote": publication.quote}, "provenance": publication.provenance}
        fixture = payload.get("fixture") or {}
        identity = tracking_identity(payload)
        read_ids = links[publication.id]
        original_id = first_read.get((fixture.get("league"), fixture.get("event_id")))
        role = ("initial" if original_id in read_ids else "amendment") if read_ids else "unclassified"
        result[publication.prediction_id] = {
            "tracking_cohort": "published", "publication_role": role,
            "recommendation_id": publication.id, "recommendation_key": publication.recommendation_key,
            "first_match_read_id": original_id, "match_read_ids": sorted(read_ids),
            "match_read_versions": [{"id": rid, "version": delivered[rid].version,
                                     "stage": delivered[rid].stage} for rid in sorted(read_ids)],
            "published_at": publication.released_at.isoformat(),
            "publication_date": publication.released_at.date().isoformat(),
            "fixture_date": identity["fixture_date"],
            "market_key": identity["selection"]["market_key"],
            "market_period": identity["selection"]["period"],
            "selection_key": identity["selection_key"],
            "system_version": identity["system_version"],
            "pipeline_version": publication.pipeline_version,
            "model_version": publication.model_version,
            "input_snapshot_id": publication.input_snapshot_id,
            "quote_time": identity["quote_time"], "quote_captured_at": identity["quote_captured_at"],
            "missing_identity_fields": identity["missing_identity_fields"],
            "card_definition_status": identity["card_definition_status"],
        }
    return result


def select_publication_scope(rows, scope):
    if scope not in PUBLICATION_SCOPES:
        raise ValueError("publication_scope must be initial, amendments, or all")
    if scope == "all":
        return rows
    role = "initial" if scope == "initial" else "amendment"
    return [row for row in rows if row.get("publication_role") == role]
