"""Shared closing-price policy; safe to import without ORM/provider clients."""

WINDOW_MINUTES = 20
MAX_QUOTE_AGE_MINUTES = 15
CLV_DEFINITION = "opening_decimal_divided_by_closing_decimal_minus_one.v1"
# Closing price source written by services/closing_prices.py (post-match fetch of the last pre-kickoff refresh).
FINAL_SNAPSHOT = "api_football_final_prematch_snapshot.v1"
