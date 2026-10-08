"""Shared referee name normalization, preserving the legacy grouping rules."""
from typing import Dict, List, Optional, Tuple

def _normalize_ref_name(ref_str: Optional[str]) -> Optional[str]:
    """'Michael Oliver, England' → 'michael oliver'. Returns None if empty."""
    if not ref_str or not ref_str.strip():
        return None
    name = ref_str.split(",")[0].strip()
    return name.lower() if name else None


def _dedup_referee_names(merged: Dict[str, dict]) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Merge referee name variants that share (first_initial, last_name).

    API-Football returns "M. Oliver" for European competitions and
    "Michael Oliver" for domestic leagues.  This merges them under the
    longest (most complete) name variant.

    Returns (deduped_merged, remap) where remap maps old names → canonical name.
    """
    from collections import defaultdict

    # Group by (first_initial, last_name)
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for name in merged:
        parts = name.split()
        if len(parts) < 2:
            groups[("", name)].append(name)
            continue
        initial = parts[0].rstrip(".")[:1]
        last = parts[-1]
        groups[(initial, last)].append(name)

    deduped: Dict[str, dict] = {}
    remap: Dict[str, str] = {}

    for (_initial, _last), names in groups.items():
        if len(names) == 1:
            canonical = names[0]
            deduped[canonical] = merged[canonical]
            remap[canonical] = canonical
            continue

        # Pick canonical: longest name (prefer full name over abbreviated)
        canonical = max(names, key=len)

        # Merge all entries into canonical
        combined = {
            "name": canonical,
            "country": None,
            "leagues": set(),
            "cards": 0, "fouls": 0, "yellows": 0, "reds": 0,
            "matches": 0,
        }
        for n in names:
            m = merged[n]
            combined["leagues"] |= m["leagues"]
            combined["cards"] += m["cards"]
            combined["fouls"] += m["fouls"]
            combined["yellows"] += m["yellows"]
            combined["reds"] += m["reds"]
            combined["matches"] += m["matches"]
            if m.get("country") and not combined["country"]:
                combined["country"] = m["country"]
            remap[n] = canonical

        deduped[canonical] = combined

    n_merged = len(merged) - len(deduped)
    if n_merged > 0:
        print(f"  [dedup] Merged {n_merged} duplicate referee entries "
              f"({len(merged)} → {len(deduped)} unique referees)")

    return deduped, remap

