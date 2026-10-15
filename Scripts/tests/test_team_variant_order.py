"""Team-name variants are tried in order and the first hit wins, so their order must not depend on hashing."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = """
import json, sys
sys.path[:0] = ["Scripts/rag_ingest", "Scripts"]
from lineup_context import _team_variants
from core.team_resolution import _kb_team_variants
print(json.dumps([_team_variants("AFC Bournemouth"), _kb_team_variants("Hamburger SV")]))
"""


def run(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_variant_order_is_the_same_in_every_process():
    lineup, kb = run(1)
    assert lineup[:3] == ["AFC Bournemouth", "afc bournemouth", "Afc Bournemouth"]
    assert kb[0] == "Hamburger SV"
    for seed in (2, 3, 4, 5):
        assert run(seed) == [lineup, kb]
