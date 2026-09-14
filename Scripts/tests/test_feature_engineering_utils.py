from __future__ import annotations

import pandas as pd

from Scripts.feature_engineering_utils import defensive_position_flags


def test_defensive_position_flags_handles_api_codes_and_dirty_values():
    positions = pd.Series(["D", "df", "CB", "LWB", "M", "F", "G", None, pd.NA, 0])

    flags = defensive_position_flags(positions, index=positions.index)

    assert flags.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert str(flags.dtype) == "int8"


def test_defensive_position_flags_handles_a_missing_position_column():
    index = pd.RangeIndex(3)

    flags = defensive_position_flags(None, index=index)

    assert flags.tolist() == [0, 0, 0]
