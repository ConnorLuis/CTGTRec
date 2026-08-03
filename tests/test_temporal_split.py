from __future__ import annotations

import pandas as pd
import pytest

from preprocessing.build_temporal_split_inter import build_temporal_split


def test_split_uses_original_order_for_timestamp_ties() -> None:
    source = pd.DataFrame(
        {
            "userID": [0, 0, 0, 1, 1, 1],
            "itemID": [9, 8, 7, 0, 1, 2],
            "rating": [1.0] * 6,
            "timestamp": [10.0, 5.0, 10.0, 1.0, 2.0, 3.0],
            "x_label": [0] * 6,
        }
    )

    result = build_temporal_split(source)
    user_zero = result[result["userID"] == 0]
    assert user_zero["itemID"].tolist() == [8, 9, 7]
    assert user_zero["x_label"].tolist() == [0, 1, 2]


def test_split_rejects_short_user_histories() -> None:
    source = pd.DataFrame(
        {
            "userID": [0, 0],
            "itemID": [0, 1],
            "rating": [1.0, 1.0],
            "timestamp": [1.0, 2.0],
            "x_label": [0, 0],
        }
    )
    with pytest.raises(ValueError, match="at least three"):
        build_temporal_split(source)
