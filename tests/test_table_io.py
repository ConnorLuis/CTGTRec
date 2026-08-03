from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from preprocessing.build_continuous_time_adj import validate_side_files
from preprocessing.raw.build_interactions import process
from preprocessing.raw.reindex_amazon_metadata import load_item_mapping
from preprocessing.table_io import read_delimited_table


@pytest.mark.parametrize("separator", [",", "\t"])
def test_mapping_reader_accepts_csv_and_legacy_tsv(
    tmp_path: Path,
    separator: str,
) -> None:
    path = tmp_path / "i_id_mapping.csv"
    path.write_text(
        separator.join(["asin", "itemID"])
        + "\n"
        + separator.join(["0000000001", "0"])
        + "\n"
        + separator.join(["B000000002", "1"])
        + "\n",
        encoding="utf-8",
    )

    table, detected = read_delimited_table(
        path,
        required_columns=["asin", "itemID"],
        dtype={"asin": str},
    )
    assert detected == separator
    assert table["asin"].tolist() == ["0000000001", "B000000002"]

    mapping = load_item_mapping(path)
    assert mapping["asin"].tolist() == ["0000000001", "B000000002"]
    assert mapping["itemID"].tolist() == [0, 1]
    assert mapping.attrs["detected_separator"] == separator


def test_mapping_reader_rejects_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    path.write_text("asin,wrong_id\nA,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="could not detect"):
        read_delimited_table(
            path,
            required_columns=["asin", "itemID"],
        )


def test_item_mapping_rejects_fractional_ids(tmp_path: Path) -> None:
    path = tmp_path / "i_id_mapping.csv"
    path.write_text("asin,itemID\nA,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integer"):
        load_item_mapping(path)


def test_side_file_validation_accepts_mixed_delimiters(tmp_path: Path) -> None:
    (tmp_path / "u_id_mapping.csv").write_text(
        "user_id\tuserID\nu0\t0\nu1\t1\n",
        encoding="utf-8",
    )
    (tmp_path / "i_id_mapping.csv").write_text(
        "asin,itemID\nA,0\nB,1\nC,2\n",
        encoding="utf-8",
    )
    np.save(tmp_path / "image_feat.npy", np.zeros((3, 4), dtype=np.float32))
    np.save(tmp_path / "text_feat.npy", np.zeros((3, 2), dtype=np.float32))

    checks = validate_side_files(tmp_path, n_users=2, n_items=3)
    assert checks["u_id_mapping.csv"] == {"rows": 2, "delimiter": "tab"}
    assert checks["i_id_mapping.csv"] == {"rows": 3, "delimiter": "comma"}
    assert checks["image_feat.npy"] == [3, 4]
    assert checks["text_feat.npy"] == [3, 2]


def test_raw_builder_writes_standard_csv_mappings(tmp_path: Path) -> None:
    source = tmp_path / "ratings.csv"
    source.write_text(
        "u0,0000000001,5,1\n"
        "u0,B000000002,4,2\n"
        "u1,0000000001,5,3\n"
        "u1,B000000002,4,4\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "data" / "toy"
    args = argparse.Namespace(
        source_type="amazon-ratings-csv",
        input=source,
        output_dir=output_dir,
        dataset="toy",
        delimiter=",",
        amazon_column_order="user-item-rating-timestamp",
        user_column="userID",
        item_column="videoID",
        timestamp_column="timestamp",
        rating_column=None,
        implicit_rating=1.0,
        min_user_interactions=1,
        min_item_interactions=1,
        mapping_order="appearance",
        overwrite=False,
    )

    process(args)

    user_header = (output_dir / "u_id_mapping.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    item_header = (output_dir / "i_id_mapping.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    interaction_header = (output_dir / "toy.inter").read_text(
        encoding="utf-8"
    ).splitlines()[0]

    assert user_header == "user_id,userID"
    assert item_header == "asin,itemID"
    assert interaction_header == "userID\titemID\trating\ttimestamp\tx_label"

    item_mapping = load_item_mapping(output_dir / "i_id_mapping.csv")
    assert item_mapping["asin"].iloc[0] == "0000000001"

    manifest = json.loads(
        (output_dir / "raw_preprocessing_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["interaction_separator"] == "tab"
    assert manifest["mapping_separator"] == "comma"
