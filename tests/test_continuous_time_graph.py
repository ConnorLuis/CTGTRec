from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from preprocessing.build_continuous_time_adj import (
    compute_user_normalized_delta,
    process_dataset,
    symmetric_normalize,
)


def test_user_normalized_delta_and_zero_span_user() -> None:
    train = pd.DataFrame(
        {
            "userID": [0, 0, 1],
            "timestamp": [0.0, 10.0, 5.0],
        }
    )
    delta = compute_user_normalized_delta(train, epsilon=1e-12)
    np.testing.assert_allclose(delta, [1.0, 0.0, 0.0])


def test_graph_builder_uses_train_edges_and_validates_side_files(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()

    interactions = pd.DataFrame(
        {
            "userID": [0, 0, 0, 0, 1, 1, 1],
            "itemID": [0, 0, 1, 2, 1, 0, 2],
            "rating": [1.0] * 7,
            "timestamp": [0.0, 10.0, 20.0, 30.0, 5.0, 6.0, 7.0],
            "x_label": [0, 0, 1, 2, 0, 1, 2],
        }
    )
    interactions.to_csv(
        dataset_dir / "toy_temporal.inter",
        sep="\t",
        index=False,
    )
    (dataset_dir / "u_id_mapping.csv").write_text(
        "user_id\tuserID\nu0\t0\nu1\t1\n",
        encoding="utf-8",
    )
    (dataset_dir / "i_id_mapping.csv").write_text(
        "asin,itemID\nA,0\nB,1\nC,2\n",
        encoding="utf-8",
    )
    np.save(dataset_dir / "image_feat.npy", np.zeros((3, 4), dtype=np.float32))
    np.save(dataset_dir / "text_feat.npy", np.zeros((3, 2), dtype=np.float32))

    manifest = process_dataset(
        data_root=tmp_path,
        dataset="toy",
        input_suffix="_temporal.inter",
        output_dir_name="continuous_time_adj",
        taus=[0.5],
        epsilon=1e-12,
        separator="\t",
        save_edge_values=True,
        overwrite=False,
    )

    graph_dir = dataset_dir / "continuous_time_adj"
    raw = sp.load_npz(graph_dir / "ct_raw_adj_user_tau0p5.npz").tocsr()
    normalized = sp.load_npz(graph_dir / "ct_adj_user_tau0p5.npz").tocsr()

    assert raw.shape == (5, 5)
    assert raw.nnz == 4  # two unique train user-item pairs, two directions each
    np.testing.assert_allclose(raw.toarray(), raw.toarray().T)
    np.testing.assert_allclose(normalized.toarray(), normalized.toarray().T)
    np.testing.assert_allclose(
        normalized.toarray(),
        symmetric_normalize(raw).toarray(),
        rtol=1e-6,
        atol=1e-7,
    )

    # u0-i0 has two train interactions, so their temporal weights are summed.
    assert raw[0, 2] > 1.0
    # Validation/test pairs must not become graph edges.
    assert raw[0, 3] == 0.0
    assert raw[0, 4] == 0.0
    assert raw[1, 2] == 0.0
    assert raw[1, 4] == 0.0

    assert manifest["num_train_interactions"] == 3
    assert manifest["num_valid_interactions"] == 2
    assert manifest["num_test_interactions"] == 2
    assert manifest["side_file_checks"]["u_id_mapping.csv"]["delimiter"] == "tab"
    assert manifest["side_file_checks"]["i_id_mapping.csv"]["delimiter"] == "comma"

    disk_manifest = json.loads(
        (graph_dir / "ct_adj_manifest.json").read_text(encoding="utf-8")
    )
    assert disk_manifest["graph_source"] == "training interactions only (x_label == 0)"
    assert (graph_dir / "ct_edge_values.csv").is_file()
