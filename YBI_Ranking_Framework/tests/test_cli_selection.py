from __future__ import annotations

import pandas as pd
import pytest

from cvrank.cli import _load_selected_sweep_rows, build_parser


def test_select_lambda_defaults_to_sweep_artifacts_and_only_retrain_is_explicit() -> None:
    parser = build_parser()

    default_args = parser.parse_args(["select-lambda", "--config", "selection.yaml"])
    assert default_args.retrain is False

    retrain_args = parser.parse_args(["select-lambda", "--config", "selection.yaml", "--retrain"])
    assert retrain_args.retrain is True

    with pytest.raises(SystemExit):
        parser.parse_args(["select-lambda", "--config", "selection.yaml", "--train"])


def test_selected_sweep_rows_fall_back_to_numeric_batch_size(tmp_path) -> None:
    summary_path = tmp_path / "lambda_sweep_summary.csv"
    runs_path = tmp_path / "lambda_sweep_runs.csv"
    summary_path.write_text("placeholder\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "route": "dbscan",
                "rank_loss_mode": "ndcg_aware_lambdarank",
                "lambda_rank": 0.3,
                "fold_index": 0,
                "batch_size": 256,
                "batch_size_tag": "",
            }
        ]
    ).to_csv(runs_path, index=False)

    rows = _load_selected_sweep_rows(
        {
            "source_summary_csv": str(summary_path),
            "route": "dbscan",
            "rank_loss_mode": "ndcg_aware_lambdarank",
            "lambda_rank": 0.3,
        },
        {"batch_size": 256, "batch_size_tag": "bs_256"},
    )

    assert rows is not None
    assert sorted(rows) == [0]
