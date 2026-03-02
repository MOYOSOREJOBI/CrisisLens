from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from narrativepulse.models.stacker import MetaStacker
from narrativepulse.models.weak_supervision import fit_label_model
from narrativepulse.train.common import (
    chronological_split,
    feature_matrix,
    load_panel_from_config,
    save_split_manifest,
)
from narrativepulse.utils.calibration import TemperatureScaler
from narrativepulse.utils.conformal import conformal_coverage, conformal_interval
from narrativepulse.utils.io import ensure_dir, write_json
from narrativepulse.utils.metrics import classification_metrics, expected_calibration_error, regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _prepare_meta_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    lf = fit_label_model(df)
    base = df[
        [
            "attention_shock_index",
            "risk_shock_index",
            "gdelt_event_count_total",
            "wikipedia_pageviews",
        ]
    ].to_numpy(dtype=np.float32)
    x = np.concatenate([base, lf.probabilities[:, None]], axis=1)
    cols = [
        "attention_shock_index",
        "risk_shock_index",
        "gdelt_event_count_total",
        "wikipedia_pageviews",
        "weak_label_probability",
    ]
    return x, cols


def main() -> None:
    args = parse_args()
    cfg, panel = load_panel_from_config(args.config)
    splits = chronological_split(panel)
    artifacts_dir = ensure_dir(Path(cfg.artifacts_dir))
    save_split_manifest(artifacts_dir / "splits.json", splits)

    x_train, meta_cols = _prepare_meta_features(splits["train"])
    x_valid, _ = _prepare_meta_features(splits["valid"])
    x_test, _ = _prepare_meta_features(splits["test"])

    y_train_cls = splits["train"]["shock_label"].to_numpy(dtype=int)
    y_valid_cls = splits["valid"]["shock_label"].to_numpy(dtype=int)
    y_test_cls = splits["test"]["shock_label"].to_numpy(dtype=int)

    y_train_reg = splits["train"]["severity_t_plus_1"].to_numpy(dtype=float)
    y_valid_reg = splits["valid"]["severity_t_plus_1"].to_numpy(dtype=float)
    y_test_reg = splits["test"]["severity_t_plus_1"].to_numpy(dtype=float)

    stacker = MetaStacker().fit(x_train, y_train_cls, y_train_reg)
    valid_out = stacker.predict(x_valid)
    test_out = stacker.predict(x_test)

    logits_proxy = np.log(np.clip(valid_out.shock_probability, 1e-6, 1 - 1e-6) / np.clip(1 - valid_out.shock_probability, 1e-6, 1 - 1e-6))
    scaler = TemperatureScaler().fit(logits_proxy, y_valid_cls)
    calibrated_test_prob = scaler.transform(
        np.log(np.clip(test_out.shock_probability, 1e-6, 1 - 1e-6) / np.clip(1 - test_out.shock_probability, 1e-6, 1 - 1e-6))
    )

    lower, upper, q = conformal_interval(y_valid_reg, valid_out.severity)
    coverage = conformal_coverage(y_test_reg, test_out.severity - q, test_out.severity + q)

    cls_metrics = classification_metrics(y_test_cls, calibrated_test_prob)
    cls_metrics["ece"] = expected_calibration_error(y_test_cls, calibrated_test_prob)
    reg_metrics = regression_metrics(y_test_reg, test_out.severity)
    reg_metrics["conformal_coverage"] = coverage
    reg_metrics["conformal_half_width"] = q

    feature_attribution = {
        col: float(score)
        for col, score in zip(meta_cols, np.abs(stacker.classifier.coef_[0]))
    }

    metrics = {
        "classification": cls_metrics,
        "regression": reg_metrics,
        "ablation_placeholders": {
            "no_graph": "run by removing graph features upstream",
            "no_hawkes": "run by removing Hawkes features upstream",
            "no_macro": "run by removing macro features upstream",
            "no_text": "run by removing text branch upstream",
        },
    }
    write_json(artifacts_dir / "metrics.json", metrics)
    write_json(
        artifacts_dir / "config.json",
        {
            "config_path": args.config,
            "meta_features": meta_cols,
        },
    )
    write_json(
        artifacts_dir / "label_map.json",
        {
            "shock_label": {"0": "no_shock", "1": "shock"},
            "severity_targets": ["severity_t_plus_1", "severity_t_plus_3", "severity_t_plus_7"],
        },
    )
    write_json(
        artifacts_dir / "feature_manifest.json",
        {"meta_features": meta_cols, "feature_attribution_abs_coef": feature_attribution},
    )
    write_json(
        artifacts_dir / "stacker_model.json",
        {
            "classifier_coef": stacker.classifier.coef_[0].tolist(),
            "classifier_intercept": stacker.classifier.intercept_.tolist(),
            "regressor_coef": stacker.regressor.coef_.tolist(),
            "regressor_intercept": float(stacker.regressor.intercept_),
            "temperature": scaler.temperature,
            "conformal_half_width": q,
        },
    )


if __name__ == "__main__":
    main()
