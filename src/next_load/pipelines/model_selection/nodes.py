import io
import json
import logging
import pickle
from datetime import datetime
from typing import Any, Dict

import numpy as np
import polars as pl
from next_load.core.nl_auth import get_s3_client

logger = logging.getLogger(__name__)

# This module selects the best model based on performance metrics and promotes it to S3 storage


def select_and_promote_best_model(
    baseline_metrics: Dict[str, Any],
    baseline_model: Any,
    candidate_metrics: Dict[str, Any],
    lgbm_models: Dict[str, Any],
    neural_model: Any,
    test_df: pl.DataFrame,
    parameters: Dict[str, Any],
) -> str:
    # Function compares model metrics to identify the best performer and uploads it to S3
    s3_client = get_s3_client()
    bucket = "next-load-data"
    target_col = parameters.get("target_column", "y")

    y_true = (
        test_df.get_column(target_col)
        if target_col in test_df.columns
        else test_df.get_column("y")
    )
    y_mean = y_true.mean()

    all_model_stats = {}

    b_mape = baseline_metrics.get("baseline.mape") or baseline_metrics.get("mape")
    b_rmse = baseline_metrics.get("baseline.rmse") or baseline_metrics.get("rmse")

    if b_mape and b_rmse:
        mape_val = b_mape[-1]["value"] if isinstance(b_mape, list) else b_mape["value"]
        rmse_val = b_rmse[-1]["value"] if isinstance(b_rmse, list) else b_rmse["value"]
        nrmse_val = rmse_val / y_mean
        score = np.sqrt(mape_val * nrmse_val)

        all_model_stats["Baseline_Seasonal_Naive"] = {
            "metrics": {"mape": mape_val, "rmse": rmse_val, "nrmse": nrmse_val},
            "score": score,
            "model_obj": baseline_model,
            "type": "baseline",
        }

    candidate_model_names = set()
    for k in candidate_metrics.keys():
        clean_key = k.split(".")[-1]
        if "_mape" in clean_key:
            candidate_model_names.add(clean_key.replace("_mape", ""))
        elif "_rmse" in clean_key:
            candidate_model_names.add(clean_key.replace("_rmse", ""))

    for name in candidate_model_names:
        m_mape_key = (
            f"candidate.{name}_mape"
            if f"candidate.{name}_mape" in candidate_metrics
            else f"{name}_mape"
        )
        m_rmse_key = (
            f"candidate.{name}_rmse"
            if f"candidate.{name}_rmse" in candidate_metrics
            else f"{name}_rmse"
        )

        m_mape = candidate_metrics.get(m_mape_key)
        m_rmse = candidate_metrics.get(m_rmse_key)

        if m_mape and m_rmse:
            mape_val = (
                m_mape[-1]["value"] if isinstance(m_mape, list) else m_mape["value"]
            )
            rmse_val = (
                m_rmse[-1]["value"] if isinstance(m_rmse, list) else m_rmse["value"]
            )
            nrmse_val = rmse_val / y_mean
            score = np.sqrt(mape_val * nrmse_val)

            model_obj = None
            if name == "Neural_Ensemble":
                model_obj = neural_model
            else:
                model_obj = lgbm_models.get(name)

            if model_obj:
                all_model_stats[name] = {
                    "metrics": {"mape": mape_val, "rmse": rmse_val, "nrmse": nrmse_val},
                    "score": score,
                    "model_obj": model_obj,
                    "type": "candidate",
                }

    if not all_model_stats:
        raise ValueError("No valid models or metrics found for comparison.")

    best_model_name = min(all_model_stats, key=lambda k: all_model_stats[k]["score"])
    best_stats = all_model_stats[best_model_name]

    logger.info(
        f"Best model selected: {best_model_name} with score: {best_stats['score']:.4f}"
    )

    metadata = {
        "best_model_name": best_model_name,
        "model_type": best_stats["type"],
        "selection_metric": "Geometric Mean of MAPE and NRMSE",
        "score": float(best_stats["score"]),
        "metrics": {k: float(v) for k, v in best_stats["metrics"].items()},
        "promotion_timestamp": datetime.now().isoformat(),
        "all_compared_models": {
            name: {
                "score": float(stats["score"]),
                "metrics": {k: float(v) for k, v in stats["metrics"].items()},
            }
            for name, stats in all_model_stats.items()
        },
    }

    model_buffer = io.BytesIO()
    pickle.dump(best_stats["model_obj"], model_buffer)
    model_buffer.seek(0)
    s3_client.upload_fileobj(model_buffer, bucket, "nl_best_estimator/model.pkl")

    metadata_json = json.dumps(metadata, indent=4)
    s3_client.put_object(
        Bucket=bucket,
        Key="nl_best_estimator/model-metadata.json",
        Body=metadata_json,
        ContentType="application/json",
    )

    logger.info(
        f"Promoted best model ({best_model_name}) to s3://{bucket}/nl_best_estimator/"
    )
    return f"s3://{bucket}/nl_best_estimator/model.pkl"
