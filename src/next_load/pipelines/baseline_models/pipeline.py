"""
Kedro pipeline for baseline model training and evaluation.
Coordinates feature engineering, data imputation, and Seasonal Naive model execution.
"""

from __future__ import annotations
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    build_baseline_features,
    impute_baseline_data,
    train_evaluate_baseline_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Defines the baseline models pipeline structure.
    """
    return pipeline(
        [
            node(
                func=build_baseline_features,
                inputs=["train_dataset", "params:baseline_models"],
                outputs="train_dataset_with_features",
                name="build_baseline_features_node",
            ),
            node(
                func=impute_baseline_data,
                inputs=["train_dataset_with_features", "params:baseline_models"],
                outputs="train_dataset_imputed",
                name="impute_baseline_data_node",
            ),
            node(
                func=train_evaluate_baseline_model,
                inputs=[
                    "train_dataset_imputed",
                    "test_dataset",
                    "params:baseline_models",
                ],
                outputs=["baseline_metrics", "baseline_model"],
                name="train_evaluate_baseline_model_node",
            ),
        ]
    )
