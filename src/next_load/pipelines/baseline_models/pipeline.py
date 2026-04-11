"""
Baseline models pipeline definition.
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
    Creates the baseline models pipeline.
    """
    return pipeline(
        [
            node(
                func=train_evaluate_baseline_model,
                inputs=[
                    "train_dataset",
                    "test_dataset",
                    "params:baseline_models",
                ],
                outputs=["baseline_metrics", "baseline_model"],
                name="train_evaluate_baseline_model_node",
            ),
        ]
    )
