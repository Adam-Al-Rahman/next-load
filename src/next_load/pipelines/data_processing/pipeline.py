"""
Kedro pipeline for data processing.
Coordinates preprocessing, interval alignment, and dataset splitting.
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    preprocess_nrldc_data,
    insert_missing_intervals,
    split_train_test_by_horizon
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Defines the data processing pipeline structure.
    """
    return pipeline(
        [
            node(
                func=preprocess_nrldc_data,
                inputs="primary_nrldc_forecast",
                outputs="processed_dataset_temp",
                name="preprocess_nrldc_data_node",
            ),
            node(
                func=insert_missing_intervals,
                inputs="processed_dataset_temp",
                outputs="preprocessed_dataset",
                name="insert_missing_intervals_node",
            ),
            node(
                func=split_train_test_by_horizon,
                inputs="preprocessed_dataset",
                outputs=["train_dataset", "test_dataset"],
                name="split_train_test_node",
            ),
        ]
    )
