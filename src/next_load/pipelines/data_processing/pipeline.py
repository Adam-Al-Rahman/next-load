# data processing pipeline coordinating preprocessing and dataset splitting

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    preprocess_nrldc_data,
    insert_missing_intervals,
    split_train_test_by_horizon,
)
from next_load.pipelines.baseline_models.nodes import (
    build_baseline_features,
    impute_baseline_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    # constructs data processing pipeline with preprocessing and splitting
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
                outputs="preprocessed_dataset_raw",
                name="insert_missing_intervals_node",
            ),
            node(
                func=build_baseline_features,
                inputs=["preprocessed_dataset_raw", "params:baseline_models"],
                outputs="preprocessed_dataset_with_features",
                name="build_preprocessed_features_node",
            ),
            node(
                func=impute_baseline_data,
                inputs=["preprocessed_dataset_with_features", "params:baseline_models"],
                outputs="preprocessed_dataset",
                name="impute_preprocessed_data_node",
            ),
            node(
                func=split_train_test_by_horizon,
                inputs="preprocessed_dataset",
                outputs=["train_dataset", "test_dataset"],
                name="split_train_test_node",
            ),
        ]
    )
