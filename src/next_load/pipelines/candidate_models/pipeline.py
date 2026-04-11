"""
Pipeline for candidate model training and evaluation
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_candidate_features,
    evaluate_candidate_models,
    impute_candidate_train_data,
    train_lgbm_candidate_models,
    train_neural_candidate_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create candidate model training and evaluation pipeline
    """
    return pipeline(
        [
            node(
                func=create_candidate_features,
                inputs=["train_dataset", "test_dataset", "params:candidate_models"],
                outputs=["train_with_exog", "test_with_exog"],
                name="create_candidate_features_node",
            ),
            node(
                func=train_lgbm_candidate_models,
                inputs=["train_with_exog", "params:candidate_models"],
                outputs="lgbm_candidate_models",
                name="train_lgbm_candidate_models_node",
            ),
            node(
                func=train_neural_candidate_models,
                inputs=["train_with_exog", "params:candidate_models"],
                outputs="neural_candidate_model",
                name="train_neural_candidate_models_node",
            ),
            node(
                func=evaluate_candidate_models,
                inputs=[
                    "lgbm_candidate_models",
                    "neural_candidate_model",
                    "test_with_exog",
                    "params:candidate_models",
                ],
                outputs="candidate_metrics",
                name="evaluate_candidate_models_node",
            ),
        ]
    )
