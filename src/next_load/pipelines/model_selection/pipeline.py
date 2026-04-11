# Model selection and final promotion pipeline comparing model metrics to promote the best estimator to S3.

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import select_and_promote_best_model


def create_pipeline(**kwargs) -> Pipeline:
    # Define model selection pipeline structure.
    return pipeline(
        [
            node(
                func=select_and_promote_best_model,
                inputs=[
                    "baseline_metrics",
                    "baseline_model",
                    "candidate_metrics",
                    "lgbm_candidate_models",
                    "neural_candidate_model",
                    "test_with_exog",
                    "params:candidate_models",
                ],
                outputs="best_model_s3_path",
                name="select_and_promote_best_model_node",
            ),
        ]
    )
