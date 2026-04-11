"""
Orchestrator for energy forecast pipelines using Prefect.
Manages execution order, retries, and scheduling for data and modeling.
"""

import asyncio
import inspect
import logging
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar, cast

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from prefect import flow, get_run_logger, serve, task
from prefect.client.schemas.schedules import CronSchedule

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


@task(
    name="run-kedro-pipeline",
    retries=3,
    retry_delay_seconds=[60, 300, 600],
    description="Executes a specific Kedro pipeline step.",
    tags=["kedro", "production"],
)
def run_kedro_step(pipeline_name: str, node_names: list[str] | None = None):
    """
    Executes a Kedro pipeline or specific nodes within a Prefect task.
    """
    logger = get_run_logger()
    logger.info(f"Starting Kedro pipeline: {pipeline_name}")

    try:
        bootstrap_project(PROJECT_ROOT)
        with KedroSession.create(project_path=PROJECT_ROOT) as session:
            session.run(pipeline_name=pipeline_name, node_names=node_names)
        return f"Completed {pipeline_name}"
    except Exception as e:
        logger.error(f"Failed Kedro pipeline {pipeline_name}: {str(e)}")
        raise


@flow(name="ELT: Extract Load and Transform", log_prints=True)
def extract_load_transform_flow():
    """
    Flow for raw data ingestion and S3 persistence.
    """
    return run_kedro_step(pipeline_name="extract_load_transform")


@flow(name="DP: Data Processing", log_prints=True)
def data_processing_flow():
    """
    Flow for cleaning and train-test splitting.
    """
    return run_kedro_step(pipeline_name="data_processing")


@flow(name="BM: Baseline Models", log_prints=True)
def baseline_models_flow():
    """
    Flow for training benchmark models.
    """
    return run_kedro_step(pipeline_name="baseline_models")


@flow(name="CM: Candidate Models", log_prints=True)
def candidate_models_flow():
    """
    Flow for training production-grade advanced models.
    """
    return run_kedro_step(pipeline_name="candidate_models")


@flow(name="EDA: Data Distribution Analysis", log_prints=True)
def eda_analysis_flow():
    """
    Flow for drift and distribution analysis.
    """
    return run_kedro_step(pipeline_name="exploratory_data_analysis")


@flow(name="Daily: ETL and Data Processing", log_prints=True)
def daily_etl_dp_flow():
    """
    Daily orchestration of data ingestion and processing.
    """
    extract_load_transform_flow()
    return data_processing_flow()


@flow(name="MS: Model Selection", log_prints=True)
def model_selection_flow():
    """
    Flow for comparing models and promoting the best one.
    """
    return run_kedro_step(pipeline_name="model_selection")


@flow(name="Weekly: Holistic Energy Forecast Pipeline", log_prints=True)
def holistic_pipeline_flow():
    """
    Weekly orchestration covering the entire training lifecycle.
    """
    daily_etl_dp_flow()
    baseline = baseline_models_flow()
    candidate = candidate_models_flow()
    selection = model_selection_flow()
    eda = eda_analysis_flow()
    return {
        "status": "Holistic Run Complete",
        "baseline": baseline,
        "candidate": candidate,
        "selection": selection,
        "eda": eda,
    }


T = TypeVar("T")


def _resolve_maybe_awaitable(value: T | Awaitable[T]) -> T:
    """
    Utility to handle potentially asynchronous Prefect deployment operations.
    """
    if inspect.isawaitable(value):

        async def _await_value(awaitable: Awaitable[T]) -> T:
            return await awaitable

        return asyncio.run(_await_value(cast(Awaitable[T], value)))
    return cast(T, value)


def deploy_and_serve():
    """
    Registers and serves all deployments with defined schedules.
    """
    daily_deployment = _resolve_maybe_awaitable(
        daily_etl_dp_flow.to_deployment(
            name="daily-elt-dp-prod",
            schedules=[CronSchedule(cron="0 12 * * 1-6", timezone="UTC")],
            tags=["production", "daily", "elt"],
        )
    )

    weekly_deployment = _resolve_maybe_awaitable(
        holistic_pipeline_flow.to_deployment(
            name="weekly-holistic-training-prod",
            schedules=[CronSchedule(cron="0 12 * * 0", timezone="UTC")],
            tags=["production", "weekly", "training"],
        )
    )

    serve(
        daily_deployment,
        weekly_deployment,
        _resolve_maybe_awaitable(
            extract_load_transform_flow.to_deployment(name="manual-elt")
        ),
        _resolve_maybe_awaitable(data_processing_flow.to_deployment(name="manual-dp")),
        _resolve_maybe_awaitable(baseline_models_flow.to_deployment(name="manual-bm")),
        _resolve_maybe_awaitable(candidate_models_flow.to_deployment(name="manual-cm")),
        _resolve_maybe_awaitable(model_selection_flow.to_deployment(name="manual-ms")),
        _resolve_maybe_awaitable(eda_analysis_flow.to_deployment(name="manual-eda")),
    )


if __name__ == "__main__":
    try:
        deploy_and_serve()
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped.")
        sys.exit(0)
    except Exception:
        logger.exception("Critical failure in orchestrator.")
        sys.exit(1)
