from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from prefect import flow, task


@task(
    retries=3,
    retry_delay_seconds=60,
    task_run_name="Kedro Step: {pipeline_name} ({node_names})",
)
def run_kedro_step(pipeline_name: str, node_names: list[str] | None = None):
    project_path = Path.cwd()

    # Bootstrapping inside the task ensures safe execution across distributed workers
    bootstrap_project(project_path)

    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_name=pipeline_name, node_names=node_names)

    return f"Completed {pipeline_name} / {node_names}"


@flow(name="NRLC Data Extraction & Load (Garage S3)", log_prints=True)
def nrldc_data_extract_load_orchestrator():
    pass


if __name__ == "__main__":
    nrldc_data_extract_load_orchestrator()
