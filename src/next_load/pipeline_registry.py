"""
Kedro pipeline registry for Next Load.
Automatically discovers and registers all defined pipelines in the project.
"""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """
    Finds and registers all pipelines in the src/next_load/pipelines directory.
    Includes a default pipeline that aggregates all available pipelines.
    """
    pipelines = find_pipelines(raise_errors=True)
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
