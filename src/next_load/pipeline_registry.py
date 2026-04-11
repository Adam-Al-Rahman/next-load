"""
Kedro pipeline registry for automatic discovery and registration of pipelines
"""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """
    Discover and register all pipelines in the project including a default pipeline that combines them all
    """
    pipelines = find_pipelines(raise_errors=True)
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
