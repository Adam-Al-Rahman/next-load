"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/configure/configuration_basics/#configuration"""

import os
from kedro.config import OmegaConfigLoader

# MLflow 3.10+ Integration Settings
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Observability (GenAI Tracing & Costs)
os.environ.setdefault("MLFLOW_TRACING_ENABLED", "true")
os.environ.setdefault("MLFLOW_TRACING_CAPTURE_COSTS", "true")
os.environ.setdefault("MLFLOW_AUTOLOGGING_ENABLED", "true")

# Workspace configuration (New in 3.10)
# Note: If MLFLOW_ENABLE_WORKSPACES is true, a workspace MUST be active.
# We set a default to prevent "Active workspace is required" error.
os.environ.setdefault("MLFLOW_ENABLE_WORKSPACES", "false") # Disable until needed
os.environ.setdefault("MLFLOW_DEFAULT_WORKSPACE", "default")
os.environ.setdefault("MLFLOW_WORKFLOW_SELECTOR", "true")

# Class that manages how configuration is loaded.
CONFIG_LOADER_CLASS = OmegaConfigLoader

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "custom_resolvers": {
        # Usage: ${env:VAR, default_value}
        "env": lambda var, default=None: os.getenv(var, default),
    },
}

# Instantiated project hooks.
# HOOKS = (ProjectHooks(),)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages Kedro's library components.
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# DATA_CATALOG_CLASS = DataCatalog
