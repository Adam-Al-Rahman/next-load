"""
Kedro project settings for Next Load.
Configures environment variables for AWS and MLflow, and defines custom resolvers for secret management.
"""

import os

from kedro.config import OmegaConfigLoader
from kedro_mlflow.framework.hooks import MlflowHook

from next_load.core.nl_auth import get_infisical_secret, get_mlflow_tracking_uri

os.environ["AWS_ACCESS_KEY_ID"] = get_infisical_secret("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = get_infisical_secret("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = get_infisical_secret(
    "AWS_DEFAULT_REGION", default="asia-south1"
)
os.environ["AWS_ENDPOINT_URL"] = get_infisical_secret(
    "AWS_ENDPOINT_URL", default="http://localhost:3900"
)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ["AWS_ENDPOINT_URL"]
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

tracking_uri = get_mlflow_tracking_uri()
os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
os.environ["MLFLOW_REGISTRY_URI"] = tracking_uri

CONFIG_LOADER_CLASS = OmegaConfigLoader
HOOKS = (MlflowHook(),)


def infisical_resolver(secret_name, default=None):
    """
    Custom OmegaConf resolver to retrieve secrets directly from Infisical.
    """
    return get_infisical_secret(secret_name, default=default)


CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "custom_resolvers": {
        "env": lambda var, default=None: os.getenv(var, default),
        "infisical": infisical_resolver,
        "turso_mlflow": lambda: tracking_uri,
    },
}

DISABLE_HOOKS_FOR_PLUGINS = ("kedro_mlflow",)
