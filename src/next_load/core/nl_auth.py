"""
Authentication and secret management module providing Infisical integration and configurations for S3 and MLflow
"""

import os
from functools import lru_cache

import boto3
import requests
from dotenv import load_dotenv
from infisical_sdk import InfisicalSDKClient


@lru_cache(maxsize=1)
def get_infisical_client(
    infisical_host="https://app.infisical.com",
) -> InfisicalSDKClient:
    """
    Initialize and authenticate the Infisical SDK client using OIDC for GitHub Actions or Universal Auth for local environments
    """
    load_dotenv(override=True)

    client = InfisicalSDKClient(host=infisical_host)

    if os.environ.get("GITHUB_ACTIONS") == "true":
        identity_id = os.environ.get("INFISICAL_OIDC_IDENTITY_ID")
        request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
        request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")

        if not request_url or not request_token:
            raise ValueError(
                "OIDC environment variables missing. "
                "Ensure 'permissions: id-token: write' is set in your GitHub workflow."
            )

        response = requests.get(
            request_url, headers={"Authorization": f"Bearer {request_token}"}
        )
        response.raise_for_status()
        jwt_token = response.json()["value"]

        client.auth.oidc_auth.login(identity_id=str(identity_id), jwt=jwt_token)
    else:
        client.auth.universal_auth.login(
            client_id=str(os.environ.get("INFISICAL_ELT_MACHINE_ID")),
            client_secret=str(os.environ.get("INFISICAL_ELT_MACHINE_SECRET")),
        )

    return client


def get_infisical_secret(
    secret_name: str,
    default: str = None,
    project_id: str = "7286983c-eb71-4a94-8ffb-724d15eb5a2b",
    env_slug: str = "prod",
    secret_path: str = "/extract_load_transform_pipeline/aws",
) -> str:
    """
    Retrieve a named secret from Infisical with optional fallback to defaults or environment variables
    """
    client = get_infisical_client()
    try:
        return client.secrets.get_secret_by_name(
            secret_name=secret_name,
            project_id=project_id,
            environment_slug=env_slug,
            secret_path=secret_path,
        ).secretValue
    except Exception:
        if default is not None:
            return default
        return os.environ.get(secret_name)


@lru_cache(maxsize=1)
def get_s3_client():
    """
    Configure and return a Boto3 S3 client using credentials retrieved from Infisical
    """
    aws_access_key = get_infisical_secret("AWS_ACCESS_KEY_ID")
    aws_secret_key = get_infisical_secret("AWS_SECRET_ACCESS_KEY")
    aws_region = get_infisical_secret("AWS_DEFAULT_REGION", default="asia-south1")
    aws_endpoint = get_infisical_secret(
        "AWS_ENDPOINT_URL", default="http://localhost:3900"
    )

    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region,
        endpoint_url=aws_endpoint,
    )


def get_mlflow_tracking_uri() -> str:
    """
    Construct the MLflow tracking URI for Turso database integration or fallback to a local SQLite database
    """
    db_url = get_infisical_secret("TURSO_DATABASE_URL")
    auth_token = get_infisical_secret("TURSO_AUTH_TOKEN")

    if not db_url or not auth_token:
        import logging

        logging.getLogger(__name__).warning(
            "Turso secrets missing. Falling back to local MLflow SQLite."
        )
        return "sqlite:///mlflow.db"

    hostname = db_url.replace("libsql://", "").replace("https://", "")

    return f"sqlite+libsql://{hostname}/?authToken={auth_token}&secure=true"
