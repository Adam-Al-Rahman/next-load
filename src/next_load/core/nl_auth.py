import os
from functools import lru_cache

import s3fs
from dotenv import load_dotenv
from infisical_sdk import InfisicalSDKClient


@lru_cache(maxsize=1)
def get_s3_filesystem() -> s3fs.S3FileSystem:
    """Authenticates with Infisical and returns a configured S3FileSystem for NextLoad."""
    load_dotenv(override=True)

    client = InfisicalSDKClient(host="https://app.infisical.com")

    client.auth.universal_auth.login(
        client_id=str(os.environ.get("INFISICAL_ELT_MACHINE_ID")),
        client_secret=str(os.environ.get("INFISICAL_ELT_MACHINE_SECRET")),
    )

    secret_config = {
        "project_id": "7286983c-eb71-4a94-8ffb-724d15eb5a2b",
        "environment_slug": "dev",
        "secret_path": "/extract_load_transform_pipeline/aws",
    }

    aws_access_key = client.secrets.get_secret_by_name(
        secret_name="AWS_ACCESS_KEY_ID", **secret_config
    ).secretValue

    aws_secret_key = client.secrets.get_secret_by_name(
        secret_name="AWS_SECRET_ACCESS_KEY", **secret_config
    ).secretValue

    aws_region = client.secrets.get_secret_by_name(
        secret_name="AWS_DEFAULT_REGION", **secret_config
    ).secretValue

    return s3fs.S3FileSystem(
        key=aws_access_key,
        secret=aws_secret_key,
        endpoint_url="http://localhost:3900",
        client_kwargs={
            "region_name": aws_region,
        },
    )
