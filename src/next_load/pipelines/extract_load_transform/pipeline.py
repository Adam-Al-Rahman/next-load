# Kedro pipeline for NRLDC data extraction and transformation

from __future__ import annotations
import asyncio
import io
import logging
import ssl
from collections.abc import Callable
from datetime import datetime
from typing import Any
import httpx
import mlflow
import pandas as pd
import polars as pl
from kedro.pipeline import Pipeline, node, pipeline
from next_load.pipelines.extract_load_transform.elt_config import (
    S3Config,
    ScraperConfig,
)
from .extract_nrldc_forecast import extract_nrldc_data
from .transform_nrldc_forecast import (
    transform_single_partition,
    validate_raw_excel_dataframe,
    validate_transformed_dataframe,
)

logger = logging.getLogger(__name__)


# SSL context for legacy government portals
def create_unsafe_ssl_context():
    context = ssl.create_default_context()
    context.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
    return context


# Asynchronous file downloader
async def download_file(client: httpx.AsyncClient, url: str) -> bytes:
    try:
        response = await client.get(url, timeout=60.0)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return b""


# Scrapes and partitions NRLDC data
async def run_scraper_partitioned(
    scraper_config: ScraperConfig,
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    metadata_partitions = {}
    file_partitions = {}
    ssl_context = create_unsafe_ssl_context()

    async with httpx.AsyncClient(verify=ssl_context) as client:
        async for raw_data_batch, year, month in extract_nrldc_data(scraper_config):
            pl_df = pl.DataFrame(raw_data_batch)

            try:
                m_num = datetime.strptime(month[:3], "%b").month
                partition_key = f"{year}_{m_num:02d}"
            except ValueError:
                partition_key = f"{year}_{month}"

            metadata_partitions[partition_key] = pl_df

            for item in raw_data_batch:
                link = item.get("download_link")
                name = item.get("file_name")
                if link and name:
                    file_name_clean = name.replace(".xlsx", "").replace(".XLSX", "")
                    file_key = f"{year}/{month}/{file_name_clean}"
                    file_content = await download_file(client, link)
                    if file_content:
                        try:
                            raw_df = pl.read_excel(
                                io.BytesIO(file_content),
                                engine="calamine",
                                has_header=False,
                            )
                            file_partitions[file_key] = raw_df
                        except Exception as e:
                            logger.error(f"Failed to parse Excel {name}: {e}")

    return metadata_partitions, file_partitions


# Kedro node for scraping
def scrape_and_partition_nrldc_node(
    params: dict[str, Any],
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    from next_load.core.nl_auth import get_infisical_secret

    s3_config = S3Config(
        endpoint_url=get_infisical_secret("AWS_ENDPOINT_URL"),
        region_name=get_infisical_secret("AWS_DEFAULT_REGION"),
        bucket_name=get_infisical_secret("BUCKET_NAME"),
        access_key=get_infisical_secret("AWS_ACCESS_KEY_ID"),
        secret_key=get_infisical_secret("AWS_SECRET_ACCESS_KEY"),
    )

    scraper_config = ScraperConfig(
        live_extraction=params.get("live_extraction", False),
        start_year=params.get("start_year", 2024),
        s3=s3_config,
    )

    return asyncio.run(run_scraper_partitioned(scraper_config))


# Validates raw data partitions
def validate_raw_partitions_node(
    partitioned_input: dict[str, Callable[[], pl.DataFrame]],
) -> dict[str, pl.DataFrame]:
    validated_partitions = {}
    success_count = 0
    fail_count = 0

    for partition_key, loader in partitioned_input.items():
        try:
            df = loader()
            if validate_raw_excel_dataframe(df, partition_key):
                validated_partitions[partition_key] = df
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"Error loading partition {partition_key}: {e}")
            fail_count += 1

    mlflow.log_metric("raw_partitions_total", len(partitioned_input))
    mlflow.log_metric("raw_partitions_valid", success_count)
    mlflow.log_metric("raw_partitions_invalid", fail_count)

    return validated_partitions


# Transforms validated partitions
def transform_forecast_partitions_node(
    validated_input: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    all_dfs = []
    for partition_key, df in validated_input.items():
        transformed_pl_df = transform_single_partition(df, partition_key)
        if transformed_pl_df is not None:
            all_dfs.append(transformed_pl_df)

    if not all_dfs:
        return pl.DataFrame()

    return pl.concat(all_dfs).sort(["date", "period"])


# ELT pipeline definition
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=scrape_and_partition_nrldc_node,
                inputs="params:scraper_config",
                outputs=["nrldc_metadata_partitioned", "nrldc_raw_files_partitioned"],
                name="scrape_and_partition_nrldc_node",
            ),
            node(
                func=validate_raw_partitions_node,
                inputs="nrldc_raw_files_partitioned",
                outputs="validated_nrldc_raw_files",
                name="validate_raw_data_contract_node",
            ),
            node(
                func=transform_forecast_partitions_node,
                inputs="validated_nrldc_raw_files",
                outputs="intermediate_primary_forecast",
                name="transform_forecast_partitions_node",
            ),
            node(
                func=validate_transformed_dataframe,
                inputs="intermediate_primary_forecast",
                outputs="primary_nrldc_forecast",
                name="validate_transformed_data_contract_node",
            ),
        ]
    )
