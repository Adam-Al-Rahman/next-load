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


def create_unsafe_ssl_context():
    """
    Creates an SSL context that allows legacy server connections.
    """
    context = ssl.create_default_context()
    context.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
    return context


async def download_file(client: httpx.AsyncClient, url: str) -> bytes:
    """Helper to download a file asynchronously."""
    try:
        response = await client.get(url, timeout=60.0)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return b""


async def run_scraper_partitioned(
    scraper_config: ScraperConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Executes the Marimo async generator and partitions results by Month/Year.
    """
    metadata_partitions = {}
    file_partitions = {}

    ssl_context = create_unsafe_ssl_context()

    async with httpx.AsyncClient(verify=ssl_context) as client:
        async for raw_data_batch, year, month in extract_nrldc_data(scraper_config):
            # 1. Metadata Partition (YYYY_MM)
            pl_df = pl.DataFrame(raw_data_batch)

            try:
                m_num = datetime.strptime(month[:3], "%b").month
                partition_key = f"{year}_{m_num:02d}"
            except ValueError:
                partition_key = f"{year}_{month}"

            metadata_partitions[partition_key] = pl_df.to_pandas()

            # 2. File Partitions (data/YYYY/Month/filename.xlsx)
            for item in raw_data_batch:
                link = item.get("download_link")
                name = item.get("file_name")
                if link and name:
                    file_name_clean = name.replace(".xlsx", "").replace(".XLSX", "")
                    file_key = f"{year}/{month}/{file_name_clean}"

                    logger.info(f"Downloading and parsing {name}...")
                    file_content = await download_file(client, link)
                    if file_content:
                        try:
                            raw_df = pl.read_excel(
                                io.BytesIO(file_content),
                                engine="calamine",
                                has_header=False,
                            )
                            file_partitions[file_key] = raw_df.to_pandas()
                        except Exception as e:
                            logger.error(f"Failed to parse Excel {name}: {e}")

    return metadata_partitions, file_partitions


def scrape_and_partition_nrldc_node(
    params: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Kedro node to run the scraping and downloading process.
    """
    s3_params = params.get("s3", {})

    s3_config = S3Config(
        endpoint_url=s3_params.get("endpoint_url", "http://localhost:3900"),
        region_name=s3_params.get("region_name", "asia-south1"),
        bucket_name=s3_params.get("bucket_name", "next-load-data"),
        access_key=s3_params.get("access_key", ""),
        secret_key=s3_params.get("secret_key", ""),
    )

    scraper_config = ScraperConfig(
        live_extraction=params.get("live_extraction", False),
        start_year=params.get("start_year", 2024),
        s3=s3_config,
    )

    return asyncio.run(run_scraper_partitioned(scraper_config))


def validate_raw_partitions_node(
    partitioned_input: dict[str, Callable[[], pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """
    Separate Node: Data Contract 'Before Transformation'.
    Filters partitions that do not match the expected Excel structure.
    """
    validated_partitions = {}
    success_count = 0
    fail_count = 0

    logger.info(f"Validating {len(partitioned_input)} raw partitions.")

    for partition_key, loader in partitioned_input.items():
        # loader is a callable for S3-sourced PartitionedDataset
        # but for MemoryDataset it might be the data itself if passed incorrectly
        # Kedro passes callables for PartitionedDataset.
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


def transform_forecast_partitions_node(
    validated_input: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Separate Node: Transformation logic.
    Converts multiple Excel-sourced DataFrames into a single primary DataFrame.
    Note: Input is from MemoryDataset, so it's a dict of DataFrames (NOT callables).
    """
    all_dfs = []

    logger.info(f"Transforming {len(validated_input)} validated partitions.")

    for partition_key, df in validated_input.items():
        transformed_pl_df = transform_single_partition(df, partition_key)
        if transformed_pl_df is not None:
            all_dfs.append(transformed_pl_df)

    if not all_dfs:
        return pd.DataFrame()

    primary_df = pl.concat(all_dfs).sort(["date", "period"])
    return primary_df.to_pandas()


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
