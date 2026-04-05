"""
Marimo notebook for evaluating data integrity of raw primary datasets.
Performs checks for missing values, time gaps, duplicates, and schema consistency.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import os
    import marimo as mo
    import s3fs
    import polars as pl
    import pyarrow.parquet as pq
    from datetime import datetime
    import altair as alt
    from next_load.core.nl_auth import get_infisical_secret


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Raw Data Inspection: Data Integrity & EDA
    """)
    return


@app.function
def DATA_INTEGRITY_INSIGHTS():
    """
    Returns high-level findings regarding dataset health, including identified time gaps and data consistency.
    """
    INSIGHTS = [
        {
            "Category": "Data Integrity",
            "Operation": "Missing Data",
            "Insight": "Zero missing values found across all rows and columns.",
            "Action": "Ready for Analysis",
        },
        {
            "Category": "Data Integrity",
            "Operation": "Time Gap",
            "Insight": "Significant gaps identified across 2025 and early 2026.",
            "Action": "Insert missing timestamps as nulls and perform per-split imputation during processing.",
        },
        {
            "Category": "Data Integrity",
            "Operation": "Missing Months Seasonal Pattern",
            "Insight": "Initial months of 2024 are missing, but 15-minute resolution provides sufficient detail.",
            "Action": "No remediation required for missing early 2024 data.",
        },
        {
            "Category": "Data Integrity",
            "Operation": "Duplication",
            "Insight": "No duplicate records found.",
            "Action": "No deduplication required.",
        },
        {
            "Category": "Data Integrity",
            "Operation": "Data Type Consistency",
            "Insight": "Columns match the domain schema.",
            "Action": "Schema validated.",
        },
        {
            "Category": "Data Integrity",
            "Operation": "Consider nrldc_intraday_forecasted_demand_mw?",
            "Insight": "NRLDC intraday forecasts won't be available at prediction time in production.",
            "Action": "Remove this column from the modeling dataset.",
        },
    ]

    return INSIGHTS


@app.cell
def _(get_infisical_secret, s3fs):
    """
    Load the primary dataset from S3 using cloud credentials.
    """
    s3_fs = s3fs.S3FileSystem(
        key=get_infisical_secret("AWS_ACCESS_KEY_ID"),
        secret=get_infisical_secret("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=get_infisical_secret("AWS_ENDPOINT_URL") or "http://localhost:3900",
        client_kwargs={"region_name": get_infisical_secret("AWS_DEFAULT_REGION")},
        config_kwargs={"s3": {"addressing_style": "path"}}
    )
    primary_dataset = pl.from_arrow(
        pq.ParquetDataset(
            "next-load-data/processed/01_primary/nrldc_forecast_primary.parquet",
            filesystem=s3_fs,
        ).read_pandas()
    )
    return primary_dataset, s3_fs


@app.cell(hide_code=True)
def _(primary_dataset):
    mo.stop(primary_dataset.is_empty())

    mo.md(f"""
    - **Total Records:** {primary_dataset.height}
    - **Estimated Memory Size:** {primary_dataset.estimated_size(unit="mb"):.2f} MB
    """)
    return


@app.cell
def _(primary_dataset):
    primary_dataset.head(25)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Is there missing data?
    """)
    return


@app.cell
def _(primary_dataset):
    primary_dataset.null_count()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Is there time gap?
    """)
    return


@app.cell
def _(primary_dataset):
    """
    Transform raw primary data into a unified time-series for gap analysis.
    """
    processed_dataset = (
        primary_dataset.with_columns(
            start_time=pl.col("period").str.split(" - ").list.first()
        )
        .with_columns(
            timestamp_str=(
                pl.col("date").cast(pl.Datetime).dt.strftime("%Y-%m-%d")
                + " "
                + pl.col("start_time")
            )
        )
        .with_columns(
            timestamp=pl.col("timestamp_str").str.to_datetime("%Y-%m-%d %H:%M")
        )
        .select(
            ["timestamp", "actual_demand_mw", "nrldc_intraday_forecasted_demand_mw"]
        )
        .unique(subset=["timestamp"], keep="first")
        .sort("timestamp")
    )

    processed_dataset.head(25)
    return (processed_dataset,)


@app.cell
def _(processed_dataset):
    processed_dataset["timestamp"].is_sorted()
    return


@app.cell
def _(processed_dataset):
    processed_dataset["timestamp"][0], processed_dataset["timestamp"][-1]
    return


@app.cell
def _(processed_dataset):
    """
    Identify missing 15-minute intervals by comparing dataset against a full range.
    """
    start_timestamp = processed_dataset["timestamp"][0]
    end_timestamp = processed_dataset["timestamp"][-1]

    expected_range = pl.DataFrame(
        {
            "expected_timestamp": pl.datetime_range(
                start=start_timestamp, end=end_timestamp, interval="15m", eager=True
            )
        }
    )

    missing_timestamps = expected_range.join(
        processed_dataset,
        left_on="expected_timestamp",
        right_on="timestamp",
        how="anti",
    )

    is_complete = missing_timestamps.height == 0

    print(f"Is every 15-minute interval present? {is_complete}")
    return (missing_timestamps,)


@app.cell
def _(missing_timestamps):
    print(f"\nMissing {missing_timestamps.height} timestamp(s)")
    missing_timestamps
    return


@app.cell
def _(missing_timestamps):
    """
    Group missing timestamps into contiguous blocks to visualize the duration of data loss.
    """
    missing_blocks = (
        missing_timestamps.sort("expected_timestamp")
        .with_columns(time_diff=pl.col("expected_timestamp").diff())
        .with_columns(
            is_new_block=(pl.col("time_diff") != pl.duration(minutes=15)).fill_null(
                True
            )
        )
        .with_columns(block_id=pl.col("is_new_block").cum_sum())
        .group_by("block_id")
        .agg(
            [
                pl.col("expected_timestamp").min().alias("missing_start"),
                pl.col("expected_timestamp").max().alias("missing_end"),
                pl.len().alias("missing_data_points"),
            ]
        )
        .sort("missing_start")
        .drop("block_id")
    )

    missing_blocks
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Are there duplicates or near-duplicates?
    """)
    return


@app.cell
def _(processed_dataset):
    not processed_dataset.filter(processed_dataset.is_duplicated()).is_empty()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Are data types consistent with domain definitions?
    """)
    return


@app.cell
def _(processed_dataset):
    processed_dataset.head()
    return


@app.cell
def _(processed_dataset):
    processed_dataset.schema
    return


if __name__ == "__main__":
    app.run()
