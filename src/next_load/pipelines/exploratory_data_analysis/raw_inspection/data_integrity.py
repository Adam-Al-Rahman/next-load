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

    from next_load.core.nl_auth import get_s3_filesystem


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Raw Data Inspection: Data Integrity & EDA
    """)
    return


@app.function
def DATA_INTEGRITY_INSIGHTS():
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
            "Insight": "Gaps: [('2025-04-28', '2025-04-30'), ('2025-05-21', '2025-07-26'), ('2025-07-29', '2025-07-29'), ('2025-08-29', '2025-09-14'), ('2025-10-31', '2025-10-31'), ('2026-02-28', '2026-02-28')]",
            "Action": "Insert the missing timestamps, but leave the values as nulls. In processing stage after data split perform machine learning imputation on each dataset independently",
        },
        {
            "Category": "Data Integrity",
            "Operation": "Missing Months Seasonal Pattern",
            "Insight": "('2024-01-01', '2024-03-31'), a model needs to see a seasonal pattern at least twice to learn it effectively. Is data heavily influenced by the time of year? => No (15 min)",
            "Action": "No need to bother about the missing months ('2024-01-01', '2024-03-31')",
        },
        {
            "Category": "Data Integrity",
            "Operation": "Duplication",
            "Insight": "Zero duplicate records identified in the dataset.",
            "Action": "No Deduplication Required",
        },
        {
            "Category": "Data Integrity",
            "Operation": "Data Type Consistency",
            "Insight": "All columns match the required domain schema.",
            "Action": "Schema Validated",
        },
        {
            "Category": "Data Integrity",
            "Operation": "Consider nrldc_intraday_forecasted_demand_mw?",
            "Insight": "At the exact moment my model needs to generate a forecast for tomorrow (or the next hour), will the NRLDC intraday forecast for that same future time already be published and sitting in my database? => No",
            "Action": "Remove the nrldc_intraday_forecasted_demand_mw",
        },
    ]

    return INSIGHTS


@app.cell
def _():
    primary_dataset = pl.from_arrow(
        pq.ParquetDataset(
            "next-load-data/processed/01_primary/nrldc_forecast_primary.parquet",
            filesystem=get_s3_filesystem(),
        ).read_pandas()
    )
    return (primary_dataset,)


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
    start_timestamp = processed_dataset["timestamp"][0]
    end_timestamp = processed_dataset["timestamp"][-1]

    # Generate a DataFrame containing the expected 15-minute intervals
    expected_range = pl.DataFrame(
        {
            "expected_timestamp": pl.datetime_range(
                start=start_timestamp, end=end_timestamp, interval="15m", eager=True
            )
        }
    )

    # Find missing timestamps using an anti-join
    # This keeps only the rows from 'expected_range' that DO NOT exist in 'processed_dataset'
    missing_timestamps = expected_range.join(
        processed_dataset,
        left_on="expected_timestamp",
        right_on="timestamp",
        how="anti",
    )

    # Check if the dataset is complete
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
    missing_blocks = (
        missing_timestamps.sort("expected_timestamp")
        # Calculate the time difference between the current row and the previous row
        .with_columns(time_diff=pl.col("expected_timestamp").diff())
        # Mark the start of a new block if the difference is NOT exactly 15 minutes
        # (The first row will be null, so we fill it with True to start Block 1)
        .with_columns(
            is_new_block=(pl.col("time_diff") != pl.duration(minutes=15)).fill_null(
                True
            )
        )
        # Create a unique block ID by cumulatively summing the True/False values
        .with_columns(block_id=pl.col("is_new_block").cum_sum())
        # Group by this block ID and find the min (start) and max (end) timestamp
        .group_by("block_id")
        .agg(
            [
                pl.col("expected_timestamp").min().alias("missing_start"),
                pl.col("expected_timestamp").max().alias("missing_end"),
                pl.len().alias("missing_data_points"),
            ]
        )
        # Sort the final output by the start dates
        .sort("missing_start")
        # Drop the internal 'block_id' as we don't need to see it
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
