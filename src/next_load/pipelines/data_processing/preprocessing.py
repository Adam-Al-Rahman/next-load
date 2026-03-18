import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import marimo as mo

    import polars as pl
    import pyarrow.parquet as pq

    from datetime import datetime
    import altair as alt

    from next_load.core.nl_auth import get_s3_filesystem


@app.cell
def _():
    from next_load.pipelines.exploratory_data_analysis.raw_inspection.data_integrity import (
        DATA_INTEGRITY_INSIGHTS,
    )

    pl.DataFrame(DATA_INTEGRITY_INSIGHTS())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Data Preprocessing
    """)
    return


@app.function
def DATA_PREPROCESSING_INSIGHTS():
    INSIGHTS = [
        {
            "Category": "Data Preprocess",
            "Operation": "train-test split",
            "Insight": "Test set should not be a percentage (like 20% or 30%). It should be exactly the length of time you intend to predict in production (your Forecast Horizon, H)",
            "Action": "Time-Series Test Set Horizon",
        },
        {
            "Category": "Data Preprocess",
            "Operation": "train-validation split",
            "Insight": "To tune your model and evaluate its stability without leaking data, you should use a Walk-Forward Validation (also known as Time-Series Cross-Validation).",
            "Action": "Rolling (or Expanding) Training Window",
        },
    ]

    return INSIGHTS


@app.cell
def _():
    s3_fs = get_s3_filesystem()
    primary_dataset = pl.from_arrow(
        pq.ParquetDataset(
            "next-load-data/processed/01_primary/nrldc_forecast_primary.parquet",
            filesystem=s3_fs,
        ).read_pandas()
    )
    return primary_dataset, s3_fs


@app.cell
def _(primary_dataset):
    primary_dataset.head()
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
    return (processed_dataset,)


@app.cell
def _(processed_dataset):
    processed_dataset.head(100)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Insert Missing TimeStamp
    """)
    return


@app.cell
def _(processed_dataset):
    processed_dataset["timestamp"].is_sorted()
    return


@app.cell
def _(processed_dataset):
    START_TIMESTAMP = processed_dataset["timestamp"][0]
    END_TIMESTAMP = processed_dataset["timestamp"][-1]

    START_TIMESTAMP, END_TIMESTAMP
    return END_TIMESTAMP, START_TIMESTAMP


@app.cell
def _(END_TIMESTAMP, START_TIMESTAMP, processed_dataset):
    # Generate a DataFrame containing the expected 15-minute intervals
    expected_range = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=START_TIMESTAMP, end=END_TIMESTAMP, interval="15m", eager=True
            )
        }
    )

    inserted_ts = expected_range.join(
        processed_dataset, on="timestamp", how="left"
    ).fill_null(0.0)
    return expected_range, inserted_ts


@app.cell
def _(inserted_ts):
    inserted_ts.filter(pl.col("actual_demand_mw") == 0)
    return


@app.cell
def _(expected_range, inserted_ts):
    # Find missing timestamps using an anti-join
    # This keeps only the rows from 'expected_range' that DO NOT exist in 'inserted_ts'
    missing_timestamps = expected_range.join(
        inserted_ts,
        left_on="timestamp",
        right_on="timestamp",
        how="anti",
    )

    print(f"Is every 15-minute interval present? {missing_timestamps.height == 0}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Remove `nrldc_intraday_forecasted_demand_mw`
    """)
    return


@app.cell
def _(inserted_ts):
    preprocessed_dataset = inserted_ts.drop("nrldc_intraday_forecasted_demand_mw")

    preprocessed_dataset
    return (preprocessed_dataset,)


@app.cell
def _(preprocessed_dataset):
    preprocessed_dataset["timestamp"].is_sorted()
    return


@app.cell
def _(preprocessed_dataset, s3_fs, s3_path):
    s3_preprocessed_path = (
        "next-load-data/processed/01-primary/preprocessed_dataset.parquet"
    )

    print(f"Writing DataFrame to {s3_preprocessed_path}...")
    with s3_fs.open(s3_path, "wb") as f:
        preprocessed_dataset.write_parquet(f)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train-Test Split: Time-Series Test Set Horizon
    """)
    return


@app.cell
def _(preprocessed_dataset):
    H = 96  # One day
    test_days = 14
    test_size = H * test_days

    train = preprocessed_dataset[:-test_size]
    test = preprocessed_dataset.tail(test_size)
    return test, train


@app.cell
def _(train):
    train
    return


@app.cell
def _(test):
    test
    return


@app.cell
def _(s3_fs, train):
    s3_train_path = "next-load-data/processed/01-primary/train_dataset.parquet"

    print(f"Writing DataFrame to {s3_train_path}...")
    with s3_fs.open(s3_train_path, "wb") as f_train:
        train.write_parquet(f_train)
    return


@app.cell
def _(s3_fs, test):
    s3_test_path = "next-load-data/processed/01-primary/test_dataset.parquet"

    print(f"Writing DataFrame to {s3_test_path}...")
    with s3_fs.open(s3_test_path, "wb") as f_test:
        test.write_parquet(f_test)
    return


if __name__ == "__main__":
    app.run()
