"""
Processing nodes for the data processing pipeline.
Provides functions for data cleaning, interval alignment, and dataset splitting.
"""
import polars as pl


def preprocess_nrldc_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cleans and transforms raw primary data into a structured time-series format.
    Extracts timestamps from period strings and ensures unique, sorted data.
    """
    processed_df = (
        df.with_columns(start_time=pl.col("period").str.split(" - ").list.first())
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

    return processed_df


def insert_missing_intervals(df: pl.DataFrame, interval: str = "15m") -> pl.DataFrame:
    """
    Ensures the time series contains all expected intervals within its range.
    Joins the input data with a generated full range of timestamps.
    """
    if df.is_empty():
        return df

    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()

    expected_range = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=start_ts, end=end_ts, interval=interval, eager=True
            )
        }
    ).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))

    df = df.with_columns(pl.col("timestamp").dt.cast_time_unit("us"))

    return expected_range.join(df, on="timestamp", how="left")


def split_train_test_by_horizon(
    df: pl.DataFrame, test_days: int = 14, horizon: int = 96
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splits the dataset into training and testing sets based on the specified forecast horizon.
    Calculates the test size by multiplying the horizon by the number of test days.
    """
    test_size = horizon * test_days

    train = df.slice(0, df.height - test_size)
    test = df.tail(test_size)

    return train, test
