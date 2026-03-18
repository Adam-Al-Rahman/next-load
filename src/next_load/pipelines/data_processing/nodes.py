import pandas as pd
import polars as pl


def preprocess_nrldc_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw primary data into a clean time-series format.
    Accepts Pandas, processes with Polars, returns Pandas.
    """
    # Convert to Polars
    pl_df = pl.from_pandas(df)

    processed_df = (
        pl_df.with_columns(start_time=pl.col("period").str.split(" - ").list.first())
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
        .select(["timestamp", "actual_demand_mw"])
        .unique(subset=["timestamp"], keep="first")
        .sort("timestamp")
    )

    # Convert back to Pandas
    return processed_df.to_pandas()


def insert_missing_intervals(df: pd.DataFrame, interval: str = "15m") -> pd.DataFrame:
    """
    Ensures that every time interval is present in the dataset.
    Accepts Pandas, processes with Polars, returns Pandas.
    """
    if df.empty:
        return df

    # Convert to Polars
    pl_df = pl.from_pandas(df)

    start_ts = pl_df["timestamp"].min()
    end_ts = pl_df["timestamp"].max()

    expected_range = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=start_ts, end=end_ts, interval=interval, eager=True
            )  # ty:ignore[no-matching-overload]
        }
    )

    full_df = expected_range.join(pl_df, on="timestamp", how="left")

    # Convert back to Pandas
    return full_df.to_pandas()


def split_train_test_by_horizon(
    df: pd.DataFrame, test_days: int = 14, horizon: int = 96
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train and test sets based on a fixed forecast horizon.
    Accepts Pandas, processes with Polars, returns Tuple of Pandas.
    """
    # Convert to Polars
    pl_df = pl.from_pandas(df)

    test_size = horizon * test_days

    train = pl_df[:-test_size]
    test = pl_df.tail(test_size)

    # Convert back to Pandas
    return train.to_pandas(), test.to_pandas()
