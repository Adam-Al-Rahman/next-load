import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    import polars as pl
    import pyarrow.parquet as pq

    from datetime import datetime
    import altair as alt
    import matplotlib.pyplot as plt
    import plotly.express as px
    import seaborn as sns

    from utilsforecast.plotting import plot_series
    from statsforecast.models import MSTL
    from statsforecast import StatsForecast

    from statsmodels.tsa.stattools import adfuller

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    import nannyml as nml

    from next_load.core.nl_auth import get_s3_filesystem

    return datetime, get_s3_filesystem, mo, pl, plot_series, plt, pq, px, sns


@app.cell
def _(pl):
    from next_load.pipelines.data_processing.preprocessing import (
        DATA_PREPROCESSING_INSIGHTS,
    )

    from next_load.pipelines.exploratory_data_analysis.raw_inspection.data_integrity import (
        DATA_INTEGRITY_INSIGHTS,
    )

    from next_load.pipelines.exploratory_data_analysis.preprocessed_inspection.univariate_analysis import (
        UNIVARIATE_ANALYSIS_INSIGHTS,
    )

    pl.DataFrame(
        UNIVARIATE_ANALYSIS_INSIGHTS()
        + DATA_PREPROCESSING_INSIGHTS()
        + DATA_INTEGRITY_INSIGHTS()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bivariate Analysis
    """)
    return


@app.cell
def _(get_s3_filesystem, pl, pq):
    S3_FS = get_s3_filesystem()
    train_dataset = pl.from_arrow(
        pq.ParquetDataset(
            "next-load-data/processed/01_primary/train_dataset.parquet",
            filesystem=S3_FS,
        ).read_pandas()
    )

    test_dataset = pl.from_arrow(
        pq.ParquetDataset(
            "next-load-data/processed/01_primary/test_dataset.parquet",
            filesystem=S3_FS,
        ).read_pandas()
    )
    return test_dataset, train_dataset


@app.cell
def _(train_dataset):
    train_dataset.head(100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time Gap
    """)
    return


@app.cell
def _(pl, train_dataset):
    missing_timestamps = train_dataset.filter(pl.col("actual_demand_mw").is_null())
    return (missing_timestamps,)


@app.cell
def _(missing_timestamps, pl):
    missing_blocks = (
        missing_timestamps.sort("timestamp")
        .with_columns(time_diff=pl.col("timestamp").diff())
        .with_columns(
            is_new_block=(pl.col("time_diff") != pl.duration(minutes=15)).fill_null(
                True
            )
        )
        .with_columns(block_id=pl.col("is_new_block").cum_sum())
        .group_by("block_id")
        .agg(
            [
                pl.col("timestamp").min().alias("missing_start"),
                pl.col("timestamp").max().alias("missing_end"),
                pl.len().alias("missing_data_points"),
            ]
        )
        .sort("missing_start")
        .drop("block_id")
    )

    missing_blocks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fake Imputation

    Imputation only for the EDA Univariate Analysis
    """)
    return


@app.cell
def _(pl, train_dataset):
    train_resampled = train_dataset.upsample(time_column="timestamp", every="15m")

    train_flagged = train_resampled.with_columns(
        pl.col("actual_demand_mw").is_null().alias("is_imputed")
    )

    train_scaffold = train_flagged.with_columns(
        pl.col("actual_demand_mw").interpolate().alias("naive_bridge")
    )

    # Extract the 'Macro Trend' using a 7-day rolling average over the bridge.
    # 7 days * 24 hours * 4 (15-min intervals) = 672 periods.
    train_trend = train_scaffold.with_columns(
        pl.col("naive_bridge")
        .rolling_mean(window_size=672, min_periods=1, center=True)
        .alias("macro_trend")
    )

    # Isolate the 'Pure' Seasonality (Actual Demand minus the Macro Trend)
    # This removes the summer/winter baseline and leaves only the daily schedule, centered near 0.
    train_detrended = train_trend.with_columns(
        (pl.col("actual_demand_mw") - pl.col("macro_trend")).alias(
            "detrended_signal"
        ),
        pl.col("timestamp").dt.weekday().alias("weekday"),
        pl.col("timestamp").dt.time().alias("time_of_day"),
    )

    # Calculate the seasonal median of ONLY the detrended swings
    train_swings = train_detrended.with_columns(
        pl.col("detrended_signal")
        .median()
        .over(["weekday", "time_of_day"])
        .alias("seasonal_swing")
    )

    # Fill the gaps by adding the slowly moving Trend to the highly localized Seasonal Swing.
    train_imputed_clean = train_swings.with_columns(
        pl.col("actual_demand_mw")
        .fill_null(pl.col("macro_trend") + pl.col("seasonal_swing"))
        .alias("actual_demand_mw_filled")
    ).drop(
        [
            "naive_bridge",
            "macro_trend",
            "detrended_signal",
            "weekday",
            "time_of_day",
            "seasonal_swing",
        ]
    )

    train_imputed_clean
    return (train_imputed_clean,)


@app.cell
def _(mo):
    mo.callout(
        """Even though we are just doing this for EDA, remember that calculating .median().over(["weekday", "time_of_day"]) across the whole training set means the imputed points in your massive May–July 2025 gap are technically 'looking ahead' at actual demand data from late 2025 and early 2026. This is perfectly fine for maintaining an unbroken timeline to visualize seasonal rhythms during EDA—especially since we safely flagged those fake points with is_imputed -- but it reinforces exactly why you must build a separate, strict, time-aware imputer for your actual cross-validation pipeline later.""",
        kind="danger",
    )
    return


@app.cell
def _(pl, test_dataset, train_imputed_clean):
    # utilsforecast requires 'unique_id', 'ds' (datetime), and 'y' (target) columns
    train_uf = train_imputed_clean.rename(
        {"timestamp": "ds", "actual_demand_mw_filled": "y"}
    ).with_columns(pl.lit("demand").alias("unique_id"))
    test_uf = test_dataset.rename(
        {"timestamp": "ds", "actual_demand_mw": "y"}
    ).with_columns(pl.lit("demand").alias("unique_id"))
    return test_uf, train_uf


@app.cell
def _(datetime, pl, train_uf):
    target_dates = [
        datetime(2024, 11, 21, 14, 45, 0),
        datetime(2025, 3, 24, 16, 0, 0),
    ]

    # 1. Convert the list to a Polars Series for safer datatype matching
    target_series = pl.Series(target_dates)

    train_uf_ol = train_uf.with_columns(
        # Interpolate the 'y' values for the target dates
        y=pl.when(pl.col("ds").is_in(target_series))
        .then(
            pl.lit(None, dtype=pl.Float64)
        )  # Explicitly type the None to prevent Schema errors
        .otherwise(pl.col("y"))
        .interpolate(),
        # Update or create the 'is_imputed' flag
        is_imputed=pl.when(pl.col("ds").is_in(target_series))
        .then(True)
        # Fallback to False if the column is brand new, otherwise keep existing values
        .otherwise(
            pl.col("is_imputed") if "is_imputed" in train_uf.columns else False
        ),
    )
    return (train_uf_ol,)


@app.cell
def _(plot_series, test_uf, train_uf_ol):
    plot_series(
        train_uf_ol.select(["ds", "y", "unique_id"]),
        test_uf,
        engine="plotly",
    ).show()
    return


@app.cell
def _(train_uf_ol):
    train_uf_ol.head(100)
    return


@app.cell
def _(px, train_uf_ol):
    fig = px.scatter(
        train_uf_ol,
        x="ds",
        y="y",
        color="is_imputed",
        title="Time Series Demand (y) vs Timestamp (ds)",
        labels={
            "ds": "Timestamp",
            "y": "Actual Demand (MW)",
            "is_imputed": "Is Imputed?",
        },
        color_discrete_map={False: "blue", True: "red"},
    )

    fig.update_traces(mode="lines+markers")

    # Display the interactive plot
    fig.show()
    return


@app.cell
def _(pl, train_uf_ol):
    # Create a feature that flags the high-variance months
    df_features = train_uf_ol.with_columns(
        pl.col("ds")
        .dt.month()
        .is_in([11, 12, 1, 2, 3])
        .cast(pl.Int8)
        .alias("is_high_variance_season"),
        # Calculate a rolling 7-day standard deviation to feed the model the recent volatility
        # (96 intervals * 7 days = 672)
        pl.col("y").rolling_std(window_size=672).alias("rolling_7d_volatility"),
    )

    df_features
    return (df_features,)


@app.cell
def _(df_features, pl):
    df_features_hw = df_features.with_columns(
        hour=pl.col("ds").dt.hour(), day_of_week=pl.col("ds").dt.strftime("%A")
    )
    return (df_features_hw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demand vs Hour
    """)
    return


@app.cell
def _(df_features_hw, mo, plt, sns):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="hour", y="y", data=df_features_hw)
    plt.title("Demand Distribution by Hour of the Day")
    mo.ui.matplotlib(plt.gca())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demand vs Day of Week
    """)
    return


@app.cell
def _(df_features_hw, mo, plt, sns):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="day_of_week", y="y", data=df_features_hw)
    plt.title("Demand Distribution by Day of the Week")
    mo.ui.matplotlib(plt.gca())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Autocorrelation (Lag=1)
    """)
    return


@app.cell
def _(df_features_hw, mo, plt):
    import pandas as pd

    plt.figure(figsize=(6, 6))
    pd.plotting.lag_plot(df_features_hw["y"].to_pandas(), lag=1)
    plt.title("Lag Plot (15-minute intervals)")
    mo.ui.matplotlib(plt.gca())
    return


if __name__ == "__main__":
    app.run()
