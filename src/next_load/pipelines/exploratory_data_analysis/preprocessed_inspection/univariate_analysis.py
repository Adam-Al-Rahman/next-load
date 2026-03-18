import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import marimo as mo

    import polars as pl
    import pyarrow.parquet as pq

    from datetime import datetime
    import altair as alt
    import matplotlib.pyplot as plt

    from utilsforecast.plotting import plot_series
    from statsforecast.models import MSTL
    from statsforecast import StatsForecast

    from statsmodels.tsa.stattools import adfuller

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    import nannyml as nml

    from next_load.core.nl_auth import get_s3_filesystem


@app.cell
def _():
    from next_load.pipelines.data_processing.preprocessing import (
        DATA_PREPROCESSING_INSIGHTS,
    )

    from next_load.pipelines.exploratory_data_analysis.raw_inspection.data_integrity import (
        DATA_INTEGRITY_INSIGHTS,
    )

    pl.DataFrame(DATA_PREPROCESSING_INSIGHTS() + DATA_INTEGRITY_INSIGHTS())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Univariate Analysis
    """)
    return


@app.function
def UNIVARIATE_ANALYSIS_INSIGHTS():
    INSIGHTS = [
        {
            "Category": "EDA Univariate Analysis",
            "Operation": "Outlier",
            "Insight": "['2024-11-21T14:45:00 => 8.785M', '2025-03-24T16:00:00 => -2.433M']",
            "Action": "Remove and perform machine learing imputation",
        },
        {
            "Category": "EDA Univariate Analysis",
            "Operation": "Variance",
            "Insight": "In year 2025 & 2026 the variance increase in duration from start of November to next year end of March",
            "Action": "Apply a variance-stabilizing transformation (e.g., Box-Cox or Log) to the target variable before statistical forecasting, and engineer a 'high-volatility season' boolean feature for ML pipelines.",
        },
        {
            "Category": "EDA Univariate Analysis",
            "Operation": "Phillips-Perron Unit Root Test",
            "Insight": "PP statistic: -5.0461, Series is stationary",
            "Action": "No need to perform further stationarity operations",
        },
        {
            "Category": "EDA Univariate Analysis",
            "Operation": "New Features",
            "Insight": "We have time gap & increase in variance in NOV-MARCH months",
            "Action": "We use machine learning imputation but add feature `is_imputed` and `is_high_volatility_season`",
        },
    ]

    return INSIGHTS


@app.cell
def _():
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


@app.cell
def _(train_dataset):
    train_dataset.tail()
    return


@app.cell
def _(test_dataset):
    test_dataset
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Time Gap
    """)
    return


@app.cell
def _(train_dataset):
    missing_timestamps = train_dataset.filter(pl.col("actual_demand_mw").is_null())
    return (missing_timestamps,)


@app.cell
def _(missing_timestamps):
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
def _():
    mo.md(r"""
    ## Fake Imputation

    Imputation only for the EDA Univariate Analysis
    """)
    return


@app.cell
def _(train_dataset):
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
def _():
    mo.callout(
        """Even though we are just doing this for EDA, remember that calculating .median().over(["weekday", "time_of_day"]) across the whole training set means the imputed points in your massive May–July 2025 gap are technically 'looking ahead' at actual demand data from late 2025 and early 2026. This is perfectly fine for maintaining an unbroken timeline to visualize seasonal rhythms during EDA—especially since we safely flagged those fake points with is_imputed -- but it reinforces exactly why you must build a separate, strict, time-aware imputer for your actual cross-validation pipeline later.""",
        kind="danger",
    )
    return


@app.cell
def _(train_imputed_clean):
    train_imputed_clean.filter(pl.col("actual_demand_mw").is_null())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Visual Exploration
    """)
    return


@app.cell
def _(test_dataset, train_imputed_clean):
    # utilsforecast requires 'unique_id', 'ds' (datetime), and 'y' (target) columns
    train_uf = train_imputed_clean.rename(
        {"timestamp": "ds", "actual_demand_mw_filled": "y"}
    ).with_columns(pl.lit("demand").alias("unique_id"))
    test_uf = test_dataset.rename(
        {"timestamp": "ds", "actual_demand_mw": "y"}
    ).with_columns(pl.lit("demand").alias("unique_id"))
    return test_uf, train_uf


@app.cell
def _(train_uf):
    train_uf
    return


@app.cell
def _(test_uf):
    test_uf
    return


@app.cell
def _(test_uf, train_uf):
    fig_series = plot_series(train_uf, test_uf, engine="plotly")
    fig_series.show()
    return


@app.cell
def _(train_uf):
    train_uf_ol = train_uf.with_columns(
        pl.when(
            pl.col("ds").is_in(
                [datetime(2024, 11, 21, 14, 45, 0), datetime(2025, 3, 24, 16, 0, 0)]
            )
        )
        .then(None)
        .otherwise(pl.col("y"))
        .interpolate()
        .alias("y")
    )
    return (train_uf_ol,)


@app.cell
def _(test_uf, train_uf_ol):
    plot_series(
        train_uf_ol.select(["ds", "y", "unique_id"]),
        test_uf,
        engine="plotly",
    ).show()
    return


@app.cell
def _():
    # from eda data integrity
    Gaps = [
        ("2025-04-28", "2025-04-30"),
        ("2025-05-21", "2025-07-26"),
        ("2025-07-29", "2025-07-29"),
        ("2025-08-29", "2025-09-14"),
        ("2025-10-31", "2025-10-31"),
        ("2026-02-28", "2026-02-28"),
    ]

    Gaps
    return


@app.cell
def _(train_uf_ol):
    train_uf_ol.head(100)
    return


@app.cell
def _(train_uf_ol):
    train_uf_ol.filter(
        (pl.col("ds") >= datetime(2025, 6, 1, 0, 0, 0))
        & (pl.col("ds") <= datetime(2025, 7, 27, 23, 59, 59))
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Time Series Decomposition
    """)
    return


@app.cell
def _(train_uf_ol):
    # Pass multiple seasonalities: Daily (96) and Weekly (96 * 7 = 672)
    mstl = MSTL(season_length=[96, 672, 20160])
    sf = StatsForecast(models=[mstl], freq="15min", n_jobs=-1)

    decomposition = sf.fit_predict(
        df=train_uf_ol.select(
            ["unique_id", "ds", "y"]
        ).to_pandas(),  # SF prefers pandas
        h=96,
        level=[90],
    )

    sf.plot(train_uf_ol.to_pandas(), decomposition, engine="plotly").show()
    return


@app.cell
def _(train_uf_ol):
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
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Stationarity Testing
    """)
    return


@app.cell
def _(train_uf_ol):
    # Using standard statsmodels for the statistical test output
    adf_result = adfuller(train_uf_ol["y"])
    print(f"ADF p-value: {adf_result[1]:.4f}")
    print(f"ADF statistics: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        print("Series is stationary")
    else:
        print("Series is non-stationary")
    return


@app.cell
def _(train_uf_ol):
    from arch.unitroot import PhillipsPerron

    pp_test = PhillipsPerron(train_uf_ol["y"].to_numpy(), lags=96)

    print(f"PP p-value: {pp_test.pvalue:.4f}")
    print(f"PP statistics: {pp_test.stat:.4f}")

    if pp_test.pvalue < 0.05:
        print("Series is stationary")
    else:
        print("Series is non-stationary")

    print("\nFull Summary:")
    print(pp_test.summary().as_text())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Autocorrelation (ACF / PACF)
    """)
    return


@app.cell
def _(train_uf_ol):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train_uf_ol["y"].to_numpy(), ax=ax1, lags=50)
    plot_pacf(train_uf_ol["y"].to_numpy(), ax=ax2, lags=50)
    mo.ui.matplotlib(plt.gca())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Drift Check
    """)
    return


@app.cell
def _(test_uf, train_uf_ol):
    ref_df = train_uf_ol.to_pandas()
    ana_df = test_uf.to_pandas()

    # Instantiate the Univariate Drift Calculator
    calc = nml.UnivariateDriftCalculator(
        column_names=["y"],
        timestamp_column_name="ds",
        continuous_methods=["kolmogorov_smirnov", "jensen_shannon"],
        chunk_size=96,
    )

    calc.fit(ref_df)
    drift_results = calc.calculate(ana_df)

    drift_fig = drift_results.filter(
        column_names=["y"], methods=["jensen_shannon"]
    ).plot()

    drift_fig.show()
    return


@app.cell
def _(test_uf, train_uf_ol):
    train_sr_clean = train_uf_ol.with_columns(
        (pl.col("y") - pl.col("y").shift(96)).alias("seasonal_residual")
    ).drop_nulls()

    test_sr_clean = test_uf.with_columns(
        (pl.col("y") - pl.col("y").shift(96)).alias("seasonal_residual")
    ).drop_nulls()

    # Convert back to Pandas for NannyML
    ref_sr_df = train_sr_clean.to_pandas()
    ana_sr_df = test_sr_clean.to_pandas()

    # Instantiate the Calculator ON THE RESIDUALS, not the raw 'y'
    calc_sr = nml.UnivariateDriftCalculator(
        column_names=["seasonal_residual"],
        timestamp_column_name="ds",
        continuous_methods=["kolmogorov_smirnov", "jensen_shannon"],
        chunk_size=96,
    )

    calc_sr.fit(ref_sr_df)
    drift_sr_results = calc_sr.calculate(ana_sr_df)

    drift_sr_fig = drift_sr_results.filter(
        column_names=["seasonal_residual"], methods=["jensen_shannon"]
    ).plot()

    drift_sr_fig.show()
    return


@app.cell
def _(train_uf_ol):
    import numpy as np
    from scipy.stats import boxcox

    df_engineered = train_uf_ol.with_columns(
        # Create the 'high-volatility season' boolean feature for ML
        # Months 11, 12, 1, 2, 3 map to True (1), all others to False (0)
        pl.col("ds")
        .dt.month()
        .is_in([11, 12, 1, 2, 3])
        .cast(pl.Int8)
        .alias("is_high_volatility_season"),
        # Variance Stabilization: Natural Log transformation
        # We use log1p (log(1+x)) as a best practice to safely handle any exact zeros
        pl.col("y").log1p().alias("y_log"),
    )

    # Extract the 'y' column as a numpy array
    y_array = train_uf_ol.get_column("y").to_numpy()

    # Calculate Box-Cox and capture the optimal lambda value
    y_boxcox, optimal_lambda = boxcox(y_array)
    print(f"Optimal Box-Cox lambda found: {optimal_lambda}")

    # Add the Box-Cox transformed array as a new column in our Polars DataFrame
    df_engineered = df_engineered.with_columns(pl.Series("y_boxcox", y_boxcox))

    df_engineered.head()
    return (df_engineered,)


@app.cell
def _(df_engineered, test_uf):
    df_engineered_m = df_engineered.rename({"y": "y_prev", "y_boxcox": "y"})

    plot_series(
        df_engineered_m,
        test_uf,
        engine="plotly",
    ).show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Temporal Analysis of Electricity Demand

    This section visualizes the `train_uf_ol` dataset to explore recurring patterns in power consumption, specifically focusing on monthly trends, daily fluctuations in April 2024, and the average demand across different days of the week.
    """)
    return


@app.cell
def _(train_uf_ol):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 1. Preprocessing: Ensure 'ds' is cast to a Datetime type.
    # Polars does NOT use an index, so we just keep everything as standard columns.
    # (If 'ds' is already a datetime type in your DataFrame, you can skip the cast)
    df = train_uf_ol.with_columns(pl.col("ds").cast(pl.Datetime))

    # 2. Calculate the grouped data

    # Monthly Average using group_by_dynamic
    monthly_mean = df.group_by_dynamic("ds", every="1mo").agg(pl.col("y").mean())

    # Daily Average for April 2024
    daily_mean_apr = (
        df.filter(
            (pl.col("ds") >= datetime(2024, 4, 1))
            & (pl.col("ds") < datetime(2024, 5, 1))
        )
        .group_by_dynamic("ds", every="1d")
        .agg(pl.col("y").mean())
    )

    # Day of Week Average for 2024
    # dt.weekday() returns 1 (Monday) to 7 (Sunday)
    dow_mean = (
        df.filter(pl.col("ds").dt.year() == 2024)
        .group_by(pl.col("ds").dt.weekday().alias("weekday"))
        .agg(pl.col("y").mean())
        .sort("weekday")  # Crucial: ensures days stay in Mon-Sun order
    )

    dow_labels = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # 3. Create the Subplots layout (3 rows, 1 column)
    time_fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Monthly Mean Demand",
            "Daily Mean Demand in April 2024",
            "Mean Demand by Day of Week in 2024",
        ),
        vertical_spacing=0.1,
    )

    # 4. Add the traces
    # Trace 1: Monthly
    time_fig.add_trace(
        go.Scatter(
            x=monthly_mean["ds"],  # Passing Polars Series directly to Plotly
            y=monthly_mean["y"],
            mode="lines+markers",
            line=dict(color="limegreen"),
            name="Monthly Mean",
        ),
        row=1,
        col=1,
    )

    # Trace 2: Daily (April 2024)
    time_fig.add_trace(
        go.Scatter(
            x=daily_mean_apr["ds"],
            y=daily_mean_apr["y"],
            mode="lines+markers",
            line=dict(color="limegreen"),
            name="Daily Mean",
        ),
        row=2,
        col=1,
    )

    # Trace 3: Day of Week
    time_fig.add_trace(
        go.Scatter(
            x=dow_labels,
            y=dow_mean["y"],
            mode="lines+markers",
            line=dict(color="limegreen"),
            name="DOW Mean",
        ),
        row=3,
        col=1,
    )

    # 5. Update formatting and axis labels
    time_fig.update_layout(
        height=1000,
        showlegend=False,
        title_text="Electricity Demand Analysis",
        title_x=0.5,
        template="plotly_white",
    )

    time_fig.update_xaxes(title_text="Month", row=1, col=1)
    time_fig.update_yaxes(title_text="Mean Demand (MW)", row=1, col=1)

    time_fig.update_xaxes(title_text="Day", row=2, col=1)
    time_fig.update_yaxes(title_text="Mean Demand (MW)", row=2, col=1)

    time_fig.update_xaxes(title_text="Day of Week", row=3, col=1)
    time_fig.update_yaxes(title_text="Mean Demand (MW)", row=3, col=1)

    time_fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
