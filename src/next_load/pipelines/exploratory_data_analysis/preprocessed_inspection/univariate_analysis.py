"""
Marimo notebook for univariate analysis of energy demand.
Covers outlier detection, variance analysis, stationarity testing, and data drift calculation.
"""

import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import marimo as mo
    import polars as pl
    import pyarrow.parquet as pq
    import s3fs
    from datetime import datetime
    import altair as alt
    import matplotlib.pyplot as plt
    from utilsforecast.plotting import plot_series
    from statsforecast.models import MSTL
    from statsforecast import StatsForecast
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import nannyml as nml
    from next_load.core.nl_auth import get_infisical_secret


@app.cell
def _():
    """
    Load previous insights from data processing and integrity checks.
    """
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
    """
    Returns summarized findings from the univariate analysis phase.
    Notes significant outliers, seasonal variance increases, and stationarity results.
    """
    INSIGHTS = [
        {
            "Category": "EDA Univariate Analysis",
            "Operation": "Outlier",
            "Insight": "Identified two extreme spikes in Nov 2024 and March 2025.",
            "Action": "Apply predictive imputation using LightGBM.",
        },
        {
            "Category": "EDA Univariate Analysis",
            "Operation": "Variance",
            "Insight": "Increased volatility observed from November to March.",
            "Action": "Create a 'high-volatility season' feature and consider log transformations.",
        },
        {
            "Category": "EDA Univariate Analysis",
            "Operation": "Phillips-Perron Unit Root Test",
            "Insight": "Series confirmed as stationary.",
            "Action": "No further stationarity adjustments needed.",
        },
        {
            "Category": "EDA Univariate Analysis",
            "Operation": "New Features",
            "Insight": "Combining time gaps and seasonal variance requires specific indicators.",
            "Action": "Implement 'is_imputed' and 'is_high_volatility_season' flags.",
        },
    ]

    return INSIGHTS


@app.cell
def _(get_infisical_secret, s3fs):
    """
    Initialize S3 connection and load train/test datasets for inspection.
    """
    S3_FS = s3fs.S3FileSystem(
        key=get_infisical_secret("AWS_ACCESS_KEY_ID"),
        secret=get_infisical_secret("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=get_infisical_secret("AWS_ENDPOINT_URL") or "http://localhost:3900",
        client_kwargs={"region_name": get_infisical_secret("AWS_DEFAULT_REGION")},
        config_kwargs={"s3": {"addressing_style": "path"}}
    )
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
    """
    Locate missing demand values in the training set.
    """
    missing_timestamps = train_dataset.filter(pl.col("actual_demand_mw").is_null())
    return (missing_timestamps,)


@app.cell
def _(missing_timestamps):
    """
    Group missing timestamps into duration blocks for better visibility of data loss events.
    """
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
    """)
    return


@app.cell
def _(train_dataset):
    """
    Apply a seasonal median imputation strategy for visual exploration only.
    Combines rolling macro trends with localized seasonal swings.
    """
    train_resampled = train_dataset.upsample(time_column="timestamp", every="15m")

    train_flagged = train_resampled.with_columns(
        pl.col("actual_demand_mw").is_null().alias("is_imputed")
    )

    train_scaffold = train_flagged.with_columns(
        pl.col("actual_demand_mw").interpolate().alias("naive_bridge")
    )

    train_trend = train_scaffold.with_columns(
        pl.col("naive_bridge")
        .rolling_mean(window_size=672, min_periods=1, center=True)
        .alias("macro_trend")
    )

    train_detrended = train_trend.with_columns(
        (pl.col("actual_demand_mw") - pl.col("macro_trend")).alias(
            "detrended_signal"
        ),
        pl.col("timestamp").dt.weekday().alias("weekday"),
        pl.col("timestamp").dt.time().alias("time_of_day"),
    )

    train_swings = train_detrended.with_columns(
        pl.col("detrended_signal")
        .median()
        .over(["weekday", "time_of_day"])
        .alias("seasonal_swing")
    )

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
        """Seasonal median imputation is used solely for EDA visualizations and contains data leakage. It should not be used for final modeling.""",
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
    """
    Format data for visualization libraries.
    """
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
    """
    Visualize the energy demand series.
    """
    fig_series = plot_series(train_uf, test_uf, engine="plotly")
    fig_series.show()
    return


@app.cell
def _(train_uf):
    """
    Apply manual outlier correction via interpolation for visual clarity.
    """
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
    """
    Perform MSTL decomposition to extract trend and seasonal components.
    """
    mstl = MSTL(season_length=[96, 672, 20160])
    sf = StatsForecast(models=[mstl], freq="15min", n_jobs=-1)

    decomposition = sf.fit_predict(
        df=train_uf_ol.select(["unique_id", "ds", "y"]).to_pandas(),
        h=96,
        level=[90],
    )

    sf.plot(train_uf_ol.to_pandas(), decomposition, engine="plotly").show()
    return


@app.cell
def _(train_uf_ol):
    """
    Generate seasonal variance features.
    """
    df_features = train_uf_ol.with_columns(
        pl.col("ds")
        .dt.month()
        .is_in([11, 12, 1, 2, 3])
        .cast(pl.Int8)
        .alias("is_high_variance_season"),
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
    """
    Apply Augmented Dickey-Fuller (ADF) test for stationarity.
    """
    adf_result = adfuller(train_uf_ol["y"])
    print(f"ADF p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        print("Series is stationary")
    else:
        print("Series is non-stationary")
    return


@app.cell
def _(train_uf_ol):
    """
    Apply Phillips-Perron test for stationarity.
    """
    from arch.unitroot import PhillipsPerron

    pp_test = PhillipsPerron(train_uf_ol["y"].to_numpy(), lags=96)

    print(f"PP p-value: {pp_test.pvalue:.4f}")
    if pp_test.pvalue < 0.05:
        print("Series is stationary")
    else:
        print("Series is non-stationary")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Autocorrelation (ACF / PACF)
    """)
    return


@app.cell
def _(train_uf_ol):
    """
    Plot ACF and PACF to identify autoregressive patterns and seasonal dependencies.
    """
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
    """
    Calculate univariate data drift between training and testing sets using NannyML.
    Uses Kolmogorov-Smirnov and Jensen-Shannon methods.
    """
    ref_df = train_uf_ol.to_pandas()
    ana_df = test_uf.to_pandas()

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
    """
    Evaluate drift specifically on seasonal residuals to detect underlying pattern changes.
    """
    train_sr_clean = train_uf_ol.with_columns(
        (pl.col("y") - pl.col("y").shift(96)).alias("seasonal_residual")
    ).drop_nulls()

    test_sr_clean = test_uf.with_columns(
        (pl.col("y") - pl.col("y").shift(96)).alias("seasonal_residual")
    ).drop_nulls()

    ref_sr_df = train_sr_clean.to_pandas()
    ana_sr_df = test_sr_clean.to_pandas()

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
    """
    Apply Box-Cox and Log transformations for variance stabilization.
    """
    import numpy as np
    from scipy.stats import boxcox

    df_engineered = train_uf_ol.with_columns(
        pl.col("ds")
        .dt.month()
        .is_in([11, 12, 1, 2, 3])
        .cast(pl.Int8)
        .alias("is_high_volatility_season"),
        pl.col("y").log1p().alias("y_log"),
    )

    y_array = train_uf_ol.get_column("y").to_numpy()
    y_boxcox, optimal_lambda = boxcox(y_array)
    print(f"Optimal Box-Cox lambda found: {optimal_lambda}")

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
    """)
    return


@app.cell
def _(train_uf_ol):
    """
    Visualize mean energy demand across different time scales: monthly, daily, and day of week.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = train_uf_ol.with_columns(pl.col("ds").cast(pl.Datetime))

    monthly_mean = df.group_by_dynamic("ds", every="1mo").agg(pl.col("y").mean())

    daily_mean_apr = (
        df.filter(
            (pl.col("ds") >= datetime(2024, 4, 1))
            & (pl.col("ds") < datetime(2024, 5, 1))
        )
        .group_by_dynamic("ds", every="1d")
        .agg(pl.col("y").mean())
    )

    dow_mean = (
        df.filter(pl.col("ds").dt.year() == 2024)
        .group_by(pl.col("ds").dt.weekday().alias("weekday"))
        .agg(pl.col("y").mean())
        .sort("weekday")
    )

    dow_labels = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]

    time_fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Monthly Mean Demand",
            "Daily Mean Demand in April 2024",
            "Mean Demand by Day of Week in 2024",
        ),
        vertical_spacing=0.1,
    )

    time_fig.add_trace(
        go.Scatter(
            x=monthly_mean["ds"],
            y=monthly_mean["y"],
            mode="lines+markers",
            line=dict(color="limegreen"),
            name="Monthly Mean",
        ),
        row=1, col=1,
    )

    time_fig.add_trace(
        go.Scatter(
            x=daily_mean_apr["ds"],
            y=daily_mean_apr["y"],
            mode="lines+markers",
            line=dict(color="limegreen"),
            name="Daily Mean",
        ),
        row=2, col=1,
    )

    time_fig.add_trace(
        go.Scatter(
            x=dow_labels,
            y=dow_mean["y"],
            mode="lines+markers",
            line=dict(color="limegreen"),
            name="DOW Mean",
        ),
        row=3, col=1,
    )

    time_fig.update_layout(
        height=1000,
        showlegend=False,
        title_text="Electricity Demand Analysis",
        title_x=0.5,
        template="plotly_white",
    )

    time_fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
