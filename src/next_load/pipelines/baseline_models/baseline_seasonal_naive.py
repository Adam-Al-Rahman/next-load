"""
Marimo notebook for building and evaluating the Seasonal Naive baseline model.
Includes data loading, gap analysis, time-aware imputation, cross-validation, and performance visualization.
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    """
    Import necessary libraries for data processing, modeling, and visualization.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import datetime
    import altair as alt
    import lightgbm as lgb
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import pyarrow.parquet as pq
    import s3fs
    from plotly.subplots import make_subplots
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive
    from tqdm import tqdm
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mae, mape, rmse
    from utilsforecast.plotting import plot_series
    from next_load.core.nl_auth import get_infisical_secret

    return (
        SeasonalNaive,
        StatsForecast,
        datetime,
        evaluate,
        get_infisical_secret,
        go,
        lgb,
        mae,
        make_subplots,
        mape,
        mo,
        np,
        pl,
        plot_series,
        pq,
        px,
        rmse,
        s3fs,
    )


@app.cell
def _(pl):
    """
    Load and display previous insights from data integrity and univariate analysis.
    """
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
    # Baseline Model: Seasonal Naive
    """)
    return


@app.cell
def _(get_infisical_secret, pl, pq, s3fs):
    """
    Establish S3 connection and load the train and test datasets.
    """
    S3_FS = s3fs.S3FileSystem(
        key=get_infisical_secret("AWS_ACCESS_KEY_ID"),
        secret=get_infisical_secret("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=get_infisical_secret("AWS_ENDPOINT_URL")
        or "http://localhost:3900",
        client_kwargs={"region_name": get_infisical_secret("AWS_DEFAULT_REGION")},
        config_kwargs={"s3": {"addressing_style": "path"}},
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time Gap
    """)
    return


@app.cell
def _(pl, train_dataset):
    """
    Identify missing timestamps in the training dataset.
    """
    missing_timestamps = train_dataset.filter(pl.col("actual_demand_mw").is_null())
    return (missing_timestamps,)


@app.cell
def _(missing_timestamps, pl):
    """
    Aggregate missing timestamps into continuous blocks for analysis.
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
def _(mo):
    mo.md(r"""
    ## Fake Imputation
    """)
    return


@app.cell
def _(pl, train_dataset):
    """
    Apply a simple median-based seasonal imputation for exploratory visualization.
    Calculates median demand per weekday and time-of-day after removing macro trends.
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
        .rolling_mean(window_size=672, min_samples=1, center=True)
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
def _(mo):
    mo.callout(
        """Median-based seasonal imputation is only for visualization during EDA. This approach uses global information from the future and should not be used for model training or cross-validation.""",
        kind="danger",
    )
    return


@app.cell
def _(pl, test_dataset, train_imputed_clean):
    """
    Format datasets for the utilsforecast library.
    Requires columns: unique_id, ds (timestamp), and y (target).
    """
    train_uf = train_imputed_clean.rename(
        {"timestamp": "ds", "actual_demand_mw_filled": "y"}
    ).with_columns(pl.lit("demand").alias("unique_id"))
    test_uf = test_dataset.rename(
        {"timestamp": "ds", "actual_demand_mw": "y"}
    ).with_columns(pl.lit("demand").alias("unique_id"))
    return test_uf, train_uf


@app.cell
def _(datetime, pl, train_uf):
    """
    Manually flag and interpolate known outliers identified in univariate analysis.
    """
    target_dates = [
        datetime(2024, 11, 21, 14, 45, 0),
        datetime(2025, 3, 24, 16, 0, 0),
    ]
    target_series = pl.Series(target_dates)

    train_uf_ol = train_uf.with_columns(
        y=pl.when(pl.col("ds").is_in(target_series.implode()))
        .then(
            pl.lit(None, dtype=pl.Float64)
        )
        .otherwise(pl.col("y"))
        .interpolate(),
        is_imputed=pl.when(pl.col("ds").is_in(target_series.implode()))
        .then(True)
        .otherwise(
            pl.col("is_imputed") if "is_imputed" in train_uf.columns else False
        ),
    )
    return (train_uf_ol,)


@app.cell
def _(plot_series, test_uf, train_uf_ol):
    """
    Visualize the training and testing series using Plotly.
    """
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
    """
    Plot demand values highlighting imputed points.
    """
    dm_fig = px.scatter(
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

    dm_fig.update_traces(mode="lines+markers")
    dm_fig.show()
    return


@app.cell
def _(pl, train_uf_ol):
    """
    Engineer seasonal features and calculate rolling volatility.
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
def _(mo):
    mo.md(r"""
    ## Model: Train
    """)
    return


@app.cell
def _(datetime, pl, train_dataset):
    """
    Prepare the main dataset for training by applying outlier nullification.
    """
    dataset = train_dataset.with_columns(
        [
            pl.lit("demand_zone_1").alias("unique_id"),
            pl.col("timestamp").alias("ds"),
            pl.col("actual_demand_mw").alias("y"),
        ]
    ).select(["unique_id", "ds", "y"])

    dataset_ol = dataset.with_columns(
        pl.when(
            pl.col("ds").is_in(
                [datetime(2024, 11, 21, 14, 45, 0), datetime(2025, 3, 24, 16, 0, 0)]
            )
        )
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    return (dataset_ol,)


@app.cell
def _(np, pl):
    def build_ts_features(df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies cyclical encoding and historical anchors to the DataFrame.
        Includes weekday, hour, and day of year sin/cos encodings plus 1-day, 1-week, and 1-year lags.
        """
        df = df.with_columns(
            [
                (pl.col("ds").dt.hour() + pl.col("ds").dt.minute() / 60.0).alias(
                    "decimal_hour"
                ),
                pl.col("ds").dt.weekday().alias("weekday_int"),
                pl.col("ds").dt.ordinal_day().alias("dayofyear_int"),
            ]
        ).with_columns(
            [
                np.sin(2 * np.pi * pl.col("decimal_hour") / 24).alias("sin_hour"),
                np.cos(2 * np.pi * pl.col("decimal_hour") / 24).alias("cos_hour"),
                np.sin(2 * np.pi * pl.col("weekday_int") / 7).alias("sin_weekday"),
                np.cos(2 * np.pi * pl.col("weekday_int") / 7).alias("cos_weekday"),
                np.sin(2 * np.pi * pl.col("dayofyear_int") / 365).alias(
                    "sin_dayofyear"
                ),
                np.cos(2 * np.pi * pl.col("dayofyear_int") / 365).alias(
                    "cos_dayofyear"
                ),
                pl.col("y").shift(96).alias("y_yesterday"),
                pl.col("y").shift(672).alias("y_last_week"),
                pl.col("y").shift(34944).alias("y_last_year"),
            ]
        )
        return df

    return (build_ts_features,)


@app.cell
def _(lgb, pl):
    def impute_time_series_fold(train_fold: pl.DataFrame) -> pl.DataFrame:
        """
        Performs iterative imputation for a single cross-validation fold using LightGBM.
        Identifies outliers using IQR, nullifies them, and then predicts values based on features.
        """
        q1 = train_fold["y"].quantile(0.25)
        q3 = train_fold["y"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        train_fold = train_fold.with_columns(
            pl.when((pl.col("y") < lower_bound) | (pl.col("y") > upper_bound))
            .then(None)
            .otherwise(pl.col("y"))
            .alias("y")
        )

        known_data = train_fold.drop_nulls(subset=["y"]).with_columns(
            pl.lit(False).alias("is_imputed")
        )
        missing_data = train_fold.filter(pl.col("y").is_null()).with_columns(
            pl.lit(True).alias("is_imputed")
        )

        if missing_data.height > 0 and known_data.height > 0:
            features = [
                "sin_hour",
                "cos_hour",
                "sin_weekday",
                "cos_weekday",
                "sin_dayofyear",
                "cos_dayofyear",
                "y_yesterday",
                "y_last_week",
                "y_last_year",
            ]

            gbm = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                random_state=42,
                device_type="cpu",
                n_jobs=-1,
                verbose=-1,
            )

            gbm.fit(
                known_data.select(features).to_numpy(), known_data["y"].to_numpy()
            )
            imputed_y = gbm.predict(missing_data.select(features).to_numpy())

            imputed_df = missing_data.with_columns(pl.Series("y", imputed_y))
            return pl.concat([known_data, imputed_df]).sort("ds")

        return train_fold

    return (impute_time_series_fold,)


@app.cell
def _(
    SeasonalNaive,
    StatsForecast,
    build_ts_features,
    dataset_ol,
    impute_time_series_fold,
    mo,
    pl,
):
    """
    Execute a Walk-Forward Cross-Validation loop for the Seasonal Naive model.
    Applies per-fold imputation to ensure realistic evaluation without data leakage.
    """
    DAILY_SEASON = 96
    HORIZON = 8
    INITIAL_TRAIN_SIZE = 96
    STEP_SIZE = 4

    model = StatsForecast(
        models=[SeasonalNaive(season_length=DAILY_SEASON)],
        freq="15m",
        n_jobs=1,
    )

    df = build_ts_features(dataset_ol)

    all_predictions = []
    total_rows = df.height
    cv_steps = list(range(INITIAL_TRAIN_SIZE, total_rows - HORIZON, STEP_SIZE))
    total_folds = len(cv_steps)

    final_imputed_state = None

    with mo.status.progress_bar(
        total=total_folds,
        title="Baseline (SeasonalNaive) CV With Imputation",
        subtitle="Initializing...",
    ) as bar:
        for fold_idx, step in enumerate(cv_steps, start=1):
            bar.update(
                increment=1,
                subtitle=f"Processing Fold {fold_idx}/{total_folds} | Train Size: {step}",
            )

            raw_train_fold = df.slice(0, step)
            test_fold = df.slice(step, HORIZON)

            imputed_train_fold = impute_time_series_fold(raw_train_fold)

            if fold_idx == total_folds:
                final_imputed_state = imputed_train_fold.clone()

            model_ready_train = imputed_train_fold.select(["unique_id", "ds", "y"])

            model.fit(model_ready_train)
            predictions = model.predict(h=HORIZON)

            merged_results = predictions.join(
                test_fold, on=["unique_id", "ds"], how="inner"
            ).with_columns(pl.lit(f"Fold_{step}").alias("fold"))

            all_predictions.append(merged_results)
    return all_predictions, df, final_imputed_state


@app.cell
def _(all_predictions, evaluate, mae, mape, pl, rmse):
    """
    Aggregate cross-validation results and calculate performance metrics.
    """
    results_df = pl.concat(all_predictions)

    evaluation_df = evaluate(
        df=results_df.to_pandas(),
        metrics=[mae, rmse, mape],
        models=["SeasonalNaive"],
        id_col="fold",
    )
    global_metrics = evaluation_df.groupby(["fold", "metric"]).mean("SeasonalNaive")
    global_metrics
    return evaluation_df, results_df


@app.cell
def _(evaluation_df):
    """
    Analyze performance across folds to identify the worst cases.
    """
    mape_only = evaluation_df[evaluation_df["metric"] == "mape"]
    worst_folds = mape_only.sort_values(by="SeasonalNaive", ascending=False)

    worst_folds.head(5)
    return


@app.cell
def _(df, final_imputed_state, go, make_subplots, pl, results_df):
    """
    Generate diagnostic plots for imputation verification and forecast evaluation.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Modular Imputation Verification (Entire Dataset)",
            "SeasonalNaive Forecasting (All CV Folds)",
        ),
        vertical_spacing=0.15,
    )

    if final_imputed_state is not None:
        imp_viz_df = final_imputed_state

        actuals = imp_viz_df.filter(~pl.col("is_imputed"))
        imputed = imp_viz_df.filter(pl.col("is_imputed"))

        fig.add_trace(
            go.Scatter(
                x=actuals["ds"],
                y=actuals["y"],
                mode="lines",
                name="Known Demand",
                line=dict(color="rgba(150, 150, 150, 0.6)", width=1),
            ),
            row=1,
            col=1,
        )

        if imputed.height > 0:
            fig.add_trace(
                go.Scatter(
                    x=imputed["ds"],
                    y=imputed["y"],
                    mode="markers",
                    name="ML Imputed Values",
                    marker=dict(color="red", size=4),
                ),
                row=1,
                col=1,
            )

    df_plot = df
    fig.add_trace(
        go.Scatter(
            x=df_plot["ds"],
            y=df_plot["y"],
            mode="lines",
            name="Actual Demand (Target)",
            line=dict(color="black", width=2),
        ),
        row=2,
        col=1,
    )

    all_folds = results_df["fold"].unique().to_list()
    colors = ["#FF9900", "#3366CC", "#DC3912", "#109618", "#990099"]

    for idx, fold_name in enumerate(all_folds):
        fold_data = results_df.filter(pl.col("fold") == fold_name)
        fig.add_trace(
            go.Scatter(
                x=fold_data["ds"],
                y=fold_data["SeasonalNaive"],
                mode="lines",
                name=f"Pred {fold_name.split('_')[1]}",
                line=dict(width=2, color=colors[idx % len(colors)]),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=800,
        title_text="Robust Time Series Pipeline (Complete Dataset)",
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
    )
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
