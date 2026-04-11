"""
Notebook for training and evaluating LightGBM and neural models
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import marimo as mo
    import altair as alt
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import polars as pl
    import pandas as pd
    import numpy as np
    import pyarrow.parquet as pq
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    import optuna
    from mlforecast import MLForecast
    from mlforecast.auto import AutoLightGBM, AutoMLForecast
    from mlforecast.lag_transforms import (
        RollingMean,
        RollingStd,
        RollingMin,
        RollingMax,
        ExponentiallyWeightedMean,
    )
    import torch
    from mlforecast.target_transforms import Differences
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, TFT, TiDE
    from neuralforecast.losses.pytorch import MAE, HuberLoss
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mae, mape, rmse
    from utilsforecast.plotting import plot_series
    from datetime import datetime
    from next_load.core.nl_auth import get_infisical_secret


@app.cell
def _():
    """
    Aggregate insights from data preprocessing and integrity modules
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
def _():
    mo.md(r"""
    # Candidate Model: LightGMB
    """)
    return


@app.function
def get_polars_storage_options():
    """
    Retrieve storage credentials for S3 operations
    """
    return {
        "aws_access_key_id": get_infisical_secret("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": get_infisical_secret("AWS_SECRET_ACCESS_KEY"),
        "aws_region": get_infisical_secret("AWS_DEFAULT_REGION", default="asia-south1"),
        "aws_endpoint_url": get_infisical_secret(
            "AWS_ENDPOINT_URL", default="http://localhost:3900"
        ),
    }


@app.cell
def _():
    """
    Load training and testing datasets from storage
    """
    pl_storage_options = get_polars_storage_options()

    r_train_dataset = pl.read_parquet(
        "s3://next-load-data/processed/01_primary/train_dataset.parquet",
        storage_options=pl_storage_options,
    )

    r_test_dataset = pl.read_parquet(
        "s3://next-load-data/processed/01_primary/test_dataset.parquet",
        storage_options=pl_storage_options,
    )

    train_dataset = r_train_dataset.drop("nrldc_intraday_forecasted_demand_mw")
    test_dataset = r_test_dataset.drop("nrldc_intraday_forecasted_demand_mw")
    return test_dataset, train_dataset


@app.cell
def _(train_dataset):
    train_dataset
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
    Identify missing target values in training data
    """
    missing_timestamps = train_dataset.filter(pl.col("actual_demand_mw").is_null())
    return (missing_timestamps,)


@app.cell
def _(missing_timestamps):
    """
    Analyze gaps in training data by grouping missing timestamps
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


@app.cell
def _(test_dataset):
    """
    Identify missing target values in testing data
    """
    test_missing_timestamps = test_dataset.filter(pl.col("actual_demand_mw").is_null())
    return (test_missing_timestamps,)


@app.cell
def _(test_missing_timestamps):
    """
    Analyze gaps in testing data by grouping missing timestamps
    """
    test_missing_blocks = (
        test_missing_timestamps.sort("timestamp")
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

    test_missing_blocks
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Imputation
    """)
    return


@app.function
def build_features(df: pl.DataFrame, parameters: dict) -> pl.DataFrame:
    """
    Create historical and cyclical features for demand forecasting
    """
    date_col = parameters["date_column"]
    target_col = parameters["target_column"]

    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    outlier_dates = [
        datetime(2024, 11, 21, 14, 45, 0),
        datetime(2025, 3, 24, 16, 0, 0),
    ]

    df = df.with_columns(
        pl.when(pl.col(date_col).is_in(outlier_dates))
        .then(None)
        .otherwise(pl.col(target_col))
        .alias(target_col)
    )

    df = df.with_columns(
        [
            (pl.col(date_col).dt.hour() + pl.col(date_col).dt.minute() / 60.0).alias(
                "decimal_hour"
            ),
            pl.col(date_col).dt.weekday().alias("weekday_int"),
            pl.col(date_col).dt.ordinal_day().alias("dayofyear_int"),
        ]
    ).with_columns(
        [
            np.sin(2 * np.pi * pl.col("decimal_hour") / 24).alias("sin_hour"),
            np.cos(2 * np.pi * pl.col("decimal_hour") / 24).alias("cos_hour"),
            np.sin(2 * np.pi * pl.col("weekday_int") / 7).alias("sin_weekday"),
            np.cos(2 * np.pi * pl.col("weekday_int") / 7).alias("cos_weekday"),
            np.sin(2 * np.pi * pl.col("dayofyear_int") / 365).alias("sin_dayofyear"),
            np.cos(2 * np.pi * pl.col("dayofyear_int") / 365).alias("cos_dayofyear"),
            pl.col(target_col).shift(96).alias("y_yesterday"),
            pl.col(target_col).shift(672).alias("y_last_week"),
            pl.col(target_col).shift(34944).alias("y_last_year"),
        ]
    )
    return df


@app.function
def impute_data(df: pl.DataFrame, parameters: dict) -> pl.DataFrame:
    """
    Impute missing values using outlier detection and LightGBM
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    target_col = parameters["target_column"]
    date_col = parameters["date_column"]
    iqr_mult = parameters["outlier_iqr_multiplier"]
    features = parameters["features"]
    lgb_params = parameters["lgb_params"]

    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_mult * iqr
    upper_bound = q3 + iqr_mult * iqr

    df = df.with_columns(
        pl.when((pl.col(target_col) < lower_bound) | (pl.col(target_col) > upper_bound))
        .then(None)
        .otherwise(pl.col(target_col))
        .alias(target_col)
    )

    known_data = df.drop_nulls(subset=[target_col]).with_columns(
        pl.lit(False).alias("is_imputed")
    )
    missing_data = df.filter(pl.col(target_col).is_null()).with_columns(
        pl.lit(True).alias("is_imputed")
    )

    if missing_data.height > 0 and known_data.height > 0:
        gbm = lgb.LGBMRegressor(**lgb_params)
        gbm.fit(
            known_data.select(features).to_numpy(),
            known_data[target_col].to_numpy(),
        )
        imputed_y = gbm.predict(missing_data.select(features).to_numpy())

        imputed_df = missing_data.with_columns(pl.Series(target_col, imputed_y))
        return pl.concat([known_data, imputed_df]).sort(date_col)

    return df.with_columns(pl.lit(False).alias("is_imputed"))


@app.cell
def _(train_dataset):
    train_ft = build_features(
        train_dataset,
        {
            "target_column": "actual_demand_mw",
            "date_column": "timestamp",
        },
    )

    train_ft
    return (train_ft,)


@app.cell
def _(train_ft):
    impute_params = {
        "target_column": "actual_demand_mw",
        "date_column": "timestamp",
        "outlier_iqr_multiplier": 1.5,
        "lgb_params": {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        "features": [
            "sin_hour",
            "cos_hour",
            "sin_weekday",
            "cos_weekday",
            "sin_dayofyear",
            "cos_dayofyear",
            "y_yesterday",
            "y_last_week",
            "y_last_year",
        ],
    }

    imputed_train_data = impute_data(train_ft, impute_params)
    imputed_train_data
    return impute_params, imputed_train_data


@app.cell
def _(imputed_train_data):
    """
    Flag high volatility seasons in training data
    """
    train_ft_data = imputed_train_data.with_columns(
        pl.col("timestamp")
        .dt.month()
        .is_in([11, 12, 1, 2, 3])
        .alias("is_high_volatility_season")
    )

    train_ft_data
    return (train_ft_data,)


@app.cell
def _(test_dataset):
    test_ft = build_features(
        test_dataset,
        {
            "target_column": "actual_demand_mw",
            "date_column": "timestamp",
        },
    )

    test_ft
    return (test_ft,)


@app.cell
def _(impute_params, test_ft):
    imputed_test_data = impute_data(test_ft, impute_params)
    imputed_test_data
    return (imputed_test_data,)


@app.cell
def _(imputed_test_data):
    """
    Flag high volatility seasons in testing data
    """
    test_ft_data = imputed_test_data.with_columns(
        pl.col("timestamp")
        .dt.month()
        .is_in([11, 12, 1, 2, 3])
        .alias("is_high_volatility_season")
    )

    test_ft_data
    return (test_ft_data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model Architecture: Train
    """)
    return


@app.cell
def _(test_ft_data, train_ft_data):
    """
    Train and evaluate neural models using Fourier features
    """
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision("medium")

    def build_fourier_features(df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate Fourier series features for seasonal patterns
        """
        exprs = []
        for k in range(1, 6):
            phase = (2 * np.pi * k * pl.col("decimal_hour")) / 24.0
            exprs.extend(
                [phase.sin().alias(f"d_sin_k{k}"), phase.cos().alias(f"d_cos_k{k}")]
            )

        for k in range(1, 5):
            phase = (2 * np.pi * k * pl.col("dayofyear_int")) / 365.25
            exprs.extend(
                [phase.sin().alias(f"y_sin_k{k}"), phase.cos().alias(f"y_cos_k{k}")]
            )

        return df.with_columns(exprs)

    train_df = train_ft_data.clone().rename(
        {"timestamp": "ds", "actual_demand_mw": "y"}
    )
    test_df = test_ft_data.clone().rename({"timestamp": "ds", "actual_demand_mw": "y"})

    train_df = train_df.with_columns(
        [pl.col("ds").cast(pl.Datetime("us")), pl.lit("grid_1").alias("unique_id")]
    ).sort("ds")

    test_df = test_df.with_columns(
        [pl.col("ds").cast(pl.Datetime("us")), pl.lit("grid_1").alias("unique_id")]
    ).sort("ds")

    train_df = build_fourier_features(train_df)
    test_df = build_fourier_features(test_df)

    fourier_cols = [
        col
        for col in train_df.columns
        if col.startswith(("d_sin", "d_cos", "y_sin", "y_cos"))
    ]
    dynamic_features = ["weekday_int", "is_high_volatility_season"] + fourier_cols

    train_prepared = train_df.select(["unique_id", "ds", "y"] + dynamic_features)

    horizon = 96
    lookback = 96 * 7

    nhits_model = NHITS(
        h=horizon,
        input_size=lookback,
        futr_exog_list=dynamic_features,
        loss=MAE(),
        scaler_type="robust",
        learning_rate=1e-3,
        max_steps=1000,
        batch_size=32,
        random_seed=42,
    )

    tide_model = TiDE(
        h=horizon,
        input_size=lookback,
        futr_exog_list=dynamic_features,
        hidden_size=256,
        decoder_output_dim=16,
        loss=HuberLoss(),
        scaler_type="robust",
        learning_rate=1e-3,
        max_steps=1000,
        batch_size=32,
        random_seed=42,
    )

    nf = NeuralForecast(models=[nhits_model, tide_model], freq="15min")

    with mo.status.spinner(
        title="[Production Mode] Training final weights on historical data..."
    ):
        nf.fit(df=train_prepared.to_pandas())

    with mo.status.spinner(title="Generating 24-hour ahead forecast..."):
        expected_future = nf.make_future_dataframe()
        expected_pl = pl.from_pandas(expected_future).with_columns(
            [pl.col("ds").cast(pl.Datetime("us"))]
        )

        expected_pl = expected_pl.with_columns(
            [
                ((pl.col("ds").dt.hour() * 60 + pl.col("ds").dt.minute()) / 60.0).alias(
                    "decimal_hour"
                ),
                pl.col("ds").dt.weekday().alias("weekday_int"),
                pl.col("ds").dt.ordinal_day().alias("dayofyear_int"),
                pl.lit(
                    train_prepared["is_high_volatility_season"].tail(1).item()
                ).alias("is_high_volatility_season"),
            ]
        )

        expected_pl = build_fourier_features(expected_pl)
        future_exog = expected_pl.select(
            ["unique_id", "ds"] + dynamic_features
        ).to_pandas()

        predictions_df = nf.predict(futr_df=future_exog)

    preds_pl = pl.from_pandas(predictions_df.reset_index()).with_columns(
        pl.col("ds").cast(pl.Datetime("us"))
    )

    eval_df = preds_pl.join(
        test_df.select(["unique_id", "ds", "y"]), on=["unique_id", "ds"], how="left"
    )

    eval_df = eval_df.with_columns(
        ((pl.col("NHITS") + pl.col("TiDE")) / 2).alias("Final_SOTA_Forecast")
    )

    df_plot = eval_df.to_pandas().sort_values("ds")

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(
        go.Scatter(
            x=df_plot["ds"],
            y=df_plot["y"],
            name="Actual Demand (Ground Truth)",
            line=dict(color="black", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot["ds"],
            y=df_plot["Final_SOTA_Forecast"],
            name="SOTA Blended Forecast",
            line=dict(color="#2ca02c", dash="solid", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot["ds"],
            y=df_plot["TiDE"],
            name="TiDE Engine",
            line=dict(color="rgba(31, 119, 180, 0.5)", dash="dot"),
        )
    )
    return fig
