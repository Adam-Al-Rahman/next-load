# Nodes for baseline models pipeline using StatsForecast.

from __future__ import annotations
import logging
from datetime import datetime
from typing import Any
import lightgbm as lgb
import mlflow
import mlflow.pyfunc
import numpy as np
import polars as pl
from mlflow.models import infer_signature
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mape, rmse

logger = logging.getLogger(__name__)


class StatsForecastWrapper(mlflow.pyfunc.PythonModel):
    # Custom MLflow wrapper for StatsForecast models.

    def __init__(self, model: StatsForecast):
        self.model = model

    def predict(
        self,
        context: Any,
        model_input: pl.DataFrame | np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        # Generates predictions for a given horizon or input DataFrame.
        if isinstance(model_input, int | np.integer):
            h = int(model_input)
        else:
            h = len(model_input)

        forecasts = self.model.predict(h=h)
        return forecasts["SeasonalNaive"].to_numpy()


def build_baseline_features(df: pl.DataFrame, parameters: dict) -> pl.DataFrame:
    # Generates baseline features and handles outlier detection.
    date_col = parameters["date_column"]
    target_col = parameters["target_column"]

    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    outlier_dates = [datetime(2024, 11, 21, 14, 45, 0), datetime(2025, 3, 24, 16, 0, 0)]

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


def impute_baseline_data(df: pl.DataFrame, parameters: dict) -> pl.DataFrame:
    # Imputes missing target values using LightGBM.
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
        X_train = known_data.select(features).to_numpy()
        y_train = known_data[target_col].to_numpy()
        X_miss = missing_data.select(features).to_numpy()

        if np.isnan(X_train).any() or np.isnan(X_miss).any():
            X_train = np.nan_to_num(X_train)
            X_miss = np.nan_to_num(X_miss)

        train_ds = lgb.Dataset(X_train, label=y_train)
        num_boost_round = lgb_params.pop("n_estimators", 100)
        native_params = {
            k: v
            for k, v in lgb_params.items()
            if k not in ["silent", "importance_type"]
        }

        gbm = lgb.train(native_params, train_ds, num_boost_round=num_boost_round)
        imputed_y = gbm.predict(X_miss)

        imputed_df = missing_data.with_columns(pl.Series(target_col, imputed_y))
        df_final = pl.concat([known_data, imputed_df]).sort(date_col)

        mlflow.log_metric("imputed_values_count", missing_data.height)
        return df_final

    return df.with_columns(pl.lit(False).alias("is_imputed"))


def train_evaluate_baseline_model(
    train_df: pl.DataFrame, test_df: pl.DataFrame, parameters: dict
) -> tuple[dict, StatsForecastWrapper]:
    # Trains and evaluates a Seasonal Naive model.
    if not isinstance(train_df, pl.DataFrame):
        train_df = pl.from_pandas(train_df)
    if not isinstance(test_df, pl.DataFrame):
        test_df = pl.from_pandas(test_df)

    daily_season = parameters["daily_season"]
    target_col = parameters["target_column"]
    date_col = parameters["date_column"]

    train_sf = (
        train_df.rename({date_col: "ds", target_col: "y"})
        .with_columns(pl.lit("demand_zone_1").alias("unique_id"))
        .select(["unique_id", "ds", "y"])
    )

    test_sf = (
        test_df.rename({date_col: "ds", target_col: "y"})
        .with_columns(pl.lit("demand_zone_1").alias("unique_id"))
        .select(["unique_id", "ds", "y"])
    )

    model = StatsForecast(
        models=[SeasonalNaive(season_length=daily_season)],
        freq="15min",
        n_jobs=-1,
    )

    model.fit(train_sf.to_pandas())
    h = test_sf.height
    predictions = pl.from_pandas(model.predict(h=h))

    merged = predictions.join(test_sf, on=["unique_id", "ds"], how="inner")
    evaluation_df = evaluate(
        df=merged.to_pandas(),
        metrics=[mae, rmse, mape],
        models=["SeasonalNaive"],
    )

    metrics = {
        "mae": float(
            evaluation_df.loc[evaluation_df["metric"] == "mae", "SeasonalNaive"].values[
                0
            ]
        ),
        "rmse": float(
            evaluation_df.loc[
                evaluation_df["metric"] == "rmse", "SeasonalNaive"
            ].values[0]
        ),
        "mape": float(
            evaluation_df.loc[
                evaluation_df["metric"] == "mape", "SeasonalNaive"
            ].values[0]
        ),
    }

    mlflow_metrics = {k: {"value": v, "step": 0} for k, v in metrics.items()}

    input_example = test_sf.head(5).to_pandas()
    output_sample = predictions.head(5)["SeasonalNaive"].to_numpy()
    signature = infer_signature(input_example, output_sample)

    tags = {
        "project": "next_load",
        "team": "data_science",
        "stage": "baseline",
        "algorithm": "seasonal_naive",
        "daily_season": str(daily_season),
        "framework": "statsforecast",
    }

    run_description = f"""
    ### Baseline Model: Seasonal Naive
    - **Season Length**: {daily_season}
    - **MAE**: {metrics["mae"]:.4f}
    - **RMSE**: {metrics["rmse"]:.4f}
    - **MAPE**: {metrics["mape"]:.4f}
    """

    wrapped_model = StatsForecastWrapper(model)
    model_name = "baseline_seasonal_naive"

    with mlflow.start_run(
        nested=True, run_name=f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        model_info = mlflow.pyfunc.log_model(
            python_model=wrapped_model,
            name=model_name,
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
            tags=tags,
            metadata={"mae": str(metrics["mae"]), "rmse": str(metrics["rmse"])},
        )

        client = mlflow.tracking.MlflowClient()
        try:
            client.get_registered_model(model_name)
            client.update_registered_model(
                name=model_name,
                description="Next Load Energy Forecasting: Baseline Seasonal Naive Models",
            )
        except Exception:
            pass

        client.update_model_version(
            name=model_name,
            version=model_info.registered_model_version,
            description=run_description,
        )

        mlflow.set_tag("mlflow.note.content", run_description)
        mlflow.set_tags(tags)
        mlflow.log_metrics(metrics)

    return mlflow_metrics, wrapped_model
