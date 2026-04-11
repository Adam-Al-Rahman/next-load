# Nodes for candidate models pipeline using LightGBM and NeuralForecast.

import logging
import shutil
from datetime import datetime
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import mlflow
import mlflow.pyfunc
import numpy as np
import optuna
import pandas as pd
import polars as pl
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    ExponentiallyWeightedMean,
    RollingMean,
    RollingStd,
)
from mlforecast.target_transforms import Differences
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mape, rmse

from next_load.core.nl_auth import get_mlflow_tracking_uri

logger = logging.getLogger(__name__)


class MLForecastWrapper(mlflow.pyfunc.PythonModel):
    # MLflow wrapper for MLForecast objects to enable model registry and standardized inference.

    def __init__(self, model, variant_name, is_ensemble=False):
        self.model = model
        self.variant_name = variant_name
        self.is_ensemble = is_ensemble

    def predict(self, context, model_input, params=None):
        # Generates predictions for a given horizon or input DataFrame.
        if isinstance(model_input, (int, np.integer)):
            h = int(model_input)
            preds = self.model.predict(h=h)
        else:
            if isinstance(model_input, pl.DataFrame):
                model_input = model_input.to_pandas()
            h = len(model_input)
            preds = self.model.predict(h=h, X_df=model_input)

        if isinstance(preds, pd.DataFrame):
            preds = pl.from_pandas(preds)

        if (
            self.is_ensemble
            and "LGBM_Huber" in preds.columns
            and "LGBM_MAE" in preds.columns
        ):
            preds = preds.with_columns(
                ((pl.col("LGBM_Huber") + pl.col("LGBM_MAE")) / 2).alias(
                    self.variant_name
                )
            )

        return preds[self.variant_name].to_numpy()


class NeuralForecastWrapper(mlflow.pyfunc.PythonModel):
    # MLflow wrapper for NeuralForecast objects.

    def load_context(self, context):
        # Loads the NeuralForecast model from artifacts upon initialization.
        from neuralforecast import NeuralForecast

        self.model = NeuralForecast.load(path=context.artifacts["nf_model"])

    def predict(self, context, model_input, params=None):
        # Generates predictions for deep learning models.
        if isinstance(model_input, pl.DataFrame):
            model_input = model_input.to_pandas()

        if isinstance(model_input, (int, np.integer)):
            h = int(model_input)
            preds_nf = self.model.predict(h=h)
        else:
            preds_nf = self.model.predict(futr_df=model_input)

        preds_nf_pl = pl.from_pandas(preds_nf.reset_index())
        preds_nf_pl = preds_nf_pl.with_columns(
            ((pl.col("NHITS") + pl.col("TiDE")) / 2).alias("Neural_Ensemble")
        )

        return preds_nf_pl["Neural_Ensemble"].to_numpy()


class SafeLGBMRegressor(LGBMRegressor):
    # Enhanced LGBMRegressor for Polars compatibility and fallback fitting.

    def fit(self, X, y, **kwargs):
        # Fits the LightGBM model with type safety and error handling.
        if isinstance(X, (pl.DataFrame, pd.DataFrame)):
            X_numeric = (
                X.to_numpy()
                if isinstance(X, pl.DataFrame)
                else X.select_dtypes(include=[np.number]).to_numpy()
            )
            if np.isnan(X_numeric).any():
                X = np.nan_to_num(X_numeric)
            else:
                X = X.to_numpy()
        elif hasattr(X, "to_numpy"):
            X = X.to_numpy()
            if np.issubdtype(X.dtype, np.number) and np.isnan(X).any():
                X = np.nan_to_num(X)

        if isinstance(y, (pl.Series, pd.Series)):
            y = y.to_numpy()
        elif hasattr(y, "to_numpy"):
            y = y.to_numpy()

        try:
            return super().fit(X, y, **kwargs)
        except TypeError as e:
            if "force_all_finite" in str(e):
                logger.warning("Bypassing broken LGBM sklearn wrapper fit.")
                params = self.get_params()
                n_estimators = params.pop("n_estimators", 100)
                native_params = {
                    k: v
                    for k, v in params.items()
                    if k not in ["silent", "importance_type", "n_estimators"]
                }
                train_ds = lgb.Dataset(X, label=y)
                self._Booster = lgb.train(
                    native_params, train_ds, num_boost_round=n_estimators
                )
                return self
            raise e

    def predict(self, X, **kwargs):
        # Predicts using the fitted model or native booster.
        if isinstance(X, (pl.DataFrame, pd.DataFrame)):
            X_numeric = (
                X.to_numpy()
                if isinstance(X, pl.DataFrame)
                else X.select_dtypes(include=[np.number]).to_numpy()
            )
            if np.isnan(X_numeric).any():
                X = np.nan_to_num(X_numeric)
            else:
                X = X.to_numpy()
        elif hasattr(X, "to_numpy"):
            X = X.to_numpy()
            if np.issubdtype(X.dtype, np.number) and np.isnan(X).any():
                X = np.nan_to_num(X)

        if hasattr(self, "_Booster") and self._Booster is not None:
            return self._Booster.predict(X, **kwargs)
        return super().predict(X, **kwargs)


def _build_fourier_features(df: pl.DataFrame) -> pl.DataFrame:
    # Vectorizes Fourier Harmonics for daily and yearly cycles.
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


def create_candidate_features(
    train: Any, test: Any, parameters: Dict[str, Any]
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    # Generates advanced features including Fourier harmonics and seasonality flags.
    if not isinstance(train, pl.DataFrame):
        if isinstance(train, pd.DataFrame):
            train = pl.from_pandas(train)
        elif hasattr(train, "to_pandas"):
            train = pl.from_pandas(train.to_pandas())
        else:
            train = pl.DataFrame(train)

    if not isinstance(test, pl.DataFrame):
        if isinstance(test, pd.DataFrame):
            test = pl.from_pandas(test)
        elif hasattr(test, "to_pandas"):
            test = pl.from_pandas(test.to_pandas())
        else:
            test = pl.DataFrame(test)

    date_col = parameters["date_column"]
    target_col = parameters["target_column"]

    def _generate_base_features(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            [
                (
                    pl.col(date_col).dt.hour() + pl.col(date_col).dt.minute() / 60.0
                ).alias("decimal_hour"),
                pl.col(date_col).dt.weekday().alias("weekday_int"),
                pl.col(date_col).dt.ordinal_day().alias("dayofyear_int"),
                pl.col(date_col).dt.month().alias("month_int"),
                pl.col(date_col).dt.day().alias("dayofmonth_int"),
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
                pl.col(date_col)
                .dt.month()
                .is_in([11, 12, 1, 2, 3])
                .alias("is_high_volatility_season"),
            ]
        )
        return _build_fourier_features(df)

    train_ft = _generate_base_features(train)
    test_ft = _generate_base_features(test)

    outlier_dates = [datetime(2024, 11, 21, 14, 45, 0), datetime(2025, 3, 24, 16, 0, 0)]

    train_ft = train_ft.with_columns(
        pl.when(pl.col(date_col).is_in(outlier_dates))
        .then(None)
        .otherwise(pl.col(target_col))
        .alias(target_col)
    )

    return train_ft, test_ft


def impute_candidate_train_data(
    train: pl.DataFrame, parameters: Dict[str, Any]
) -> pl.DataFrame:
    # Imputes missing demand data using LightGBM and historical anchors.
    target_col = parameters["target_column"]
    date_col = parameters["date_column"]
    prep_params = parameters["preprocessing"]
    iqr_mult = prep_params["outlier_iqr_multiplier"]
    features = prep_params["imputation_features"]
    lgb_params = prep_params["imputation_lgb_params"]

    train = train.with_columns(
        [
            pl.col(target_col).shift(96).alias("y_yesterday"),
            pl.col(target_col).shift(672).alias("y_last_week"),
            pl.col(target_col).shift(34944).alias("y_last_year"),
        ]
    )

    q1 = train[target_col].quantile(0.25)
    q3 = train[target_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_mult * iqr
    upper_bound = q3 + iqr_mult * iqr

    train = train.with_columns(
        pl.when((pl.col(target_col) < lower_bound) | (pl.col(target_col) > upper_bound))
        .then(None)
        .otherwise(pl.col(target_col))
        .alias(target_col)
    )

    known_data = train.drop_nulls(subset=[target_col]).with_columns(
        pl.lit(False).alias("is_imputed")
    )
    missing_data = train.filter(pl.col(target_col).is_null()).with_columns(
        pl.lit(True).alias("is_imputed")
    )

    if missing_data.height > 0 and known_data.height > 0:
        gbm = SafeLGBMRegressor(**lgb_params)
        X_train = known_data.select(features).to_numpy()
        y_train = known_data[target_col].to_numpy()
        gbm.fit(X_train, y_train)

        X_missing = missing_data.select(features).to_numpy()
        imputed_y = gbm.predict(X_missing)

        imputed_df = missing_data.with_columns(pl.Series(target_col, imputed_y))
        return pl.concat([known_data, imputed_df]).sort(date_col)

    return train.with_columns(pl.lit(False).alias("is_imputed"))


def train_lgbm_candidate_models(
    train_df: pl.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    # Trains LightGBM variants using MLForecast and Optuna.
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    target_col = parameters["target_column"]
    date_col = parameters["date_column"]
    id_col = parameters["id_column"]
    freq = parameters["freq"]
    h = parameters["forecast_horizon"]
    lgbm_params = parameters["lgbm_mlforecast"]
    trials = lgbm_params["optuna_trials"]

    df_prepared = (
        train_df.rename({date_col: "ds", target_col: "y"})
        .with_columns([pl.col("ds").cast(pl.Datetime), pl.lit("grid_1").alias(id_col)])
        .unique(subset=["ds"], keep="last")
        .sort("ds")
    )

    feature_cols = [
        "decimal_hour",
        "weekday_int",
        "dayofyear_int",
        "month_int",
        "dayofmonth_int",
        "sin_hour",
        "cos_hour",
        "sin_weekday",
        "cos_weekday",
        "sin_dayofyear",
        "cos_dayofyear",
        "is_high_volatility_season",
    ]
    fourier_cols = [
        c
        for c in df_prepared.columns
        if c.startswith(("d_sin", "d_cos", "y_sin", "y_cos"))
    ]
    exog_cols = feature_cols + fourier_cols

    df_prepared = (
        df_prepared.drop_nulls(subset=exog_cols + ["y"])
        .upsample(time_column="ds", every=freq)
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .select([id_col, "ds", "y"] + exog_cols)
    )

    lgbm_freq = freq.replace("m", "min") if freq.endswith("m") else freq

    best_models = {}

    for variant in lgbm_params["variants"]:
        variant_name = variant["name"]
        variant_type = variant["type"]

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 400, 2000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.08, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 5, 12),
                "num_leaves": trial.suggest_int("num_leaves", 31, 256),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.5, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.95),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
                "n_jobs": 1,
                "verbose": -1,
            }

            if variant_type in ["anchored", "stable"]:
                params["objective"] = "mae"
            elif variant_type == "ensemble":
                params["objective"] = "huber"

            model = SafeLGBMRegressor(**params)
            lags = []
            lag_transforms = {}
            target_transforms = []

            if variant_type != "stable":
                lags = lgbm_params["lags"]
                lag_transforms = {
                    1: [
                        RollingMean(window_size=4),
                        ExponentiallyWeightedMean(alpha=0.5),
                    ],
                    96: [RollingMean(window_size=4), RollingStd(window_size=8)],
                }
                target_transforms = [Differences([96])]

            fcst = MLForecast(
                models={"LGBM": model},
                freq=lgbm_freq,
                lags=lags,
                lag_transforms=lag_transforms,
                target_transforms=target_transforms,
            )

            val_size = h
            train_opt = df_prepared.head(-val_size).select(
                [id_col, "ds", "y"] + exog_cols
            )
            val_opt = df_prepared.tail(val_size).select([id_col, "ds", "y"] + exog_cols)

            fcst.fit(
                train_opt.to_pandas(),
                id_col=id_col,
                time_col="ds",
                target_col="y",
                static_features=[],
            )
            future_exog = val_opt.select([id_col, "ds"] + exog_cols)
            preds = fcst.predict(h=val_size, X_df=future_exog.to_pandas())

            eval_df = val_opt.select([id_col, "ds", "y"]).join(
                pl.from_pandas(preds), on=[id_col, "ds"], how="inner"
            )
            score = evaluate(
                eval_df, metrics=[mape], models=["LGBM"], id_col=id_col, target_col="y"
            )
            return score["LGBM"].item()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials)

        with mlflow.start_run(run_name=variant_name, nested=True):
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_mape", study.best_value)

            best_p = study.best_params
            if variant_type in ["anchored", "stable"]:
                best_p["objective"] = "mae"

            models = {}
            if variant_type == "ensemble":
                models = {
                    "LGBM_Huber": SafeLGBMRegressor(
                        **best_p, objective="huber", random_state=42
                    ),
                    "LGBM_MAE": SafeLGBMRegressor(
                        **best_p, objective="mae", random_state=123
                    ),
                }
            else:
                models = {variant_name: SafeLGBMRegressor(**best_p)}

            lags = []
            lag_transforms = {}
            target_transforms = []
            if variant_type != "stable":
                lags = lgbm_params["lags"]
                lag_transforms = {
                    1: [
                        RollingMean(window_size=4),
                        ExponentiallyWeightedMean(alpha=0.5),
                    ],
                    96: [RollingMean(window_size=4), RollingStd(window_size=8)],
                }
                target_transforms = [Differences([96])]

            final_fcst = MLForecast(
                models=models,
                freq=lgbm_freq,
                lags=lags,
                lag_transforms=lag_transforms,
                target_transforms=target_transforms,
            )

            final_fcst.fit(
                df_prepared.to_pandas(),
                id_col=id_col,
                time_col="ds",
                target_col="y",
                static_features=[],
            )

            input_example = df_prepared.head(5).to_pandas()
            signature = infer_signature(
                input_example, np.zeros(len(input_example), dtype=np.float32)
            )

            wrapped_model = MLForecastWrapper(
                model=final_fcst,
                variant_name=variant_name,
                is_ensemble=(variant_type == "ensemble"),
            )

            mlflow.pyfunc.log_model(
                python_model=wrapped_model,
                name=f"model_{variant_name}",
                registered_model_name=f"candidate_{variant_name}",
                signature=signature,
                input_example=input_example,
            )

            best_models[variant_name] = final_fcst

    return best_models


def train_neural_candidate_models(
    train_df: pl.DataFrame, parameters: Dict[str, Any]
) -> Any:
    # Trains deep learning models using NeuralForecast.
    import torch
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE, HuberLoss
    from neuralforecast.models import NHITS, TiDE

    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    target_col = parameters["target_column"]
    date_col = parameters["date_column"]
    id_col = parameters["id_column"]
    h = parameters["forecast_horizon"]
    nf_params = parameters["neuralforecast"]

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    df_prepared = (
        train_df.rename({date_col: "ds", target_col: "y"})
        .with_columns(
            [pl.col("ds").cast(pl.Datetime("us")), pl.lit("grid_1").alias(id_col)]
        )
        .unique(subset=["ds"], keep="last")
        .sort("ds")
    )

    feature_cols = [
        "decimal_hour",
        "weekday_int",
        "dayofyear_int",
        "month_int",
        "dayofmonth_int",
        "sin_hour",
        "cos_hour",
        "sin_weekday",
        "cos_weekday",
        "sin_dayofyear",
        "cos_dayofyear",
        "is_high_volatility_season",
    ]
    fourier_cols = [
        c
        for c in df_prepared.columns
        if c.startswith(("d_sin", "d_cos", "y_sin", "y_cos"))
    ]
    exog_cols = feature_cols + fourier_cols

    freq = parameters.get("freq", "15m")
    nf_freq = freq.replace("m", "min") if freq.endswith("m") else freq

    df_prepared = (
        df_prepared.drop_nulls(subset=exog_cols + ["y"])
        .upsample(time_column="ds", every=freq)
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .select([id_col, "ds", "y"] + exog_cols)
    )

    lookback = nf_params["lookback"]

    nhits_model = NHITS(
        h=h,
        input_size=lookback,
        futr_exog_list=exog_cols,
        loss=MAE(),
        scaler_type="robust",
        learning_rate=nf_params["learning_rate"],
        max_steps=nf_params["max_steps"],
        batch_size=nf_params["batch_size"],
        random_seed=nf_params["random_seed"],
    )

    tide_model = TiDE(
        h=h,
        input_size=lookback,
        futr_exog_list=exog_cols,
        hidden_size=256,
        decoder_output_dim=16,
        loss=HuberLoss(),
        scaler_type="robust",
        learning_rate=nf_params["learning_rate"],
        max_steps=nf_params["max_steps"],
        batch_size=nf_params["batch_size"],
        random_seed=nf_params["random_seed"],
    )

    nf = NeuralForecast(models=[nhits_model, tide_model], freq=nf_freq)

    with mlflow.start_run(run_name="neural_ensemble", nested=True):
        mlflow.log_params(nf_params)
        nf.fit(df=df_prepared.to_pandas())

        nf_path = "./nf_model"
        nf.save(path=nf_path, overwrite=True)

        input_example = df_prepared.head(5).to_pandas()
        signature = infer_signature(
            input_example, np.zeros(len(input_example), dtype=np.float32)
        )

        mlflow.pyfunc.log_model(
            python_model=NeuralForecastWrapper(),
            name="neural_model",
            registered_model_name="candidate_neural_ensemble",
            artifacts={"nf_model": nf_path},
            signature=signature,
            input_example=input_example,
        )
        shutil.rmtree(nf_path, ignore_errors=True)

    return nf


def evaluate_candidate_models(
    lgbm_models: Dict[str, Any],
    neural_model: Any,
    test_df: pl.DataFrame,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    # Evaluates candidate models on the testing dataset.
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    target_col = parameters["target_column"]
    date_col = parameters["date_column"]
    id_col = parameters["id_column"]

    df_test = (
        test_df.rename({date_col: "ds", target_col: "y"})
        .with_columns(
            [
                pl.col("ds").cast(pl.Datetime("us")),
                pl.lit("grid_1").alias(id_col),
            ]
        )
        .unique(subset=["ds"], keep="last")
        .sort("ds")
    )

    freq = parameters.get("freq", "15m")
    df_test = (
        df_test.upsample(time_column="ds", every=freq)
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
    )

    df_test_nf = df_test.clone()

    all_results = {}
    feature_cols = [
        "decimal_hour",
        "weekday_int",
        "dayofyear_int",
        "month_int",
        "dayofmonth_int",
        "sin_hour",
        "cos_hour",
        "sin_weekday",
        "cos_weekday",
        "sin_dayofyear",
        "cos_dayofyear",
        "is_high_volatility_season",
    ]
    fourier_cols = [
        c for c in df_test.columns if c.startswith(("d_sin", "d_cos", "y_sin", "y_cos"))
    ]
    exog_cols = feature_cols + fourier_cols

    with mlflow.start_run(run_name="candidate_evaluation", nested=True):
        for name, fcst in lgbm_models.items():
            future_exog = df_test.select([id_col, "ds"] + exog_cols)
            preds = fcst.predict(h=len(df_test), X_df=future_exog.to_pandas())
            preds_pl = pl.from_pandas(preds)

            if "LGBM_Huber" in preds_pl.columns and "LGBM_MAE" in preds_pl.columns:
                preds_pl = preds_pl.with_columns(
                    ((pl.col("LGBM_Huber") + pl.col("LGBM_MAE")) / 2).alias(name)
                )

            eval_df = df_test.select([id_col, "ds", "y"]).join(
                preds_pl, on=[id_col, "ds"], how="inner"
            )

            logger.info(f"Evaluation rows for {name}: {eval_df.height}")

            metrics = evaluate(
                eval_df,
                metrics=[rmse, mae, mape],
                models=[name],
                id_col=id_col,
                target_col="y",
            )

            for _, row in metrics.to_pandas().iterrows():
                mlflow.log_metric(f"{name}_{row['metric']}", row[name])
                all_results[f"{name}_{row['metric']}"] = {"value": row[name], "step": 0}

        future_exog_nf_pl = df_test_nf.select([id_col, "ds"] + exog_cols).to_pandas()

        future_df = neural_model.make_future_dataframe()

        future_exog_nf_pl["ds"] = pd.to_datetime(future_exog_nf_pl["ds"])
        future_df["ds"] = pd.to_datetime(future_df["ds"])
        future_exog_nf_pl[id_col] = future_exog_nf_pl[id_col].astype(str)
        future_df[id_col] = future_df[id_col].astype(str)

        future_exog_nf = future_df.merge(
            future_exog_nf_pl, on=[id_col, "ds"], how="left"
        )
        future_exog_nf = future_exog_nf.ffill().bfill().fillna(0)

        preds_nf = neural_model.predict(futr_df=future_exog_nf)
        preds_nf_pl = pl.from_pandas(preds_nf.reset_index()).with_columns(
            pl.col("ds").cast(pl.Datetime("us"))
        )

        if "NHITS" in preds_nf_pl.columns and "TiDE" in preds_nf_pl.columns:
            preds_nf_pl = preds_nf_pl.with_columns(
                ((pl.col("NHITS") + pl.col("TiDE")) / 2).alias("Neural_Ensemble")
            )
        else:
            available_model = [
                c for c in preds_nf_pl.columns if c not in [id_col, "ds"]
            ][0]
            preds_nf_pl = preds_nf_pl.with_columns(
                pl.col(available_model).alias("Neural_Ensemble")
            )

        eval_df_nf = df_test.select([id_col, "ds", "y"]).join(
            preds_nf_pl, on=[id_col, "ds"], how="inner"
        )

        logger.info(f"Evaluation rows for Neural_Ensemble: {eval_df_nf.height}")

        if eval_df_nf.height > 0:
            metrics_nf = evaluate(
                eval_df_nf,
                metrics=[rmse, mae, mape],
                models=["Neural_Ensemble"],
                id_col=id_col,
                target_col="y",
            )

            for _, row in metrics_nf.to_pandas().iterrows():
                mlflow.log_metric(
                    f"Neural_Ensemble_{row['metric']}", row["Neural_Ensemble"]
                )
                all_results[f"Neural_Ensemble_{row['metric']}"] = {
                    "value": row["Neural_Ensemble"],
                    "step": 0,
                }
        else:
            logger.warning(
                "Neural_Ensemble evaluation resulting in 0 rows! Check timestamp alignment."
            )

    return all_results
