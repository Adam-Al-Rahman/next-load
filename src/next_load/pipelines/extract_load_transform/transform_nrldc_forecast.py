# /// script
# dependencies = [
#     "marimo",
#     "polars>=1.38.1",
#     "pandera[polars]>=0.29.0",
#     "python-calamine>=0.2.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import datetime
    import logging
    import re
    from typing import Optional

    import pandas as pd
    import pandera as pa
    import polars as pl
    from pandera.polars import DataFrameModel, Field

    logger = logging.getLogger(__name__)


@app.class_definition
class RawNRLDCForecastMetadataSchema(DataFrameModel):
    """Data Contract: Raw Excel Metadata structure."""

    report_title: str = Field(eq="Northern Regional Load Despatch Centre (NRLDC)")
    date_value: str = Field(nullable=True)
    period: str = Field(eq="Period")
    intra_day_demand: str = Field(eq="Intraday Forcasted Demand (B)")
    actual_demand: str = Field(eq="Actual Demand (C)")


@app.class_definition
class RawNRLDCForecastTabularDataSchema(DataFrameModel):
    """Data Contract: Raw Excel Tabular structure."""

    period: str = Field(str_matches=r"^\d{2}:\d{2}\s*-\s*\d{2}:\d{2}$")
    nrldc_intraday_forecasted_demand_mw: str = Field(
        str_matches=r"^(-?\d+(\.\d+)?)?$", nullable=True
    )
    actual_demand_mw: str = Field(str_matches=r"^(-?\d+(\.\d+)?)?$", nullable=True)


@app.class_definition
class TransformedRawNRLDCForecastDataSchema(DataFrameModel):
    """Data Contract: Final transformed unified table."""

    # We use DateTime to accommodate the automatic casting when moving between Pandas/Polars
    date: datetime.datetime
    period: str = Field(str_matches=r"^\d{2}:\d{2}\s*-\s*\d{2}:\d{2}$")
    actual_demand_mw: float = Field(nullable=True)
    nrldc_intraday_forecasted_demand_mw: float = Field(nullable=True)

    class Config:
        strict = True
        ordered = True


@app.function
def validate_raw_excel_dataframe(df: pd.DataFrame, partition_key: str) -> bool:
    """
    Node-level logic for 'Before Transformation' contract validation.
    """
    try:
        dataset = pl.from_pandas(df).select(pl.all().cast(pl.Utf8)).fill_null("")

        # Validate Metadata Cells
        metadata_cells = pl.DataFrame(
            {
                "report_title": [str(dataset.item(0, 0)).strip()],
                "date_value": [str(dataset.item(2, 1)).strip()],
                "period": [str(dataset.item(4, 1)).strip()],
                "intra_day_demand": [str(dataset.item(3, 3)).strip()],
                "actual_demand": [str(dataset.item(3, 4)).strip()],
            }
        ).select(pl.all().cast(pl.Utf8))

        RawNRLDCForecastMetadataSchema.validate(metadata_cells)

        # Validate Tabular
        raw_tabular_data = dataset.slice(5).select(
            [dataset.columns[i] for i in [1, 3, 4]]
        )
        raw_tabular_data = raw_tabular_data.rename(
            dict(
                zip(
                    raw_tabular_data.columns,
                    [
                        "period",
                        "nrldc_intraday_forecasted_demand_mw",
                        "actual_demand_mw",
                    ],
                )
            )
        )

        RawNRLDCForecastTabularDataSchema.validate(raw_tabular_data)
        return True
    except Exception as e:
        logger.warning(f"Raw Validation Failed for {partition_key}: {e}")
        return False


@app.function
def transform_single_partition(
    df: pd.DataFrame, partition_key: str
) -> Optional[pl.DataFrame]:
    """
    Core transformation logic.
    """
    try:
        dataset = pl.from_pandas(df).select(pl.all().cast(pl.Utf8)).fill_null("")

        # Parse Date from partition_key (filename)
        # Expected format in filename: dd-mm-yyyy (e.g., 14-03-2026)
        match = re.search(r"(\d{2})-(\d{2})-(\d{4})", partition_key)
        if match:
            day, month, year = match.groups()
            file_date = datetime.datetime.strptime(f"{day}-{month}-{year}", "%d-%m-%Y")
        else:
            # Fallback to Excel cell if filename date is missing
            logger.warning(
                f"Could not find dd-mm-yyyy date in partition_key: {partition_key}. Falling back to Excel cell."
            )
            raw_date = dataset.item(2, 1).strip()
            try:
                file_date = datetime.datetime.strptime(raw_date, "%d-%b-%Y")
            except ValueError:
                if "-Sept-" in raw_date:
                    raw_date = raw_date.replace("-Sept-", "-Sep-")
                    file_date = datetime.datetime.strptime(raw_date, "%d-%b-%Y")
                else:
                    raise

        # Extract Tabular
        raw_tabular_data = dataset.slice(5).select(
            [dataset.columns[i] for i in [1, 3, 4]]
        )
        raw_tabular_data = raw_tabular_data.rename(
            dict(
                zip(
                    raw_tabular_data.columns,
                    [
                        "period",
                        "nrldc_intraday_forecasted_demand_mw",
                        "actual_demand_mw",
                    ],
                )
            )
        )

        # Cast and Add Date
        return raw_tabular_data.with_columns(
            pl.col("nrldc_intraday_forecasted_demand_mw").cast(
                pl.Float64, strict=False
            ),
            pl.col("actual_demand_mw").cast(pl.Float64, strict=False),
            pl.lit(file_date).alias("date"),
        ).select(
            [
                "date",
                "period",
                "actual_demand_mw",
                "nrldc_intraday_forecasted_demand_mw",
            ]
        )
    except Exception as e:
        logger.error(f"Transformation Failed for {partition_key}: {e}")
        return None


@app.function
def validate_transformed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Node-level logic for 'After Transformation' contract validation.
    """
    # Ensure column 'date' is actually datetime for Pandera validation
    df["date"] = pd.to_datetime(df["date"])

    pl_df = pl.from_pandas(df)
    TransformedRawNRLDCForecastDataSchema.validate(pl_df)
    return df


if __name__ == "__main__":
    app.run()
