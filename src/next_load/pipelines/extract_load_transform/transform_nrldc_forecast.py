"""
Transformation logic and data validation for NRLDC forecast Excel files.
Uses Polars for data manipulation and Pandera for schema enforcement.
"""

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
    """
    Schema definition for validating the metadata headers in the raw Excel files.
    """

    report_title: str = Field(eq="Northern Regional Load Despatch Centre (NRLDC)")
    date_value: str = Field(nullable=True)
    period: str = Field(eq="Period")
    intra_day_demand: str = Field(eq="Intraday Forcasted Demand (B)")
    actual_demand: str = Field(eq="Actual Demand (C)")


@app.class_definition
class RawNRLDCForecastTabularDataSchema(DataFrameModel):
    """
    Schema definition for validating the tabular demand data in the raw Excel files.
    """

    period: str = Field(str_matches=r"^\d{2}:\d{2}\s*-\s*\d{2}:\d{2}$")
    nrldc_intraday_forecasted_demand_mw: str = Field(
        str_matches=r"^(-?\d+(\.\d+)?)?$", nullable=True
    )
    actual_demand_mw: str = Field(str_matches=r"^(-?\d+(\.\d+)?)?$", nullable=True)


@app.class_definition
class TransformedRawNRLDCForecastDataSchema(DataFrameModel):
    """
    Schema definition for the final transformed and unified data table.
    """

    date: datetime.datetime
    period: str = Field(str_matches=r"^\d{2}:\d{2}\s*-\s*\d{2}:\d{2}$")
    actual_demand_mw: float = Field(nullable=True)
    nrldc_intraday_forecasted_demand_mw: float = Field(nullable=True)

    class Config:
        strict = True
        ordered = True


@app.function
def validate_raw_excel_dataframe(df: pl.DataFrame, partition_key: str) -> bool:
    """
    Validates the structure of a raw Excel file before transformation.
    Checks both metadata cells and tabular data alignment.
    """
    try:
        dataset = df.select(pl.all().cast(pl.Utf8)).fill_null("")

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

        raw_tabular_data = dataset.slice(5).select(
            [dataset.columns[i] for i in [1, 3, 4]]
        )
        raw_tabular_data = raw_tabular_data.rename(
            {
                raw_tabular_data.columns[0]: "period",
                raw_tabular_data.columns[1]: "nrldc_intraday_forecasted_demand_mw",
                raw_tabular_data.columns[2]: "actual_demand_mw",
            }
        )

        RawNRLDCForecastTabularDataSchema.validate(raw_tabular_data)
        return True
    except Exception as e:
        logger.warning(f"Raw Validation Failed for {partition_key}: {e}")
        return False


@app.function
def transform_single_partition(
    df: pl.DataFrame, partition_key: str
) -> Optional[pl.DataFrame]:
    """
    Transforms a single partition of raw NRLDC data into a cleaned format.
    Extracts the date from the partition key or file content and casts demand values to floats.
    """
    try:
        dataset = df.select(pl.all().cast(pl.Utf8)).fill_null("")

        match = re.search(r"(\d{2})-(\d{2})-(\d{4})", partition_key)
        if match:
            day, month, year = match.groups()
            file_date = datetime.datetime.strptime(f"{day}-{month}-{year}", "%d-%m-%Y")
        else:
            raw_date = dataset.item(2, 1).strip()
            try:
                file_date = datetime.datetime.strptime(raw_date, "%d-%b-%Y")
            except ValueError:
                if "-Sept-" in raw_date:
                    raw_date = raw_date.replace("-Sept-", "-Sep-")
                    file_date = datetime.datetime.strptime(raw_date, "%d-%b-%Y")
                else:
                    raise

        raw_tabular_data = dataset.slice(5).select(
            [dataset.columns[i] for i in [1, 3, 4]]
        )
        raw_tabular_data = raw_tabular_data.rename(
            {
                raw_tabular_data.columns[0]: "period",
                raw_tabular_data.columns[1]: "nrldc_intraday_forecasted_demand_mw",
                raw_tabular_data.columns[2]: "actual_demand_mw",
            }
        )

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
def validate_transformed_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Validates the structure and data types of the transformed DataFrame.
    Ensures the final output matches the TransformedRawNRLDCForecastDataSchema.
    """
    if df["date"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("date").cast(pl.Datetime))

    TransformedRawNRLDCForecastDataSchema.validate(df)
    return df


if __name__ == "__main__":
    app.run()
