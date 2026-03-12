import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    from dataclasses import dataclass

    import marimo as mo


@app.class_definition
@dataclass
class S3Config:
    endpoint_url: str
    region_name: str
    bucket_name: str
    access_key: str
    secret_key: str


@app.class_definition
@dataclass
class ScraperConfig:
    live_extraction: bool
    start_year: int
    s3: S3Config


if __name__ == "__main__":
    app.run()
