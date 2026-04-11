"""
Extracts power grid data from the KPTCL dashboard and saves it to a local file.
"""

import asyncio
import csv
import json
import logging
import os
from datetime import datetime

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOCAL_FILE_PATH = "kptcl_grid_data.csv"

CSV_HEADERS = [
    "ingestion_timestamp",
    "dashboard_timestamp",
    "frequency",
    "state_ui",
    "state_demand",
    "thermal",
    "thermal_ipp",
    "hydro",
    "wind",
    "solar",
    "other_ncep",
    "cgs_drawal",
    "pavagada_kspdcl",
    "bescom",
    "hescom",
    "gescom",
    "cesc",
    "mescom",
]


async def fetch_complete_table_kptcl():
    """
    Scrapes data from the specified URL and appends it to a CSV file.
    """
    url = "https://kptclsldc.in/Default.aspx"

    schema = {
        "name": "KPTCL_Complete_Table",
        "baseSelector": "body",
        "fields": [
            {"name": "dashboard_timestamp", "selector": "#Label6", "type": "text"},
            {"name": "frequency", "selector": "#Label1", "type": "text"},
            {"name": "state_ui", "selector": "#Label12", "type": "text"},
            {"name": "state_demand", "selector": "#Label5", "type": "text"},
            {"name": "thermal", "selector": "#lbl_thermal", "type": "text"},
            {"name": "thermal_ipp", "selector": "#lbl_thrmipp", "type": "text"},
            {"name": "hydro", "selector": "#lbl_hydro", "type": "text"},
            {"name": "wind", "selector": "#lbl_wind", "type": "text"},
            {"name": "solar", "selector": "#lbl_solar", "type": "text"},
            {"name": "other_ncep", "selector": "#lbl_other", "type": "text"},
            {"name": "cgs_drawal", "selector": "#Label3", "type": "text"},
            {"name": "pavagada_kspdcl", "selector": "#lblpvgslr", "type": "text"},
            {"name": "bescom", "selector": "#Label7", "type": "text"},
            {"name": "hescom", "selector": "#Label8", "type": "text"},
            {"name": "gescom", "selector": "#Label9", "type": "text"},
            {"name": "cesc", "selector": "#Label10", "type": "text"},
            {"name": "mescom", "selector": "#Label11", "type": "text"},
        ],
    }

    extraction_strategy = JsonCssExtractionStrategy(schema)

    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
        exclude_external_links=True,
    )

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config)

            if result.success and result.extracted_content:
                data = json.loads(result.extracted_content)

                if data:
                    metrics = data[0]
                    current_time = datetime.now()

                    def safe_int(val):
                        try:
                            return int(str(val).replace(",", "").strip()) if val else 0
                        except ValueError:
                            return 0

                    def safe_float(val):
                        try:
                            return (
                                float(str(val).replace(",", "").strip()) if val else 0.0
                            )
                        except ValueError:
                            return 0.0

                    row_data = {
                        "ingestion_timestamp": current_time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "dashboard_timestamp": metrics.get(
                            "dashboard_timestamp", ""
                        ).strip(),
                        "frequency": safe_float(metrics.get("frequency")),
                        "state_ui": safe_int(metrics.get("state_ui")),
                        "state_demand": safe_int(metrics.get("state_demand")),
                        "thermal": safe_int(metrics.get("thermal")),
                        "thermal_ipp": safe_int(metrics.get("thermal_ipp")),
                        "hydro": safe_int(metrics.get("hydro")),
                        "wind": safe_int(metrics.get("wind")),
                        "solar": safe_int(metrics.get("solar")),
                        "other_ncep": safe_int(metrics.get("other_ncep")),
                        "cgs_drawal": safe_int(metrics.get("cgs_drawal")),
                        "pavagada_kspdcl": safe_int(metrics.get("pavagada_kspdcl")),
                        "bescom": safe_int(metrics.get("bescom")),
                        "hescom": safe_int(metrics.get("hescom")),
                        "gescom": safe_int(metrics.get("gescom")),
                        "cesc": safe_int(metrics.get("cesc")),
                        "mescom": safe_int(metrics.get("mescom")),
                    }

                    file_exists = os.path.isfile(LOCAL_FILE_PATH)

                    with open(
                        LOCAL_FILE_PATH, mode="a", newline="", encoding="utf-8"
                    ) as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)

                        if not file_exists:
                            writer.writeheader()
                            logging.info(
                                f"Created new local file with headers: {LOCAL_FILE_PATH}"
                            )

                        writer.writerow(row_data)

                    logging.info(
                        f"[v] State Demand: {row_data['state_demand']} | Solar: {row_data['solar']} -> Appended to {LOCAL_FILE_PATH}"
                    )

                    return row_data
            else:
                logging.error(f"[x] Extraction Failed: {result.error_message}")

    except Exception as e:
        logging.error(f"[-] Pipeline Exception: {str(e)}")


async def local_daemon_loop():
    """
    Periodically triggers the data extraction process.
    """
    logging.info(
        "[@] Daemon Active: Orchestrating local state capture every 10 minutes..."
    )
    while True:
        await fetch_complete_table_kptcl()
        await asyncio.sleep(600)


if __name__ == "__main__":
    asyncio.run(local_daemon_loop())
