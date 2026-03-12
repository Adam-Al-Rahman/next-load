import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import datetime
    import json

    import marimo as mo
    from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
    from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

    from next_load.pipelines.extract_load_transform.elt_config import ScraperConfig


@app.function
async def extract_nrldc_data(
    config: ScraperConfig,
    nrldc_base_url="https://nrldc.in",
):
    """Async generator that crawls NRLDC and yields raw data month by month."""
    base_url = f"{nrldc_base_url}/forecast/intra-day-forecast"
    session_id = "intra_day_forecast"
    current_time = datetime.datetime.now()
    history_years = [
        str(year) for year in range(config.start_year, current_time.year + 1)
    ]

    extract_month_schema = {
        "name": "MonthExtractor",
        "baseSelector": "div.accordion-item",
        "fields": [
            {
                "name": "month",
                "selector": "button[data-foldername]",
                "type": "attribute",
                "attribute": "data-foldername",
            }
        ],
    }

    extract_month_data_schema = {
        "name": "NRLDC Files",
        "baseSelector": "table tbody tr",
        "fields": [
            {
                "name": "file_name",
                "selector": "td:nth-child(1)",
                "type": "text",
            },
            {"name": "size", "selector": "td:nth-child(2)", "type": "text"},
            {
                "name": "file_date",
                "selector": "td:nth-child(3)",
                "type": "text",
            },
            {
                "name": "download_link",
                "selector": "td:nth-child(4) a",
                "type": "attribute",
                "attribute": "href",
            },
        ],
    }

    async with AsyncWebCrawler() as crawler:
        print("Loading root directory...")
        await crawler.arun(
            url=base_url,
            cache_mode=CacheMode.BYPASS,
            config=CrawlerRunConfig(session_id=session_id),
        )

        target_years = (
            [current_time.strftime("%Y")] if config.live_extraction else history_years
        )

        for year in target_years:
            print(f"\n{'=' * 40}\n--- Processing Year: {year} ---\n{'=' * 40}")
            year_config = CrawlerRunConfig(
                js_code=f"document.querySelector('button[data-foldername=\"{year}\"]')?.click();",
                cache_mode=CacheMode.BYPASS,
                wait_for=f"js:() => document.querySelector('.dir-breadcrumbs')?.innerText.includes('{year}')",
                session_id=session_id,
                js_only=True,
            )
            await crawler.arun(url=base_url, config=year_config)

            extracted_month_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                delay_before_return_html=1.0,
                extraction_strategy=JsonCssExtractionStrategy(extract_month_schema),
                session_id=session_id,
                js_only=True,
            )
            mentioned_month_result = await crawler.arun(
                url=base_url, config=extracted_month_config
            )

            months = []
            if (
                mentioned_month_result.success
                and mentioned_month_result.extracted_content
            ):
                extracted_data = json.loads(mentioned_month_result.extracted_content)
                months = [
                    item["month"]
                    for item in extracted_data
                    if item.get("month") and item["month"] not in history_years
                ]
            else:
                print(f"Failed to extract months for {year}")
                continue

            target_months = (
                [current_time.strftime("%b")] if config.live_extraction else months
            )

            for month in sorted(
                target_months,
                key=lambda m: datetime.datetime.strptime(m[:3], "%b").month,
            ):
                print(f"\n  -> Navigating to Month: {month}")
                month_config = CrawlerRunConfig(
                    js_code=f"document.querySelector('button[data-foldername=\"{month}\"]')?.click();",
                    cache_mode=CacheMode.BYPASS,
                    wait_for=f"js:() => document.querySelector('.dir-breadcrumbs')?.innerText.includes('{month}')",
                    js_only=True,
                    session_id=session_id,
                )
                await crawler.arun(url=base_url, config=month_config)

                month_pagination_config = CrawlerRunConfig(
                    js_code="""
                    const selectBox = document.getElementById('dt-length-0');
                    if (selectBox) {
                        selectBox.value = '50';
                        selectBox.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    """,
                    cache_mode=CacheMode.BYPASS,
                    delay_before_return_html=2.0,
                    session_id=session_id,
                    js_only=True,
                    extraction_strategy=JsonCssExtractionStrategy(
                        extract_month_data_schema
                    ),
                )
                month_result = await crawler.arun(
                    url=base_url, config=month_pagination_config
                )

                if month_result.success and month_result.extracted_content:
                    raw_data = json.loads(month_result.extracted_content)

                    # Normalize links
                    for item in raw_data:
                        link = item.get("download_link", "")
                        if link and link.startswith("/"):
                            item["download_link"] = f"{nrldc_base_url}{link}"
                        elif link and not link.startswith("http"):
                            item["download_link"] = f"{nrldc_base_url}/{link}"

                    yield raw_data, year, month

                # Navigate BACK
                back_to_year_config = CrawlerRunConfig(
                    js_code=f"""
                        const breadcrumbLinks = Array.from(document.querySelectorAll('.dir-breadcrumbs a'));
                        const targetLink = breadcrumbLinks.find(a => a.textContent.trim() === '{year}');
                        if (targetLink) {{ targetLink.click(); }}
                    """,
                    cache_mode=CacheMode.BYPASS,
                    wait_for=f"js:() => !document.querySelector('.dir-breadcrumbs')?.innerText.includes('{month}')",
                    js_only=True,
                    session_id=session_id,
                )
                await crawler.arun(url=base_url, config=back_to_year_config)

            print(f"\nFinished {year}. Navigating back to root directory...")
            exist_current_year_config = CrawlerRunConfig(
                js_code="""
                    const breadcrumbLinks = Array.from(document.querySelectorAll('.dir-breadcrumbs a'));
                    const targetLink = breadcrumbLinks.find(a => a.textContent.trim() === 'Intra Day Forecast');
                    if (targetLink) { targetLink.click(); }
                """,
                cache_mode=CacheMode.BYPASS,
                wait_for=f"js:() => !document.querySelector('.dir-breadcrumbs')?.innerText.includes('{year}')",
                js_only=True,
                session_id=session_id,
            )
            await crawler.arun(url=base_url, config=exist_current_year_config)


if __name__ == "__main__":
    app.run()
