# kbo-rl

Reinforcement learning KBO

## A. Validated KBO Data Sources

1. **KBO Data Portal – Collector (GitHub)** [S1]
   - **Data provided:** game results, schedules, and player statistics.
   - **Access method:** scraper/collector tooling with year-based extraction and multiple output formats.
   - **Strengths:** purpose-built for KBO extraction and supports year-based collection.
   - **Limitations:** scraper-based workflows can break if source pages change; CLI/runtime setup is required.
   - **Licensing considerations:** verify the repository license before redistribution; if MIT is declared in the repo, follow MIT terms.

2. **NK-Datasets (PyPI)** [S2]
   - **Data provided:** standardized KBO datasets including batting, pitching, fielding, and people tables.
   - **Access method:** Python loader functions (for example, `load_kbo_batting()`, `load_kbo_pitching()`).
   - **Strengths:** easiest local workflow for Python users and aligns well with pandas pipelines.
   - **Limitations:** depends on package maintenance and its published schema/version.
   - **Licensing considerations:** check package/repository metadata for license terms; use MIT terms where applicable.

3. **MYKBO Scraping (GitHub)** [S3]
   - **Data provided:** full KBO game data.
   - **Access method:** Scrapy spiders with optional Kafka/MariaDB pipeline integration.
   - **Strengths:** broad game-level coverage and production pipeline integration options.
   - **Limitations:** heavier infrastructure footprint when using Kafka/MariaDB components.
   - **Licensing considerations:** verify repository license and comply with MIT where applicable.

4. **KBO Data Portal Organization (GitHub)** [S4]
   - **Data provided:** collector, pipeline, and API-server repositories for ingestion/access.
   - **Access method:** organization-level repository suite (collector + pipeline + API server).
   - **Strengths:** end-to-end architecture options (collection through serving).
   - **Limitations:** multi-repo setup can require additional operational effort.
   - **Licensing considerations:** review each repo's license file and apply MIT requirements where applicable.

### Citations

- [S1] Provided search result: *"KBO Data Portal – Collector (GitHub): Provides scraping tools for game results, schedules, and player statistics. Supports year-based extraction and multiple formats."*
- [S2] Provided search result: *"NK-Datasets (PyPI): Provides standardized KBO datasets with pandas loaders (batting, pitching, fielding, people)."*
- [S3] Provided search result: *"MYKBO Scraping (GitHub): Scrapy-based project for scraping full KBO game data, with Kafka/MariaDB integration."*
- [S4] Provided search result: *"KBO Data Portal Organization (GitHub): Includes collector, pipeline, and API-server repositories for KBO data ingestion and access."*

## B. Python Script to Download Last 5 Years of KBO Data

A complete script is available at:

- `scripts/download_kbo_data.py`

It uses `nk-datasets` loaders (`load_kbo_batting`, `load_kbo_pitching`, `load_kbo_fielding`, `load_kbo_people`) and saves data by season for the most recent 5 years by default.

## C. Usage Instructions

### Quick start (PowerShell — recommended)

A single PowerShell script handles the entire setup-and-download workflow on both Windows and Linux/macOS:

```powershell
# Default: last 5 complete seasons, CSV output, saved to data/kbo
.\scripts\setup_and_download.ps1

# Custom year range, parquet format
.\scripts\setup_and_download.ps1 -StartYear 2021 -EndYear 2025 -Format parquet -OutputDir data/kbo

# Enable verbose/debug output
.\scripts\setup_and_download.ps1 -Verbose
```

The script will:
1. Detect a suitable `python` / `python3` interpreter on your `PATH`.
2. Create (or reuse) a `.venv` virtual environment in the repository root.
3. Install all dependencies listed in `requirements.txt`.
4. Run `scripts/download_kbo_data.py` and save the data files to `--OutputDir`.

> **Prerequisite:** PowerShell 5.1+ (Windows) or [PowerShell 7+](https://aka.ms/powershell) (cross-platform). Python 3.9+ must be installed and on your `PATH`.

---

### Manual setup (shell / bash)

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   > `pyarrow` (listed in `requirements.txt`) is needed only for `--format parquet`.

2. Run with default range (last 5 complete seasons, CSV output):

   ```bash
   python scripts/download_kbo_data.py
   ```

3. Run with custom years and Parquet output:

   ```bash
   python scripts/download_kbo_data.py \
     --start-year 2021 \
     --end-year 2025 \
     --format parquet \
     --output-dir data/kbo
   ```

4. Modify the year range:
   - Use `--start-year` and `--end-year`.
   - If omitted, the script computes `end_year = current_year - 1` and downloads 5 seasons (`end_year - 4` through `end_year`).
