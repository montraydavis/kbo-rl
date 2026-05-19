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

## B. Data Schemas

Files are written to `data/kbo/<year>/<table>.csv` (or `.parquet`). The `people` table has no year column and is written to `data/kbo/all/people.csv`.

### box-scores / batting (identical schema)

| Column | Type | Description |
| ------ | ---- | ----------- |
| `kbobID` | int | KBO batting record ID |
| `playerID` | int | Player identifier (join key to `people`) |
| `yearID` | int | Season year |
| `stint` | int | Stint number within season |
| `lgID` | str | League ID (always `KBO`) |
| `teamID` | str | Team abbreviation |
| `G` | int | Games played |
| `AB` | int | At-bats |
| `R` | int | Runs scored |
| `H` | int | Hits |
| `2B` | int | Doubles |
| `3B` | int | Triples |
| `HR` | int | Home runs |
| `RBI` | int | Runs batted in |
| `SB` | int | Stolen bases |
| `CS` | int | Caught stealing |
| `BB` | int | Walks |
| `SO` | int | Strikeouts |
| `GIDP` | int | Grounded into double plays |
| `AVG` | str | Batting average |
| `SLG` | str | Slugging percentage |
| `OBP` | str | On-base percentage |
| `PA` | int | Plate appearances |
| `TB` | int | Total bases |

### pitching

| Column | Type | Description |
| ------ | ---- | ----------- |
| `kbopID` | int | KBO pitching record ID |
| `playerID` | int | Player identifier (join key to `people`) |
| `yearID` | int | Season year |
| `stint` | int | Stint number within season |
| `lgID` | str | League ID (always `KBO`) |
| `teamID` | str | Team abbreviation |
| `G` | int | Games pitched |
| `CG` | int | Complete games |
| `SHO` | int | Shutouts |
| `W` | int | Wins |
| `L` | int | Losses |
| `SV` | int | Saves |
| `HLD` | int | Holds |
| `WPCT` | str | Winning percentage |
| `BFP` | int | Batters faced |
| `IPouts` | int | Innings pitched × 3 (outs recorded) |
| `H` | int | Hits allowed |
| `HR` | int | Home runs allowed |
| `BB` | int | Walks allowed |
| `HBP` | int | Hit batters |
| `SO` | int | Strikeouts |
| `R` | int | Runs allowed |
| `ER` | int | Earned runs allowed |
| `ERA` | str | Earned run average |

### fielding

| Column | Type | Description |
| ------ | ---- | ----------- |
| `kbofID` | int | KBO fielding record ID |
| `playerID` | int | Player identifier (join key to `people`) |
| `yearID` | int | Season year |
| `stint` | int | Stint number within season |
| `lgID` | str | League ID (always `KBO`) |
| `teamID` | str | Team abbreviation |
| `POS` | str | Position (e.g. `P`, `C`, `1B`, `SS`, `OF`) |
| `G` | int | Games played at position |
| `GS` | int | Games started at position |
| `InnOuts` | int | Innings played × 3 (outs) |
| `E` | int | Errors |
| `PKO` | int | Pickoff outs |
| `PO` | int | Putouts |
| `A` | int | Assists |
| `DP` | int | Double plays |
| `FPCT` | str | Fielding percentage |
| `PB` | int | Passed balls (catchers only) |
| `SB` | int | Stolen bases allowed (catchers only) |
| `CS` | int | Runners caught stealing (catchers only) |
| `CS_pct` | str | Caught-stealing percentage (catchers only; `-` when not applicable) |

### people (`data/kbo/all/people.csv`)

| Column | Type | Description |
| ------ | ---- | ----------- |
| `playerID` | int | Player identifier (primary key) |
| `birthYear` | int | Birth year |
| `birthMonth` | int | Birth month |
| `birthDay` | int | Birth day |
| `nationality` | str | Player nationality |
| `nameLast` | str | Last name (romanized) |
| `nameFirst` | str | First name (romanized) |
| `nameGiven` | str | Full name (romanized) |
| `weight` | int | Weight in pounds |
| `height` | int | Height in inches |
| `bats` | str | Batting handedness (`R`, `L`, `B`) |
| `throws` | str | Throwing handedness (`R`, `L`) |
| `position` | str | Primary position (nullable; 503 nulls in full dataset) |

---

## C. Python Script to Download Last 5 Years of KBO Data

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
