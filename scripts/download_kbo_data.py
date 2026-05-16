#!/usr/bin/env python3
"""Download the most recent KBO seasons using nk-datasets loaders."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

DataFrame = Any
Loader = Callable[[], DataFrame]

YEAR_COLUMNS: Tuple[str, ...] = (
    "year",
    "season",
    "Season",
    "Year",
    "SEASON",
)


def build_dataset_loaders() -> Dict[str, Loader]:
    """Import nk-datasets loaders at runtime."""
    # Import lazily so --help works even when dependencies are not yet installed.
    try:
        from nk_datasets import (
            load_kbo_batting,
            load_kbo_fielding,
            load_kbo_people,
            load_kbo_pitching,
        )
    except ImportError as exc:
        raise RuntimeError(
            "nk-datasets is required. Install it with `pip install nk-datasets`."
        ) from exc

    return {
        "batting": load_kbo_batting,
        "pitching": load_kbo_pitching,
        "fielding": load_kbo_fielding,
        "people": load_kbo_people,
    }


def configure_logging(verbose: bool) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def compute_year_range(start_year: Optional[int], end_year: Optional[int]) -> list[int]:
    """Return a 5-year window when explicit bounds are not provided."""
    default_end = datetime.now(timezone.utc).year - 1
    final_end = end_year if end_year is not None else default_end
    final_start = start_year if start_year is not None else final_end - 4

    if final_start > final_end:
        raise ValueError("start-year must be less than or equal to end-year")

    return list(range(final_start, final_end + 1))


def find_year_column(df: DataFrame) -> Optional[str]:
    """Find a likely season column in the loaded DataFrame."""
    for col in YEAR_COLUMNS:
        if col in df.columns:
            return col
    return None


def filter_for_year(df: DataFrame, year_column: str, year: int) -> DataFrame:
    """Filter records for a single season."""
    return df[df[year_column] == year].copy()


def write_frame(df: DataFrame, output_path: Path, file_format: str) -> None:
    """Persist a DataFrame in the selected format."""
    if file_format == "csv":
        df.to_csv(output_path, index=False)
    elif file_format == "parquet":
        df.to_parquet(output_path, index=False)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unsupported format: {file_format}")


def process_dataset(
    dataset_name: str,
    loader: Loader,
    years: Iterable[int],
    output_dir: Path,
    file_format: str,
) -> None:
    """Load, filter, and save one dataset for each requested season."""
    logging.info("Loading dataset: %s", dataset_name)

    try:
        frame = loader()
    except Exception as exc:  # broad catch to keep other datasets running
        logging.exception("Failed to load '%s': %s", dataset_name, exc)
        return

    year_column = find_year_column(frame)
    if year_column is None:
        # Fallback path when schema does not expose a recognizable season column.
        logging.warning(
            "No recognized year column found for '%s'. Saving complete dataset once.",
            dataset_name,
        )
        extension = "csv" if file_format == "csv" else "parquet"
        output_path = output_dir / f"kbo_{dataset_name}_all_years.{extension}"
        write_frame(frame, output_path, file_format)
        logging.info("Saved %s rows to %s", len(frame), output_path)
        return

    for year in years:
        # Save one file per season to make downstream training slices easier.
        year_frame = filter_for_year(frame, year_column, year)
        if year_frame.empty:
            logging.warning("No rows for %s in %s", dataset_name, year)
            continue

        extension = "csv" if file_format == "csv" else "parquet"
        output_path = output_dir / f"kbo_{dataset_name}_{year}.{extension}"
        write_frame(year_frame, output_path, file_format)
        logging.info("Saved %s rows to %s", len(year_frame), output_path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Download KBO data for the last 5 seasons using nk-datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/kbo",
        help="Directory to write output files (default: data/kbo)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Start season year (default: end-year - 4)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End season year (default: current year - 1)",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
        help="Output file format (default: csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        years = compute_year_range(args.start_year, args.end_year)
    except ValueError as exc:
        logging.error("Invalid year range: %s", exc)
        return 2

    output_dir = Path(args.output_dir)
    # Ensure target directory exists before writing files.
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving files to %s", output_dir.resolve())
    logging.info("Processing seasons: %s", years)

    try:
        dataset_loaders = build_dataset_loaders()
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 1

    # Process each table independently so one failure doesn't block others.
    for dataset_name, loader in dataset_loaders.items():
        try:
            process_dataset(dataset_name, loader, years, output_dir, args.format)
        except Exception as exc:  # safeguard so one failure doesn't halt all output
            logging.exception("Unexpected error while processing '%s': %s", dataset_name, exc)

    logging.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
