"""
Player metadata enrichment: age, handedness, foreign player flag.
"""
from __future__ import annotations

import pandas as pd

from src.data.loader import load_table


def load_people(data_root=None) -> pd.DataFrame:
    return load_table("people", data_root=data_root)


def enrich_with_people(df: pd.DataFrame,
                        people: pd.DataFrame | None = None,
                        year_col: str = "yearID") -> pd.DataFrame:
    """Left-join player metadata onto any DataFrame with a playerID column.

    Adds: age, bats, throws, position, is_foreign, nationality.
    """
    if people is None:
        people = load_people()

    if people.empty:
        return df

    meta = people[["playerID", "birthYear", "nationality",
                   "bats", "throws", "position"]].copy()
    meta = meta.drop_duplicates("playerID")

    merged = df.merge(meta, on="playerID", how="left")

    if year_col in merged.columns and "birthYear" in merged.columns:
        merged["age"] = merged[year_col] - merged["birthYear"]

    if "nationality" in merged.columns:
        merged["is_foreign"] = (merged["nationality"] != "KOR").astype(int)

    return merged
