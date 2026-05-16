"""
Collections (flat layout):
- kbo_npb
- kbo_npb_mlb
- nk_merged
- nk_merged_tall

Each subpackage exposes:
  - list_csv()
  - load_csv(csv_name, **read_csv_kwargs)
"""

from __future__ import annotations
import importlib

__version__ = "0.1.0"

_COLLECTION_PACKAGES = {
    "kbo_npb": "kbo_npb",
    "kbo_npb_mlb": "kbo_npb_mlb",
    "nk_merged": "nk_merged",
    "nk_merged_tall": "nk_merged_tall",
}

def show_collections() -> list[str]:
    return list(_COLLECTION_PACKAGES.keys())

def _get_pkg_module(collection: str):
    try:
        pkg_name = _COLLECTION_PACKAGES[collection]
    except KeyError:
        raise KeyError(f"Unknown collection: {collection}. Available: {show_collections()}")
    # Import subpackages relative to 'mnk'
    return importlib.import_module(f".{pkg_name}", __name__)

def list_csv(collection: str):
    return _get_pkg_module(collection).list_csv()

def load_csv(collection: str, csv_name: str, **read_csv_kwargs):
    return _get_pkg_module(collection).load_csv(csv_name, **read_csv_kwargs)

# KBO+NPB (originals)
from .kbo_npb.api import (
    load_kbo_batting,
    load_kbo_pitching,
    load_kbo_fielding,
    load_kbo_people,
    load_npb_batting,
    load_npb_pitching,
    load_npb_fielding,
    load_npb_people,
)

# KBO+NPB+MLB (aligned columns across 3 leagues)
from .kbo_npb_mlb.api import (
    load_kbo_batting as load_kbo_batting_aligned,
    load_kbo_pitching as load_kbo_pitching_aligned,
    load_kbo_fielding as load_kbo_fielding_aligned,
    load_kbo_people as load_kbo_people_aligned,
    load_npb_batting as load_npb_batting_aligned,
    load_npb_pitching as load_npb_pitching_aligned,
    load_npb_fielding as load_npb_fielding_aligned,
    load_npb_people as load_npb_people_aligned,
    load_mlb_batting as load_mlb_batting_aligned,
    load_mlb_pitching as load_mlb_pitching_aligned,
    load_mlb_fielding as load_mlb_fielding_aligned,
    load_mlb_people as load_mlb_people_aligned,
)

# NK_merged (wide)
from .nk_merged.api import (
    load_nk_batting,
    load_nk_pitching,
    load_nk_fielding,
    load_nk_people,
    load_nk_league,
    load_nk_team,
    load_nk_merged_people,
)

# MNK_merged (tall)
from .nk_merged_tall.api import (
    load_nk_playerstats_CL,
    load_nk_playerstats_JBL,
    load_nk_playerstats_KBO,
    load_nk_playerstats_PL,
    load_nk_league as load_nk_league_tall,
    load_nk_team as load_nk_team_tall,
    load_nk_stats,
)

__all__ = [
    # interface
    "show_collections", "list_csv", "load_csv",

    # KBO/NPB originals
    "load_kbo_batting", 
    "load_kbo_pitching", 
    "load_kbo_fielding", 
    "load_kbo_people",
    "load_npb_batting", 
    "load_npb_pitching",
    "load_npb_fielding", 
    "load_npb_people",

    # aligned (3-league)
    "load_kbo_batting_aligned", 
    "load_kbo_pitching_aligned",
    "load_kbo_fielding_aligned", 
    "load_kbo_people_aligned",
    "load_npb_batting_aligned", 
    "load_npb_pitching_aligned", 
    "load_npb_fielding_aligned", 
    "load_npb_people_aligned",
    "load_mlb_batting_aligned", 
    "load_mlb_pitching_aligned", 
    "load_mlb_fielding_aligned", 
    "load_mlb_people_aligned",

    # NK merged
    "load_nk_batting", 
    "load_nk_pitching", 
    "load_nk_fielding", 
    "load_nk_people",
    "load_nk_league", 
    "load_nk_team", 
    "load_nk_merged_people",

    # NK merged tall
    "load_nk_playerstats_CL",
    "load_nk_playerstats_JBL",
    "load_nk_playerstats_KBO",
    "load_nk_playerstats_PL",
    "load_nk_league_tall", 
    "load_nk_team_tall", 
    "load_nk_stats",

    "__version__",
]
