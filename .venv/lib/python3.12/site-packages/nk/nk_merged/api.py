from __future__ import annotations
from . import load_csv, list_csv as _base_list_csv  # from nk_merged/__init__.py

__all__ = [
    "list_csv", "load_csv",
    "load_nk_batting", "load_nk_pitching", "load_nk_fielding",
    "load_nk_people", "load_nk_league", "load_nk_team",
    "load_nk_merged_people",
]

def list_csv():
    return _base_list_csv()

# call the package-level load_csv
def load_nk_batting(**kw):        
    return load_csv("nk_batting", **kw)
def load_nk_pitching(**kw):       
    return load_csv("nk_pitching", **kw)
def load_nk_fielding(**kw):       
    return load_csv("nk_fielding", **kw)
def load_nk_people(**kw):         
    return load_csv("nk_people", **kw)
def load_nk_league(**kw):         
    return load_csv("nk_league", **kw)
def load_nk_team(**kw):           
    return load_csv("nk_team", **kw)
def load_nk_merged_people(**kw):  
    return load_csv("merged_player_records", **kw)
