from . import load_csv

# KBO
def load_kbo_batting(**kwargs):
    return load_csv("kbo_batting", **kwargs)

def load_kbo_pitching(**kwargs):
    return load_csv("kbo_pitching", **kwargs)

def load_kbo_fielding(**kwargs):
    return load_csv("kbo_fielding", **kwargs)

def load_kbo_people(**kwargs):
    return load_csv("kbo_people", **kwargs)

# NPB
def load_npb_batting(**kwargs):
    return load_csv("npb_batting", **kwargs)

def load_npb_pitching(**kwargs):
    return load_csv("npb_pitching", **kwargs)

def load_npb_fielding(**kwargs):
    return load_csv("npb_fielding", **kwargs)

def load_npb_people(**kwargs):
    return load_csv("npb_people", **kwargs)

__all__ = [
    "load_kbo_batting",
    "load_kbo_pitching",
    "load_kbo_fielding",
    "load_kbo_people",
    "load_npb_batting",
    "load_npb_pitching",
    "load_npb_fielding",
    "load_npb_people",
    ]
