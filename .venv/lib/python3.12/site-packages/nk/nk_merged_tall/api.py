from . import load_csv

# NK tall
def load_nk_playerstats_CL (**kwargs):
    return load_csv("nk_playerstats_CL", **kwargs)

def load_nk_playerstats_JBL (**kwargs):
    return load_csv("nk_playerstats_JBL", **kwargs)

def load_nk_playerstats_KBO (**kwargs):
    return load_csv("nk_playerstats_KBO", **kwargs)

def load_nk_playerstats_PL (**kwargs):
    return load_csv("nk_playerstats_PL", **kwargs)

def load_nk_league(**kwargs):
    return load_csv("nk_league", **kwargs)

def load_nk_team(**kwargs):
    return load_csv("nk_team", **kwargs)

def load_nk_stats(**kwargs):
    return load_csv("nk_stats", **kwargs)

__all__ = [
    "load_nk_playerstats_CL",
    "load_nk_playerstats_JBL",
    "load_nk_playerstats_KBO",
    "load_nk_playerstats_PL",
    "load_nk_stats",
    "load_nk_league",
    "load_nk_team",
    ]
