from importlib.resources import files, as_file
from pathlib import Path
import pandas as pd

def list_csv():
    base = files(__package__).joinpath("data")
    with as_file(base) as base_path:
        return sorted(p.name for p in Path(base_path).glob("*.csv"))

def load_csv(csv_name, **kwargs):
    name = csv_name if csv_name.endswith(".csv") else f"{csv_name}.csv"
    resource = files(__package__).joinpath("data").joinpath(name)
    with as_file(resource) as path:
        return pd.read_csv(path, **kwargs)

__all__ = ["list_csv", "load_csv"]
