"""Shared CSV validation for event-triggered maintenance plots."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def read_table(path, required=(), optional=False, unique=None):
    path=Path(path)
    if not path.exists():
        if optional: return pd.DataFrame(columns=required)
        raise FileNotFoundError(f'Missing result file: {path}')
    try: df=pd.read_csv(path)
    except pd.errors.EmptyDataError:
        if optional: return pd.DataFrame(columns=required)
        raise ValueError(f'Empty result file: {path.name}')
    missing=set(required)-set(df.columns)
    if missing: raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
    if unique and df.duplicated(unique).any():
        raise ValueError(f'{path.name}: duplicate keys {unique}')
    return df

def bool_values(series):
    def convert(x):
        if pd.isna(x): raise ValueError('Missing boolean value')
        if isinstance(x,(bool,np.bool_)): return bool(x)
        if str(x).strip().lower() in ('true','1','1.0'): return True
        if str(x).strip().lower() in ('false','0','0.0'): return False
        raise ValueError(f'Invalid boolean: {x!r}')
    return series.map(convert).astype(bool)

def config(directory):
    path=Path(directory)/'config.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

def numeric(df, columns, allow_missing=False):
    for col in columns:
        df[col]=pd.to_numeric(df[col],errors='raise')
        values=df[col].dropna() if allow_missing else df[col]
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f'Non-finite {col}')
        if not allow_missing and df[col].isna().any(): raise ValueError(f'Missing {col}')
    return df
