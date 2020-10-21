from typing import Any, Dict, Optional

import pandas as pd


def fill_dt_all(df, ts_id=["category", "cost_center"], **kwargs) -> pd.DataFrame:
    ts = df.groupby(ts_id, as_index=False, group_keys=False).apply(fill_dt, **kwargs)
    return ts


def fill_dt(
    df,
    dates=pd.date_range("2017-01-01", "2019-12-31", freq="D"),
    freq="D",
    fillna_kwargs: Optional[Dict[str, Any]] = None,
    resample: str = "sum",
    resample_kwargs={},
) -> pd.DataFrame:
    """Make sure each timeseries has contiguous days, then optionally downsampled.

    Dataframe must either has column "x", or indexed with "x" only.

    Arguments:
        dates (pd.DatetimeIndex or Tuple[str, str, str]): new timestamp index. A bit complicated, so please pay attention.
            - If pd.DatetimeIndex, then this is typically created by pd.date_range("yyyy-mm-dd", "yyyy-mm-dd", freq="D")).
            - If Tuple[str, str, str], then dates[0] == "yyyy-mm-dd" or "min", dates[1] == "yyyy-mm-dd" or "max", and
              dates[2] = frequency of the original index.

        freq (str): after df is reindexed, further downsample to this freq.
        fillna_kwargs (Dict[str, Any], optional):  Use None for demand, dict(method='ffill') for price. Defaults to None.
        resample_fn (str, optional): Use "sum" for demand, "max" for price curves. Defaults to "sum".
        resample_kwargs (dict, optional): [description]. Defaults to {}.

    Returns a dataframe indexed by X.
    """
    X = "x"
    if X in df.columns:
        df = df.set_index(X).copy()

    if not isinstance(dates, pd.DatetimeIndex):
        # Must be Tuple[str, str, str]
        start, end, freq_ori = dates
        if start == "min":
            start = df.index.min()
        if end == "max":
            end = df.index.max()
        dates = pd.date_range(start, end, freq=freq_ori)

    # Pre-compute nan-filler.
    # - number columns: fillna with 0.0
    # - non-number columns: fillna with the 1st non-NA
    nan_repl = df.iloc[0:1, :].reset_index(drop=True)
    for i in range(nan_repl.shape[1]):
        if pd.api.types.is_numeric_dtype(type(nan_repl.iloc[0, i])):
            nan_repl.iloc[0, i] = 0.0
    nan_repl = {k: v[0] for k, v in nan_repl.to_dict().items()}

    # Re-index timeseries to contiguous days
    if fillna_kwargs is None:
        daily_binpat = df.reindex(dates).fillna(value=nan_repl)
    else:
        daily_binpat = df.reindex(dates).fillna(**fillna_kwargs)
        # For non-number columns, always use the value from the first row
        col_to_refill = {k: v for k, v in nan_repl.items() if not pd.api.types.is_numeric_dtype(type(v))}
        for k, v in col_to_refill.items():
            daily_binpat[k] = v
    daily_binpat.index.name = df.index.name

    if freq == "D":
        return daily_binpat.reset_index()

    # Downsample y if necessary.
    downsampled_binpat = daily_binpat.resample(freq)
    resample_fn = getattr(downsampled_binpat, resample)
    downsampled_binpat = resample_fn(**resample_kwargs)

    # Resample will drop non-number columns, so we need to restore them.
    col_to_reinsert = {k: v for k, v in nan_repl.items() if not pd.api.types.is_numeric_dtype(type(v))}
    for k, v in col_to_reinsert.items():
        downsampled_binpat[k] = v

    return downsampled_binpat.reset_index()
