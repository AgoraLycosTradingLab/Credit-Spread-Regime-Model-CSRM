"""
Shared utility functions for the Credit Spread Regime Model (CSRM).

This module contains small, reusable helpers that are generic enough to be
shared across validation, transforms, features, scoring, interpretation,
tests, and example scripts.

Design goals
------------
- deterministic
- side-effect free
- minimal and reusable
- no domain logic duplication
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def clip_series(
    series: pd.Series,
    lower: float,
    upper: float,
) -> pd.Series:
    """
    Clip a pandas Series to the provided bounds.

    Parameters
    ----------
    series : pd.Series
        Input series.
    lower : float
        Lower clip bound.
    upper : float
        Upper clip bound.

    Returns
    -------
    pd.Series
        Clipped series.
    """
    return series.clip(lower=lower, upper=upper)


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Safely divide one aligned Series by another.

    Division by zero is converted to NaN.

    Parameters
    ----------
    numerator : pd.Series
        Numerator series.
    denominator : pd.Series
        Denominator series.

    Returns
    -------
    pd.Series
        Resulting ratio series.
    """
    aligned_num, aligned_den = numerator.align(denominator, join="outer")
    safe_den = aligned_den.where(aligned_den != 0.0, np.nan)
    result = aligned_num / safe_den
    result.name = None
    return result


def safe_log(
    series: pd.Series,
) -> pd.Series:
    """
    Compute the natural log of a Series with nonpositive values mapped to NaN.

    Parameters
    ----------
    series : pd.Series
        Input series.

    Returns
    -------
    pd.Series
        Log-transformed series.
    """
    safe_series = series.where(series > 0.0, np.nan)
    result = np.log(safe_series)
    result.name = None
    return result


def compute_pct_change(
    series: pd.Series,
) -> pd.Series:
    """
    Compute percentage change with explicit no-fill behavior.

    Parameters
    ----------
    series : pd.Series
        Input series.

    Returns
    -------
    pd.Series
        Percentage-change series.
    """
    result = series.pct_change(fill_method=None)
    result.name = None
    return result


def compute_log_return(
    series: pd.Series,
) -> pd.Series:
    """
    Compute log returns for a positive-valued series.

    Nonpositive values are mapped to NaN before calculation.

    Parameters
    ----------
    series : pd.Series
        Input level series.

    Returns
    -------
    pd.Series
        Log-return series.
    """
    safe_series = series.where(series > 0.0, np.nan)
    result = np.log(safe_series / safe_series.shift(1))
    result.name = None
    return result


def rolling_mean(
    series: pd.Series,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """
    Compute a rolling mean.

    Parameters
    ----------
    series : pd.Series
        Input series.
    window : int
        Rolling window.
    min_periods : int, default 1
        Minimum observations required.

    Returns
    -------
    pd.Series
        Rolling mean series.
    """
    return series.rolling(window=window, min_periods=min_periods).mean()


def rolling_std(
    series: pd.Series,
    window: int,
    min_periods: int = 1,
    ddof: int = 0,
) -> pd.Series:
    """
    Compute a rolling standard deviation.

    Parameters
    ----------
    series : pd.Series
        Input series.
    window : int
        Rolling window.
    min_periods : int, default 1
        Minimum observations required.
    ddof : int, default 0
        Delta degrees of freedom.

    Returns
    -------
    pd.Series
        Rolling standard deviation series.
    """
    return series.rolling(window=window, min_periods=min_periods).std(ddof=ddof)


def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """
    Compute a rolling z-score.

    Parameters
    ----------
    series : pd.Series
        Input series.
    window : int
        Rolling lookback window.
    min_periods : int, default 1
        Minimum observations required.

    Returns
    -------
    pd.Series
        Rolling z-score series.
    """
    mean = rolling_mean(series=series, window=window, min_periods=min_periods)
    std = rolling_std(series=series, window=window, min_periods=min_periods, ddof=0)
    std = std.where(std > 0.0, np.nan)
    result = (series - mean) / std
    result.name = None
    return result


def rolling_drawdown(
    series: pd.Series,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """
    Compute rolling drawdown relative to the trailing rolling maximum.

    Parameters
    ----------
    series : pd.Series
        Input level series.
    window : int
        Rolling lookback window.
    min_periods : int, default 1
        Minimum observations required.

    Returns
    -------
    pd.Series
        Drawdown series in [-inf, 0].
    """
    rolling_max = series.rolling(window=window, min_periods=min_periods).max()
    rolling_max = rolling_max.where(rolling_max > 0.0, np.nan)
    result = (series / rolling_max) - 1.0
    result.name = None
    return result


def rowwise_mean(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """
    Compute a row-wise mean across selected columns.

    Missing columns are ignored. If none of the requested columns exist,
    an all-NaN series is returned.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str]
        Requested columns.

    Returns
    -------
    pd.Series
        Row-wise mean.
    """
    existing_columns = [col for col in columns if col in df.columns]
    if not existing_columns:
        return make_nan_series(df.index)
    result = df[existing_columns].mean(axis=1, skipna=True)
    result.name = None
    return result


def make_nan_series(
    index: pd.Index,
) -> pd.Series:
    """
    Create an all-NaN float Series aligned to an index.

    Parameters
    ----------
    index : pd.Index
        Target index.

    Returns
    -------
    pd.Series
        All-NaN series.
    """
    return pd.Series(np.nan, index=index, dtype=float)


def deduplicate_preserve_order(
    values: Iterable[str],
) -> list[str]:
    """
    Deduplicate an iterable of strings while preserving first occurrence order.

    Parameters
    ----------
    values : Iterable[str]
        Input sequence.

    Returns
    -------
    list[str]
        Deduplicated ordered list.
    """
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Raise an error if required columns are missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str]
        Required columns.

    Raises
    ------
    ValueError
        If one or more columns are missing.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def is_numeric_frame(
    df: pd.DataFrame,
) -> bool:
    """
    Check whether all columns in a DataFrame are numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    bool
        True if all columns are numeric, otherwise False.
    """
    return all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)


def coerce_numeric_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Coerce all DataFrame columns to numeric using errors='coerce'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Numeric-coerced copy.
    """
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out