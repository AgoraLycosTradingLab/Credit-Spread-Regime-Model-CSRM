"""
Transform utilities for the Credit Spread Regime Model (CSRM).

This module constructs the canonical transformed inputs used by the
feature-engineering layer:
- credit proxy ratios
- optional log-ratios
- return series for ratios and selected raw inputs

Design goals
------------
- deterministic
- config-driven
- explicit column construction
- no hidden fill logic
- modular and testable
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credit_spread_regime_model.config import build_config
from credit_spread_regime_model.validate import validate_inputs


def build_transforms(
    prices: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build the transformed dataset for CSRM.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw input DataFrame containing required ETF and cross-asset columns,
        plus optional macro columns.
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - validated raw inputs
        - ratio columns
        - optional log-ratio columns
        - return columns for ratios and selected raw inputs
    """
    effective_config = build_config(config)
    df = validate_inputs(prices=prices, config=effective_config)

    transformed = df.copy()

    ratio_df = build_ratio_frame(
        df=transformed,
        ratio_config=effective_config["transforms"]["ratios"],
        log_ratio=effective_config["transforms"]["log_ratio"],
    )
    transformed = transformed.join(ratio_df, how="left")

    return_df = build_return_frame(
        df=transformed,
        ratio_columns=list(effective_config["transforms"]["ratios"].keys()),
        return_method=effective_config["transforms"]["return_method"],
    )
    transformed = transformed.join(return_df, how="left")

    return transformed


def build_ratio_frame(
    df: pd.DataFrame,
    ratio_config: dict[str, dict[str, str]],
    log_ratio: bool = False,
) -> pd.DataFrame:
    """
    Construct configured ratio columns.

    Parameters
    ----------
    df : pd.DataFrame
        Validated input DataFrame.
    ratio_config : dict[str, dict[str, str]]
        Mapping of ratio name to numerator/denominator column definitions.
        Example:
        {
            "hyg_ief": {"numerator": "HYG", "denominator": "IEF"},
            ...
        }
    log_ratio : bool, default False
        If True, also create log-transformed ratio columns with prefix 'log_'.

    Returns
    -------
    pd.DataFrame
        DataFrame of ratio columns and optional log-ratio columns.
    """
    ratio_frame = pd.DataFrame(index=df.index)

    for ratio_name, ratio_spec in ratio_config.items():
        numerator_col = ratio_spec["numerator"]
        denominator_col = ratio_spec["denominator"]

        ratio_series = compute_ratio(
            numerator=df[numerator_col],
            denominator=df[denominator_col],
        )
        ratio_frame[ratio_name] = ratio_series

        if log_ratio:
            ratio_frame[f"log_{ratio_name}"] = compute_log_ratio(ratio_series)

    return ratio_frame


def build_return_frame(
    df: pd.DataFrame,
    ratio_columns: list[str],
    return_method: str = "pct_change",
) -> pd.DataFrame:
    """
    Construct return columns for canonical ratios and selected raw assets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw inputs and ratio columns.
    ratio_columns : list[str]
        Ratio columns to convert into returns.
    return_method : str, default 'pct_change'
        Return method. Supported:
        - 'pct_change'
        - 'log_return'

    Returns
    -------
    pd.DataFrame
        DataFrame of return columns.

    Raises
    ------
    ValueError
        If return_method is unsupported.
    """
    asset_columns = ["HYG", "JNK", "LQD", "SHY", "IEF", "TLT", "SPY", "VIX"]
    available_asset_columns = [col for col in asset_columns if col in df.columns]
    available_ratio_columns = [col for col in ratio_columns if col in df.columns]

    columns_to_transform = available_asset_columns + available_ratio_columns
    return_frame = pd.DataFrame(index=df.index)

    for col in columns_to_transform:
        return_frame[f"{col.lower()}_ret"] = compute_return_series(
            series=df[col],
            method=return_method,
        )

    return return_frame


def compute_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Compute a ratio series with safe handling of invalid denominators.

    Parameters
    ----------
    numerator : pd.Series
        Numerator series.
    denominator : pd.Series
        Denominator series.

    Returns
    -------
    pd.Series
        Ratio series with division by zero handled as NaN.
    """
    aligned_num, aligned_den = numerator.align(denominator, join="outer")

    safe_den = aligned_den.where(aligned_den != 0.0, np.nan)
    ratio = aligned_num / safe_den
    ratio.name = None
    return ratio


def compute_log_ratio(ratio: pd.Series) -> pd.Series:
    """
    Compute the natural log of a ratio series.

    Parameters
    ----------
    ratio : pd.Series
        Ratio series.

    Returns
    -------
    pd.Series
        Log-ratio series where nonpositive values are converted to NaN.
    """
    safe_ratio = ratio.where(ratio > 0.0, np.nan)
    logged = np.log(safe_ratio)
    logged.name = None
    return logged


def compute_return_series(
    series: pd.Series,
    method: str = "pct_change",
) -> pd.Series:
    """
    Compute a return series from a price or ratio series.

    Parameters
    ----------
    series : pd.Series
        Input series.
    method : str, default 'pct_change'
        Return method. Supported:
        - 'pct_change'
        - 'log_return'

    Returns
    -------
    pd.Series
        Return series.

    Raises
    ------
    ValueError
        If method is unsupported.
    """
    if method == "pct_change":
        result = series.pct_change(fill_method=None)
        result.name = None
        return result

    if method == "log_return":
        safe_series = series.where(series > 0.0, np.nan)
        result = np.log(safe_series / safe_series.shift(1))
        result.name = None
        return result

    raise ValueError(
        f"Unsupported return_method: {method}. "
        "Supported methods are ['pct_change', 'log_return']."
    )


def get_ratio_columns(config: dict[str, Any] | None = None) -> list[str]:
    """
    Return the configured ratio column names.

    Parameters
    ----------
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    list[str]
        Configured ratio column names.
    """
    effective_config = build_config(config)
    return list(effective_config["transforms"]["ratios"].keys())


def get_return_columns(config: dict[str, Any] | None = None) -> list[str]:
    """
    Return the canonical return column names expected from the transform layer.

    Parameters
    ----------
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    list[str]
        Canonical return column names for ratios and core assets.
    """
    effective_config = build_config(config)

    asset_columns = ["HYG", "JNK", "LQD", "SHY", "IEF", "TLT", "SPY", "VIX"]
    ratio_columns = list(effective_config["transforms"]["ratios"].keys())

    return [f"{col.lower()}_ret" for col in asset_columns + ratio_columns]