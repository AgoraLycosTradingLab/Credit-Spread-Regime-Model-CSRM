"""
Validation utilities for the Credit Spread Regime Model (CSRM).

This module validates and standardizes raw model inputs before they enter
the transform, feature, and scoring pipeline.

Design goals
------------
- deterministic
- explicit failure modes
- no hidden preprocessing
- modular and testable
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from credit_spread_regime_model.config import build_config
from credit_spread_regime_model.schema import get_default_schema


def validate_inputs(
    prices: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Validate and standardize the raw input DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw input DataFrame containing required ETF and cross-asset columns,
        plus optional macro columns.
    config : dict[str, Any] | None, default None
        Optional config overrides merged on top of DEFAULT_CONFIG.

    Returns
    -------
    pd.DataFrame
        Validated, standardized copy of the input DataFrame.

    Raises
    ------
    TypeError
        If prices is not a pandas DataFrame.
    ValueError
        If required columns are missing, if missingness exceeds limits, if
        nonpositive prices are found where forbidden, or if the final dataset
        is empty after cleaning.
    """
    effective_config = build_config(config)
    schema = get_default_schema()

    df = _ensure_dataframe(prices)
    df = _standardize_index(
        df=df,
        sort_index=effective_config["inputs"]["sort_index"],
        drop_duplicate_index=effective_config["inputs"]["drop_duplicate_index"],
        allow_duplicates=schema["index"]["allow_duplicates"],
    )

    df = _check_required_columns(
        df=df,
        required_columns=schema["prices"]["required_columns"],
    )

    df = _filter_allowed_columns(
        df=df,
        required_columns=schema["prices"]["required_columns"],
        optional_columns=schema["prices"]["optional_columns"],
        allow_extra_columns=effective_config["inputs"]["allow_extra_columns"],
    )

    if effective_config["validation"]["enforce_numeric"]:
        df = _coerce_numeric_columns(df)

    df = _check_numeric_columns(df)

    if effective_config["validation"]["drop_all_na_rows"]:
        df = _drop_all_na_rows(df)

    df = _check_missing_fraction(
        df=df,
        columns=schema["prices"]["required_columns"],
        max_missing_fraction=effective_config["validation"]["max_missing_fraction"],
    )

    if effective_config["validation"]["reject_nonpositive_prices"]:
        df = _check_strictly_positive(
            df=df,
            columns=schema["prices"]["strictly_positive_required"],
        )

    df = _check_min_history(
        df=df,
        min_history=effective_config["inputs"]["price_min_history"],
        required_columns=schema["prices"]["required_columns"],
    )

    return df


def _ensure_dataframe(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the input is a pandas DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Input object.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame.

    Raises
    ------
    TypeError
        If input is not a DataFrame.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    return prices.copy()


def _standardize_index(
    df: pd.DataFrame,
    sort_index: bool,
    drop_duplicate_index: bool,
    allow_duplicates: bool,
) -> pd.DataFrame:
    """
    Standardize and validate the DataFrame index.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    sort_index : bool
        Whether to sort the index ascending.
    drop_duplicate_index : bool
        Whether to drop duplicate index values, keeping the last observation.
    allow_duplicates : bool
        Whether duplicate index values are permitted after processing.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized index.

    Raises
    ------
    TypeError
        If the index cannot be converted to datetime.
    ValueError
        If duplicates remain when not allowed, or if the DataFrame becomes empty.
    """
    df = df.copy()

    try:
        df.index = pd.to_datetime(df.index)
    except Exception as exc:
        raise TypeError("Input index must be datetime-like or convertible to datetime.") from exc

    if sort_index:
        df = df.sort_index()

    if drop_duplicate_index:
        df = df.loc[~df.index.duplicated(keep="last")]

    if not allow_duplicates and df.index.has_duplicates:
        raise ValueError("Input index contains duplicate timestamps.")

    if df.empty:
        raise ValueError("Input DataFrame is empty after index standardization.")

    return df


def _check_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    """
    Validate that all required columns exist.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    required_columns : list[str]
        Required input columns.

    Returns
    -------
    pd.DataFrame
        Unmodified DataFrame.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _filter_allowed_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    optional_columns: list[str],
    allow_extra_columns: bool,
) -> pd.DataFrame:
    """
    Restrict columns to required and optional schema columns if configured.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    required_columns : list[str]
        Required input columns.
    optional_columns : list[str]
        Optional input columns.
    allow_extra_columns : bool
        If False, drops columns not in required + optional.

    Returns
    -------
    pd.DataFrame
        DataFrame with allowed columns if filtering is enabled.
    """
    if allow_extra_columns:
        return df.copy()

    allowed_columns = required_columns + optional_columns
    existing_allowed = [col for col in allowed_columns if col in df.columns]
    return df.loc[:, existing_allowed].copy()


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce all columns to numeric where possible.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns coerced to numeric using errors='coerce'.
    """
    converted = df.copy()
    for col in converted.columns:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")
    return converted


def _check_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all columns are numeric after coercion.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Unmodified DataFrame.

    Raises
    ------
    TypeError
        If one or more columns remain non-numeric.
    """
    non_numeric = [
        col for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise TypeError(f"Non-numeric columns detected after coercion: {non_numeric}")
    return df


def _drop_all_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that are entirely NaN across all columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with all-NaN rows removed.

    Raises
    ------
    ValueError
        If all rows are removed.
    """
    cleaned = df.dropna(how="all")
    if cleaned.empty:
        raise ValueError("Input DataFrame contains no usable rows after dropping all-NaN rows.")
    return cleaned


def _check_missing_fraction(
    df: pd.DataFrame,
    columns: list[str],
    max_missing_fraction: float,
) -> pd.DataFrame:
    """
    Validate missing-data fraction for required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str]
        Columns to inspect.
    max_missing_fraction : float
        Maximum allowed fraction of missing values per required column.

    Returns
    -------
    pd.DataFrame
        Unmodified DataFrame.

    Raises
    ------
    ValueError
        If any required column exceeds the allowed missing fraction.
    """
    violations: dict[str, float] = {}

    for col in columns:
        missing_fraction = float(df[col].isna().mean())
        if missing_fraction > max_missing_fraction:
            violations[col] = missing_fraction

    if violations:
        raise ValueError(
            "Required columns exceed maximum missing fraction: "
            f"{violations}"
        )

    return df


def _check_strictly_positive(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Enforce strictly positive values for specified columns where non-missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str]
        Columns that must be strictly positive.

    Returns
    -------
    pd.DataFrame
        Unmodified DataFrame.

    Raises
    ------
    ValueError
        If any non-missing value is less than or equal to zero.
    """
    violations: dict[str, int] = {}

    for col in columns:
        invalid_mask = df[col].notna() & (df[col] <= 0)
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            violations[col] = invalid_count

    if violations:
        raise ValueError(
            "Strictly positive price validation failed for columns: "
            f"{violations}"
        )

    return df


def _check_min_history(
    df: pd.DataFrame,
    min_history: int,
    required_columns: list[str],
) -> pd.DataFrame:
    """
    Validate that sufficient history exists for required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    min_history : int
        Minimum number of non-null observations required.
    required_columns : list[str]
        Columns that must satisfy the minimum history requirement.

    Returns
    -------
    pd.DataFrame
        Unmodified DataFrame.

    Raises
    ------
    ValueError
        If one or more required columns have insufficient valid history.
    """
    counts = df[required_columns].notna().sum(axis=0)
    violations = counts[counts < min_history]

    if not violations.empty:
        raise ValueError(
            "Insufficient history for required columns: "
            f"{violations.to_dict()} (min required: {min_history})"
        )

    return df