"""
Example runner for the Credit Spread Regime Model (CSRM).

This script demonstrates how to:
- download market proxy data with yfinance
- download optional macro data from FRED via fredapi
- format inputs for the CSRM package
- run the full model
- inspect the latest model snapshot

Execution
---------
PowerShell example:
    $env:PYTHONPATH = "src"
    $env:FRED_API_KEY = "YOUR_FRED_API_KEY"
    python examples\\run_csrm_example.py

Notes
-----
- This file is an executable example, not part of core model logic.
- Core package functions remain deterministic and side-effect free.
- FRED integration is optional. If no API key is supplied, the script runs
  with market data only.
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
import yfinance as yf

from credit_spread_regime_model import DEFAULT_CONFIG, run_csrm
from credit_spread_regime_model.model import get_latest_snapshot


YFINANCE_TICKER_MAP: dict[str, str] = {
    "HYG": "HYG",
    "JNK": "JNK",
    "LQD": "LQD",
    "SHY": "SHY",
    "IEF": "IEF",
    "TLT": "TLT",
    "SPY": "SPY",
    "VIX": "^VIX",
}

FRED_SERIES_MAP: dict[str, str] = {
    "HY_OAS": "BAMLH0A0HYM2",
    "IG_OAS": "BAMLC0A0CM",
    "NFCI": "NFCI",
    "ANFCI": "ANFCI",
}


def download_price_history(
    ticker_map: dict[str, str],
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Download close history for the configured market tickers.

    Parameters
    ----------
    ticker_map : dict[str, str]
        Mapping from internal CSRM column names to yfinance tickers.
    start_date : str, default "2018-01-01"
        Download start date.
    end_date : str | None, default None
        Optional download end date.
    auto_adjust : bool, default False
        Passed through to yfinance.download.

    Returns
    -------
    pd.DataFrame
        Price DataFrame indexed by date and using CSRM canonical column names.

    Raises
    ------
    ValueError
        If downloaded data is empty or required columns cannot be formed.
    """
    yf_tickers = list(ticker_map.values())

    raw = yf.download(
        tickers=yf_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )

    if raw.empty:
        raise ValueError("No price data was downloaded from yfinance.")

    prices = _extract_close_frame(raw=raw, ticker_map=ticker_map)

    if prices.empty:
        raise ValueError("Failed to construct a usable price DataFrame from yfinance output.")

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices = prices.sort_index()
    prices = prices.dropna(how="all")

    required_columns = list(ticker_map.keys())
    missing_columns = [col for col in required_columns if col not in prices.columns]
    if missing_columns:
        raise ValueError(f"Missing expected price columns after download: {missing_columns}")

    return prices


def _extract_close_frame(
    raw: pd.DataFrame,
    ticker_map: dict[str, str],
) -> pd.DataFrame:
    """
    Extract close-price data from yfinance output and rename columns to CSRM names.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw DataFrame returned by yfinance.download.
    ticker_map : dict[str, str]
        Mapping from internal CSRM names to downloaded ticker symbols.

    Returns
    -------
    pd.DataFrame
        Close-price DataFrame with canonical CSRM column names.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        level_0 = raw.columns.get_level_values(0)

        if "Close" in level_0:
            close_frame = raw["Close"].copy()
        elif "Adj Close" in level_0:
            close_frame = raw["Adj Close"].copy()
        else:
            raise ValueError("Could not find Close or Adj Close in yfinance output.")
    else:
        if "Close" in raw.columns:
            close_frame = raw[["Close"]].copy()
        elif "Adj Close" in raw.columns:
            close_frame = raw[["Adj Close"]].copy()
        else:
            raise ValueError("Could not find Close or Adj Close in yfinance output.")

        if len(ticker_map) != 1:
            raise ValueError("Single-ticker yfinance output received for a multi-ticker request.")

        only_internal_name = next(iter(ticker_map.keys()))
        close_frame.columns = [only_internal_name]
        return close_frame

    reverse_map = {external: internal for internal, external in ticker_map.items()}
    available_columns = [col for col in close_frame.columns if col in reverse_map]

    if not available_columns:
        raise ValueError("No expected tickers were present in the yfinance close-price output.")

    close_frame = close_frame.loc[:, available_columns].rename(columns=reverse_map)

    expected_order = [internal for internal in ticker_map.keys() if internal in close_frame.columns]
    close_frame = close_frame.loc[:, expected_order]

    return close_frame


def download_fred_macro_data(
    index: pd.Index,
    fred_api_key: str,
    fred_series_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Download FRED macro series and align them to the provided trading-day index.

    Parameters
    ----------
    index : pd.Index
        Target DateTimeIndex, typically the price DataFrame index.
    fred_api_key : str
        FRED API key.
    fred_series_map : dict[str, str] | None, default None
        Mapping from CSRM macro column names to FRED series IDs.

    Returns
    -------
    pd.DataFrame
        Macro DataFrame aligned to the provided index.

    Raises
    ------
    ImportError
        If fredapi is not installed.
    ValueError
        If the provided index is empty.
    """
    if len(index) == 0:
        raise ValueError("Cannot download FRED data for an empty index.")

    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError(
            "fredapi is required for FRED downloads. "
            "Install with: python -m pip install fredapi"
        ) from exc

    series_map = fred_series_map or FRED_SERIES_MAP
    fred = Fred(api_key=fred_api_key)

    start_date = pd.to_datetime(index.min()).date().isoformat()
    end_date = pd.to_datetime(index.max()).date().isoformat()

    macro_frames: list[pd.Series] = []

    for internal_name, fred_series_id in series_map.items():
        series = fred.get_series(
            fred_series_id,
            observation_start=start_date,
            observation_end=end_date,
        )

        series = pd.Series(series, name=internal_name)
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()
        macro_frames.append(series)

    if not macro_frames:
        return pd.DataFrame(index=index)

    macro = pd.concat(macro_frames, axis=1)
    macro = macro.sort_index()

    aligned = macro.reindex(index).ffill()
    aligned.index = pd.to_datetime(aligned.index)
    aligned.index.name = index.name

    return aligned


def add_fred_macro_data(
    prices: pd.DataFrame,
    fred_api_key: str | None,
    fred_series_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Append FRED macro series to the price DataFrame when an API key is provided.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame with CSRM canonical price columns.
    fred_api_key : str | None
        FRED API key. If None or empty, prices are returned unchanged.
    fred_series_map : dict[str, str] | None, default None
        Mapping from CSRM macro column names to FRED series IDs.

    Returns
    -------
    pd.DataFrame
        Combined price + macro DataFrame.
    """
    if fred_api_key is None or fred_api_key.strip() == "":
        return prices.copy()

    macro = download_fred_macro_data(
        index=prices.index,
        fred_api_key=fred_api_key,
        fred_series_map=fred_series_map,
    )

    combined = prices.join(macro, how="left")
    combined.index.name = prices.index.name

    return combined


def build_input_frame(
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    auto_adjust: bool = False,
    fred_api_key: str | None = None,
) -> pd.DataFrame:
    """
    Build the complete CSRM input frame from market and optional FRED data.

    Parameters
    ----------
    start_date : str, default "2018-01-01"
        Market-data start date.
    end_date : str | None, default None
        Optional market-data end date.
    auto_adjust : bool, default False
        Passed through to yfinance.download.
    fred_api_key : str | None, default None
        Optional FRED API key.

    Returns
    -------
    pd.DataFrame
        Full CSRM input DataFrame.
    """
    prices = download_price_history(
        ticker_map=YFINANCE_TICKER_MAP,
        start_date=start_date,
        end_date=end_date,
        auto_adjust=auto_adjust,
    )

    full_input = add_fred_macro_data(
        prices=prices,
        fred_api_key=fred_api_key,
        fred_series_map=FRED_SERIES_MAP,
    )

    return full_input


def select_output_columns(
    result: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """
    Select a subset of columns if they exist.

    Parameters
    ----------
    result : pd.DataFrame
        CSRM output DataFrame.
    columns : Iterable[str]
        Requested output columns.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only available requested columns.
    """
    keep = [col for col in columns if col in result.columns]
    return result.loc[:, keep].copy()


def print_run_metadata(
    input_df: pd.DataFrame,
    fred_api_key: str | None,
) -> None:
    """
    Print lightweight run metadata for the example script.

    Parameters
    ----------
    input_df : pd.DataFrame
        Final input DataFrame sent to the model.
    fred_api_key : str | None
        FRED API key used for the run.
    """
    has_fred = fred_api_key is not None and fred_api_key.strip() != ""
    macro_columns = [col for col in FRED_SERIES_MAP.keys() if col in input_df.columns]

    print("=== CSRM Run Metadata ===")
    print(f"Date range: {input_df.index.min().date()} to {input_df.index.max().date()}")
    print(f"Rows: {len(input_df)}")
    print(f"Using FRED macro data: {has_fred}")
    print(f"Macro columns present: {macro_columns}")
    print()


def main() -> None:
    """
    Run the example CSRM workflow with optional FRED integration.
    """
    fred_api_key = os.getenv("FRED_API_KEY")

    input_df = build_input_frame(
        start_date="2018-01-01",
        end_date=None,
        auto_adjust=False,
        fred_api_key=fred_api_key,
    )

    result = run_csrm(prices=input_df, config=DEFAULT_CONFIG)

    summary_columns = [
        "composite_score",
        "regime",
        "early_warning_flag",
        "stress_acceleration_flag",
        "recovery_flag",
        "confirmed_risk_on_flag",
        "regime_conviction",
        "model_state_summary",
    ]
    summary = select_output_columns(result=result, columns=summary_columns)
    latest = get_latest_snapshot(summary)

    print_run_metadata(input_df=input_df, fred_api_key=fred_api_key)

    print("=== CSRM Latest Snapshot ===")
    print(latest.to_string())
    print()
    print("=== CSRM Tail ===")
    print(summary.tail(10).to_string())


if __name__ == "__main__":
    main()