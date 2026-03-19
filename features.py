"""
Feature engineering for the Credit Spread Regime Model (CSRM).

This module builds the model's interpretable feature layer on top of the
validated transform dataset.

Feature families
----------------
- Trend
- Stress
- Confirmation
- Divergence
- Persistence
- VIX

Design goals
------------
- deterministic
- config-driven
- explicit feature formulas
- modular and testable
- no hidden preprocessing
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credit_spread_regime_model.config import build_config
from credit_spread_regime_model.transforms import build_transforms, get_ratio_columns


def build_features(
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build the full CSRM feature layer.

    Parameters
    ----------
    data : pd.DataFrame
        Either:
        - raw validated input data with required price columns, or
        - transformed data already containing canonical ratio columns.
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    pd.DataFrame
        Input data joined with feature columns.

    Notes
    -----
    If canonical ratio columns are not present, the transform layer is
    constructed automatically.
    """
    effective_config = build_config(config)

    df = _ensure_transform_layer(data=data, config=effective_config)

    vix_frame = build_vix_features(df=df, config=effective_config)
    trend_frame = build_trend_features(df=df, config=effective_config)
    stress_frame = build_stress_features(df=df, config=effective_config)
    confirmation_frame = build_confirmation_features(
        df=df,
        vix_frame=vix_frame,
        config=effective_config,
    )
    divergence_frame = build_divergence_features(
        trend_frame=trend_frame,
        confirmation_frame=confirmation_frame,
        config=effective_config,
    )
    persistence_frame = build_persistence_features(
        trend_frame=trend_frame,
        stress_frame=stress_frame,
        config=effective_config,
    )

    feature_frames = [
        trend_frame,
        stress_frame,
        confirmation_frame,
        divergence_frame,
        persistence_frame,
        vix_frame,
    ]

    output = df.copy()
    for frame in feature_frames:
        output = output.join(frame, how="left")

    return output


def build_trend_features(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build ratio-based trend features.

    Parameters
    ----------
    df : pd.DataFrame
        Transformed dataset containing canonical ratio columns.
    config : dict[str, Any]
        Effective merged config.

    Returns
    -------
    pd.DataFrame
        Trend feature columns and aggregate trend feature.
    """
    ratio_columns = config["features"]["trend"]["ratio_columns"]
    fast_window = int(config["windows"]["trend_fast"])
    slow_window = int(config["windows"]["trend_slow"])
    lower = float(config["features"]["trend"]["clip_lower"])
    upper = float(config["features"]["trend"]["clip_upper"])

    trend_frame = pd.DataFrame(index=df.index)

    for ratio_col in ratio_columns:
        trend_col = f"trend_{ratio_col}"
        trend_frame[trend_col] = compute_trend_feature(
            series=df[ratio_col],
            fast_window=fast_window,
            slow_window=slow_window,
            lower=lower,
            upper=upper,
        )

    trend_columns = [f"trend_{col}" for col in ratio_columns]
    trend_frame["credit_trend_feature"] = _rowwise_mean(trend_frame, trend_columns)

    return trend_frame


def build_stress_features(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build stress features from ratio weakness, ratio drawdown, and optional
    macro stress series.

    Parameters
    ----------
    df : pd.DataFrame
        Transformed dataset containing canonical ratio columns.
    config : dict[str, Any]
        Effective merged config.

    Returns
    -------
    pd.DataFrame
        Stress feature columns and aggregate stress feature.

    Notes
    -----
    Higher stress values indicate more risk-off behavior.
    """
    ratio_columns = config["features"]["trend"]["ratio_columns"]
    z_window = int(config["windows"]["zscore"])
    drawdown_window = int(config["windows"]["drawdown"])
    zscore_clip = float(config["features"]["stress"]["zscore_clip"])
    drawdown_clip = float(config["features"]["stress"]["drawdown_clip"])
    oas_zscore_clip = float(config["features"]["stress"]["oas_zscore_clip"])
    nfci_zscore_clip = float(config["features"]["stress"]["nfci_zscore_clip"])
    use_oas_if_available = bool(config["features"]["stress"]["use_oas_if_available"])

    stress_frame = pd.DataFrame(index=df.index)

    for ratio_col in ratio_columns:
        stress_col = f"stress_{ratio_col}"
        ratio_stress = compute_ratio_stress_feature(
            series=df[ratio_col],
            z_window=z_window,
            drawdown_window=drawdown_window,
            zscore_clip=zscore_clip,
            drawdown_clip=drawdown_clip,
        )
        stress_frame[stress_col] = ratio_stress

    if use_oas_if_available:
        stress_frame["hy_oas_stress"] = compute_positive_zscore_feature(
            series=df["HY_OAS"] if "HY_OAS" in df.columns else _nan_series(df.index),
            window=z_window,
            zscore_clip=oas_zscore_clip,
        )
        stress_frame["ig_oas_stress"] = compute_positive_zscore_feature(
            series=df["IG_OAS"] if "IG_OAS" in df.columns else _nan_series(df.index),
            window=z_window,
            zscore_clip=oas_zscore_clip,
        )
    else:
        stress_frame["hy_oas_stress"] = _nan_series(df.index)
        stress_frame["ig_oas_stress"] = _nan_series(df.index)

    stress_frame["nfci_stress"] = compute_positive_zscore_feature(
        series=df["NFCI"] if "NFCI" in df.columns else _nan_series(df.index),
        window=z_window,
        zscore_clip=nfci_zscore_clip,
    )
    stress_frame["anfci_stress"] = compute_positive_zscore_feature(
        series=df["ANFCI"] if "ANFCI" in df.columns else _nan_series(df.index),
        window=z_window,
        zscore_clip=nfci_zscore_clip,
    )

    aggregate_cols = [
        *(f"stress_{col}" for col in ratio_columns),
        "hy_oas_stress",
        "ig_oas_stress",
        "nfci_stress",
        "anfci_stress",
    ]
    stress_frame["credit_stress_feature"] = _rowwise_mean(stress_frame, aggregate_cols)

    return stress_frame


def build_confirmation_features(
    df: pd.DataFrame,
    vix_frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build cross-asset confirmation features from SPY and VIX.

    Parameters
    ----------
    df : pd.DataFrame
        Transformed dataset.
    vix_frame : pd.DataFrame
        VIX feature frame from build_vix_features.
    config : dict[str, Any]
        Effective merged config.

    Returns
    -------
    pd.DataFrame
        Confirmation feature columns and aggregate confirmation feature.

    Notes
    -----
    Higher confirmation values indicate more risk-on confirmation.
    """
    fast_window = int(config["windows"]["trend_fast"])
    slow_window = int(config["windows"]["trend_slow"])
    lower = -1.0
    upper = 1.0

    confirmation_frame = pd.DataFrame(index=df.index)

    if bool(config["features"]["confirmation"]["use_spy_trend"]):
        confirmation_frame["spy_trend_feature"] = compute_trend_feature(
            series=df["SPY"],
            fast_window=fast_window,
            slow_window=slow_window,
            lower=lower,
            upper=upper,
        )
    else:
        confirmation_frame["spy_trend_feature"] = _nan_series(df.index)

    if bool(config["features"]["confirmation"]["use_vix_level"]):
        confirmation_frame["vix_level_feature"] = _clip_series(
            -vix_frame["vix_level_zscore"],
            lower=-1.0,
            upper=1.0,
        )
    else:
        confirmation_frame["vix_level_feature"] = _nan_series(df.index)

    if bool(config["features"]["confirmation"]["use_vix_trend"]):
        confirmation_frame["vix_trend_feature"] = _clip_series(
            -vix_frame["vix_trend_value"],
            lower=-1.0,
            upper=1.0,
        )
    else:
        confirmation_frame["vix_trend_feature"] = _nan_series(df.index)

    confirmation_cols = [
        "spy_trend_feature",
        "vix_level_feature",
        "vix_trend_feature",
    ]
    confirmation_frame["credit_confirmation_feature"] = _rowwise_mean(
        confirmation_frame,
        confirmation_cols,
    )

    return confirmation_frame


def build_divergence_features(
    trend_frame: pd.DataFrame,
    confirmation_frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build divergence features.

    Parameters
    ----------
    trend_frame : pd.DataFrame
        Trend feature frame.
    confirmation_frame : pd.DataFrame
        Confirmation feature frame.
    config : dict[str, Any]
        Effective merged config.

    Returns
    -------
    pd.DataFrame
        Divergence feature columns.

    Notes
    -----
    Higher divergence values indicate greater disagreement / instability and
    should be treated as a risk signal.
    """
    threshold = float(config["features"]["divergence"]["large_divergence_threshold"])

    divergence_frame = pd.DataFrame(index=trend_frame.index)

    divergence_frame["credit_equity_divergence"] = _clip_series(
        (
            trend_frame["credit_trend_feature"]
            - confirmation_frame["spy_trend_feature"]
        ).abs()
        / threshold,
        lower=0.0,
        upper=1.0,
    )

    divergence_frame["hy_ig_divergence"] = _clip_series(
        trend_frame["trend_hyg_lqd"].abs() / threshold,
        lower=0.0,
        upper=1.0,
    )

    divergence_frame["credit_divergence_feature"] = _rowwise_mean(
        divergence_frame,
        ["credit_equity_divergence", "hy_ig_divergence"],
    )

    return divergence_frame


def build_persistence_features(
    trend_frame: pd.DataFrame,
    stress_frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build persistence features from trailing risk-on vs risk-off dominance.

    Parameters
    ----------
    trend_frame : pd.DataFrame
        Trend feature frame.
    stress_frame : pd.DataFrame
        Stress feature frame.
    config : dict[str, Any]
        Effective merged config.

    Returns
    -------
    pd.DataFrame
        Persistence feature columns.

    Notes
    -----
    The persistence proxy is:
        state_proxy = credit_trend_feature - credit_stress_feature

    Positive values imply risk-on dominance.
    Negative values imply risk-off dominance.
    """
    lookback = int(config["features"]["persistence"]["lookback"])

    persistence_frame = pd.DataFrame(index=trend_frame.index)

    state_proxy = trend_frame["credit_trend_feature"] - stress_frame["credit_stress_feature"]

    risk_on_state = (state_proxy > 0.0).astype(float)
    risk_off_state = (state_proxy < 0.0).astype(float)

    persistence_frame["risk_on_persistence"] = risk_on_state.rolling(
        window=lookback,
        min_periods=1,
    ).mean()

    persistence_frame["risk_off_persistence"] = risk_off_state.rolling(
        window=lookback,
        min_periods=1,
    ).mean()

    persistence_frame["credit_persistence_feature"] = _clip_series(
        persistence_frame["risk_on_persistence"]
        - persistence_frame["risk_off_persistence"],
        lower=-1.0,
        upper=1.0,
    )

    return persistence_frame


def build_vix_features(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build VIX-specific feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Transformed dataset.
    config : dict[str, Any]
        Effective merged config.

    Returns
    -------
    pd.DataFrame
        VIX feature frame containing level z-score and trend value.
    """
    z_window = int(config["windows"]["zscore"])
    fast_window = int(config["windows"]["vix_trend"])
    slow_window = int(config["windows"]["trend_slow"])
    level_zscore_clip = float(config["features"]["vix"]["level_zscore_clip"])
    trend_clip = float(config["features"]["vix"]["trend_clip"])

    vix_frame = pd.DataFrame(index=df.index)

    vix_frame["vix_level_zscore"] = _clip_series(
        _rolling_zscore(df["VIX"], window=z_window) / level_zscore_clip,
        lower=-1.0,
        upper=1.0,
    )

    vix_frame["vix_trend_value"] = compute_trend_feature(
        series=df["VIX"],
        fast_window=fast_window,
        slow_window=slow_window,
        lower=-trend_clip,
        upper=trend_clip,
    )

    return vix_frame


def compute_trend_feature(
    series: pd.Series,
    fast_window: int,
    slow_window: int,
    lower: float = -1.0,
    upper: float = 1.0,
    normalization_scale: float = 0.05,
) -> pd.Series:
    """
    Compute a normalized moving-average trend feature.

    Parameters
    ----------
    series : pd.Series
        Input level series.
    fast_window : int
        Fast moving-average window.
    slow_window : int
        Slow moving-average window.
    lower : float, default -1.0
        Lower clip bound.
    upper : float, default 1.0
        Upper clip bound.
    normalization_scale : float, default 0.05
        Scale used to normalize the MA spread.

    Returns
    -------
    pd.Series
        Trend feature in approximately [-1, 1].

    Notes
    -----
    Formula:
        ((MA_fast / MA_slow) - 1) / normalization_scale
    """
    fast_ma = series.rolling(window=fast_window, min_periods=1).mean()
    slow_ma = series.rolling(window=slow_window, min_periods=1).mean()

    raw = ((fast_ma / slow_ma) - 1.0) / normalization_scale
    return _clip_series(raw, lower=lower, upper=upper)


def compute_ratio_stress_feature(
    series: pd.Series,
    z_window: int,
    drawdown_window: int,
    zscore_clip: float,
    drawdown_clip: float,
) -> pd.Series:
    """
    Compute a ratio stress feature.

    Parameters
    ----------
    series : pd.Series
        Ratio level series.
    z_window : int
        Rolling z-score window.
    drawdown_window : int
        Rolling drawdown window.
    zscore_clip : float
        Clip scale for the level z-score component.
    drawdown_clip : float
        Clip scale for the drawdown component.

    Returns
    -------
    pd.Series
        Stress feature where higher values indicate more stress.

    Notes
    -----
    Stress is modeled as the average of:
    - weakness in the ratio level (negative z-score)
    - rolling drawdown pressure
    """
    level_z = _rolling_zscore(series, window=z_window)
    drawdown = _rolling_drawdown(series, window=drawdown_window)

    z_component = _clip_series(-level_z / zscore_clip, lower=-1.0, upper=1.0)
    drawdown_component = _clip_series(
        (-drawdown) / drawdown_clip,
        lower=0.0,
        upper=1.0,
    )

    stress = 0.5 * z_component + 0.5 * drawdown_component
    return _clip_series(stress, lower=-1.0, upper=1.0)


def compute_positive_zscore_feature(
    series: pd.Series,
    window: int,
    zscore_clip: float,
) -> pd.Series:
    """
    Convert a level series into a positive stress-style z-score feature.

    Parameters
    ----------
    series : pd.Series
        Input level series.
    window : int
        Rolling z-score window.
    zscore_clip : float
        Scale used before clipping.

    Returns
    -------
    pd.Series
        Feature where higher values indicate greater stress.
    """
    z = _rolling_zscore(series, window=window)
    return _clip_series(z / zscore_clip, lower=-1.0, upper=1.0)


def _ensure_transform_layer(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Ensure canonical transform columns exist.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    config : dict[str, Any]
        Effective merged config.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the transform layer.
    """
    ratio_columns = get_ratio_columns(config)
    if all(col in data.columns for col in ratio_columns):
        return data.copy()
    return build_transforms(prices=data, config=config)


def _rolling_zscore(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """
    Compute a rolling z-score.

    Parameters
    ----------
    series : pd.Series
        Input series.
    window : int
        Rolling lookback window.

    Returns
    -------
    pd.Series
        Rolling z-score series.
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std(ddof=0)
    rolling_std = rolling_std.where(rolling_std > 0.0, np.nan)
    return (series - rolling_mean) / rolling_std


def _rolling_drawdown(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """
    Compute rolling drawdown relative to the trailing rolling maximum.

    Parameters
    ----------
    series : pd.Series
        Input level series.
    window : int
        Rolling lookback window.

    Returns
    -------
    pd.Series
        Drawdown series in [-inf, 0].
    """
    rolling_max = series.rolling(window=window, min_periods=1).max()
    rolling_max = rolling_max.where(rolling_max > 0.0, np.nan)
    return (series / rolling_max) - 1.0


def _rowwise_mean(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """
    Compute a row-wise mean across the specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str]
        Column names.

    Returns
    -------
    pd.Series
        Row-wise mean with skipna=True.
    """
    existing_columns = [col for col in columns if col in df.columns]
    if not existing_columns:
        return _nan_series(df.index)
    return df[existing_columns].mean(axis=1, skipna=True)


def _clip_series(
    series: pd.Series,
    lower: float,
    upper: float,
) -> pd.Series:
    """
    Clip a series to the provided bounds.

    Parameters
    ----------
    series : pd.Series
        Input series.
    lower : float
        Lower bound.
    upper : float
        Upper bound.

    Returns
    -------
    pd.Series
        Clipped series.
    """
    return series.clip(lower=lower, upper=upper)


def _nan_series(index: pd.Index) -> pd.Series:
    """
    Create an all-NaN series aligned to the provided index.

    Parameters
    ----------
    index : pd.Index
        Target index.

    Returns
    -------
    pd.Series
        NaN series.
    """
    return pd.Series(np.nan, index=index, dtype=float)