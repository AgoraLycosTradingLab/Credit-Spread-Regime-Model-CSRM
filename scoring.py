"""
Scoring utilities for the Credit Spread Regime Model (CSRM).

This module converts the feature layer into:
- sub-scores
- composite score
- regime label
- regime numeric state

Design goals
------------
- deterministic
- config-driven
- interpretable
- modular and testable
- no hidden weighting logic
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credit_spread_regime_model.config import build_config
from credit_spread_regime_model.features import build_features


def build_scores(
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build the full scoring layer for CSRM.

    Parameters
    ----------
    data : pd.DataFrame
        Either:
        - raw input data,
        - transformed data, or
        - data already containing feature columns.
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    pd.DataFrame
        Input data joined with score columns, composite score, and regime fields.
    """
    effective_config = build_config(config)
    df = _ensure_feature_layer(data=data, config=effective_config)

    score_frame = pd.DataFrame(index=df.index)

    score_frame["credit_trend_score"] = score_trend_feature(
        feature=df["credit_trend_feature"],
        config=effective_config,
    )

    score_frame["credit_stress_score"] = score_stress_feature(
        feature=df["credit_stress_feature"],
        config=effective_config,
    )

    score_frame["credit_confirmation_score"] = score_confirmation_feature(
        feature=df["credit_confirmation_feature"],
        config=effective_config,
    )

    score_frame["credit_divergence_score"] = score_divergence_feature(
        feature=df["credit_divergence_feature"],
        config=effective_config,
    )

    score_frame["credit_persistence_score"] = score_persistence_feature(
        feature=df["credit_persistence_feature"],
        config=effective_config,
    )

    composite = compute_composite_score(
        score_frame=score_frame,
        config=effective_config,
    )

    score_frame["composite_score"] = composite
    score_frame["composite_score_clipped"] = composite.clip(
        lower=float(effective_config["scoring"]["composite_clip_lower"]),
        upper=float(effective_config["scoring"]["composite_clip_upper"]),
    )
    score_frame["regime"] = map_score_to_regime(
        composite_score=score_frame["composite_score_clipped"],
        config=effective_config,
    )
    score_frame["regime_numeric"] = map_regime_to_numeric(
        regime=score_frame["regime"],
        config=effective_config,
    )
    score_frame["risk_on_signal"] = (
        score_frame["composite_score_clipped"] > 0.0
    ).astype(float)
    score_frame["risk_off_signal"] = (
        score_frame["composite_score_clipped"] < 0.0
    ).astype(float)

    return df.join(score_frame, how="left")


def score_trend_feature(
    feature: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """
    Convert aggregate trend feature into a sub-score.

    Parameters
    ----------
    feature : pd.Series
        Aggregate trend feature.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.Series
        Trend sub-score in [-1, 1].
    """
    return _clip_to_subscore_bounds(
        series=feature,
        subscore_name="credit_trend_score",
        config=config,
    )


def score_stress_feature(
    feature: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """
    Convert aggregate stress feature into a sub-score.

    Parameters
    ----------
    feature : pd.Series
        Aggregate stress feature, where higher means more stress.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.Series
        Stress sub-score in [-1, 1], where higher means more risk-on.

    Notes
    -----
    Stress is inverted because the scoring convention is:
    positive score = risk-on contribution
    negative score = risk-off contribution
    """
    inverted = -feature
    return _clip_to_subscore_bounds(
        series=inverted,
        subscore_name="credit_stress_score",
        config=config,
    )


def score_confirmation_feature(
    feature: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """
    Convert aggregate confirmation feature into a sub-score.

    Parameters
    ----------
    feature : pd.Series
        Aggregate confirmation feature.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.Series
        Confirmation sub-score in [-1, 1].
    """
    return _clip_to_subscore_bounds(
        series=feature,
        subscore_name="credit_confirmation_score",
        config=config,
    )


def score_divergence_feature(
    feature: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """
    Convert aggregate divergence feature into a sub-score.

    Parameters
    ----------
    feature : pd.Series
        Aggregate divergence feature, where higher means more disagreement.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.Series
        Divergence sub-score in [-1, 1], where higher means more risk-on.

    Notes
    -----
    Divergence is a negative contributor to confidence / risk appetite,
    so it is inverted.
    """
    inverted = -feature
    return _clip_to_subscore_bounds(
        series=inverted,
        subscore_name="credit_divergence_score",
        config=config,
    )


def score_persistence_feature(
    feature: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """
    Convert aggregate persistence feature into a sub-score.

    Parameters
    ----------
    feature : pd.Series
        Aggregate persistence feature.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.Series
        Persistence sub-score in [-1, 1].
    """
    return _clip_to_subscore_bounds(
        series=feature,
        subscore_name="credit_persistence_score",
        config=config,
    )


def compute_composite_score(
    score_frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """
    Compute the final weighted composite score.

    Parameters
    ----------
    score_frame : pd.DataFrame
        DataFrame containing all configured sub-score columns.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.Series
        Composite score in the configured bounds.

    Notes
    -----
    Missing sub-scores are handled row-wise by dividing by the sum of the
    available weights for non-missing values when normalization is enabled.
    """
    subscore_config = config["scoring"]["subscores"]
    subscore_names = list(subscore_config.keys())

    weight_map = {
        name: float(subscore_config[name]["weight"])
        for name in subscore_names
    }

    weights = pd.Series(weight_map, dtype=float)

    weighted_scores = score_frame[subscore_names].mul(weights, axis=1)
    weighted_sum = weighted_scores.sum(axis=1, skipna=True)

    if bool(config["scoring"]["normalize_weights"]):
        valid_mask = score_frame[subscore_names].notna().astype(float)
        effective_weight_sum = valid_mask.mul(weights, axis=1).sum(axis=1)
        composite = weighted_sum / effective_weight_sum.replace(0.0, np.nan)
    else:
        composite = weighted_sum

    composite = composite.clip(
        lower=float(config["scoring"]["composite_clip_lower"]),
        upper=float(config["scoring"]["composite_clip_upper"]),
    )

    return composite


def map_score_to_regime(
    composite_score: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """
    Map the composite score to a categorical regime label.

    Parameters
    ----------
    composite_score : pd.Series
        Composite score series.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.Series
        Regime label series.
    """
    risk_on_min = float(config["regime"]["risk_on_min"])
    risk_off_max = float(config["regime"]["risk_off_max"])
    labels = config["regime"]["labels"]

    regime = pd.Series(labels["neutral"], index=composite_score.index, dtype=object)
    regime = regime.where(composite_score > risk_off_max, labels["risk_off"])
    regime = regime.where(composite_score < risk_on_min, labels["risk_on"])

    regime = regime.where(composite_score.notna(), np.nan)
    return regime


def map_regime_to_numeric(
    regime: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """
    Map regime labels to numeric state values.

    Parameters
    ----------
    regime : pd.Series
        Regime label series.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.Series
        Numeric regime state:
        -  1.0 for Risk-On
        -  0.0 for Neutral
        - -1.0 for Risk-Off
    """
    labels = config["regime"]["labels"]
    mapping = {
        labels["risk_on"]: 1.0,
        labels["neutral"]: 0.0,
        labels["risk_off"]: -1.0,
    }
    return regime.map(mapping).astype(float)


def get_subscore_weights(
    config: dict[str, Any] | None = None,
) -> pd.Series:
    """
    Return configured sub-score weights.

    Parameters
    ----------
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    pd.Series
        Weight series indexed by sub-score name.
    """
    effective_config = build_config(config)
    subscore_config = effective_config["scoring"]["subscores"]

    return pd.Series(
        {
            name: float(spec["weight"])
            for name, spec in subscore_config.items()
        },
        dtype=float,
    )


def _ensure_feature_layer(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Ensure aggregate feature columns exist.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the feature layer.
    """
    required_feature_columns = [
        "credit_trend_feature",
        "credit_stress_feature",
        "credit_confirmation_feature",
        "credit_divergence_feature",
        "credit_persistence_feature",
    ]

    if all(col in data.columns for col in required_feature_columns):
        return data.copy()

    return build_features(data=data, config=config)


def _clip_to_subscore_bounds(
    series: pd.Series,
    subscore_name: str,
    config: dict[str, Any],
) -> pd.Series:
    """
    Clip a sub-score series to its configured bounds.

    Parameters
    ----------
    series : pd.Series
        Input score-like series.
    subscore_name : str
        Name of the configured sub-score.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.Series
        Clipped sub-score series.
    """
    lower = float(config["scoring"]["subscores"][subscore_name]["clip_lower"])
    upper = float(config["scoring"]["subscores"][subscore_name]["clip_upper"])
    return series.clip(lower=lower, upper=upper)