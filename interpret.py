"""
Interpretation utilities for the Credit Spread Regime Model (CSRM).

This module builds the post-scoring interpretation layer:
- boolean flags
- component contribution diagnostics
- explanation columns for auditability

Design goals
------------
- deterministic
- transparent
- config-driven
- modular and testable
- no hidden narrative logic
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credit_spread_regime_model.config import build_config
from credit_spread_regime_model.scoring import build_scores, get_subscore_weights


def build_interpretation(
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build the full interpretation layer for CSRM.

    Parameters
    ----------
    data : pd.DataFrame
        Either:
        - raw input data,
        - transformed data,
        - feature-layer data, or
        - score-layer data.
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    pd.DataFrame
        Input data joined with flags, contributions, and explanation columns.
    """
    effective_config = build_config(config)
    df = _ensure_score_layer(data=data, config=effective_config)

    flags_frame = build_flag_frame(df=df, config=effective_config)
    contributions_frame = build_contribution_frame(df=df, config=effective_config)
    explanation_frame = build_explanation_frame(
        df=df,
        contributions_frame=contributions_frame,
        config=effective_config,
    )

    output = df.copy()
    output = output.join(flags_frame, how="left")
    output = output.join(contributions_frame, how="left")
    output = output.join(explanation_frame, how="left")

    return output


def build_flag_frame(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build boolean model flags.

    Parameters
    ----------
    df : pd.DataFrame
        Score-layer DataFrame.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.DataFrame
        DataFrame containing boolean flag columns.
    """
    flags_frame = pd.DataFrame(index=df.index)

    flags_frame["early_warning_flag"] = compute_early_warning_flag(
        df=df,
        config=config,
    )
    flags_frame["stress_acceleration_flag"] = compute_stress_acceleration_flag(
        df=df,
        config=config,
    )
    flags_frame["recovery_flag"] = compute_recovery_flag(
        df=df,
        config=config,
    )
    flags_frame["confirmed_risk_on_flag"] = compute_confirmed_risk_on_flag(
        df=df,
        config=config,
    )

    return flags_frame


def build_contribution_frame(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build weighted component contribution diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Score-layer DataFrame containing sub-score columns.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.DataFrame
        Contribution and ranking diagnostics for each timestamp.
    """
    weights = get_subscore_weights(config)
    subscore_names = list(weights.index)

    contributions = pd.DataFrame(index=df.index)

    for name in subscore_names:
        contributions[f"{name}_contribution"] = df[name] * weights[name]

    contribution_columns = [f"{name}_contribution" for name in subscore_names]

    contributions["positive_contribution_sum"] = (
        contributions[contribution_columns]
        .clip(lower=0.0)
        .sum(axis=1, skipna=True)
    )

    contributions["negative_contribution_sum"] = (
        contributions[contribution_columns]
        .clip(upper=0.0)
        .sum(axis=1, skipna=True)
    )

    contributions["dominant_positive_component"] = _rowwise_idxmax_label(
        contributions[contribution_columns]
    )
    contributions["dominant_negative_component"] = _rowwise_idxmin_label(
        contributions[contribution_columns]
    )

    contributions["contribution_dispersion"] = (
        contributions[contribution_columns]
        .std(axis=1, ddof=0)
    )

    return contributions


def build_explanation_frame(
    df: pd.DataFrame,
    contributions_frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build explicit explanation columns for downstream diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Score-layer DataFrame.
    contributions_frame : pd.DataFrame
        Contribution diagnostics.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.DataFrame
        Explanation-oriented columns with no hidden logic.
    """
    explanation = pd.DataFrame(index=df.index)

    explanation["regime_signal_direction"] = np.where(
        df["composite_score_clipped"] > 0.0,
        "Risk-On Bias",
        np.where(
            df["composite_score_clipped"] < 0.0,
            "Risk-Off Bias",
            "Neutral Bias",
        ),
    )

    explanation["regime_conviction"] = classify_regime_conviction(
        composite_score=df["composite_score_clipped"],
        config=config,
    )

    explanation["stress_state"] = classify_stress_state(
        stress_score=df["credit_stress_score"],
    )

    explanation["trend_state"] = classify_trend_state(
        trend_score=df["credit_trend_score"],
    )

    explanation["confirmation_state"] = classify_confirmation_state(
        confirmation_score=df["credit_confirmation_score"],
    )

    explanation["divergence_state"] = classify_divergence_state(
        divergence_score=df["credit_divergence_score"],
    )

    explanation["persistence_state"] = classify_persistence_state(
        persistence_score=df["credit_persistence_score"],
    )

    explanation["primary_driver"] = contributions_frame["dominant_positive_component"]
    explanation["primary_drag"] = contributions_frame["dominant_negative_component"]

    explanation["model_state_summary"] = build_model_state_summary(
        composite_score=df["composite_score_clipped"],
        trend_score=df["credit_trend_score"],
        stress_score=df["credit_stress_score"],
        confirmation_score=df["credit_confirmation_score"],
        divergence_score=df["credit_divergence_score"],
        persistence_score=df["credit_persistence_score"],
    )

    return explanation


def compute_early_warning_flag(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """
    Compute the early warning flag.

    Logic
    -----
    Trigger when:
    - composite score is mildly negative
    - trend score is negative
    - stress score is deteriorating / risk-off leaning
    """
    spec = config["flags"]["early_warning"]
    if not bool(spec["enabled"]):
        return pd.Series(False, index=df.index, dtype=bool)

    flag = (
        (df["composite_score_clipped"] <= float(spec["composite_max"]))
        & (df["credit_trend_score"] <= float(spec["trend_max"]))
        & (df["credit_stress_score"] <= -float(spec["stress_min"]))
    )
    return flag.fillna(False).astype(bool)


def compute_stress_acceleration_flag(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """
    Compute the stress acceleration flag.

    Logic
    -----
    Trigger when:
    - stress score worsens sharply
    - VIX trend also worsens
    """
    spec = config["flags"]["stress_acceleration"]
    if not bool(spec["enabled"]):
        return pd.Series(False, index=df.index, dtype=bool)

    stress_delta = (-df["credit_stress_score"]).diff()
    vix_delta = df["vix_trend_value"].diff()

    flag = (
        (stress_delta >= float(spec["stress_delta_min"]))
        & (vix_delta >= float(spec["vix_delta_min"]))
    )
    return flag.fillna(False).astype(bool)


def compute_recovery_flag(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """
    Compute the recovery flag.

    Logic
    -----
    Trigger when:
    - composite improves into positive territory
    - trend is positive
    - stress score is supportive
    """
    spec = config["flags"]["recovery"]
    if not bool(spec["enabled"]):
        return pd.Series(False, index=df.index, dtype=bool)

    flag = (
        (df["composite_score_clipped"] >= float(spec["composite_min"]))
        & (df["credit_trend_score"] >= float(spec["trend_min"]))
        & (df["credit_stress_score"] >= float(spec["stress_max"]))
    )
    return flag.fillna(False).astype(bool)


def compute_confirmed_risk_on_flag(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """
    Compute the confirmed risk-on flag.

    Logic
    -----
    Trigger when:
    - composite score is decisively positive
    - confirmation score is positive
    - persistence score is positive
    """
    spec = config["flags"]["confirmed_risk_on"]
    if not bool(spec["enabled"]):
        return pd.Series(False, index=df.index, dtype=bool)

    flag = (
        (df["composite_score_clipped"] >= float(spec["composite_min"]))
        & (df["credit_confirmation_score"] >= float(spec["confirmation_min"]))
        & (df["credit_persistence_score"] >= float(spec["persistence_min"]))
    )
    return flag.fillna(False).astype(bool)


def classify_regime_conviction(
    composite_score: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """
    Classify the strength of the current regime signal.

    Parameters
    ----------
    composite_score : pd.Series
        Composite score series.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.Series
        Categorical conviction label.
    """
    risk_on_min = float(config["regime"]["risk_on_min"])
    risk_off_max = float(config["regime"]["risk_off_max"])

    abs_score = composite_score.abs()
    threshold = max(abs(risk_on_min), abs(risk_off_max))

    conviction = pd.Series("Low", index=composite_score.index, dtype=object)
    conviction = conviction.where(abs_score < threshold, "Moderate")
    conviction = conviction.where(abs_score < threshold + 0.20, "High")
    conviction = conviction.where(composite_score.notna(), np.nan)

    return conviction


def classify_stress_state(
    stress_score: pd.Series,
) -> pd.Series:
    """
    Classify stress state from the stress sub-score.

    Higher stress score = less stressed.
    Lower stress score = more stressed.
    """
    state = pd.Series("Balanced", index=stress_score.index, dtype=object)
    state = state.where(stress_score >= -0.30, "Elevated Stress")
    state = state.where(stress_score <= 0.30, "Low Stress")
    state = state.where(stress_score.notna(), np.nan)
    return state


def classify_trend_state(
    trend_score: pd.Series,
) -> pd.Series:
    """
    Classify trend state from the trend sub-score.
    """
    state = pd.Series("Flat", index=trend_score.index, dtype=object)
    state = state.where(trend_score >= -0.30, "Negative Trend")
    state = state.where(trend_score <= 0.30, "Positive Trend")
    state = state.where(trend_score.notna(), np.nan)
    return state


def classify_confirmation_state(
    confirmation_score: pd.Series,
) -> pd.Series:
    """
    Classify cross-asset confirmation state.
    """
    state = pd.Series("Mixed", index=confirmation_score.index, dtype=object)
    state = state.where(confirmation_score >= -0.30, "Negative Confirmation")
    state = state.where(confirmation_score <= 0.30, "Positive Confirmation")
    state = state.where(confirmation_score.notna(), np.nan)
    return state


def classify_divergence_state(
    divergence_score: pd.Series,
) -> pd.Series:
    """
    Classify divergence state.

    Higher divergence feature becomes lower divergence score after inversion.
    Therefore:
    - low score => high divergence
    - high score => low divergence
    """
    state = pd.Series("Moderate Divergence", index=divergence_score.index, dtype=object)
    state = state.where(divergence_score >= -0.30, "High Divergence")
    state = state.where(divergence_score <= 0.30, "Low Divergence")
    state = state.where(divergence_score.notna(), np.nan)
    return state


def classify_persistence_state(
    persistence_score: pd.Series,
) -> pd.Series:
    """
    Classify persistence state.
    """
    state = pd.Series("Mixed Persistence", index=persistence_score.index, dtype=object)
    state = state.where(persistence_score >= -0.30, "Risk-Off Persistent")
    state = state.where(persistence_score <= 0.30, "Risk-On Persistent")
    state = state.where(persistence_score.notna(), np.nan)
    return state


def build_model_state_summary(
    composite_score: pd.Series,
    trend_score: pd.Series,
    stress_score: pd.Series,
    confirmation_score: pd.Series,
    divergence_score: pd.Series,
    persistence_score: pd.Series,
) -> pd.Series:
    """
    Build a compact deterministic summary string.

    Parameters
    ----------
    composite_score : pd.Series
        Composite score series.
    trend_score : pd.Series
        Trend sub-score series.
    stress_score : pd.Series
        Stress sub-score series.
    confirmation_score : pd.Series
        Confirmation sub-score series.
    divergence_score : pd.Series
        Divergence sub-score series.
    persistence_score : pd.Series
        Persistence sub-score series.

    Returns
    -------
    pd.Series
        Pipe-delimited summary string.
    """
    summary = pd.DataFrame(
        {
            "bias": np.where(
                composite_score > 0.0,
                "risk_on",
                np.where(composite_score < 0.0, "risk_off", "neutral"),
            ),
            "trend": np.where(
                trend_score > 0.15,
                "positive_trend",
                np.where(trend_score < -0.15, "negative_trend", "flat_trend"),
            ),
            "stress": np.where(
                stress_score > 0.15,
                "low_stress",
                np.where(stress_score < -0.15, "elevated_stress", "balanced_stress"),
            ),
            "confirm": np.where(
                confirmation_score > 0.15,
                "confirmed",
                np.where(confirmation_score < -0.15, "unconfirmed", "mixed_confirm"),
            ),
            "divergence": np.where(
                divergence_score > 0.15,
                "low_divergence",
                np.where(divergence_score < -0.15, "high_divergence", "moderate_divergence"),
            ),
            "persistence": np.where(
                persistence_score > 0.15,
                "risk_on_persistent",
                np.where(
                    persistence_score < -0.15,
                    "risk_off_persistent",
                    "mixed_persistence",
                ),
            ),
        },
        index=composite_score.index,
    )

    out = (
        summary["bias"]
        + " | "
        + summary["trend"]
        + " | "
        + summary["stress"]
        + " | "
        + summary["confirm"]
        + " | "
        + summary["divergence"]
        + " | "
        + summary["persistence"]
    )
    out = out.where(composite_score.notna(), np.nan)
    return out


def _ensure_score_layer(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Ensure score-layer columns exist.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the score layer.
    """
    required_score_columns = [
        "credit_trend_score",
        "credit_stress_score",
        "credit_confirmation_score",
        "credit_divergence_score",
        "credit_persistence_score",
        "composite_score",
        "composite_score_clipped",
        "regime",
        "regime_numeric",
    ]

    if all(col in data.columns for col in required_score_columns):
        return data.copy()

    return build_scores(data=data, config=config)


def _rowwise_idxmax_label(df: pd.DataFrame) -> pd.Series:
    """
    Return the label of the largest contribution per row.

    If all values are missing, returns NaN.
    """
    out = pd.Series(np.nan, index=df.index, dtype=object)
    valid_rows = df.notna().any(axis=1)
    out.loc[valid_rows] = df.loc[valid_rows].idxmax(axis=1)
    return out


def _rowwise_idxmin_label(df: pd.DataFrame) -> pd.Series:
    """
    Return the label of the smallest contribution per row.

    If all values are missing, returns NaN.
    """
    out = pd.Series(np.nan, index=df.index, dtype=object)
    valid_rows = df.notna().any(axis=1)
    out.loc[valid_rows] = df.loc[valid_rows].idxmin(axis=1)
    return out