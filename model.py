"""
Top-level model orchestration for the Credit Spread Regime Model (CSRM).

This module provides the package entrypoint for running the full model
pipeline from raw inputs to final interpreted outputs.

Pipeline stages
---------------
1. Validation
2. Transforms
3. Features
4. Scoring
5. Interpretation
6. Output assembly

Design goals
------------
- deterministic
- modular
- auditable
- config-driven
- no I/O or side effects
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from credit_spread_regime_model.config import build_config
from credit_spread_regime_model.features import build_features
from credit_spread_regime_model.interpret import build_interpretation
from credit_spread_regime_model.scoring import build_scores
from credit_spread_regime_model.schema import (
    get_feature_columns,
    get_final_output_columns,
    get_flag_columns,
    get_ratio_columns,
    get_subscore_columns,
)
from credit_spread_regime_model.transforms import build_transforms
from credit_spread_regime_model.validate import validate_inputs


def run_csrm(
    prices: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run the full Credit Spread Regime Model pipeline.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw input DataFrame containing required ETF and cross-asset columns
        and optional macro columns.
    config : dict[str, Any] | None, default None
        Optional config overrides merged on top of DEFAULT_CONFIG.

    Returns
    -------
    pd.DataFrame
        Final CSRM output DataFrame containing the configured combination of:
        raw inputs, ratios, features, sub-scores, flags, diagnostics, and
        final regime outputs.

    Notes
    -----
    This is the canonical package entrypoint intended for research pipelines,
    tests, notebooks, and production integrations.
    """
    effective_config = build_config(config)

    validated = run_validation_stage(prices=prices, config=effective_config)
    transformed = run_transform_stage(prices=validated, config=effective_config)
    featured = run_feature_stage(data=transformed, config=effective_config)
    scored = run_scoring_stage(data=featured, config=effective_config)
    interpreted = run_interpretation_stage(data=scored, config=effective_config)

    return assemble_output(df=interpreted, config=effective_config)


def run_validation_stage(
    prices: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run the validation stage.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw input DataFrame.
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    pd.DataFrame
        Validated input DataFrame.
    """
    effective_config = build_config(config)
    return validate_inputs(prices=prices, config=effective_config)


def run_transform_stage(
    prices: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run the transform stage.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw or validated input DataFrame.
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    pd.DataFrame
        DataFrame including the transform layer.
    """
    effective_config = build_config(config)
    return build_transforms(prices=prices, config=effective_config)


def run_feature_stage(
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run the feature stage.

    Parameters
    ----------
    data : pd.DataFrame
        Raw, transformed, or partially processed data.
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    pd.DataFrame
        DataFrame including the feature layer.
    """
    effective_config = build_config(config)
    return build_features(data=data, config=effective_config)


def run_scoring_stage(
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run the scoring stage.

    Parameters
    ----------
    data : pd.DataFrame
        Raw, transformed, featured, or partially processed data.
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    pd.DataFrame
        DataFrame including the scoring layer.
    """
    effective_config = build_config(config)
    return build_scores(data=data, config=effective_config)


def run_interpretation_stage(
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run the interpretation stage.

    Parameters
    ----------
    data : pd.DataFrame
        Raw, transformed, featured, scored, or partially processed data.
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    pd.DataFrame
        DataFrame including the interpretation layer.
    """
    effective_config = build_config(config)
    return build_interpretation(data=data, config=effective_config)


def assemble_output(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Assemble the final output DataFrame according to output configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Fully processed CSRM DataFrame.
    config : dict[str, Any] | None, default None
        Optional config overrides.

    Returns
    -------
    pd.DataFrame
        Final output DataFrame with configured column groups.

    Notes
    -----
    The returned DataFrame preserves index order and includes columns in a
    stable, deterministic order.
    """
    effective_config = build_config(config)

    output_cfg = effective_config["outputs"]
    input_columns = _get_input_columns_from_config(effective_config)
    ratio_columns = get_ratio_columns()
    feature_columns = get_feature_columns()
    subscore_columns = get_subscore_columns()
    final_output_columns = get_final_output_columns()
    flag_columns = get_flag_columns()
    diagnostic_columns = _get_diagnostic_columns(df)

    ordered_columns: list[str] = []

    if bool(output_cfg["include_raw_inputs"]):
        ordered_columns.extend([col for col in input_columns if col in df.columns])

    if bool(output_cfg["include_ratios"]):
        ordered_columns.extend([col for col in ratio_columns if col in df.columns])

    if bool(output_cfg["include_features"]):
        ordered_columns.extend([col for col in feature_columns if col in df.columns])

    if bool(output_cfg["include_subscores"]):
        ordered_columns.extend([col for col in subscore_columns if col in df.columns])

    ordered_columns.extend([col for col in final_output_columns if col in df.columns])

    if bool(output_cfg["include_flags"]):
        ordered_columns.extend([col for col in flag_columns if col in df.columns])

    if bool(output_cfg["include_diagnostics"]):
        ordered_columns.extend([col for col in diagnostic_columns if col in df.columns])

    ordered_columns = _deduplicate_preserve_order(ordered_columns)

    return df.loc[:, ordered_columns].copy()


def get_latest_snapshot(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Return the most recent row of a CSRM output DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        CSRM output DataFrame.

    Returns
    -------
    pd.Series
        Last available row.

    Raises
    ------
    ValueError
        If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Cannot extract latest snapshot from an empty DataFrame.")
    return df.iloc[-1].copy()


def _get_input_columns_from_config(
    config: dict[str, Any],
) -> list[str]:
    """
    Get canonical input columns from config.

    Parameters
    ----------
    config : dict[str, Any]
        Effective config.

    Returns
    -------
    list[str]
        Required and optional input column names.
    """
    required_columns = list(config["inputs"]["required_price_columns"])
    optional_columns = list(config["inputs"]["optional_macro_columns"])
    return required_columns + optional_columns


def _get_diagnostic_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Get interpretation and diagnostic columns that are not part of the
    canonical feature, sub-score, final output, or flag groups.

    Parameters
    ----------
    df : pd.DataFrame
        Fully processed CSRM DataFrame.

    Returns
    -------
    list[str]
        Ordered diagnostic column names.
    """
    canonical_columns = set(
        get_ratio_columns()
        + get_feature_columns()
        + get_subscore_columns()
        + get_final_output_columns()
        + get_flag_columns()
    )

    preferred_diagnostic_order = [
        "composite_score_clipped",
        "regime_numeric",
        "risk_on_signal",
        "risk_off_signal",
        "credit_trend_score_contribution",
        "credit_stress_score_contribution",
        "credit_confirmation_score_contribution",
        "credit_divergence_score_contribution",
        "credit_persistence_score_contribution",
        "positive_contribution_sum",
        "negative_contribution_sum",
        "dominant_positive_component",
        "dominant_negative_component",
        "contribution_dispersion",
        "regime_signal_direction",
        "regime_conviction",
        "stress_state",
        "trend_state",
        "confirmation_state",
        "divergence_state",
        "persistence_state",
        "primary_driver",
        "primary_drag",
        "model_state_summary",
    ]

    diagnostics = [
        col for col in preferred_diagnostic_order
        if col in df.columns and col not in canonical_columns
    ]

    return diagnostics


def _deduplicate_preserve_order(
    columns: list[str],
) -> list[str]:
    """
    Deduplicate a column list while preserving first occurrence order.

    Parameters
    ----------
    columns : list[str]
        Input column names.

    Returns
    -------
    list[str]
        Deduplicated column names.
    """
    seen: set[str] = set()
    ordered: list[str] = []

    for col in columns:
        if col not in seen:
            seen.add(col)
            ordered.append(col)

    return ordered