"""
Schema definitions for the Credit Spread Regime Model (CSRM).

This module provides the canonical column contract used across the package.
It contains declarative schema metadata only; validation logic belongs in
validate.py.

Design goals
------------
- deterministic
- explicit
- centralized
- easy to reuse across modules
- no hidden column assumptions
"""

from __future__ import annotations

from typing import Any


PRICE_COLUMNS: list[str] = [
    "HYG",
    "JNK",
    "LQD",
    "SHY",
    "IEF",
    "TLT",
    "SPY",
    "VIX",
]

OPTIONAL_MACRO_COLUMNS: list[str] = [
    "HY_OAS",
    "IG_OAS",
    "NFCI",
    "ANFCI",
]

CANONICAL_INPUT_COLUMNS: list[str] = PRICE_COLUMNS + OPTIONAL_MACRO_COLUMNS


RATIO_COLUMNS: list[str] = [
    "hyg_ief",
    "lqd_ief",
    "hyg_lqd",
    "hyg_shy",
    "jnk_ief",
    "lqd_shy",
]


TREND_FEATURE_COLUMNS: list[str] = [
    "trend_hyg_ief",
    "trend_lqd_ief",
    "trend_hyg_lqd",
    "trend_hyg_shy",
    "trend_jnk_ief",
    "trend_lqd_shy",
    "credit_trend_feature",
]

STRESS_FEATURE_COLUMNS: list[str] = [
    "stress_hyg_ief",
    "stress_lqd_ief",
    "stress_hyg_lqd",
    "stress_hyg_shy",
    "stress_jnk_ief",
    "stress_lqd_shy",
    "credit_stress_feature",
    "hy_oas_stress",
    "ig_oas_stress",
    "nfci_stress",
    "anfci_stress",
]

CONFIRMATION_FEATURE_COLUMNS: list[str] = [
    "spy_trend_feature",
    "vix_level_feature",
    "vix_trend_feature",
    "credit_confirmation_feature",
]

DIVERGENCE_FEATURE_COLUMNS: list[str] = [
    "credit_equity_divergence",
    "hy_ig_divergence",
    "credit_divergence_feature",
]

PERSISTENCE_FEATURE_COLUMNS: list[str] = [
    "risk_on_persistence",
    "risk_off_persistence",
    "credit_persistence_feature",
]

VIX_FEATURE_COLUMNS: list[str] = [
    "vix_level_zscore",
    "vix_trend_value",
]

FEATURE_COLUMNS: list[str] = (
    TREND_FEATURE_COLUMNS
    + STRESS_FEATURE_COLUMNS
    + CONFIRMATION_FEATURE_COLUMNS
    + DIVERGENCE_FEATURE_COLUMNS
    + PERSISTENCE_FEATURE_COLUMNS
    + VIX_FEATURE_COLUMNS
)


SUBSCORE_COLUMNS: list[str] = [
    "credit_trend_score",
    "credit_stress_score",
    "credit_confirmation_score",
    "credit_divergence_score",
    "credit_persistence_score",
]

FINAL_OUTPUT_COLUMNS: list[str] = [
    "composite_score",
    "regime",
]

FLAG_COLUMNS: list[str] = [
    "early_warning_flag",
    "stress_acceleration_flag",
    "recovery_flag",
    "confirmed_risk_on_flag",
]

DIAGNOSTIC_COLUMNS: list[str] = [
    "composite_score_clipped",
    "regime_numeric",
    "risk_on_signal",
    "risk_off_signal",
]


COLUMN_GROUPS: dict[str, list[str]] = {
    "prices": PRICE_COLUMNS,
    "macro_optional": OPTIONAL_MACRO_COLUMNS,
    "canonical_inputs": CANONICAL_INPUT_COLUMNS,
    "ratios": RATIO_COLUMNS,
    "trend_features": TREND_FEATURE_COLUMNS,
    "stress_features": STRESS_FEATURE_COLUMNS,
    "confirmation_features": CONFIRMATION_FEATURE_COLUMNS,
    "divergence_features": DIVERGENCE_FEATURE_COLUMNS,
    "persistence_features": PERSISTENCE_FEATURE_COLUMNS,
    "vix_features": VIX_FEATURE_COLUMNS,
    "features": FEATURE_COLUMNS,
    "subscores": SUBSCORE_COLUMNS,
    "final_outputs": FINAL_OUTPUT_COLUMNS,
    "flags": FLAG_COLUMNS,
    "diagnostics": DIAGNOSTIC_COLUMNS,
}


DEFAULT_SCHEMA: dict[str, Any] = {
    "index": {
        "type": "datetime_like",
        "name": None,
        "monotonic_increasing": True,
        "allow_duplicates": False,
    },
    "prices": {
        "required_columns": PRICE_COLUMNS,
        "optional_columns": OPTIONAL_MACRO_COLUMNS,
        "dtype": "numeric",
        "strictly_positive_required": [
            "HYG",
            "JNK",
            "LQD",
            "SHY",
            "IEF",
            "TLT",
            "SPY",
            "VIX",
        ],
    },
    "transforms": {
        "ratio_columns": RATIO_COLUMNS,
    },
    "features": {
        "feature_columns": FEATURE_COLUMNS,
        "subscore_columns": SUBSCORE_COLUMNS,
    },
    "outputs": {
        "final_columns": FINAL_OUTPUT_COLUMNS,
        "flag_columns": FLAG_COLUMNS,
        "diagnostic_columns": DIAGNOSTIC_COLUMNS,
    },
}


def get_price_columns() -> list[str]:
    """
    Return required raw price columns.

    Returns
    -------
    list[str]
        Required ETF and cross-asset price columns.
    """
    return list(PRICE_COLUMNS)


def get_optional_macro_columns() -> list[str]:
    """
    Return optional macro columns.

    Returns
    -------
    list[str]
        Optional macro/FRED input columns.
    """
    return list(OPTIONAL_MACRO_COLUMNS)


def get_ratio_columns() -> list[str]:
    """
    Return canonical ratio column names.

    Returns
    -------
    list[str]
        Ratio columns expected from transforms.py.
    """
    return list(RATIO_COLUMNS)


def get_feature_columns() -> list[str]:
    """
    Return all feature column names.

    Returns
    -------
    list[str]
        Canonical feature columns across all feature families.
    """
    return list(FEATURE_COLUMNS)


def get_subscore_columns() -> list[str]:
    """
    Return sub-score column names.

    Returns
    -------
    list[str]
        Canonical sub-score columns used in scoring.py.
    """
    return list(SUBSCORE_COLUMNS)


def get_flag_columns() -> list[str]:
    """
    Return flag column names.

    Returns
    -------
    list[str]
        Canonical regime and stress flag columns.
    """
    return list(FLAG_COLUMNS)


def get_final_output_columns() -> list[str]:
    """
    Return final output columns.

    Returns
    -------
    list[str]
        Final top-level model output columns.
    """
    return list(FINAL_OUTPUT_COLUMNS)


def get_column_groups() -> dict[str, list[str]]:
    """
    Return all named column groups.

    Returns
    -------
    dict[str, list[str]]
        Mapping of schema group names to column lists.
    """
    return {key: list(value) for key, value in COLUMN_GROUPS.items()}


def get_default_schema() -> dict[str, Any]:
    """
    Return the default schema dictionary.

    Returns
    -------
    dict[str, Any]
        Declarative package schema.
    """
    return {
        "index": dict(DEFAULT_SCHEMA["index"]),
        "prices": {
            "required_columns": list(DEFAULT_SCHEMA["prices"]["required_columns"]),
            "optional_columns": list(DEFAULT_SCHEMA["prices"]["optional_columns"]),
            "dtype": DEFAULT_SCHEMA["prices"]["dtype"],
            "strictly_positive_required": list(
                DEFAULT_SCHEMA["prices"]["strictly_positive_required"]
            ),
        },
        "transforms": {
            "ratio_columns": list(DEFAULT_SCHEMA["transforms"]["ratio_columns"]),
        },
        "features": {
            "feature_columns": list(DEFAULT_SCHEMA["features"]["feature_columns"]),
            "subscore_columns": list(DEFAULT_SCHEMA["features"]["subscore_columns"]),
        },
        "outputs": {
            "final_columns": list(DEFAULT_SCHEMA["outputs"]["final_columns"]),
            "flag_columns": list(DEFAULT_SCHEMA["outputs"]["flag_columns"]),
            "diagnostic_columns": list(
                DEFAULT_SCHEMA["outputs"]["diagnostic_columns"]
            ),
        },
    }